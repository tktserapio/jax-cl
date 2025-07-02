"""
Incremental Learning Trainer for CIFAR-100 Experiments
Handles all training logic, learning rate scheduling, model updates, 
checkpointing, and results tracking.
"""

import time
import os
import pickle
from typing import Dict, Any, Optional, List
from copy import deepcopy
from functools import partialmethod

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mlproj_manager.problems import CifarDataSet
from mlproj_manager.file_management.file_and_directory_management import (
    store_object_with_several_attempts,
)

@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Compute cross-entropy loss with support for both one-hot and integer labels."""
    if labels.ndim > 1:  # One-hot labels
        return optax.softmax_cross_entropy(logits, labels).mean()
    else:  # Integer labels
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Compute metrics with support for both one-hot and integer labels."""
    loss = cross_entropy_loss(logits, labels)
    
    # Handle both one-hot and integer labels for accuracy
    if labels.ndim > 1:  # One-hot labels
        accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    else:  # Integer labels
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    
    return {"loss": loss, "accuracy": accuracy}


class TrainState(train_state.TrainState):
    """Extended TrainState with batch statistics for batch normalization."""
    batch_stats: Dict[str, Any] = None


class IncrementalTrainer:
    """Handles training logic for incremental CIFAR-100 experiments."""
    
    def __init__(
        self,
        model,
        stepsize: float,
        momentum: float,
        weight_decay: float,
        noise_std: float = 0.0,
        use_cbp: bool = False,
        replacement_rate: float = 0.0,
        utility_function: str = "weight",
        maturity_threshold: int = 0,
        reset_head: bool = False,
        reset_network: bool = False,
        early_stopping: bool = False,
        results_dir: str = None,
        run_index: int = 0,
        verbose_print_fn=print,
        # Experiment constants
        num_epochs: int = 4000,
        num_classes: int = 100,
        class_increase_frequency: int = 200,
        batch_sizes: Dict[str, int] = None,
        image_dims: tuple = (32, 32, 3),
        random_seed: int = 42,
        device=None
    ):
        self.model = model
        self.stepsize = stepsize
        self.base_lr = stepsize
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.noise_std = noise_std
        self.perturb_weights_indicator = noise_std > 0.0
        
        # CBP parameters
        self.use_cbp = use_cbp
        self.replacement_rate = replacement_rate
        self.utility_function = utility_function
        self.maturity_threshold = maturity_threshold
        
        # Reset flags
        self.reset_head = reset_head
        self.reset_network = reset_network
        self.early_stopping = early_stopping
        
        # Experiment setup
        self.results_dir = results_dir
        self.run_index = run_index
        self._print = verbose_print_fn
        
        # Constants
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.class_increase_frequency = class_increase_frequency
        self.batch_sizes = batch_sizes or {"train": 90, "validation": 50, "test": 100}
        self.image_dims = image_dims
        self.device = device
        
        # Initialize random states
        self.key = jax.random.PRNGKey(random_seed)
        
        # Tracking variables
        self.current_epoch = 0
        self.current_num_classes = 5
        self.best_accuracy = 0.0
        self.best_params = {}
        
        # Data order
        self.all_classes = np.random.permutation(self.num_classes)
        
        # Initialize CBP if enabled
        self.resgnt = None
        self.current_features = None
        if self.use_cbp:
            self.resgnt = ResGnT(
                net=self.model, 
                hidden_activation="relu",
                replacement_rate=self.replacement_rate,
                decay_rate=0.99,
                util_type=self.utility_function,
                maturity_threshold=self.maturity_threshold,
                device=self.device
            )
            self.current_features = []
        
        # Initialize training state (will be set when training starts)
        self.state = None
        
        # Initialize results tracking
        self._initialize_results_tracking()
        
        # Checkpointing setup
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints") if self.results_dir else None
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency
        self.delete_old_checkpoints = True
        self.noise_std = noise_std
        self.use_cbp = use_cbp
        self.reset_head = reset_head
        self.reset_network = reset_network
        self.early_stopping = early_stopping
        self.results_dir = results_dir
        self.run_index = run_index
        self._print = verbose_print_fn
        
        # Training state
        self.state = None
        self.key = None
        self.current_epoch = 0
        self.current_num_classes = 5
        self.all_classes = None
        
        # CBP related
        self.current_features = [] if use_cbp else None
        self.resgnt = None
        
        # Best model tracking
        self.best_accuracy = 0.0
        self.best_params = {}
        
        # Constants
        self.class_increase_frequency = 200
        self.running_avg_window = 25
        self.image_dims = (32, 32, 3)
        self.num_epochs = 4000
        self.num_classes = 100
        self.num_images_per_class = 450
        
        # Running averages
        self.current_running_avg_step = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        
        # Results tracking
        self.results_dict = {}
        self._initialize_results()
        
        # Checkpointing
        if self.results_dir:
            self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
            self.checkpoint_save_frequency = self.class_increase_frequency
            self.delete_old_checkpoints = True
        
        # Create optimizer
        self.optim = optax.sgd(learning_rate=self.stepsize, momentum=self.momentum)
        
    def _initialize_results(self):
        """Initialize results tracking dictionaries."""
        number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
        class_increase = 5
        number_of_image_per_task = self.num_images_per_class * class_increase
        
        batch_sizes = {"train": 90, "validation": 50, "test": 100}
        bin_size = (self.running_avg_window * batch_sizes["train"])
        total_checkpoints = int(np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size))

        # Use numpy arrays instead of torch tensors for JAX compatibility
        train_prototype_array = np.zeros(total_checkpoints, dtype=np.float32)
        self.results_dict["train_loss_per_checkpoint"] = np.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = np.zeros_like(train_prototype_array)

        prototype_array = np.zeros(self.num_epochs, dtype=np.float32)
        self.results_dict["epoch_runtime"] = np.zeros_like(prototype_array)
        
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = np.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = np.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = np.zeros_like(prototype_array)
        
        # Will be set when all_classes is available
        self.results_dict["class_order"] = None
        
    def _initialize_results_tracking(self):
        """Initialize results tracking dictionaries."""
        self.results_dict = {}
        
        # Training summaries setup
        self.running_avg_window = 25
        self.current_running_avg_step = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        
        # Initialize result arrays
        self._initialize_summaries()
    
    def _initialize_summaries(self):
        """Initialize the summaries for the experiment."""
        number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
        class_increase = 5
        number_of_image_per_task = 450 * class_increase  # num_images_per_class * class_increase
        
        bin_size = (self.running_avg_window * self.batch_sizes["train"])
        total_checkpoints = int(np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size))

        # Use numpy arrays instead of torch tensors
        train_prototype_array = np.zeros(total_checkpoints, dtype=np.float32)
        self.results_dict["train_loss_per_checkpoint"] = np.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = np.zeros_like(train_prototype_array)

        prototype_array = np.zeros(self.num_epochs, dtype=np.float32)
        self.results_dict["epoch_runtime"] = np.zeros_like(prototype_array)
        
        # Test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = np.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = np.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = np.zeros_like(prototype_array)
        self.results_dict["class_order"] = self.all_classes

    def _store_training_summaries(self):
        """Store training summaries matching PyTorch/NNX behavior."""
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window
        
        self._print(f"\t\tOnline accuracy: {self.running_accuracy / self.running_avg_window:.2f}")
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_dl, val_dl, epoch_number: int, epoch_runtime: float):
        """Store test summaries matching PyTorch/NNX behavior."""
        
        self.results_dict["epoch_runtime"][epoch_number] += epoch_runtime

        for data_name, data_loader, compare_to_best in [("test", test_dl, False), ("validation", val_dl, True)]:
            evaluation_start_time = time.perf_counter()
            loss, accuracy = self.evaluate_network(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_params = deepcopy(self.state.params)

            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] += evaluation_time
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] += float(loss)
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] += float(accuracy)

            self._print(f"\t\t{data_name} accuracy: {accuracy:.4f}")

        self._print(f"\t\tEpoch run time in seconds: {epoch_runtime:.4f}")

    def initialize_training_state(self, key: jax.random.PRNGKey, all_classes: np.ndarray):
        """Initialize the training state with model parameters."""
        self.key = key
        self.all_classes = all_classes
        
        # Initialize model
        dummy = jnp.ones((1, *self.image_dims), jnp.float32)
        variables = self.model.init(key, dummy, feature_list=None, train=True)
        params, batch_stats = variables["params"], variables.get("batch_stats", {})
        
        # Create training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optim
        )
        # Add batch stats manually since TrainState doesn't include them by default
        self.state = self.state.replace(batch_stats=batch_stats)
        
    def set_cbp_module(self, resgnt):
        """Set the CBP (ResGnT) module."""
        self.resgnt = resgnt
        
    @staticmethod
    @jax.jit
    def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute cross-entropy loss handling both one-hot and integer labels."""
        if labels.ndim > 1:  # One-hot labels
            return optax.softmax_cross_entropy(logits, labels).mean()
        else:  # Integer labels
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

    @staticmethod
    def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute loss and accuracy metrics."""
        loss = IncrementalTrainer.cross_entropy_loss(logits, labels)
        
        # Handle both one-hot and integer labels for accuracy
        if labels.ndim > 1:  # One-hot labels
            accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
        else:  # Integer labels
            accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        
        return {"loss": loss, "accuracy": accuracy}
    
    def _current_lr(self, epoch: int) -> Optional[float]:
        """Piece-wise decay identical to original PyTorch logic."""
        mod = epoch % self.class_increase_frequency
        if mod == 0:
            factor = 1.0
            if mod == 0 and epoch > 0:  # Task boundary
                self._print(f"\tTask boundary: LR reset to {self.base_lr:.5f}")
        elif mod == 60:
            factor = 0.2
        elif mod == 120:
            factor = 0.2 ** 2
        elif mod == 160:
            factor = 0.2 ** 3
        else:
            return None  # keep previous
        return self.base_lr * factor
    
    def _inject_noise(self, params, key):
        """Inject noise into model parameters if enabled."""
        if self.noise_std <= 0.0:
            return params
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(leaves) + 1)
        new_leaves = [
            leaf + self.noise_std * jax.random.normal(k, leaf.shape, leaf.dtype) 
            for leaf, k in zip(leaves, keys[:-1])
        ]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)
    
    def _reset_head(self):
        """Reset final classification layer."""
        # Reinitialize final layer with new random key
        self.key, sub = jax.random.split(self.key)
        # Implementation depends on your model structure
        # This is a placeholder - adjust based on your ResNet structure
        self._print("Head reset - implementation needed based on model structure")

    def _reset_network(self):
        """Reset entire network."""
        self.key, sub = jax.random.split(self.key)
        dummy = jnp.ones((1, *self.image_dims), jnp.float32)
        variables = self.model.init(sub, dummy, feature_list=None, train=True)
        params, batch_stats = variables["params"], variables.get("batch_stats", {})
        self.state = self.state.replace(params=params, batch_stats=batch_stats)
        self._print("Network reset completed")
    
    def create_jit_functions(self):
        """Create JIT-compiled training and evaluation functions."""
        
        @jax.jit
        def train_step_jit(state, images, labels, rng, lr_scale):
            """JIT-compiled training step function."""
            def loss_fn(params):
                variables = {"params": params, "batch_stats": state.batch_stats}
                (logits, new_batch_stats) = self.model.apply(
                    variables,
                    images,
                    feature_list=self.current_features if self.use_cbp else None,
                    train=True,
                    rngs={"dropout": rng},
                    mutable=["batch_stats"],
                )
                new_stats = new_batch_stats["batch_stats"]
                
                # Select only current classes for both logits and labels
                current_logits = logits[:, self.all_classes[:self.current_num_classes]]
                if labels.ndim > 1:  # One-hot labels
                    current_labels = labels[:, self.all_classes[:self.current_num_classes]]
                else:  # Integer labels
                    current_labels = labels
                
                loss = self.cross_entropy_loss(current_logits, current_labels)
                return loss, (current_logits, new_stats)

            (loss, (logits, new_stats)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            # Scale gradients
            grads = jax.tree_util.tree_map(lambda g: g * lr_scale, grads)
            state = state.apply_gradients(grads=grads, batch_stats=new_stats)
            
            # Compute metrics on current classes only
            if labels.ndim > 1:  # One-hot labels
                current_labels = labels[:, self.all_classes[:self.current_num_classes]]
            else:
                current_labels = labels
            metrics = self.compute_metrics(logits, current_labels)
            return state, metrics

        @jax.jit
        def eval_step_jit(state, images, labels):
            """JIT-compiled evaluation step function."""
            variables = {"params": state.params, "batch_stats": state.batch_stats}
            logits = self.model.apply(variables, images, feature_list=None, train=False, mutable=False)
            
            # Select only current classes for both logits and labels
            current_logits = logits[:, self.all_classes[:self.current_num_classes]]
            if labels.ndim > 1:  # One-hot labels
                current_labels = labels[:, self.all_classes[:self.current_num_classes]]
            else:
                current_labels = labels
                
            return self.compute_metrics(current_logits, current_labels)
        
        return train_step_jit, eval_step_jit
    
    def evaluate_network(self, test_dl: DataLoader, eval_step_jit):
        """Evaluate network on test data."""
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in test_dl:
            images = jnp.asarray(batch["image"].numpy())
            labels = jnp.asarray(batch["label"].numpy())
            
            metrics = eval_step_jit(self.state, images, labels)
            total_loss += float(metrics["loss"])
            total_acc += float(metrics["accuracy"])
            num_batches += 1

        return total_loss / num_batches, total_acc / num_batches
    
    def train_epoch(
        self, 
        train_dl: DataLoader, 
        test_dl: DataLoader, 
        val_dl: DataLoader,
        epoch: int,
        summary_callbacks: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Train for one epoch and return metrics."""
        
        # Create JIT functions
        train_step_jit, eval_step_jit = self.create_jit_functions()
        
        # Learning rate scheduling
        lr_candidate = self._current_lr(epoch)
        if lr_candidate is not None:
            self._print(f"\tCurrent stepsize: {lr_candidate:.5f}")
        current_lr = lr_candidate if lr_candidate is not None else self.base_lr

        epoch_start_time = time.perf_counter()
        
        for step, batch in enumerate(train_dl):
            # Convert to JAX arrays
            images = jnp.asarray(batch["image"].numpy())
            labels = jnp.asarray(batch["label"].numpy())

            # CBP feature extraction if enabled
            if self.use_cbp:
                # Clear features from previous step
                self.current_features = []
                
                # Forward pass for feature extraction (no gradients)
                variables = {"params": self.state.params, "batch_stats": self.state.batch_stats}
                _ = self.model.apply(
                    variables,
                    images,
                    feature_list=self.current_features,
                    train=True,
                    mutable=False  # Don't update batch stats in feature extraction
                )

            # RNG split
            self.key, sub = jax.random.split(self.key)

            # Training step
            self.state, metrics = train_step_jit(self.state, images, labels, sub, current_lr)

            # CBP processing if enabled
            if self.use_cbp and self.current_features and self.resgnt:
                try:
                    self.resgnt.gen_and_test(self.current_features)
                except Exception as cbp_error:
                    self._print(f"DEBUG Linen CBP ERROR: {cbp_error}")
                    raise cbp_error

            # Optional noise injection
            if self.noise_std > 0.0:
                self.key, nkey = jax.random.split(self.key)
                new_params = self._inject_noise(self.state.params, nkey)
                self.state = self.state.replace(params=new_params)

            # Accumulate for running averages
            self.running_loss += float(metrics["loss"])
            self.running_accuracy += float(metrics["accuracy"])
            
            # Store summaries every running_avg_window steps
            if (step + 1) % self.running_avg_window == 0:
                self._print(f"\t\tStep Number: {step + 1}")
                if summary_callbacks and "training" in summary_callbacks:
                    summary_callbacks["training"](self.running_loss, self.running_accuracy)
                self.running_loss = 0.0
                self.running_accuracy = 0.0
                self.current_running_avg_step += 1

        epoch_end_time = time.perf_counter()
        epoch_runtime = epoch_end_time - epoch_start_time
        
        # Evaluate on validation and test sets
        val_loss, val_accuracy = self.evaluate_network(val_dl, eval_step_jit)
        test_loss, test_accuracy = self.evaluate_network(test_dl, eval_step_jit)
        
        # Update best model if validation accuracy improved
        if val_accuracy > self.best_accuracy:
            self.best_accuracy = val_accuracy
            self.best_params = deepcopy(self.state.params)
        
        # Call summary callbacks
        if summary_callbacks and "evaluation" in summary_callbacks:
            summary_callbacks["evaluation"](
                test_loss, test_accuracy, val_loss, val_accuracy, epoch_runtime, epoch
            )
        
        self._print(f"\t\tvalidation accuracy: {val_accuracy:.4f}")
        self._print(f"\t\ttest accuracy: {test_accuracy:.4f}")
        self._print(f"\t\tEpoch run time in seconds: {epoch_runtime:.4f}")
        
        self.current_epoch += 1
        
        return {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_runtime": epoch_runtime
        }
    
    def should_add_classes(self) -> bool:
        """Check if it's time to add new classes."""
        return self.current_epoch % self.class_increase_frequency == 0
    
    def extend_classes(self) -> int:
        """Add new classes at task boundaries. Returns new class count."""
        if self.should_add_classes():
            self._print(f"Best accuracy in the task: {self.best_accuracy:.4f}")
            
            if self.early_stopping and self.best_params:
                self.state = self.state.replace(params=self.best_params)
            
            # Reset best accuracy tracking for new task
            self.best_accuracy = 0.0
            self.best_params = {}

            if self.current_num_classes >= 100:  # Max classes reached
                return self.current_num_classes

            increase = 5
            self.current_num_classes += increase
            
            self._print("\tNew class added...")
            self._print(f"\tCurrent classes: {self.current_num_classes}/100")
            
            if self.reset_head:
                self._reset_head()
            if self.reset_network:
                self._reset_network()
                
        return self.current_num_classes
    
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state for checkpointing."""
        return {
            "model_state": self.state.params,
            "batch_stats": self.state.batch_stats,
            "optim_state": self.state.opt_state,
            "jax_rng_state": self.key,
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "best_accuracy": self.best_accuracy,
            "best_params": self.best_params,
            "resgnt": self.resgnt if self.use_cbp else None
        }
    
    def load_training_state(self, state_dict: Dict[str, Any]):
        """Load training state from checkpoint."""
        self.state = self.state.replace(
            params=state_dict["model_state"],
            batch_stats=state_dict["batch_stats"],
            opt_state=state_dict["optim_state"]
        )
        self.key = state_dict["jax_rng_state"]
        self.current_epoch = state_dict["epoch_number"]
        self.current_num_classes = state_dict["current_num_classes"]
        self.current_running_avg_step = state_dict["current_running_avg_step"]
        self.best_accuracy = state_dict.get("best_accuracy", 0.0)
        self.best_params = state_dict.get("best_params", {})
        
        if self.use_cbp and "resgnt" in state_dict and state_dict["resgnt"]:
            self.resgnt = state_dict["resgnt"]
    
    def train(self, train_dl: DataLoader, test_dl: DataLoader, val_dl: DataLoader,
              train_ds: CifarDataSet, test_ds: CifarDataSet, val_ds: CifarDataSet):
        """Main training loop for the incremental learning experiment."""
        
        # Initialize data partitions
        train_ds.select_new_partition(self.all_classes[: self.current_num_classes])
        test_ds.select_new_partition(self.all_classes[: self.current_num_classes])
        val_ds.select_new_partition(self.all_classes[: self.current_num_classes])
        self._save_model_parameters()

        # Create JIT-compiled functions
        train_step_jit, eval_step_jit = self.create_jit_functions()

        for epoch in tqdm(range(self.current_epoch, self.num_epochs)):
            self._print(f"\tEpoch number: {epoch + 1}")
            
            # Learning rate scheduling
            lr_candidate = self._current_lr(epoch)
            if lr_candidate is not None:
                self._print(f"\tCurrent stepsize: {lr_candidate:.5f}")
            current_lr = lr_candidate if lr_candidate is not None else self.base_lr

            epoch_start_time = time.perf_counter()
            
            for step, batch in enumerate(train_dl):
                # Convert to JAX arrays
                images = jnp.asarray(batch["image"].numpy())
                labels = jnp.asarray(batch["label"].numpy())

                # CBP feature extraction if enabled
                if self.use_cbp:
                    # Clear features from previous step
                    self.current_features = []
                    
                    # Forward pass for feature extraction (no gradients)
                    variables = {"params": self.state.params, "batch_stats": self.state.batch_stats}
                    _ = self.model.apply(
                        variables,
                        images,
                        feature_list=self.current_features,
                        train=True,
                        mutable=False  # Don't update batch stats in feature extraction
                    )
                    
                    # Debug output for first step
                    if step == 0 and epoch == self.current_epoch:
                        self._print(f"DEBUG Linen CBP: Number of feature layers: {len(self.current_features)}")
                        for i, feat in enumerate(self.current_features):
                            self._print(f"DEBUG Linen CBP: Feature {i} shape: {feat.shape}")

                # RNG split
                self.key, sub = jax.random.split(self.key)

                # Training step
                self.state, metrics = train_step_jit(self.state, images, labels, sub, current_lr)

                # CBP processing if enabled
                if self.use_cbp and self.current_features:
                    try:
                        self.resgnt.gen_and_test(self.current_features)
                    except Exception as cbp_error:
                        self._print(f"DEBUG Linen CBP ERROR: {cbp_error}")
                        self._print(f"  Feature shapes: {[f.shape for f in self.current_features]}")
                        if hasattr(self.resgnt, 'weight_layers'):
                            self._print(f"  ResGnT weight layers: {len(self.resgnt.weight_layers)}")
                        raise cbp_error

                # Optional noise injection
                if self.perturb_weights_indicator:
                    self.key, nkey = jax.random.split(self.key)
                    new_params = self._inject_noise(self.state.params, nkey)
                    self.state = self.state.replace(params=new_params)

                # Accumulate for running averages
                self.running_loss += float(metrics["loss"])
                self.running_accuracy += float(metrics["accuracy"])
                
                # Store summaries every running_avg_window steps
                if (step + 1) % self.running_avg_window == 0:
                    self._print(f"\t\tStep Number: {step + 1}")
                    self._store_training_summaries()

            epoch_end_time = time.perf_counter()
            epoch_runtime = epoch_end_time - epoch_start_time
            
            # Store test summaries
            self._store_test_summaries(test_dl, val_dl, epoch_number=epoch, epoch_runtime=epoch_runtime)

            self.current_epoch += 1
            self._extend_classes(train_ds, test_ds, val_ds)

            # Save checkpoint periodically
            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()

    def get_experiment_checkpoint(self):
        """Creates a dictionary with all the necessary information to pause and resume the experiment."""
        partial_results = {}
        for k, v in self.results_dict.items():
            # Ensure consistent numpy conversion for all array types
            if hasattr(v, 'shape'):  # Any array-like object (JAX or numpy)
                partial_results[k] = np.asarray(v)
            else:
                partial_results[k] = v

        checkpoint = {
            "model_state": self.state.params,  # Flax model parameters
            "batch_stats": self.state.batch_stats,  # Batch norm stats
            "optim_state": self.state.opt_state,  # Optimizer state
            "jax_rng_state": self.key,  # JAX random state
            "numpy_rng_state": np.random.get_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "partial_results": partial_results,
            "best_accuracy": self.best_accuracy,
            "best_params": self.best_params
        }

        # Add CBP state if using CBP
        if self.use_cbp and hasattr(self, 'resgnt'):
            checkpoint["resgnt"] = self.resgnt

        return checkpoint

    def save_experiment_checkpoint(self):
        """Save experiment checkpoint."""
        if not self.experiment_checkpoints_dir_path:
            return
            
        os.makedirs(self.experiment_checkpoints_dir_path, exist_ok=True)
        
        checkpoint = self.get_experiment_checkpoint()
        checkpoint_file_name = f"{self.checkpoint_identifier_name}-{self.current_epoch}.pkl"
        checkpoint_file_path = os.path.join(self.experiment_checkpoints_dir_path, checkpoint_file_name)
        
        with open(checkpoint_file_path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        self._print(f"Checkpoint saved: {checkpoint_file_path}")

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values.
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were successfully loaded
        """
        try:
            with open(file_path, mode="rb") as experiment_checkpoint_file:
                checkpoint = pickle.load(experiment_checkpoint_file)

            # Restore Flax model and optimizer state
            self.state = self.state.replace(
                params=checkpoint["model_state"],
                batch_stats=checkpoint["batch_stats"],
                opt_state=checkpoint["optim_state"]
            )
            self.key = checkpoint["jax_rng_state"]
            np.random.set_state(checkpoint["numpy_rng_state"])
            
            self.current_epoch = checkpoint["epoch_number"]
            self.current_num_classes = checkpoint["current_num_classes"]
            self.all_classes = checkpoint["all_classes"]
            self.current_running_avg_step = checkpoint["current_running_avg_step"]
            self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
            self.best_params = checkpoint.get("best_params", {})

            partial_results = checkpoint["partial_results"]
            for k, v in self.results_dict.items():
                if k in partial_results:
                    self.results_dict[k] = partial_results[k]

            # Restore CBP state if using CBP
            if self.use_cbp and "resgnt" in checkpoint:
                self.resgnt = checkpoint["resgnt"]

            return True
        except Exception as e:
            self._print(f"Failed to load checkpoint: {e}")
            return False
