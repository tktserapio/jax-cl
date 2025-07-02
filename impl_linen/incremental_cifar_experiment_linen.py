import argparse
import dataclasses
import json
import os
import pickle
import time
from copy import deepcopy
from functools import partialmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

# third‑party
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax

# mlproj‑manager helpers (unchanged)
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import (
    access_dict,
    get_random_seeds,
    turn_off_debugging_processes,
)
from mlproj_manager.util.data_preprocessing_and_transformations import (
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotator,
    ToTensor,
)
from mlproj_manager.file_management.file_and_directory_management import (
    store_object_with_several_attempts,
)
from torchvision_modified_resnet_linen import build_resnet18
from res_gnt_linen import ResGnT
import torchvision.transforms

# --------------------------------------------------------------------------------------
# Small helper to sub‑sample the CIFAR dataset (unchanged apart from .numpy() removal)
# --------------------------------------------------------------------------------------

def subsample_cifar_data_set(sub_sample_indices: torch.Tensor, cifar_data: CifarDataSet):
    """In‑place sub‑sampling identical to the PyTorch version."""
    idx = sub_sample_indices.cpu().numpy()
    cifar_data.data["data"] = cifar_data.data["data"][idx]
    cifar_data.data["labels"] = cifar_data.data["labels"][idx]
    cifar_data.integer_labels = torch.as_tensor(cifar_data.integer_labels)[idx].tolist()
    cifar_data.current_data = cifar_data.partition_data()


# --------------------------------------------------------------------------------------
# TrainState & utility functions
# --------------------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    batch_stats: Dict[str, Any] = dataclasses.field(default_factory=dict)


@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    # Handle both one-hot and integer labels
    if labels.ndim > 1:  # One-hot labels
        return optax.softmax_cross_entropy(logits, labels).mean()
    else:  # Integer labels
        return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    loss = cross_entropy_loss(logits, labels)
    
    # Handle both one-hot and integer labels for accuracy
    if labels.ndim > 1:  # One-hot labels
        accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
    else:  # Integer labels
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    
    return {"loss": loss, "accuracy": accuracy}


# --------------------------------------------------------------------------------------
# Incremental CIFAR Experiment (Flax / JAX)
# --------------------------------------------------------------------------------------

class IncrementalCIFARExperimentJAX(Experiment):
    """Flax/JAX port of the Incremental CIFAR‑100 continual‑learning experiment."""

    # --------------------------- initialisation --------------------------- #
    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose: bool = True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # silence/cuDNN debugging parity with original script
        debug = access_dict(exp_params, "debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)

        # reproducibility
        random_seeds = get_random_seeds()
        self.random_seed = int(random_seeds[run_index])
        np.random.seed(self.random_seed)
        self.key = jax.random.PRNGKey(self.random_seed)

        # experiment hyper‑params
        self.data_path: str = exp_params["data_path"]
        self.num_workers: int = access_dict(exp_params, "num_workers", default=0, val_type=int)  # Set to 0 for JAX compatibility
        self.stepsize: float = float(exp_params["stepsize"])
        self.weight_decay: float = float(exp_params["weight_decay"])
        self.momentum: float = float(exp_params["momentum"])

        # reset flags
        self.reset_head: bool = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network: bool = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        self.early_stopping: bool = access_dict(exp_params, "early_stopping", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting whole network supersedes resetting head."))

        # CBP flags (implementation TODO)
        self.use_cbp: bool = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.replacement_rate: float = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        self.utility_function: str = access_dict(exp_params, "utility_function", default="weight", val_type=str)
        self.maturity_threshold: int = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)

        # noise
        self.noise_std: float = access_dict(exp_params, "noise_std", default=0.0, val_type=float)
        self.perturb_weights_indicator: bool = self.noise_std > 0.0

        # constants & bookkeeping
        self.num_epochs = 4000
        self.class_increase_frequency = 200
        self.current_epoch = 0
        self.current_num_classes = 5
        self.num_classes = 100
        self.batch_sizes = {"train": 90, "validation": 50, "test": 100}
        self.image_dims = (32, 32, 3)

        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy: float = 0.0
        self.best_params: Dict[str, Any] = {}

        # For CBP feature collection
        self.current_features = [] if self.use_cbp else None

        # network
        self.model = build_resnet18(num_classes=self.num_classes)
        dummy = jnp.ones((1, *self.image_dims), jnp.float32)
        self.key, sub = jax.random.split(self.key)
        variables = self.model.init(sub, dummy, feature_list=None, train=True)
        params, batch_stats = variables["params"], variables.get("batch_stats", {})

        # Initialize CBP if enabled
        if self.use_cbp:
            # Import here to avoid circular imports
            try:
                from res_gnt_linen import ResGnT
                self.resgnt = ResGnT(
                    net=self.model,
                    hidden_activation="relu",
                    replacement_rate=self.replacement_rate,
                    decay_rate=0.99,
                    util_type=self.utility_function,
                    maturity_threshold=self.maturity_threshold
                )
                print("CBP (ResGnT) initialized")
            except ImportError as e:
                print(f"Warning: CBP not available - {e}")
                self.use_cbp = False
                self.current_features = None
        params, batch_stats = variables["params"], variables.get("batch_stats", {})

        # optimiser: we keep LR=1.0 inside optimizer, and scale grads manually → easy LR reset
        self.base_lr = self.stepsize
        tx = optax.chain(
            optax.add_decayed_weights(self.weight_decay),
            optax.sgd(learning_rate=1.0, momentum=self.momentum),
        )
        self.state: TrainState = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx, batch_stats=batch_stats)

        # containers for running‑avg summaries (trimmed)
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.running_avg_window = 25
        self.current_running_avg_step = 0

        # Initialize summaries like PyTorch/NNX versions
        self._initialize_summaries()

        # Checkpoint system
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency
        self.delete_old_checkpoints = True

    def _initialize_summaries(self):
        """Initialize logging arrays matching PyTorch/NNX versions"""
        number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
        class_increase = 5
        number_of_image_per_task = 450 * class_increase  # num_images_per_class
        bin_size = (self.running_avg_window * self.batch_sizes["train"])
        total_checkpoints = int(np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size))

        # Use numpy arrays for logging
        self.results_dict["train_loss_per_checkpoint"] = np.zeros(total_checkpoints, dtype=np.float32)
        self.results_dict["train_accuracy_per_checkpoint"] = np.zeros(total_checkpoints, dtype=np.float32)
        self.results_dict["epoch_runtime"] = np.zeros(self.num_epochs, dtype=np.float32)
        
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = np.zeros(self.num_epochs, dtype=np.float32)
            self.results_dict[set_type + "_accuracy_per_epoch"] = np.zeros(self.num_epochs, dtype=np.float32)
            self.results_dict[set_type + "_evaluation_runtime"] = np.zeros(self.num_epochs, dtype=np.float32)
        self.results_dict["class_order"] = self.all_classes

    def _store_training_summaries(self):
        """Store training summaries matching PyTorch/NNX behavior"""
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] += self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] += self.running_accuracy / self.running_avg_window
        
        self._print(f"\t\tOnline accuracy: {self.running_accuracy / self.running_avg_window:.2f}")
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_dl, val_dl, epoch_number: int, epoch_runtime: float):
        """Store test summaries matching PyTorch/NNX behavior"""
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

    def get_validation_and_train_indices(self, cifar_data: CifarDataSet):
        """Split data into train/validation matching PyTorch/NNX versions"""
        num_val_samples_per_class = 50
        num_train_samples_per_class = 450
        validation_set_size = 5000
        train_set_size = 45000

        validation_indices = torch.zeros(validation_set_size, dtype=torch.int32)
        train_indices = torch.zeros(train_set_size, dtype=torch.int32)
        current_val_samples = 0
        current_train_samples = 0
        
        for i in range(self.num_classes):
            class_indices = torch.where(torch.tensor(cifar_data.data["labels"][:, i]) == 1)[0]
            validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] = class_indices[:num_val_samples_per_class]
            train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] = class_indices[num_val_samples_per_class:]
            current_val_samples += num_val_samples_per_class
            current_train_samples += num_train_samples_per_class

        return train_indices, validation_indices

    def evaluate_network(self, test_dl):
        """Evaluate network on test data"""
        # This method is called from _store_test_summaries, which is called from train
        # So we need to access the JIT function from train - let's use a simple approach
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in test_dl:
            images = jnp.asarray(batch["image"].numpy())
            labels = jnp.asarray(batch["label"].numpy())
            
            # Use a simple non-JIT evaluation for now
            variables = {"params": self.state.params, "batch_stats": self.state.batch_stats}
            logits = self.model.apply(variables, images, feature_list=None, train=False, mutable=False)
            
            # Select only current classes for both logits and labels
            current_logits = logits[:, self.all_classes[:self.current_num_classes]]
            if labels.ndim > 1:  # One-hot labels
                current_labels = labels[:, self.all_classes[:self.current_num_classes]]
            else:
                current_labels = labels
                
            metrics = compute_metrics(current_logits, current_labels)
            total_loss += float(metrics["loss"])
            total_acc += float(metrics["accuracy"])
            num_batches += 1

        return total_loss / num_batches, total_acc / num_batches

    def _save_model_parameters(self):
        """Save model parameters for post-experiment analysis"""
        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = f"index-{self.run_index}_epoch-{self.current_epoch}.pkl"
        file_path = os.path.join(model_parameters_dir_path, file_name)

        # Convert JAX arrays to numpy for saving
        pure_params = jax.tree_util.tree_map(lambda x: np.array(x), self.state.params)
        store_object_with_several_attempts(pure_params, file_path, storing_format="pickle", num_attempts=10)

    def _reset_head(self):
        """Reset final classification layer"""
        # Reinitialize final layer with new random key
        self.key, sub = jax.random.split(self.key)
        # Implementation depends on your model structure
        # This is a placeholder - adjust based on your ResNet structure
        print("Head reset - implementation needed based on model structure")

    def _reset_network(self):
        """Reset entire network"""
        self.key, sub = jax.random.split(self.key)
        dummy = jnp.ones((1, *self.image_dims), jnp.float32)
        variables = self.model.init(sub, dummy, feature_list=None, train=True)
        params, batch_stats = variables["params"], variables.get("batch_stats", {})
        self.state = self.state.replace(params=params, batch_stats=batch_stats)
        print("Network reset completed")

    # ---------------------------------------------------------------------
    # Helper functions for JIT‑compiled step functions
    # ---------------------------------------------------------------------
    def _apply_model(self, params, batch_stats, images, train, rng):
        """Non-JIT helper for model application"""
        variables = {"params": params, "batch_stats": batch_stats}
        if train:
            (logits, new_batch_stats) = self.model.apply(
                variables,
                images,
                feature_list=self.current_features if self.use_cbp else None,
                train=True,
                rngs={"dropout": rng},
                mutable=["batch_stats"],
            )
            return logits, new_batch_stats["batch_stats"]
        else:
            logits = self.model.apply(variables, images, feature_list=None, train=False, mutable=False)
            return logits, batch_stats

    def _current_lr(self, epoch: int) -> Optional[float]:
        """Piece‑wise decay identical to original PyTorch logic."""
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
        if not self.perturb_weights_indicator:
            return params
        leaves, treedef = jax.tree_util.tree_flatten(params)
        keys = jax.random.split(key, len(leaves) + 1)
        new_leaves = [
            leaf + self.noise_std * jax.random.normal(k, leaf.shape, leaf.dtype) for leaf, k in zip(leaves, keys[:-1])
        ]
        return jax.tree_util.tree_unflatten(treedef, new_leaves)

    # ---------------------------------------------------------------------
    # Data helpers – identical to PyTorch version (DataLoader kept)
    # ---------------------------------------------------------------------
    def get_data(self, train: bool = True, validation: bool = False):
        """Loads CIFAR‑100 via CifarDataSet and returns (dataset, dataloader)."""
        cifar_data = CifarDataSet(
            root_dir=self.data_path,
            train=train,
            cifar_type=100,
            device=None,
            image_normalization="max",
            label_preprocessing="one-hot",
            use_torch=True,
        )

        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        transformations: List[Callable] = [
            ToTensor(swap_color_axis=True),
            Normalize(mean=mean, std=std),
        ]
        if not validation:
            transformations += [
                RandomHorizontalFlip(p=0.5),
                RandomCrop(size=32, padding=4, padding_mode="reflect"),
                RandomRotator(degrees=(0, 15)),
            ]
        cifar_data.set_transformation(torchvision.transforms.Compose(transformations))  # type: ignore

        if not train:
            batch_size = self.batch_sizes["test"]
            dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
            return cifar_data, dataloader

        train_indices, val_indices = self.get_validation_and_train_indices(cifar_data)
        indices = val_indices if validation else train_indices
        subsample_cifar_data_set(indices, cifar_data)
        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        return cifar_data, DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    # ---------------------------------------------------------------------
    # Training / evaluation loops (simplified)
    # ---------------------------------------------------------------------
    def train(self, train_dl: DataLoader, test_dl: DataLoader, val_dl: DataLoader,
              train_ds: CifarDataSet, test_ds: CifarDataSet, val_ds: CifarDataSet):
        
        # Define JIT-compiled functions inside train method to capture self via closure
        @jax.jit
        def train_step_jit(state, images, labels, rng, lr_scale):
            """JIT-compiled training step function with closure access to self"""
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
                
                loss = cross_entropy_loss(current_logits, current_labels)
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
            metrics = compute_metrics(logits, current_labels)
            return state, metrics

        @jax.jit
        def eval_step_jit(state, images, labels):
            """JIT-compiled evaluation step function with closure access to self"""
            variables = {"params": state.params, "batch_stats": state.batch_stats}
            logits = self.model.apply(variables, images, feature_list=None, train=False, mutable=False)
            
            # Select only current classes for both logits and labels
            current_logits = logits[:, self.all_classes[:self.current_num_classes]]
            if labels.ndim > 1:  # One-hot labels
                current_labels = labels[:, self.all_classes[:self.current_num_classes]]
            else:
                current_labels = labels
                
            return compute_metrics(current_logits, current_labels)

        # Existing train method code continues...
        train_ds.select_new_partition(self.all_classes[: self.current_num_classes])
        test_ds.select_new_partition(self.all_classes[: self.current_num_classes])
        val_ds.select_new_partition(self.all_classes[: self.current_num_classes])
        self._save_model_parameters()

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
                        print(f"DEBUG Linen CBP: Number of feature layers: {len(self.current_features)}")
                        for i, feat in enumerate(self.current_features):
                            print(f"DEBUG Linen CBP: Feature {i} shape: {feat.shape}")

                # RNG split
                self.key, sub = jax.random.split(self.key)

                # Training step
                self.state, metrics = train_step_jit(self.state, images, labels, sub, current_lr)

                # CBP processing if enabled
                if self.use_cbp and self.current_features:
                    try:
                        self.resgnt.gen_and_test(self.current_features)
                    except Exception as cbp_error:
                        print(f"DEBUG Linen CBP ERROR: {cbp_error}")
                        print(f"  Feature shapes: {[f.shape for f in self.current_features]}")
                        if hasattr(self.resgnt, 'weight_layers'):
                            print(f"  ResGnT weight layers: {len(self.resgnt.weight_layers)}")
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

    def _extend_classes(self, train_ds: CifarDataSet, test_ds: CifarDataSet, val_ds: CifarDataSet):
        """Add new classes at task boundaries"""
        if self.current_epoch % self.class_increase_frequency == 0:
            self._print(f"Best accuracy in the task: {self.best_accuracy:.4f}")
            
            if self.early_stopping and self.best_params:
                self.state = self.state.replace(params=self.best_params)
            
            # Reset best accuracy tracking for new task
            self.best_accuracy = 0.0
            self.best_params = {}
            self._save_model_parameters()

            if self.current_num_classes == self.num_classes:
                return

            increase = 5
            self.current_num_classes += increase
            train_ds.select_new_partition(self.all_classes[: self.current_num_classes])
            test_ds.select_new_partition(self.all_classes[: self.current_num_classes])
            val_ds.select_new_partition(self.all_classes[: self.current_num_classes])

            self._print("\tNew class added...")
            self._print(f"\tCurrent classes: {self.current_num_classes}/{self.num_classes}")
            
            if self.reset_head:
                self._reset_head()
            if self.reset_network:
                self._reset_network()

    def get_experiment_checkpoint(self):
        """Creates a dictionary with all the necessary information to pause and resume the experiment"""
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

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
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
            print(f"Failed to load checkpoint: {e}")
            return False

    def run(self):
        """Main experiment runner - matches PyTorch/NNX versions"""
        # Load data
        train_ds, train_dl = self.get_data(train=True, validation=False)
        val_ds, val_dl = self.get_data(train=True, validation=True)
        test_ds, test_dl = self.get_data(train=False)
        
        # Load checkpoint if one is available
        # Temporarily disabled to debug shape mismatch
        # self.load_experiment_checkpoint()
        
        # Train network
        self.train(train_dl, test_dl, val_dl, train_ds, test_ds, val_ds)
        
        # Store results using exp.store_results()
        self.store_results()
# --------------------------------------------------------------------------------------
# main entry point wrapper – mirrors original CLI
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Incremental CIFAR‑100 Experiment (Flax)")
    parser.add_argument("--config", type=str, default="./incremental_cifar/cfg/base_deep_learning_system.json")
    parser.add_argument("--experiment-index", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    base = os.path.dirname(os.path.abspath(__file__))
    
    # Fix data_path handling - check for empty string as well as missing key
    if "data_path" not in cfg or not cfg["data_path"]:
        cfg["data_path"] = os.path.join(base, "data")
    if "results_dir" not in cfg or not cfg["results_dir"]:
        cfg["results_dir"] = os.path.join(base, "results")
    if "experiment_name" not in cfg or not cfg["experiment_name"]:
        cfg["experiment_name"] = os.path.splitext(os.path.basename(args.config))[0]
    
    # Debug: Print the paths being used
    print(f"Using data_path: '{cfg['data_path']}'")
    print(f"Using results_dir: '{cfg['results_dir']}'")
    print(f"Using experiment_name: '{cfg['experiment_name']}'")
    
    # Make sure data directory exists
    os.makedirs(cfg["data_path"], exist_ok=True)

    exp = IncrementalCIFARExperimentJAX(
        cfg,
        results_dir=os.path.join(cfg["results_dir"], cfg["experiment_name"]),
        run_index=args.experiment_index,
        verbose=args.verbose,
    )

    start = time.perf_counter()
    exp.run()  # Use the run() method like PyTorch/NNX versions
    end = time.perf_counter()
    print(f"Total runtime: {(end - start) / 60:.2f} min")


if __name__ == "__main__":
    main()