import time
import os
import pickle
from copy import deepcopy
import json
import argparse
from functools import partialmethod

from tqdm import tqdm
import numpy as np

# jax imports
from flax import nnx
from functools import partial
import jax
import jax.numpy as jnp
import optax

import torch
from torch.utils.data import DataLoader # still use pytorch dataloader

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.experiments import Experiment
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator
from mlproj_manager.file_management.file_and_directory_management import store_object_with_several_attempts

# updated imports with jax implementations
# from modified_resnet_nnx import build_resnet18
from modified_resnet_nnx_cbp import build_resnet18
from res_gnt_jax import ResGnT

# for hessian computation at the start and end of each task 
from utils.hessian_computation import get_hvp_fn
from utils.lanczos import lanczos_alg
from utils.density import tridiag_to_density
from utils.optimizer import l2_regularization, adam_with_param_counts
from utils.file_system import get_results_path, numpyify, plot_hessian_spectrum

class JAXCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x

def subsample_cifar_data_set(sub_sample_indices, cifar_data: CifarDataSet):
    """
    Sub-samples the CIFAR 100 data set according to the given indices
    :param sub_sample_indices: array of indices in the same format as the cifar data set (numpy or jax)
    :param cifar_data: cifar data to be sub-sampled
    :return: None, but modifies the given cifar_dataset
    """

    cifar_data.data["data"] = cifar_data.data["data"][np.asarray(sub_sample_indices)]
    cifar_data.data["labels"] = cifar_data.data["labels"][np.asarray(sub_sample_indices)]
    cifar_data.integer_labels = jnp.array(cifar_data.integer_labels)[np.asarray(sub_sample_indices)].tolist()
    cifar_data.current_data = cifar_data.partition_data()


class IncrementalCIFARExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define jax device - handle GPU/CPU gracefully
        try:
            gpu_devices = jax.devices("gpu")
            if gpu_devices:
                self.device = gpu_devices[0]
                print(f"JAX is using GPU: {self.device}")
                print(f"   Available GPU devices: {len(gpu_devices)}")
                for i, device in enumerate(gpu_devices):
                    print(f"   GPU {i}: {device}")
            else:
                self.device = jax.devices("cpu")[0]
                print(f"⚠️  JAX is using CPU: {self.device}")
                print("   No GPU devices found")
        except RuntimeError as e:
            # No GPU available, use CPU
            self.device = jax.devices("cpu")[0]
            print(f"⚠️  JAX is using CPU: {self.device}")
            print(f"   GPU detection failed: {e}")

        # disable tqdm if verbose is enabled
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=self.verbose)

        """ For reproducibility """
        random_seeds = get_random_seeds()
        self.random_seed = int(random_seeds[self.run_index])  # Ensure it's a Python int, not torch tensor
        self.rng_key = jax.random.PRNGKey(self.random_seed)
        np.random.seed(self.random_seed)

        """ Experiment parameters """
        self.data_path = exp_params["data_path"]
        self.num_workers = access_dict(exp_params, key="num_workers", default=1, val_type=int)  # set to 1 when using cpu

        # optimization parameters
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = exp_params["weight_decay"]
        self.momentum = exp_params["momentum"]

        # network resetting parameters
        self.reset_head = access_dict(exp_params, "reset_head", default=False, val_type=bool)
        self.reset_network = access_dict(exp_params, "reset_network", default=False, val_type=bool)
        if self.reset_head and self.reset_network:
            print(Warning("Resetting the whole network supersedes resetting the head of the network. There's no need to set both to True."))
        self.early_stopping = access_dict(exp_params, "early_stopping", default=False, val_type=bool)

        # cbp parameters
        self.use_cbp = access_dict(exp_params, "use_cbp", default=False, val_type=bool)
        self.replacement_rate = access_dict(exp_params, "replacement_rate", default=0.0, val_type=float)
        assert (not self.use_cbp) or (self.replacement_rate > 0.0), "Replacement rate should be greater than 0."
        self.utility_function = access_dict(exp_params, "utility_function", default="weight", val_type=str,
                                            choices=["weight", "contribution"])
        self.maturity_threshold = access_dict(exp_params, "maturity_threshold", default=0, val_type=int)
        assert (not self.use_cbp) or (self.maturity_threshold > 0), "Maturity threshold should be greater than 0."

        # shrink and perturb parameters
        self.noise_std = access_dict(exp_params, "noise_std", default=0.0, val_type=float)
        self.perturb_weights_indicator = self.noise_std > 0.0

        """ Training constants """
        self.num_epochs = 4000
        self.current_num_classes = 5
        self.batch_sizes = {"train": 90, "test": 100, "validation":50}
        self.num_classes = 100
        self.image_dims = (32, 32, 3)
        self.num_images_per_class = 450

        """ Network set up """
        # initialize network with proper JAX rngs
        rngs = nnx.Rngs(self.random_seed)
        self.net = build_resnet18(num_classes=self.num_classes, rngs=rngs)

        tx = optax.chain(
            optax.add_decayed_weights(self.weight_decay),
            optax.sgd(learning_rate=self.stepsize, momentum=self.momentum, nesterov=False)
        )

        self.optim = nnx.Optimizer(self.net, tx)

        # JAX doesn't need explicit device movement like PyTorch
        self.current_epoch = 0

        # for cbp
        self.resgnt = None
        if self.use_cbp:
            self.resgnt = ResGnT(net=self.net,
                                 hidden_activation="relu",
                                 replacement_rate=self.replacement_rate,
                                 decay_rate=0.99,
                                 util_type=self.utility_function,
                                 maturity_threshold=self.maturity_threshold,
                                 device=self.device)
        self.current_features = [] if self.use_cbp else None

        """ For data partitioning """
        self.class_increase_frequency = 200
        self.all_classes = np.random.permutation(self.num_classes)
        self.best_accuracy = jnp.array(0.0, dtype=jnp.float32)
        self.best_accuracy_model_parameters = {}

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "current_epoch"
        self.checkpoint_save_frequency = self.class_increase_frequency  # save every time a new class is added
        self.delete_old_checkpoints = True

        """ For summaries """
        self.running_avg_window = 25
        self.current_running_avg_step, self.running_loss, self.running_accuracy = (0, 0.0, 0.0)
        self._initialize_summaries()

        self.compute_hessian          = access_dict(exp_params, "compute_hessian", default=False,  val_type=bool)
        self.compute_hessian_interval = access_dict(exp_params, "compute_hessian_interval", default=1, val_type=int)
        self.compute_hessian_size     = access_dict(exp_params, "compute_hessian_size",     default=512, val_type=int)

    # ------------------------------ Methods for initializing the experiment ------------------------------#
    def _initialize_summaries(self):
        """
        Initializes the summaries for the experiment
        """
        number_of_tasks = np.arange(self.num_epochs // self.class_increase_frequency) + 1
        class_increase = 5
        number_of_image_per_task = self.num_images_per_class * class_increase
        bin_size = (self.running_avg_window * self.batch_sizes["train"])
        total_checkpoints = int(np.sum(number_of_tasks * self.class_increase_frequency * number_of_image_per_task // bin_size))

        # JAX arrays instead of torch tensors
        train_prototype_array = jnp.zeros(total_checkpoints, dtype=np.float32)
        self.results_dict["train_loss_per_checkpoint"] = np.zeros_like(train_prototype_array)
        self.results_dict["train_accuracy_per_checkpoint"] = np.zeros_like(train_prototype_array)

        prototype_array = jnp.zeros(self.num_epochs, dtype=np.float32)
        self.results_dict["epoch_runtime"] = np.zeros_like(prototype_array)
        # test and validation summaries
        for set_type in ["test", "validation"]:
            self.results_dict[set_type + "_loss_per_epoch"] = np.zeros_like(prototype_array)
            self.results_dict[set_type + "_accuracy_per_epoch"] = np.zeros_like(prototype_array)
            self.results_dict[set_type + "_evaluation_runtime"] = np.zeros_like(prototype_array)
        self.results_dict["class_order"] = self.all_classes

    # ----------------------------- For saving and loading experiment checkpoints ----------------------------- #
    def get_experiment_checkpoint(self):
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        partial_results = {}
        for k, v in self.results_dict.items():
            if isinstance(v, jnp.ndarray):
                partial_results[k] = np.array(v)
            else:
                partial_results[k] = v 

        checkpoint = {
            "model_state": nnx.state(self.net),  # JAX model state
            "optimizer_state": nnx.state(self.optim), # optimizer state
            "jax_rng_state": self.rng_key,       # JAX random state
            "numpy_rng_state": np.random.get_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
            "best_accuracy": float(self.best_accuracy), 
            "best_accuracy_model_parameters": self.best_accuracy_model_parameters,
            "partial_results": partial_results
        }

        if self.use_cbp:
            checkpoint["resgnt"] = self.resgnt

        return checkpoint

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were succesfully loaded
        """

        with open(file_path, mode="rb") as experiment_checkpoint_file:
            checkpoint = pickle.load(experiment_checkpoint_file)

        # Restore JAX model and optimizer state
        nnx.update(self.net, checkpoint["model_state"])
        nnx.update(self.optim, checkpoint["optimizer_state"])

        self.rng_key = checkpoint["jax_rng_state"]
        np.random.set_state(checkpoint["numpy_rng_state"])
        
        self.current_epoch = checkpoint["epoch_number"]
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

        self.best_accuracy = jnp.array(checkpoint["best_accuracy"], dtype=jnp.float32)
        self.best_accuracy_model_parameters = checkpoint["best_accuracy_model_parameters"]

        partial_results = checkpoint["partial_results"]
        for k, v in self.results_dict.items():
            self.results_dict[k] = partial_results[k] 

        if self.use_cbp:
            self.resgnt = checkpoint["resgnt"]

    # --------------------------------------- For storing summaries --------------------------------------- #
    def _store_training_summaries(self):
        # store train data - use regular numpy indexing since these are numpy arrays
        self.results_dict["train_loss_per_checkpoint"][self.current_running_avg_step] = self.running_loss / self.running_avg_window
        self.results_dict["train_accuracy_per_checkpoint"][self.current_running_avg_step] = self.running_accuracy / self.running_avg_window

        self._print("\t\tOnline accuracy: {0:.2f}".format(self.running_accuracy / self.running_avg_window))
        self.running_loss *= 0.0
        self.running_accuracy *= 0.0
        self.current_running_avg_step += 1

    def _store_test_summaries(self, test_data, val_data, epoch_number: int, epoch_runtime: float):
        """ Computes test summaries and stores them in results dir """

        self.results_dict["epoch_runtime"][epoch_number] = epoch_runtime

        self.net.eval()
        for data_name, data_loader, compare_to_best in [("test", test_data, False), ("validation", val_data, True)]:
            # evaluate on data
            evaluation_start_time = time.perf_counter()
            loss, accuracy = self.evaluate_network(data_loader)
            evaluation_time = time.perf_counter() - evaluation_start_time

            if compare_to_best:
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_accuracy_model_parameters = nnx.state(self.net)  # Save JAX state

            # store summaries - use regular numpy indexing
            self.results_dict[data_name + "_evaluation_runtime"][epoch_number] = evaluation_time
            self.results_dict[data_name + "_loss_per_epoch"][epoch_number] = float(loss)
            self.results_dict[data_name + "_accuracy_per_epoch"][epoch_number] = float(accuracy)

            # print progress
            self._print("\t\t{0} accuracy: {1:.4f}".format(data_name, accuracy))

        self.net.train()
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def evaluate_network(self, test_data):
        """
        Evaluates the network on the test data
        :param test_data: a data loader or iterable
        :return: (jnp.Array) test loss, (jnp.Array) test accuracy
        """
        @nnx.jit
        def eval_step_jit(model, images, labels, current_classes):
            logits_full = model(images)
            logits = logits_full[:, current_classes] 
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=labels).mean()
            return loss, logits

        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        current_classes = self.all_classes[:self.current_num_classes]

        # self._print(f"🔍 EVAL DEBUG - current_num_classes: {self.current_num_classes}")
        # self._print(f"🔍 EVAL DEBUG - current_classes shape: {current_classes.shape}")
        # self._print(f"🔍 EVAL DEBUG - active_classes: {current_classes}")

        # all_labels_seen = []
        
        # self.net.eval()
        # JAX doesn't need no_grad context
        for batch_idx, sample in enumerate(test_data):
            images = jnp.asarray(sample["image"].numpy())
            test_labels = jnp.asarray(sample["label"].numpy())

            if len(images.shape) == 4 and images.shape[1] == 3:  # (N, C, H, W) -> (N, H, W, C)
                images = jnp.transpose(images, (0, 2, 3, 1))
            
            # Convert one-hot to integer labels for JAX loss function
            test_labels_int = jnp.argmax(test_labels, axis=1)
            
            loss, logits = eval_step_jit(self.net, images, test_labels_int, current_classes)

            # Compute loss and accuracy
            avg_loss += loss
            avg_acc += jnp.mean(jnp.argmax(logits, axis=1) == test_labels_int)
            num_test_batches += 1

        return avg_loss / num_test_batches, avg_acc / num_test_batches

    # ------------------------------------- For running the experiment ------------------------------------- #
    def run(self):
        # load data
        training_data, training_dataloader = self.get_data(train=True, validation=False)
        val_data, val_dataloader = self.get_data(train=True, validation=True)
        test_data, test_dataloader = self.get_data(train=False)
        # load checkpoint if one is available
        
        self.load_experiment_checkpoint()
        
        # train network
        self.train(train_dataloader=training_dataloader, test_dataloader=test_dataloader, val_dataloader=val_dataloader,
                   test_data=test_data, training_data=training_data, val_data=val_data)
        
        # store results using exp.store_results()

    def get_data(self, train: bool = True, validation: bool = False):
        """
        Loads the data set
        :param train: (bool) indicates whether to load the train (True) or the test (False) data
        :param validation: (bool) indicates whether to return the validation set. The validation set is made up of
                           50 examples of each class of whichever set was loaded
        :return: data set, data loader
        """

        """ Loads CIFAR data set """
        cifar_data = CifarDataSet(root_dir=self.data_path,
                                  train=train,
                                  cifar_type=100,
                                  device=None,
                                  image_normalization="max",
                                  label_preprocessing="one-hot",
                                  use_torch=False)

        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)

        transformations = [
            ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
            Normalize(mean=mean, std=std),  # center by mean and divide by std
        ]

        if not validation:
            transformations.append(RandomHorizontalFlip(p=0.5))
            transformations.append(RandomCrop(size=32, padding=4, padding_mode="reflect"))
            transformations.append(RandomRotator(degrees=(0,15)))

        cifar_data.set_transformation(JAXCompose(transformations))

        if not train:
            batch_size = self.batch_sizes["test"]
            dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
            return cifar_data, dataloader

        train_indices, validation_indices = self.get_validation_and_train_indices(cifar_data)
        indices = validation_indices if validation else train_indices
        subsample_cifar_data_set(sub_sample_indices=indices, cifar_data=cifar_data)
        batch_size = self.batch_sizes["validation"] if validation else self.batch_sizes["train"]
        return cifar_data, DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)

    def get_validation_and_train_indices(self, cifar_data: CifarDataSet):
        """
        Splits the cifar data into validation and train set and returns the indices of each set with respect to the
        original dataset
        :param cifar_data: and instance of CifarDataSet
        :return: train and validation indices
        """
        num_val_samples_per_class = 50
        num_train_samples_per_class = 450
        validation_set_size = 5000
        train_set_size = 45000

        # Use numpy arrays for indices - no need for JAX here
        validation_indices = np.zeros(validation_set_size, dtype=np.int32)
        train_indices = np.zeros(train_set_size, dtype=np.int32)
        current_val_samples = 0
        current_train_samples = 0
        for i in range(self.num_classes):
            # Use numpy operations for index manipulation
            class_indices = np.where(cifar_data.data["labels"][:, i] == 1)[0]
            validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] = class_indices[:num_val_samples_per_class]
            train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] = class_indices[num_val_samples_per_class:]
            current_val_samples += num_val_samples_per_class
            current_train_samples += num_train_samples_per_class

        return train_indices, validation_indices

    # Using PyTorch DataLoader for data loading but JAX for computation
    def train(self, train_dataloader : DataLoader, test_dataloader : DataLoader, val_dataloader : DataLoader,
          test_data: CifarDataSet, training_data: CifarDataSet, val_data: CifarDataSet):

        training_data.select_new_partition(self.all_classes[:self.current_num_classes]) # example [23, 45, 67, 89...]
        test_data.select_new_partition(self.all_classes[:self.current_num_classes]) # now the dataset only contains samples from the current active classes
        val_data.select_new_partition(self.all_classes[:self.current_num_classes]) # same for validation 
        self._save_model_parameters()

        # start of task 0 - initial Hessian computation
        if self.current_epoch == 0 and 0 % self.compute_hessian_interval == 0:
            self._compute_hessian(tag="start",
                          train_loader=train_dataloader,
                          eval_loader=test_dataloader,
                          task_id=0)

        def compute_loss_and_logits(model, images, labels, current_classes):
            logits_full = model(images)
            logits = logits_full[:, current_classes]
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=labels).mean()
            return loss, logits

        # Pre-compile JIT functions outside the loop for better performance
        @nnx.jit
        def train_step_jit(model, optimizer, images, labels, class_indices):
            def loss_fn(model):
                return compute_loss_and_logits(model, images, labels, class_indices)
            
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(model)

            # this logic should be alright but not sure if it works with JIT
            if self.use_cbp:
                # TODO: fix the self reference since this is JIT-compiled
                modified_grads = self.resgnt.modify_gradients(grads, model) 
                optimizer.update(modified_grads)
            else:
                optimizer.update(grads)

            return loss, logits

        for e in tqdm(range(self.current_epoch, self.num_epochs)):
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()
            # self.net.train()
            for step_number, sample in enumerate(train_dataloader):
                # Convert data once and efficiently
                image = jnp.asarray(sample["image"].numpy())
                label = jnp.asarray(sample["label"].numpy())
                
                # Convert from NCHW to NHWC format once
                if len(image.shape) == 4 and image.shape[1] == 3:
                    image = jnp.transpose(image, (0, 2, 3, 1))

                current_classes = self.all_classes[:self.current_num_classes]
                labels_int = jnp.argmax(label, axis=1)

                current_loss, predictions = train_step_jit(
                    self.net, self.optim, image, labels_int, current_classes
                )
                
                # Inject noise (if enabled)
                self.inject_noise()

                # Compute accuracy efficiently
                current_accuracy = jnp.mean(jnp.argmax(predictions, axis=1) == labels_int)
                
                # Store summaries
                self.running_loss += float(current_loss)
                self.running_accuracy += float(current_accuracy)
                
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

            epoch_end_time = time.perf_counter()
            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=self.current_epoch,
                                    epoch_runtime=epoch_end_time - epoch_start_time)

            # HESSIAN COMPUTATION 

            # compute hessian at end of task
            if ((self.current_epoch + 1) % self.class_increase_frequency == 0 and (self.current_epoch + 1) // self.class_increase_frequency % self.compute_hessian_interval == 0):
                self._compute_hessian(tag="end",
                                      train_loader=train_dataloader,
                                      eval_loader=test_dataloader,
                                      task_id=(self.current_epoch + 1) // self.class_increase_frequency - 1)

            # do we have a new task?
            is_new_task = ((self.current_epoch + 1) % self.class_increase_frequency) == 0
            
            self.current_epoch += 1
            
            # Extend classes for new task
            self.extend_classes(training_data, test_data, val_data)
            
            # start of new task - compute Hessian after extending classes
            if is_new_task and (self.current_epoch // self.class_increase_frequency) % self.compute_hessian_interval == 0:
                self._compute_hessian(tag="start",
                              train_loader=train_dataloader,
                              eval_loader=test_dataloader,
                              task_id=self.current_epoch // self.class_increase_frequency)

            if self.current_epoch % self.checkpoint_save_frequency == 0:
                self.save_experiment_checkpoint()


    def set_lr(self):
        """ Changes the learning rate of the optimizer according to the current epoch of the task """
        current_stepsize = None
        if (self.current_epoch % self.class_increase_frequency) == 0:
            current_stepsize = self.stepsize
        elif (self.current_epoch % self.class_increase_frequency) == 60:
            current_stepsize = round(self.stepsize * 0.2, 5)
        elif (self.current_epoch % self.class_increase_frequency) == 120:
            current_stepsize = round(self.stepsize * (0.2 ** 2), 5)
        elif (self.current_epoch % self.class_increase_frequency) == 160:
            current_stepsize = round(self.stepsize * (0.2 ** 3), 5)

        if current_stepsize is not None:
            tx = optax.chain(
                optax.add_decayed_weights(self.weight_decay),
                optax.sgd(learning_rate=current_stepsize, momentum=self.momentum, nesterov=False)
            )
            self.optim = nnx.Optimizer(self.net, tx)
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))

    def inject_noise(self):
        """
        Adds a small amount of random noise to the parameters of the network
        """
        if not self.perturb_weights_indicator: 
            return

        # Optimized JAX approach - avoid tree_map for better performance
        current_state = nnx.state(self.net)
        
        # Update random key once
        self.rng_key, noise_key = jax.random.split(self.rng_key)
        
        # Efficient noise injection without tree_map
        def add_noise_to_layer(layer_params, key_offset=0):
            if isinstance(layer_params, dict):
                noisy_params = {}
                for i, (k, v) in enumerate(layer_params.items()):
                    if hasattr(v, 'shape'):  # JAX array
                        noise_subkey = jax.random.fold_in(noise_key, key_offset + i)
                        noise = jax.random.normal(noise_subkey, v.shape) * self.noise_std
                        noisy_params[k] = v + noise
                    elif isinstance(v, dict):
                        noisy_params[k] = add_noise_to_layer(v, key_offset + i * 100)
                    else:
                        noisy_params[k] = v
                return noisy_params
            elif hasattr(layer_params, 'shape'):
                noise = jax.random.normal(noise_key, layer_params.shape) * self.noise_std
                return layer_params + noise
            else:
                return layer_params
        
        noisy_state = add_noise_to_layer(current_state)
        nnx.update(self.net, noisy_state)

    def _compute_hessian(self, tag: str, train_loader: DataLoader, eval_loader: DataLoader, task_id: int):
        
        if not self.compute_hessian:
            return

        train_sample = next(iter(train_loader))
        eval_sample = next(iter(eval_loader))

        x_train = jnp.array(train_sample["image"].numpy()[:self.compute_hessian_size])
        y_train = jnp.array(train_sample["label"].numpy()[:self.compute_hessian_size])
        x_eval = jnp.array(eval_sample["image"].numpy()[:self.compute_hessian_size])
        y_eval = jnp.array(eval_sample["label"].numpy()[:self.compute_hessian_size])

        if len(x_train.shape) == 4 and x_train.shape[1] == 3:
            x_train = jnp.transpose(x_train, (0, 2, 3, 1))
            x_eval = jnp.transpose(x_eval, (0, 2, 3, 1))

        y_train_int = jnp.argmax(y_train, axis=1)
        y_eval_int = jnp.argmax(y_eval, axis=1)

        current_classes = self.all_classes[:self.current_num_classes]
        
        # Define loss functions
        def loss_fn_train(params, x_batch, y_batch):
            net_tmp = nnx.merge(nnx.graphdef(self.net), params)
            logits = net_tmp(x_batch)[:, current_classes]
            return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()

        def loss_fn_eval(params, x_batch, y_batch):
            net_tmp = nnx.merge(nnx.graphdef(self.net), params)
            logits = net_tmp(x_batch)[:, current_classes]
            return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()

        # Create HVP functions
        # self.net.train()
        hvp_fn_train, unravel, num_params = get_hvp_fn(
            loss_fn_train,
            nnx.state(self.net), # pass the params 
            (x_train, y_train_int)
        )

        # self.net.eval()
        hvp_fn_eval, _, _ = get_hvp_fn(
            loss_fn_eval,
            nnx.state(self.net), # pass the params
            (x_eval, y_eval_int)
        )
        
        # FIXED: Create linear operators that work with current state
        # self.net.train()
        hvp_train = lambda v: hvp_fn_train(nnx.state(self.net), v)

        # self.net.eval()
        hvp_eval = lambda v: hvp_fn_eval(nnx.state(self.net), v)

        # Rest of Lanczos computation remains the same
        rng_key_train = jax.random.PRNGKey(self.random_seed + task_id * 1000)
        rng_key_eval = jax.random.PRNGKey(self.random_seed + task_id * 1000 + 1)
        
        tri_train, _ = lanczos_alg(hvp_train, num_params, order=100, rng_key=rng_key_train)
        tri_eval, _ = lanczos_alg(hvp_eval, num_params, order=100, rng_key=rng_key_eval)

        # Convert to spectral density
        dens_train, grid_train = tridiag_to_density([tri_train], grid_len=10_000, sigma_squared=1e-5)
        dens_eval, grid_eval = tridiag_to_density([tri_eval], grid_len=10_000, sigma_squared=1e-5)

        # Store results and create plots
        at_init = (tag == "start")
        
        jax.debug.callback(
            plot_hessian_spectrum,
            grid_train, dens_train, grid_eval, dens_eval,
            task_id, "cifar_jax_july9.1", at_init=at_init
        )

        # Log summary information
        max_eig_train = float(grid_train[dens_train.argmax()])
        max_eig_eval = float(grid_eval[dens_eval.argmax()])
        
        self._print(f"[Hessian-{tag}] Task {task_id} — "
                f"top-eigenvalue(train) ≈ {max_eig_train:.3g}, "
                f"top-eigenvalue(eval) ≈ {max_eig_eval:.3g}")
        
        # Store results
        if not hasattr(self, 'hessian_results'):
            self.hessian_results = {}
        
        key = f"task_{task_id}_{tag}"
        self.hessian_results[key] = {
            'train_eigenvalues': np.array(grid_train),
            'train_density': np.array(dens_train),
            'eval_eigenvalues': np.array(grid_eval), 
            'eval_density': np.array(dens_eval),
            'max_eig_train': max_eig_train,
            'max_eig_eval': max_eig_eval,
            'num_params': num_params,
            'current_classes': len(current_classes)
        }

    # def _compute_hessian(self, tag: str, train_loader: DataLoader, eval_loader: DataLoader, task_id: int):
        
    #     train_sample = next(iter(train_loader))
    #     eval_sample = next(iter(eval_loader))

    #     x_train = train_sample["image"]
    #     y_train = train_sample["label"]
    #     x_eval = eval_sample["image"]
    #     y_eval = eval_sample["label"]

    #     # keep only the first N examples
    #     N = self.compute_hessian_size
    #     x_train, y_train = x_train[:N], y_train[:N]
    #     x_eval, y_eval = x_eval[:N], y_eval[:N]

    #     # convert to JAX arrays and NHWC
    #     x_train = jnp.array(x_train.numpy())
    #     y_train = jnp.array(y_train.numpy())
    #     x_eval = jnp.array(x_eval.numpy()) 
    #     y_eval = jnp.array(y_eval.numpy())

    #     # Handle NCHW -> NHWC conversion
    #     if len(x_train.shape) == 4 and x_train.shape[1] == 3:
    #         x_train = jnp.transpose(x_train, (0, 2, 3, 1))
    #     if len(x_eval.shape) == 4 and x_eval.shape[1] == 3:
    #         x_eval = jnp.transpose(x_eval, (0, 2, 3, 1))

    #     y_train_int = jnp.argmax(y_train, axis=1)
    #     y_eval_int = jnp.argmax(y_eval, axis=1)

    #             # --------------- build loss function for HVP ----------------
    #     # The get_hvp_fn utility expects loss functions with signature: loss(params, x_batch, y_batch)
    #     def loss_fn_train(params, x_batch, y_batch):
    #         net_tmp = nnx.merge(nnx.graphdef(self.net), params)
    #         logits = net_tmp(x_batch)[:, self.all_classes[:self.current_num_classes]]
    #         return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()

    #     def loss_fn_eval(params, x_batch, y_batch):
    #         net_tmp = nnx.merge(nnx.graphdef(self.net), params)
    #         logits = net_tmp(x_batch)[:, self.all_classes[:self.current_num_classes]]
    #         return optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()

    #     # Create HVP functions with correct API
    #     hvp_fn_train, unravel, num_params = get_hvp_fn(
    #         loss_fn_train,
    #         nnx.state(self.net), 
    #         (x_train, y_train_int)
    #     )
    #     hvp_fn_eval, _, _ = get_hvp_fn(
    #         loss_fn_eval,
    #         nnx.state(self.net), 
    #         (x_eval, y_eval_int)
    #     )

    #     # Create pure linear operators for Lanczos
    #     hvp_train = lambda v: hvp_fn_train(nnx.state(self.net), v)
    #     hvp_eval = lambda v: hvp_fn_eval(nnx.state(self.net), v)

    #     # --------------- Lanczos 100 steps --------------------
    #     # Use different random keys for train and eval to avoid identical results
    #     rng_key_train = jax.random.PRNGKey(self.random_seed + task_id * 1000)
    #     rng_key_eval = jax.random.PRNGKey(self.random_seed + task_id * 1000 + 1)
        
    #     tri_train, _ = lanczos_alg(hvp_train, num_params, order=100, rng_key=rng_key_train)
    #     tri_eval, _ = lanczos_alg(hvp_eval, num_params, order=100, rng_key=rng_key_eval)

    #     # Convert to spectral density
    #     dens_train, grid_train = tridiag_to_density([tri_train], grid_len=10_000, sigma_squared=1e-5)
    #     dens_eval, grid_eval = tridiag_to_density([tri_eval], grid_len=10_000, sigma_squared=1e-5)

    #     # Store results and create plots
    #     at_init = (tag == "start")
        
    #     # Use JAX debug callback for plotting (follows MNIST pattern)
    #     jax.debug.callback(
    #         plot_hessian_spectrum,
    #         grid_train, dens_train, grid_eval, dens_eval,
    #         task_id, "cifar_jax_july8.2", at_init=at_init
    #     )

    #     # Log summary information
    #     max_eig_train = grid_train[dens_train.argmax()]
    #     max_eig_eval = grid_eval[dens_eval.argmax()]
        
    #     self._print(f"[Hessian-{tag}] Task {task_id} — "
    #                f"top-eigenvalue(train) ≈ {max_eig_train:.3g}, "
    #                f"top-eigenvalue(eval) ≈ {max_eig_eval:.3g}")
        
    #     # Optional: Store results in experiment results for later analysis
    #     if not hasattr(self, 'hessian_results'):
    #         self.hessian_results = {}
        
    #     key = f"task_{task_id}_{tag}"
    #     self.hessian_results[key] = {
    #         'train_eigenvalues': grid_train,
    #         'train_density': dens_train,
    #         'eval_eigenvalues': grid_eval,
    #         'eval_density': dens_eval,
    #         'max_eig_train': float(max_eig_train),
    #         'max_eig_eval': float(max_eig_eval)
    #     }

    def extend_classes(self, training_data: CifarDataSet, test_data: CifarDataSet, val_data: CifarDataSet):
        """
        Adds one new class to the data set with certain frequency
        """
        if (self.current_epoch % self.class_increase_frequency) == 0:
            self._print("Best accuracy in the task: {0:.4f}".format(self.best_accuracy))
            if self.early_stopping:
                # In JAX, we need to handle state restoration differently
                best_state = self.best_accuracy_model_parameters
                if best_state:
                    nnx.update(self.net, best_state)
            
            # Reset best accuracy tracking for new task - this is critical!
            self.best_accuracy = jnp.array(0.0, dtype=jnp.float32)
            self.best_accuracy_model_parameters = {}
            self._save_model_parameters()

            if self.current_num_classes == self.num_classes: return

            increase = 5
            old_num_classes = self.current_num_classes
            self.current_num_classes += increase
            
            # 🔍 DEBUG: Add this debug block
            # self._print(f"🔍 TASK DEBUG - Adding new classes:")
            # self._print(f"  old_num_classes: {old_num_classes}")
            # self._print(f"  new_num_classes: {self.current_num_classes}")
            # self._print(f"  new_classes_added: {self.all_classes[old_num_classes:self.current_num_classes]}")
            
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew class added...")
            self._print(f"\tCurrent classes: {self.current_num_classes}/{self.num_classes}")
        
            if self.reset_head:
                self._reset_head()
            if self.reset_network:
                self._reset_network()

    def _save_model_parameters(self):
        """ Stores the parameters of the model, so it can be evaluated after the experiment is over """

        model_parameters_dir_path = os.path.join(self.results_dir, "model_parameters")
        os.makedirs(model_parameters_dir_path, exist_ok=True)

        file_name = "index-{0}_epoch-{1}.pkl".format(self.run_index, self.current_epoch)
        file_path = os.path.join(model_parameters_dir_path, file_name)

        # Extract and save only pure parameter arrays (compatible with post-analysis)
        model_snapshot = {
            "model_state": nnx.state(self.net),
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes[:self.current_num_classes],  # Only active classes
            "epoch": self.current_epoch,
            "accuracy": float(self.best_accuracy),
            "model_config": {
                "num_classes": self.num_classes,
                "image_dims": self.image_dims,
            }
        }
        
        # Convert JAX arrays to numpy for reliable serialization
        def jax_to_numpy(obj):
            if isinstance(obj, dict):
                return {key: jax_to_numpy(value) for key, value in obj.items()}
            elif hasattr(obj, 'shape') and hasattr(obj, '__array__'):
                return np.array(obj)
            else:
                return obj
        
        serializable_snapshot = jax_to_numpy(model_snapshot)
        store_object_with_several_attempts(serializable_snapshot, file_path, storing_format="pickle", num_attempts=10)

    def _reset_head(self):
        """Reset only the final classification layer (fc) with proper random initialization"""
        # new random key for the head layer 
        head_key = jax.random.PRNGKey(self.random_seed + self.current_num_classes)
        
        # update the final layer
        self.net.fc = nnx.Linear(
            in_features=self.net.fc.in_features,
            out_features=self.net.fc.out_features,
            kernel_init=nnx.initializers.kaiming_normal(),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(head_key)
        )
        
        # update state via nnx.Optimizer
        tx = optax.chain(
            optax.add_decayed_weights(self.weight_decay),
            optax.sgd(learning_rate=self.stepsize, momentum=self.momentum, nesterov=False)
        )
        self.optim = nnx.Optimizer(self.net, tx)

        if self.verbose:
            print(f"Reset head layer with new random key")

    def _reset_network(self):
        """Reset the entire network with proper random initialization"""
        # Generate new random key for full network initialization
        network_key = jax.random.PRNGKey(self.random_seed + self.current_num_classes + 1000)
        
        # reset the entire network - already with kaiming normal initialization
        self.net = build_resnet18(
            num_classes=self.num_classes,
            rngs=nnx.Rngs(network_key) # new random key
        )

        tx = optax.chain(
            optax.add_decayed_weights(self.weight_decay),
            optax.sgd(learning_rate=self.stepsize, momentum=self.momentum, nesterov=False)
        )
        self.optim = nnx.Optimizer(self.net, tx)
        
        if self.verbose:
            print(f"Reset entire network with new random key")

def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action="store", type=str,
                        default='./incremental_cifar/cfg/base_deep_learning_system.json',
                        help="Path to the file containing the parameters for the experiment.")
    parser.add_argument("--experiment-index", action="store", type=int, default=0,
                        help="Index for the run; this will determine the random seed and the name of the results.")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Whether to print extra information about the experiment as it's running.")
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        experiment_parameters = json.load(config_file)

    file_path = os.path.dirname(os.path.abspath(__file__))
    if "data_path" not in experiment_parameters.keys() or experiment_parameters["data_path"] == "":
        experiment_parameters["data_path"] = os.path.join(file_path, "data")
    if "results_dir" not in experiment_parameters.keys() or experiment_parameters["results_dir"] == "":
        experiment_parameters["results_dir"] = os.path.join(file_path, "results")
    if "experiment_name" not in experiment_parameters.keys() or experiment_parameters["experiment_name"] == "":
        experiment_parameters["experiment_name"] = os.path.splitext(os.path.basename(args.config_file))

    initial_time = time.perf_counter()
    exp = IncrementalCIFARExperiment(experiment_parameters,
                                     results_dir=os.path.join(experiment_parameters["results_dir"], experiment_parameters["experiment_name"]),
                                     run_index=args.experiment_index,
                                     verbose=args.verbose)
    exp.run()
    exp.store_results()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()


# python3 incremental_cifar_experiment_jax.py --config ./cfg/base_deep_learning_system.json --verbose --experiment-index 0
# python3 ./plots/plot_incremental_cifar_results.py --results_dir ./results/ --algorithms base_deep_learning_system --metric test_accuracy_per_epoch