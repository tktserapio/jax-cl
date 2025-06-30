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
from torchvision_modified_resnet_jax import build_resnet18
from res_gnt_jax import ResGnT

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
                print(f"✅ JAX is using GPU: {self.device}")
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
        
        # Print additional JAX device information
        print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
        print(f"JAX version: {jax.__version__}")
        all_devices = jax.devices()
        print(f"All available JAX devices: {all_devices}")

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
        self.net = build_resnet18(num_classes=self.num_classes, norm_layer=nnx.BatchNorm, rngs=rngs)
        # already initialized with kaiming init in the torchvision_modified_resnet_jax.py file

        # initialize optimizer - JAX/Optax style
        self.optim = optax.sgd(learning_rate=self.stepsize, momentum=self.momentum)
        # Initialize optimizer state
        self.optim_state = self.optim.init(nnx.state(self.net))

        # define loss function
        self.loss = optax.softmax_cross_entropy_with_integer_labels  # JAX loss function

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
            "optim_state": self.optim_state,     # JAX optimizer state
            "jax_rng_state": self.rng_key,       # JAX random state
            "numpy_rng_state": np.random.get_state(),
            "epoch_number": self.current_epoch,
            "current_num_classes": self.current_num_classes,
            "all_classes": self.all_classes,
            "current_running_avg_step": self.current_running_avg_step,
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
        self.optim_state = checkpoint["optim_state"]
        self.rng_key = checkpoint["jax_rng_state"]
        np.random.set_state(checkpoint["numpy_rng_state"])
        
        self.current_epoch = checkpoint["epoch_number"]
        self.current_num_classes = checkpoint["current_num_classes"]
        self.all_classes = checkpoint["all_classes"]
        self.current_running_avg_step = checkpoint["current_running_avg_step"]

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

        # JAX doesn't a dedicated eval mode like pytorch
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

        # jax doesn't need to switch back to train mode, as we never switched to dedicated eval mode
        self._print("\t\tEpoch run time in seconds: {0:.4f}".format(epoch_runtime))

    def evaluate_network(self, test_data):
        """
        Evaluates the network on the test data
        :param test_data: a data loader or iterable
        :return: (jnp.Array) test loss, (jnp.Array) test accuracy
        """

        avg_loss = 0.0
        avg_acc = 0.0
        num_test_batches = 0
        
        # JAX doesn't need no_grad context
        for _, sample in enumerate(test_data):
            # Convert PyTorch tensors from DataLoader to JAX arrays
            images = jnp.array(sample["image"].detach().cpu().numpy())
            test_labels = jnp.array(sample["label"].detach().cpu().numpy())
            
            # JAX/Flax expects NHWC format, but PyTorch gives us NCHW
            # Convert from NCHW to NHWC
            if len(images.shape) == 4 and images.shape[1] == 3:  # (N, C, H, W) -> (N, H, W, C)
                images = jnp.transpose(images, (0, 2, 3, 1))
            
            # Convert one-hot to integer labels for JAX loss function
            test_labels_int = jnp.argmax(test_labels, axis=1)
            
            # Forward pass
            test_predictions = self.net(images)[:, self.all_classes[:self.current_num_classes]]

            # Compute loss and accuracy
            avg_loss += self.loss(test_predictions, test_labels_int).mean()
            avg_acc += jnp.mean(test_predictions.argmax(axis=1) == test_labels_int)
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

        training_data.select_new_partition(self.all_classes[:self.current_num_classes])
        test_data.select_new_partition(self.all_classes[:self.current_num_classes])
        val_data.select_new_partition(self.all_classes[:self.current_num_classes])
        self._save_model_parameters()

        # JIT-compiled training step to match PyTorch logic exactly
        @jax.jit
        def train_step_jit(net_state, opt_state, images, labels, class_indices):
            def loss_fn(params):
                # Reconstruct network from graph definition and parameters
                net_temp = nnx.merge(nnx.graphdef(self.net), params)
                predictions = net_temp(images)[:, class_indices]
                labels_int = jnp.argmax(labels, axis=1)
                return optax.softmax_cross_entropy_with_integer_labels(predictions, labels_int).mean()
            
            # Compute loss and gradients
            loss_val, grads = jax.value_and_grad(loss_fn)(net_state)
            
            # Apply optimizer update (equivalent to backward() + step())
            updates, new_opt_state = self.optim.update(grads, opt_state)
            new_net_state = optax.apply_updates(net_state, updates)
            
            return new_net_state, new_opt_state, loss_val

        for e in tqdm(range(self.current_epoch, self.num_epochs)):
            self._print("\tEpoch number: {0}".format(e + 1))
            self.set_lr()

            epoch_start_time = time.perf_counter()
            for step_number, sample in enumerate(train_dataloader):
                # sample observation and target
                image = jnp.array(sample["image"].detach().cpu().numpy())
                label = jnp.array(sample["label"].detach().cpu().numpy())
                
                # JAX/Flax expects NHWC format, but PyTorch gives us NCHW
                # Convert from NCHW to NHWC
                if len(image.shape) == 4 and image.shape[1] == 3:  # (N, C, H, W) -> (N, H, W, C)
                    image = jnp.transpose(image, (0, 2, 3, 1))

                # reset gradients (implicit in JAX functional approach)
                # JAX doesn't need explicit gradient reset

                # compute prediction and loss
                current_features = [] if self.use_cbp else None
                net_state = nnx.state(self.net)
                class_indices = self.all_classes[:self.current_num_classes]
                
                # Forward pass to get predictions (matching PyTorch logic)
                predictions = self.net(image, current_features)[:, class_indices]
                labels_int = jnp.argmax(label, axis=1)
                current_reg_loss = optax.softmax_cross_entropy_with_integer_labels(predictions, labels_int).mean()
                current_loss = current_reg_loss  # JAX doesn't need .detach().clone()

                # backpropagate and update weights
                new_net_state, self.optim_state, _ = train_step_jit(
                    net_state, self.optim_state, image, label, class_indices
                )
                
                # Update network with new state (equivalent to optim.step())
                nnx.update(self.net, new_net_state)
                
                # CBP processing (matching PyTorch timing)
                if self.use_cbp: 
                    self.resgnt.gen_and_test(current_features)
                
                # Inject noise (matching PyTorch timing)
                self.inject_noise()

                # store summaries (matching PyTorch logic exactly)
                current_accuracy = jnp.mean(predictions.argmax(axis=1) == labels_int)
                self.running_loss += float(current_loss)
                self.running_accuracy += float(current_accuracy)
                if (step_number + 1) % self.running_avg_window == 0:
                    self._print("\t\tStep Number: {0}".format(step_number + 1))
                    self._store_training_summaries()

            epoch_end_time = time.perf_counter()
            self._store_test_summaries(test_dataloader, val_dataloader, epoch_number=e,
                                    epoch_runtime=epoch_end_time - epoch_start_time)

            self.current_epoch += 1
            self.extend_classes(training_data, test_data, val_data)

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
            # In JAX/Optax, we recreate the optimizer with new learning rate
            self.optim = optax.sgd(learning_rate=current_stepsize, momentum=self.momentum)
            # Note: We keep the same optimizer state, just update the learning rate
            self._print("\tCurrent stepsize: {0:.5f}".format(current_stepsize))

    def inject_noise(self):
        """
        Adds a small amount of random noise to the parameters of the network
        """
        if not self.perturb_weights_indicator: return

        # JAX approach - update parameters functionally
        def add_noise_to_params(params):
            def add_noise_to_param(param):
                if isinstance(param, jnp.ndarray):
                    noise = jax.random.normal(self.rng_key, param.shape) * self.noise_std
                    # Update the random key for next use
                    self.rng_key, _ = jax.random.split(self.rng_key)
                    return param + noise
                return param
            
            return jax.tree.map(add_noise_to_param, params)
        
        # Get current state, add noise, and update network
        current_state = nnx.state(self.net)
        noisy_state = add_noise_to_params(current_state)
        nnx.update(self.net, noisy_state)

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
            
            self.best_accuracy = jnp.zeros_like(self.best_accuracy)
            self.best_accuracy_model_parameters = {}
            self._save_model_parameters()

            if self.current_num_classes == self.num_classes: return

            increase = 5
            self.current_num_classes += increase
            training_data.select_new_partition(self.all_classes[:self.current_num_classes])
            test_data.select_new_partition(self.all_classes[:self.current_num_classes])
            val_data.select_new_partition(self.all_classes[:self.current_num_classes])

            self._print("\tNew class added...")
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
        model_state = nnx.state(self.net)
        
        def extract_pure_arrays(obj):
            """Convert NNX state to pure numpy/JAX arrays"""
            if isinstance(obj, dict):
                return {key: extract_pure_arrays(value) for key, value in obj.items()}
            elif hasattr(obj, 'shape') and hasattr(obj, '__array__'):  # JAX array
                return np.array(obj)  # Convert to numpy for reliable saving/loading
            else:
                return obj
        
        pure_params = extract_pure_arrays(model_state)
        store_object_with_several_attempts(pure_params, file_path, storing_format="pickle", num_attempts=10)

    def _reset_head(self):
        """Reset only the final classification layer (fc) with proper random initialization"""
        # Generate new random key for head initialization
        head_key = jax.random.PRNGKey(self.random_seed + self.current_num_classes)
        
        # Reinitialize the final linear layer with kaiming initialization
        self.net.fc = nnx.Linear(
            self.net.fc.in_features,
            self.net.fc.out_features,
            kernel_init=nnx.initializers.kaiming_normal(),
            rngs=nnx.Rngs(head_key)
        )
        
        # Reinitialize optimizer state for the new parameters
        self.optim_state = self.optim.init(nnx.state(self.net))
        
        if self.verbose:
            print(f"Reset head layer with new random key")

    def _reset_network(self):
        """Reset the entire network with proper random initialization"""
        # Generate new random key for full network initialization
        network_key = jax.random.PRNGKey(self.random_seed + self.current_num_classes + 1000)
        
        # Rebuild the entire network with new random keys
        from torchvision_modified_resnet_jax import build_resnet18
        self.net = build_resnet18(
            num_classes=self.num_classes, 
            norm_layer=nnx.BatchNorm, 
            rngs=nnx.Rngs(network_key)
        )
        
        # Reinitialize optimizer state for the new network
        self.optim_state = self.optim.init(nnx.state(self.net))
        
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