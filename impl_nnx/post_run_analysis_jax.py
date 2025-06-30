"""
Script for computing the effective rank, stable rank, number of dormant neurons, and average weight magnitude of the
JAX/NNX models trained during the incremental cifar experiment.

This is the JAX/NNX version of the post-run analysis script, compatible with models saved from incremental_cifar_experiment_jax.py
"""

# built-in libraries
import time
import os
import sys
import argparse

# third party libraries
from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax import nnx
import pickle
from torch.utils.data import DataLoader  # Still use PyTorch DataLoader for data loading
import numpy as np
from torchvision import transforms
from scipy.linalg import svd

# from ml project manager
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize

from torchvision_modified_resnet_jax import build_resnet18


# -------------------- For loading data and network parameters -------------------- #
def load_model_parameters(parameter_dir_path: str, index: int, epoch_number: int):
    """
    Loads the model parameters stored in parameter_dir_path corresponding to the index and epoch number
    return: Dictionary of pure JAX/numpy arrays
    """

    model_parameters_file_name = "index-{0}_epoch-{1}.pkl".format(index, epoch_number)
    model_parameters_file_path = os.path.join(parameter_dir_path, model_parameters_file_name)

    if not os.path.isfile(model_parameters_file_path):
        error_message = "Couldn't find model parameters for index {0} and epoch number {1}.".format(index, epoch_number)
        raise ValueError(error_message)

    with open(model_parameters_file_path, 'rb') as f:
        params = pickle.load(f)
        
        # Convert numpy arrays back to JAX arrays for use
        def numpy_to_jax(obj):
            if isinstance(obj, dict):
                return {key: numpy_to_jax(value) for key, value in obj.items()}
            elif isinstance(obj, np.ndarray):
                return jnp.array(obj)
            else:
                return obj
        
        return numpy_to_jax(params)


def load_classes(classes_dir_path: str, index: int):
    """
    Loads the list of ordered classes used for partitioning the datta during the experiment
    return: list
    """

    classes_file_name = "index-{0}.npy".format(index)
    classes_file_path = os.path.join(classes_dir_path, classes_file_name)

    if not os.path.isfile(classes_file_path):
        error_message = "Couldn't find list of classes for index {0}.".format(index)
        raise ValueError(error_message)

    return np.load(classes_file_path)


def load_cifar_data(data_path: str, train: bool = True):
    """
    Loads the cifar 100 data set with normalization
    :param data_path: path to the directory containing the data set
    :param train: bool that indicates whether to load the train or test data
    :return: CifarDataSet and DataLoader objects
    """
    cifar_data = CifarDataSet(root_dir=data_path,
                              train=train,
                              cifar_type=100,
                              device=None,
                              image_normalization="max",
                              label_preprocessing="one-hot",
                              use_torch=False)  # Use JAX format

    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)

    transformations = [
        ToTensor(swap_color_axis=True),  # reshape to (C x H x W)
        Normalize(mean=mean, std=std),  # center by mean and divide by std
    ]

    # Use JAX-compatible transforms
    class JAXCompose:
        def __init__(self, transforms):
            self.transforms = transforms
        
        def __call__(self, x):
            for transform in self.transforms:
                x = transform(x)
            return x

    cifar_data.set_transformation(JAXCompose(transformations))

    num_workers = 12
    batch_size = 1000
    dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return cifar_data, dataloader


# -------------------- For computing analysis of the network -------------------- #
def compute_average_weight_magnitude(net):
    """ Computes average magnitude of the weights in the JAX/NNX network """

    num_weights = 0
    sum_weight_magnitude = 0.0

    # Get network state (parameters)
    net_state = nnx.state(net)
    
    def count_and_sum_weights(obj, path=""):
        nonlocal num_weights, sum_weight_magnitude
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                count_and_sum_weights(value, path + f".{key}" if path else key)
        elif hasattr(obj, 'shape') and hasattr(obj, 'size'):  # JAX array
            num_weights += obj.size
            sum_weight_magnitude += float(jnp.sum(jnp.abs(obj)))
            # Debug print for first few weights
            if num_weights < 1000:
                print(f"  Found weights at {path}: shape={obj.shape}, size={obj.size}")
    
    count_and_sum_weights(net_state)
    
    if num_weights == 0:
        print("Warning: No weights found in network state")
        return 0.0
    
    return float(sum_weight_magnitude / num_weights)


def compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold: float = 0.01):
    """
    Computes the proportion of dormant units in a JAX/NNX ResNet. It also returns the features of the last layer 
    for the first batch of samples
    """

    features_per_layer = []
    last_layer_activations = None

    for i, sample in enumerate(cifar_data_loader):
        # Convert PyTorch tensors to JAX arrays
        image = jnp.array(sample["image"].detach().cpu().numpy())
        
        # Convert from NCHW to NHWC format for JAX/Flax
        if len(image.shape) == 4 and image.shape[1] == 3:
            image = jnp.transpose(image, (0, 2, 3, 1))
        
        # Get features during forward pass
        temp_features = []
        _ = net(image, temp_features)

        features_per_layer = temp_features
        last_layer_activations = np.array(temp_features[-1])  # Convert to numpy for analysis
        break

    # Count dead neurons
    dead_neurons = jnp.zeros(len(features_per_layer))
    
    for layer_idx in range(len(features_per_layer) - 1):
        # For convolutional layers (4D: NHWC)
        if len(features_per_layer[layer_idx].shape) == 4:
            # Mean over batch, height, width dimensions
            activation_rates = jnp.mean(features_per_layer[layer_idx] != 0, axis=(0, 1, 2))
        else:
            # For dense layers (2D: NC)
            activation_rates = jnp.mean(features_per_layer[layer_idx] != 0, axis=0)
        
        dead_neurons = dead_neurons.at[layer_idx].set(jnp.sum(activation_rates < dormant_unit_threshold))
    
    # Handle last layer (usually dense)
    if len(features_per_layer[-1].shape) == 2:
        activation_rates = jnp.mean(features_per_layer[-1] != 0, axis=0)
    else:
        activation_rates = jnp.mean(features_per_layer[-1] != 0, axis=(0, 1, 2))
    
    dead_neurons = dead_neurons.at[-1].set(jnp.sum(activation_rates < dormant_unit_threshold))
    
    # Calculate total number of features
    total_features = 0
    for layer_feats in features_per_layer:
        if len(layer_feats.shape) == 4:  # Conv layer NHWC
            total_features += layer_feats.shape[-1]  # Number of channels
        else:  # Dense layer
            total_features += layer_feats.shape[-1]  # Number of units
    
    return float(jnp.sum(dead_neurons) / total_features), last_layer_activations


def compute_effective_rank(singular_values: np.ndarray):
    """ Computes the effective rank of the representation layer """

    norm_sv = singular_values / np.sum(np.abs(singular_values))
    entropy = 0.0
    for p in norm_sv:
        if p > 0.0:
            entropy -= p * np.log(p)

    return np.e ** entropy


def compute_stable_rank(singular_values: np.ndarray):
    """ Computes the stable rank of the representation layer """
    sorted_singular_values = np.flip(np.sort(singular_values))
    cumsum_sorted_singular_values = np.cumsum(sorted_singular_values) / np.sum(singular_values)
    return np.sum(cumsum_sorted_singular_values < 0.99) + 1


def compute_last_task_accuracy_per_class_in_order(net, ordered_classes: np.ndarray,
                                                  test_data, experiment_index: int):
    """
    Computes the accuracy of each class in the order they were presented
    :param net: JAX/NNX resnet with the parameters stored at the end of the experiment
    :param ordered_classes: numpy array with the cifar 100 classes in the order they were presented
    :param test_data: cifar100 test data loader
    :return: numpy array
    """

    ordered_classes = np.int32(ordered_classes)
    num_classes = 100
    num_examples_per_class = 100

    class_correct = jnp.zeros(num_classes)
    
    for i, sample in enumerate(test_data):
        # Convert PyTorch tensors to JAX arrays
        image = jnp.array(sample["image"].detach().cpu().numpy())
        labels = jnp.array(sample["label"].detach().cpu().numpy())
        
        # Convert from NCHW to NHWC format for JAX/Flax
        if len(image.shape) == 4 and image.shape[1] == 3:
            image = jnp.transpose(image, (0, 2, 3, 1))
        
        # Forward pass
        outputs = net(image)
        predicted = jnp.argmax(outputs, axis=1)  # Get the class with the highest score
        true_labels = jnp.argmax(labels, axis=1)  # Get the class with the highest score

        # Update the counts for each class
        for i, class_label in enumerate(ordered_classes):
            mask = (true_labels == class_label)
            correct_predictions = (predicted == true_labels) & mask
            class_correct = class_correct.at[i].add(jnp.sum(correct_predictions))

    return np.array(class_correct / num_examples_per_class)


# -------------------- For storing the results of the analysis -------------------- #
def store_analysis_results(weight_magnitude_results: np.ndarray,
                           dormant_units_results: (np.ndarray, np.ndarray),
                           effective_rank_results: (np.ndarray, np.ndarray),
                           stable_rank_results: (np.ndarray, np.ndarray),
                           accuracy_per_class_in_order: np.ndarray,
                           results_dir: str, experiment_index: int):
    """
    Stores the results of the post run analysis
    :param weight_magnitude_results: np array containing the output of compute_average_weight_magnitude
    :param dormant_units_results: tuple containing the results of the dormant unit analysis for the previous tasks and
                                  the next task for each different task
    :param effective_rank_results: tuple containing the results of the effective rank analysis for the previous tasks
                                   and the next task for each different task
    :param stable_rank_results: tuple containing the results of the stable rank analysis for the previous tasks and the
                                next task for each different task
    :param accuracy_per_class_in_order: np array containing the accuracy of the final model for each class in the order
                                        they were presented
    :param results_dir: path to the results directory
    :param experiment_index: experiment index
    """

    index_file_name = "index-{0}.npy".format(experiment_index)
    result_dir_names_and_arrays = [
        ("weight_magnitude_analysis", weight_magnitude_results),
        ("previous_tasks_dormant_units_analysis", dormant_units_results[0]),
        ("next_task_dormant_units_analysis", dormant_units_results[1]),
        ("previous_tasks_effective_rank_analysis", effective_rank_results[0]),
        ("next_task_effective_rank_analysis", effective_rank_results[1]),
        ("previous_tasks_stable_rank_analysis", stable_rank_results[0]),
        ("next_task_stable_rank_analysis", stable_rank_results[1]),
        ("accuracy_per_class_in_order", accuracy_per_class_in_order)
    ]

    # store results in the corresponding dir
    for results_name, results_array in result_dir_names_and_arrays:
        temp_results_dir = os.path.join(results_dir, results_name)
        os.makedirs(temp_results_dir, exist_ok=True)
        np.save(os.path.join(temp_results_dir, index_file_name), results_array)


def analyze_results(results_dir: str, data_path: str, dormant_unit_threshold: float = 0.01):
    """
    Analyses the parameters of a run and creates files with the results of the analysis
    :param results_dir: path to directory containing the results for a parameter combination
    :param data_path: path to the cifar100 data set
    :param dormant_unit_threshold: hidden units whose activation fall bellow this threshold are considered dormant
    """

    parameter_dir_path = os.path.join(results_dir, "model_parameters")
    experiment_indices_file_path = os.path.join(results_dir, "experiment_indices.npy")
    class_order_dir_path = os.path.join(results_dir, "class_order")

    # JAX doesn't need explicit device management like PyTorch
    print("Using JAX backend:", jax.lib.xla_bridge.get_backend().platform)
    
    number_of_epochs = np.arange(21) * 200  # by design the model parameters were stored after each of these epochs
    classes_per_task = 5                    # by design each task increases the data set by 5 classes
    last_epoch = 4000
    experiment_indices = np.load(experiment_indices_file_path)

    # Initialize JAX/NNX ResNet
    rngs = nnx.Rngs(42)  # Use a fixed seed for reproducibility
    net = build_resnet18(num_classes=100, norm_layer=nnx.BatchNorm, rngs=rngs)
    
    cifar_data, cifar_data_loader = load_cifar_data(data_path, train=True)
    test_data, test_data_loader = load_cifar_data(data_path, train=False)

    for exp_index in tqdm(experiment_indices):
        ordered_classes = load_classes(class_order_dir_path, index=exp_index)

        average_weight_magnitude_per_epoch = np.zeros(number_of_epochs.size - 1, dtype=np.float32)
        dormant_units_prop_before = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_before = np.zeros_like(average_weight_magnitude_per_epoch)
        dormant_units_prop_after = np.zeros_like(average_weight_magnitude_per_epoch)
        effective_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)
        stable_rank_after = np.zeros_like(average_weight_magnitude_per_epoch)

        for i, epoch_number in enumerate(number_of_epochs[:-1]):
            # Load model parameters from before training on the task
            model_parameters = load_model_parameters(parameter_dir_path, index=exp_index, epoch_number=epoch_number)
            
            # Update JAX/NNX network with loaded parameters
            nnx.update(net, model_parameters)

            # Compute average weight magnitude
            average_weight_magnitude_per_epoch[i] = compute_average_weight_magnitude(net)

            # Compute summaries for next task
            current_classes = ordered_classes[(i * classes_per_task):((i + 1) * classes_per_task)]
            cifar_data.select_new_partition(current_classes)

            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)
            dormant_units_prop_after[i] = prop_dormant
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            effective_rank_after[i] = compute_effective_rank(singular_values)
            stable_rank_after[i] = compute_stable_rank(singular_values)

            # Compute summaries from data from previous tasks
            if i == 0: continue
            current_classes = ordered_classes[:(i * classes_per_task)]
            cifar_data.select_new_partition(current_classes)
            prop_dormant, last_layer_features = compute_dormant_units_proportion(net, cifar_data_loader, dormant_unit_threshold)

            dormant_units_prop_before[i] = prop_dormant
            singular_values = svd(last_layer_features, compute_uv=False, lapack_driver="gesvd")
            effective_rank_before[i] = compute_effective_rank(singular_values)
            stable_rank_before[i] = compute_stable_rank(singular_values)

        # Load final model and compute per-class accuracy
        final_model_parameters = load_model_parameters(parameter_dir_path, exp_index, last_epoch)
        nnx.update(net, final_model_parameters)
        accuracy_per_class_in_order = compute_last_task_accuracy_per_class_in_order(net, ordered_classes,
                                                                                    test_data_loader, exp_index)

        store_analysis_results(weight_magnitude_results=average_weight_magnitude_per_epoch,
                               dormant_units_results=(dormant_units_prop_before, dormant_units_prop_after),
                               effective_rank_results=(effective_rank_before, effective_rank_after),
                               stable_rank_results=(stable_rank_before, stable_rank_after),
                               accuracy_per_class_in_order=accuracy_per_class_in_order,
                               results_dir=results_dir,
                               experiment_index=exp_index)


def parse_arguments() -> dict:

    file_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', action="store", type=str,
                        default=os.path.join(file_path, "results", "base_deep_learning_system"),
                        help="Path to directory with the results of a parameter combination.")
    parser.add_argument('--data_path', action="store", type=str, default=os.path.join(file_path, "data"),
                        help="Path to directory with the CIFAR 100 data set.")
    parser.add_argument('--dormant_unit_threshold', action="store", type=float, default=0.01,
                        help="Units whose activations are less than this threshold are considered dormant.")

    args = parser.parse_args()
    return vars(args)


def main():

    analysis_arguments = parse_arguments()

    initial_time = time.perf_counter()
    analyze_results(results_dir=analysis_arguments["results_dir"],
                    data_path=analysis_arguments["data_path"],
                    dormant_unit_threshold=analysis_arguments["dormant_unit_threshold"])
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()

# python3 post_run_analysis_jax.py --results_dir ./results/base_deep_learning_system/
# python3 ./plots/plot_incremental_cifar_results.py --results_dir ./results/ --algorithms base_deep_learning_system --metric test_accuracy_per_epoch