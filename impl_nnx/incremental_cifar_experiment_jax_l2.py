import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax 
from torch.utils.data import DataLoader 
import time 
from tqdm import tqdm 

import json
import os
import argparse

# CifarDataSet imports (replacing torchvision CIFAR100)
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator

# Add this helper function after the imports
def access_dict(dictionary, key, default=None, val_type=None):
    """Helper function to access dictionary values with defaults and type checking"""
    if key in dictionary:
        value = dictionary[key]
        if val_type is not None:
            try:
                return val_type(value)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert {key}={value} to {val_type}, using default {default}")
                return default
        return value
    return default

# Modify the initialization section (around line 130)
def load_config(config_path="base_deep_learning_system.json"):
    """Load experiment configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
        
        # Filter out comment fields (keys starting with _comment_ or _model_description_)
        filtered_config = {k: v for k, v in config.items() 
                          if not k.startswith('_comment_') and not k.startswith('_model_description_')}
        return filtered_config
    else:
        print(f"Config file {config_path} not found, using defaults")
        config = {}
    return config

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

def get_data(data_path, train=True, validation=False, batch_sizes=None, num_workers=0):
    """
    Loads CIFAR data set - adapted from incremental_cifar_experiment_jax.py
    """
    if batch_sizes is None:
        batch_sizes = {"train": 90, "test": 100, "validation": 50}
    
    # Load CIFAR-100 with CifarDataSet (one-hot labels)
    cifar_data = CifarDataSet(root_dir=data_path,
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
        batch_size = batch_sizes["test"]
        dataloader = DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return cifar_data, dataloader

    train_indices, validation_indices = get_validation_and_train_indices(cifar_data)
    indices = validation_indices if validation else train_indices
    subsample_cifar_data_set(sub_sample_indices=indices, cifar_data=cifar_data)
    batch_size = batch_sizes["validation"] if validation else batch_sizes["train"]
    return cifar_data, DataLoader(cifar_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_validation_and_train_indices(cifar_data: CifarDataSet):
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
    for i in range(100):  # 100 classes in CIFAR-100
        # Use numpy operations for index manipulation
        class_indices = np.where(cifar_data.data["labels"][:, i] == 1)[0]
        validation_indices[current_val_samples:(current_val_samples + num_val_samples_per_class)] = class_indices[:num_val_samples_per_class]
        train_indices[current_train_samples:(current_train_samples + num_train_samples_per_class)] = class_indices[num_val_samples_per_class:]
        current_val_samples += num_val_samples_per_class
        current_train_samples += num_train_samples_per_class

    return train_indices, validation_indices

from modified_resnet_nnx import build_resnet18

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='JAX Incremental CIFAR Experiment')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., ./cfg/base_deep_learning_system.json)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--experiment-index', type=int, default=0,
                       help='Experiment index for multiple runs')
    args = parser.parse_args()

    # Load configuration from file
    config = load_config(args.config)
    
    if args.verbose:
        print(f"Running experiment {args.experiment_index}")
        print(f"Config loaded from: {args.config}")

    # Extract hyperparameters from config (using actual config keys)
    stepsize = access_dict(config, "stepsize", default=0.1, val_type=float)
    momentum = access_dict(config, "momentum", default=0.9, val_type=float)
    weight_decay = access_dict(config, "weight_decay", default=0.0005, val_type=float)
    noise_std = access_dict(config, "noise_std", default=0.0, val_type=float)
    data_path = access_dict(config, "data_path", default="./data", val_type=str)
    results_dir = access_dict(config, "results_dir", default="./results", val_type=str)
    experiment_name = access_dict(config, "experiment_name", default="base_deep_learning_system", val_type=str)
    num_workers = access_dict(config, "num_workers", default=1, val_type=int)
    
    # CBP parameters (for future use)
    use_cbp = access_dict(config, "use_cbp", default=False, val_type=bool)
    
    # Network reset parameters (for future use)
    reset_head = access_dict(config, "reset_head", default=False, val_type=bool)
    reset_network = access_dict(config, "reset_network", default=False, val_type=bool)
    early_stopping = access_dict(config, "early_stopping", default=True, val_type=bool)
    
    # Hessian computation parameters (for future use)
    compute_hessian = access_dict(config, "compute_hessian", default=False, val_type=bool)
    compute_hessian_size = access_dict(config, "compute_hessian_size", default=100, val_type=int)
    compute_hessian_interval = access_dict(config, "compute_hessian_interval", default=1, val_type=int)
    
    # Fixed experiment parameters (not in config file)
    num_epochs = 4000
    class_increase_frequency = 200
    running_avg_window = 25
    batch_size_train = 90
    batch_size_test = 100
    batch_size_validation = 50
    num_classes_start = 5
    num_classes_total = 100
    num_classes_increment = 5
    random_seed = 0

    # Use empty string as current directory if data_path is empty
    if data_path == "":
        data_path = "./data"
    if results_dir == "":
        results_dir = "./results"

    if args.verbose:
        print("Experiment Configuration:")
        print(f"  Learning rate (stepsize): {stepsize}")
        print(f"  Momentum: {momentum}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Noise std: {noise_std}")
        print(f"  Data path: {data_path}")
        print(f"  Results dir: {results_dir}")
        print(f"  Experiment name: {experiment_name}")
        print(f"  Use CBP: {use_cbp}")
        print(f"  Compute Hessian: {compute_hessian}")
        if compute_hessian:
            print(f"    Hessian size: {compute_hessian_size}")
            print(f"    Hessian interval: {compute_hessian_interval}")

    # Set random seed
    np.random.seed(random_seed)
    jax.config.update('jax_default_prng_impl', 'rbg')

    # Initialize model
    model = build_resnet18(num_classes=num_classes_total, rngs=nnx.Rngs(random_seed))

    # Initialize optimizer with optax
    tx = optax.chain(
        optax.add_decayed_weights(config['weight_decay']), 
        optax.sgd(learning_rate=config['stepsize'], momentum=config['momentum'], nesterov=False)
    )

    from flax.training import train_state 

    variables = nnx.state(model)
    params = variables['params'] if 'params' in variables else variables 

    def apply_fn(variables, x, **kwargs):
        nnx.update(model, variables)
        return model(x)

    state = train_state.TrainState.create(
        apply_fn=apply_fn, 
        params=params, 
        tx=tx,
    )

    # Initialize dataset using CifarDataSet pipeline
    batch_sizes = {
        "train": batch_size_train,
        "test": batch_size_test,
        "validation": batch_size_validation
    }
    
    # Set random seed for class ordering
    np.random.seed(random_seed)
    all_classes = np.random.permutation(100)  # Random class order for incremental learning
    current_num_classes = num_classes_start

    # Run the experiment
    results = run_incremental_experiment(
        model=model,
        state=state,
        data_path=data_path,
        all_classes=all_classes,
        current_num_classes=current_num_classes,
        batch_sizes=batch_sizes,
        num_workers=num_workers,  # Force to 0 for JAX compatibility
        config={
            'num_epochs': num_epochs,
            'class_increase_frequency': class_increase_frequency,
            'running_avg_window': running_avg_window,
            'noise_std': noise_std,
            'stepsize': stepsize,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'num_classes_increment': num_classes_increment,
            'num_classes_total': num_classes_total,
            'verbose': args.verbose,
            # Future features
            'use_cbp': use_cbp,
            'reset_head': reset_head,
            'reset_network': reset_network,
            'early_stopping': early_stopping,
            'compute_hessian': compute_hessian,
            'compute_hessian_size': compute_hessian_size,
            'compute_hessian_interval': compute_hessian_interval,
        }
    )

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_filename = f"{experiment_name}_exp{args.experiment_index}_results.json"
    results_path = os.path.join(results_dir, results_filename)
    
    # Add config and experiment info to results
    results['config'] = config
    results['experiment_index'] = args.experiment_index
    results['command_line_args'] = vars(args)
    
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0 and hasattr(value[0], 'item'):
                json_results[key] = [v.item() if hasattr(v, 'item') else v for v in value]
            else:
                json_results[key] = value
        json.dump(json_results, f, indent=2)
    
    if args.verbose:
        print(f"Results saved to {results_path}")
    
    return results

# Loss function for TrainState (matches PyTorch exactly with one-hot labels)
@jax.jit
def train_step(state, images, labels, current_classes):
    """Train for a single step using TrainState with one-hot labels"""
    # Convert from NCHW to NHWC format if needed
    if len(images.shape) == 4 and images.shape[1] == 3:
        images = jnp.transpose(images, (0, 2, 3, 1))
    
    def loss_fn(params):
        logits_full = state.apply_fn(params, images)
        logits = logits_full[:, current_classes]
        
        # Convert one-hot labels to integer labels
        labels_int = jnp.argmax(labels, axis=1)
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels_int
        ).mean()
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)
    
    # Compute accuracy
    labels_int = jnp.argmax(labels, axis=1)
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels_int)
    
    return new_state, loss, accuracy

@jax.jit
def eval_step(state, images, labels, current_classes):
    """Evaluate on current classes only using TrainState with one-hot labels"""
    # Convert from NCHW to NHWC format if needed
    if len(images.shape) == 4 and images.shape[1] == 3:
        images = jnp.transpose(images, (0, 2, 3, 1))
    
    logits_full = state.apply_fn(state.params, images)
    logits = logits_full[:, current_classes]
    
    # Convert one-hot labels to integer labels
    labels_int = jnp.argmax(labels, axis=1)
    
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels_int
    ).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=1) == labels_int)
    
    return loss, accuracy

def inject_noise(params, noise_std=0.0):
    """Add Gaussian noise to model parameters (shrink & perturb)"""
    if noise_std <= 0.0:
        return params
    
    def add_noise_to_param(param):
        if isinstance(param, jnp.ndarray):
            key = jax.random.PRNGKey(np.random.randint(0, 10000))
            noise = jax.random.normal(key, param.shape) * noise_std
            return param + noise
        return param
    
    # Apply noise to all parameters
    return jax.tree_util.tree_map(add_noise_to_param, params)

def run_incremental_experiment(model, state, data_path, all_classes, current_num_classes, batch_sizes, num_workers, config):
    # Extract config parameters
    num_epochs = config['num_epochs']
    class_increase_frequency = config['class_increase_frequency']
    running_avg_window = config['running_avg_window']
    noise_std = config['noise_std']
    num_classes_increment = config['num_classes_increment']
    num_classes_total = config['num_classes_total']
    verbose = config['verbose']
    
    # Results tracking
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "best_train_accuracy": [],
        "best_test_accuracy": [],
        "best_validation_accuracy": [],
    }

    # Track best accuracies per task
    best_train_acc = 0.0
    best_test_acc = 0.0
    best_val_acc = 0.0
    
    current_epoch = 0
    running_loss = 0.0
    running_accuracy = 0.0
    current_state = state
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\tEpoch number: {epoch + 1}")
        
        # Get current classes 
        current_classes_list = all_classes[:current_num_classes]
        current_classes = jnp.array(current_classes_list)
        
        # Get data loaders for current classes using CifarDataSet pipeline
        training_data, train_loader = get_data(data_path, train=True, validation=False, 
                                               batch_sizes=batch_sizes, num_workers=num_workers)
        val_data, val_loader = get_data(data_path, train=True, validation=True, 
                                        batch_sizes=batch_sizes, num_workers=num_workers)
        test_data, test_loader = get_data(data_path, train=False, validation=False, 
                                          batch_sizes=batch_sizes, num_workers=num_workers)
        
        # Filter data to current classes (matches PyTorch exactly)
        training_data.select_new_partition(current_classes_list)
        val_data.select_new_partition(current_classes_list)
        test_data.select_new_partition(current_classes_list)

        task_epoch = epoch % class_increase_frequency

        current_lr = None # initialize to None
        if task_epoch == 0:
            current_lr = config['stepsize']
        elif task_epoch == 60:
            current_lr = round(config['stepsize'] * 0.2, 5)
        elif task_epoch == 120:
            current_lr = round(config['stepsize'] * 0.04, 5)
        elif task_epoch == 160:
            current_lr = round(config['stepsize'] * 0.008, 5)

        if current_lr is not None:
            tx = optax.chain(
                optax.add_decayed_weights(config['weight_decay']), 
                optax.sgd(learning_rate=current_lr, momentum=config['momentum'], nesterov=False)
            )
            current_state = current_state.replace(tx=tx)
            if verbose:
                print(f"\t\tLearning rate updated to: {current_lr:.5f}")
        
        epoch_start_time = time.perf_counter()
        
        # Training phase
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        num_train_steps = 0

        for step, sample in enumerate(train_loader):
            # Convert PyTorch tensors to JAX arrays OUTSIDE the JIT function
            images = jnp.array(sample["image"].numpy())
            labels = jnp.array(sample["label"].numpy())
            
            current_state, loss, accuracy = train_step(current_state, images, labels, current_classes)
            
            # Add noise (shrink & perturb)
            if noise_std > 0.0:
                noisy_params = inject_noise(current_state.params, noise_std)
                current_state = current_state.replace(params=noisy_params)
            
            # Running averages
            running_loss += loss.item()
            running_accuracy += accuracy.item()
            epoch_train_loss += loss.item()
            epoch_train_acc += accuracy.item()
            num_train_steps += 1
            
            if verbose and (step + 1) % running_avg_window == 0:
                avg_loss = running_loss / running_avg_window
                avg_acc = running_accuracy / running_avg_window
                print(f"\t\tStep Number: {step + 1}")
                print(f"\t\tOnline accuracy: {avg_acc:.2f}")
                
                running_loss = 0.0
                running_accuracy = 0.0
        
        # Calculate epoch averages
        if num_train_steps > 0:
            epoch_train_loss /= num_train_steps
            epoch_train_acc /= num_train_steps
        
        # Evaluation phase

        test_losses, test_accs = [], []
        for sample in test_loader:
            # Convert PyTorch tensors to JAX arrays OUTSIDE the JIT function
            images = jnp.array(sample["image"].numpy())
            labels = jnp.array(sample["label"].numpy())
            
            loss, acc = eval_step(current_state, images, labels, current_classes)
            test_losses.append(loss.item())
            test_accs.append(acc.item())
        
        test_loss = np.mean(test_losses) if test_losses else 0.0
        test_acc = np.mean(test_accs) if test_accs else 0.0
        
        # Validation evaluation

        val_losses, val_accs = [], []
        for sample in val_loader:
            # Convert PyTorch tensors to JAX arrays OUTSIDE the JIT function
            images = jnp.array(sample["image"].numpy())
            labels = jnp.array(sample["label"].numpy())
            
            loss, acc = eval_step(current_state, images, labels, current_classes)
            val_losses.append(loss.item())
            val_accs.append(acc.item())
        
        val_loss = np.mean(val_losses) if val_losses else 0.0
        val_acc = np.mean(val_accs) if val_accs else 0.0
        
        epoch_time = time.perf_counter() - epoch_start_time

        # Update best accuracies
        if epoch_train_acc > best_train_acc:
            best_train_acc = epoch_train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Store results
        results["train_loss"].append(epoch_train_loss)
        results["train_accuracy"].append(epoch_train_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)
        results["validation_loss"].append(val_loss)
        results["validation_accuracy"].append(val_acc)

        # Store best accuracies
        results["best_train_accuracy"].append(best_train_acc)
        results["best_test_accuracy"].append(best_test_acc)
        results["best_validation_accuracy"].append(best_val_acc)
        
        # Print results
        if verbose:
            print(f"\t\ttrain accuracy: {epoch_train_acc:.4f} (best: {best_train_acc:.4f})")
            print(f"\t\ttest accuracy: {test_acc:.4f} (best: {best_test_acc:.4f})")
            print(f"\t\tvalidation accuracy: {val_acc:.4f} (best: {best_val_acc:.4f})")
            print(f"\t\tEpoch run time in seconds: {epoch_time:.4f}")
        
        current_epoch += 1
        
        # Add new classes every class_increase_frequency epochs
        if current_epoch % class_increase_frequency == 0:
            if current_num_classes < num_classes_total:
                current_num_classes = min(num_classes_total, current_num_classes + num_classes_increment)
                # Reset best accuracies for new task
                best_train_acc = 0.0
                best_test_acc = 0.0
                best_val_acc = 0.0
                if verbose:
                    print("\tNew classes added... (resetting best accuracies)")
    
    return results

# Run the experiment
if __name__ == "__main__":
    main()