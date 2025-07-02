import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax import nnx
import optax

sys.path.append('../')
from utils.file_system import get_results_path, numpyify, plot_hessian_spectrum
import orbax.checkpoint
from utils.optimizer import l2_regularization, adam_with_param_counts
from utils.hessian_computation import get_hvp_fn
from utils.lanczos import lanczos_alg
from utils.density import tridiag_to_density

# CIFAR experiment imports
from torchvision_modified_resnet_jax import build_resnet18
from torch.utils.data.dataloader import DataLoader
from data.CIFAR100 import CifarDataSet, subsample_cifar_data_set
from utils.data_utils import JAXCompose, ToTensor, Normalize

@jax.jit
def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)

class Agent:
    """NNX-compatible Agent class for ResNet18"""
    def __init__(self, network):
        self.network = network
        
    def predict(self, net_state, x):
        net_temp = nnx.merge(nnx.graphdef(self.network), net_state)
        return net_temp(x)
    
    def loss(self, net_state, x, y):
        net_temp = nnx.merge(nnx.graphdef(self.network), net_state)
        output = net_temp(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits=output, labels=y).mean()
    
    def train_step(self, net_state, opt_state, optimizer, batch):
        def loss_fn(params):
            net_temp = nnx.merge(nnx.graphdef(self.network), params)
            logits = net_temp(batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch['label']).mean()
            return loss, logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(net_state)
        updates, new_opt_state = optimizer.update(grads, opt_state, net_state)
        new_net_state = optax.apply_updates(net_state, updates)
        
        return new_net_state, new_opt_state, loss, logits



train_images_per_class = 450  # CIFAR-100 train samples per class
test_images_per_class = 100   # CIFAR-100 test samples per class
validation_images_per_class = 50  # CIFAR-100 validation samples per class

data_path = exp_params["data_path"]

def load_cifar100(classes=[]):
    """Load CIFAR-100 data for specified classes, similar to incremental experiment"""
    # Create CIFAR dataset
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
    
    # Set up transformations like in incremental experiment
    transformations = [ToTensor(), Normalize(mean=mean, std=std)]
    cifar_data.set_transformation(JAXCompose(transformations))
    
    # Select only the specified classes
    cifar_data.select_new_partition(classes)
    
    # Split into train/validation/test like incremental experiment
    num_val_samples_per_class = 50
    num_train_samples_per_class = 450
    validation_set_size = len(classes) * num_val_samples_per_class
    train_set_size = len(classes) * num_train_samples_per_class
    
    # Get indices for train/validation split
    validation_indices = np.zeros(validation_set_size, dtype=np.int32)
    train_indices = np.zeros(train_set_size, dtype=np.int32)
    current_val_samples = 0
    current_train_samples = 0
    
    for i, class_idx in enumerate(classes):
        indices_for_class = np.where(np.array(cifar_data.targets) == class_idx)[0]
        validation_indices[current_val_samples:current_val_samples + num_val_samples_per_class] = \
            indices_for_class[:num_val_samples_per_class]
        train_indices[current_train_samples:current_train_samples + num_train_samples_per_class] = \
            indices_for_class[num_val_samples_per_class:num_val_samples_per_class + num_train_samples_per_class]
        
        current_val_samples += num_val_samples_per_class
        current_train_samples += num_train_samples_per_class
    
    # Create train dataset
    train_cifar = CifarDataSet()
    train_cifar.set_transformation(JAXCompose(transformations))
    train_cifar.select_new_partition(classes)
    subsample_cifar_data_set(sub_sample_indices=train_indices, cifar_data=train_cifar)
    
    # Create test dataset (use original test set)
    test_cifar = CifarDataSet(train=False)
    test_cifar.set_transformation(JAXCompose(transformations))
    test_cifar.select_new_partition(classes)
    
    # Convert to JAX arrays with proper format (NHWC)
    x_train, y_train = [], []
    for i in range(len(train_cifar)):
        sample = train_cifar[i]
        x_train.append(sample['image'])
        y_train.append(i // num_train_samples_per_class)  # Relabel to 0, 1, 2, ...
    
    x_test, y_test = [], []
    for i in range(len(test_cifar)):
        sample = test_cifar[i]
        x_test.append(sample['image'])
        y_test.append(i // test_images_per_class)  # Relabel to 0, 1, 2, ...
    
    x_train = jnp.array(x_train, dtype=jnp.float32)
    y_train = jnp.array(y_train)
    x_test = jnp.array(x_test, dtype=jnp.float32)
    y_test = jnp.array(y_test)
    
    return x_train, y_train, x_test, y_test

def save_data(data, data_file):
    with open(data_file, 'wb+') as f:
        pickle.dump(data, f)

def repeat_expr(params: {}):
    agent_type = params['agent']
    num_tasks = params['num_tasks']
    num_showings = params['num_showings']

    step_size = params['step_size']
    replacement_rate = 0.0001
    decay_rate = 0.99
    maturity_threshold = 100
    util_type = 'contribution'
    opt = params['opt']
    weight_decay = 0
    use_gpu = 0
    dev='cpu'
    num_classes = 10
    total_classes = 100  # CIFAR-100 has 100 classes
    new_heads = False
    mini_batch_size = 100
    perturb_scale = 0
    momentum = 0
    net_type = 1
    compute_hessian = False
    compute_hessian_size = 50

    if 'replacement_rate' in params.keys(): replacement_rate = params['replacement_rate']
    if 'decay_rate' in params.keys(): decay_rate = params['decay_rate']
    if 'util_type' in params.keys(): util_type = params['util_type']
    if 'maturity_threshold' in params.keys():   maturity_threshold = params['maturity_threshold']
    if 'weight_decay' in params.keys(): weight_decay = params['weight_decay']       
    if 'num_classes' in params.keys():  num_classes = params['num_classes']
    if 'new_heads' in params.keys():    new_heads = params['new_heads']
    if 'mini_batch_size' in params.keys():  mini_batch_size = params['mini_batch_size']
    if 'perturb_scale' in params.keys():    perturb_scale = params['perturb_scale']
    if 'momentum' in params.keys(): momentum = params['momentum']
    if 'new_heads' in params.keys(): new_heads = params['new_heads']
    if 'net_type' in params.keys(): net_type = params['net_type']
    if 'compute_hessian' in params.keys(): compute_hessian = params['compute_hessian']
    if 'compute_hessian_size' in params.keys(): compute_hessian_size = params['compute_hessian_size']


    print(params)
    
    num_epochs = num_showings
    classes_per_task = num_classes

    rng = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(0)  # For NNX
    dummy_input = jnp.ones([1, 32, 32, 3])
    
    # Use ResNet18 like incremental experiment
    network = build_resnet18(num_classes=num_classes, norm_layer=nnx.BatchNorm, rngs=rngs)
    agent = Agent(network)
    
    # Get network state using NNX
    net_state = nnx.state(network)
    
    tx = optax.sgd(step_size, momentum)
    opt_state = tx.init(net_state)

    # CIFAR-100 class order (0-99)
    cifar_classes = list(range(100))
    np.random.seed(params.get('run_idx', 0))
    class_order = np.random.permutation(cifar_classes)
    
    num_class_repetitions_required = int(num_classes * num_tasks / total_classes) + 1
    class_order = np.concatenate([class_order]*num_class_repetitions_required)

    save_after_every_n_tasks = 1
    if num_tasks >= 10:
        save_after_every_n_tasks = int(num_tasks/10)

    examples_per_epoch = train_images_per_class * classes_per_task

    train_accuracies = np.zeros((num_tasks, num_epochs))
    test_accuracies = np.zeros((num_tasks, num_epochs))

    x_train, x_test, y_train, y_test = None, None, None, None
    for task_idx in range(num_tasks):
        del x_train, x_test, y_train, y_test
        x_train, y_train, x_test, y_test = load_cifar100(class_order[task_idx*classes_per_task:(task_idx+1)*classes_per_task])

        # if new_heads:
        #     # Reset final layer for new task (NNX style)
        #     # This would need to be implemented based on ResNet18 structure
        #     pass


       

        for epoch_idx in tqdm(range(num_epochs)):
            new_train_accuracies = []
            for start_idx in range(0, examples_per_epoch, mini_batch_size):
                batch_x = x_train[start_idx: start_idx + mini_batch_size]
                batch_y = y_train[start_idx: start_idx + mini_batch_size]
                batch = {'image': batch_x, 'label': batch_y}


                 #compute hessian at the start or end of each new epoch
                if compute_hessian and ((epoch_idx == 0 and start_idx == 0) or (epoch_idx == num_epochs - 1 and start_idx == examples_per_epoch - mini_batch_size)):
                    # Hessian computation on test set
                    # Hessian computation on train set
                    x_hessian, y_hessian = batch_x[:compute_hessian_size], batch_y[:compute_hessian_size]
                    batch_hessian = {'image': x_hessian, 'label': y_hessian}
                    hvp_fn, unravel, num_params = get_hvp_fn(agent.loss, net_state, (x_hessian, y_hessian))
                    hvp_cl = lambda v: hvp_fn(net_state, v)
                    rng, _rng = jax.random.split(rng)
                    tridiag, lanczos_vecs = lanczos_alg(
                        hvp_cl,
                        num_params,
                        order=100,
                        rng_key=rng
                    )
                    density_train, grids_train = tridiag_to_density([tridiag], grid_len=10000, sigma_squared=1e-5)

                    if start_idx == 0:
                        jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_train, density_train, task_idx, params['agent_type'], at_init=True)
                    else:
                        jax.debug.callback(plot_hessian_spectrum, grids_train, density_train, grids_train, density_train, task_idx, params['agent_type'], at_init=False)

            


                net_state, opt_state, loss, logits = agent.train_step(net_state, opt_state, tx, batch)
                
                # Update network
                nnx.update(network, net_state)
                
                new_train_accuracies.append(compute_accuracy(logits, batch_y))

            train_accuracies[task_idx][epoch_idx] = np.mean(new_train_accuracies)

            new_test_accuracies = []
            for start_idx in range(0, x_test.shape[0], mini_batch_size):
                test_batch_x = x_test[start_idx: start_idx + mini_batch_size]
                test_batch_y = y_test[start_idx: start_idx + mini_batch_size]
                logits = agent.predict(net_state, test_batch_x)
                new_test_accuracies.append(compute_accuracy(logits, test_batch_y))

            test_accuracies[task_idx][epoch_idx] = np.mean(new_test_accuracies)
            print('accuracy for task', task_idx, 'in epoch', epoch_idx, ': train, ',
                  train_accuracies[task_idx][epoch_idx], ', test,', test_accuracies[task_idx][epoch_idx])

        if task_idx % save_after_every_n_tasks == 0:
            save_data(data={
                'train_accuracies': train_accuracies,
                'test_accuracies': test_accuracies,
            }, data_file=params['data_file'])

    save_data(data={
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
    }, data_file=params['data_file'])

def main(arguments):
    # params = {
    #     'agent': 'bp',
    #     'num_tasks': 2000,
    #     'num_showings': 100,
    #     'step_size': 0.01,
    #     'opt': 'sgd',
    #     'run_idx': 0,
    #     'num_classes': 2,
    #     'data_file': 'data/imagenet_bp_jax.pkl',
    # }

    params = {
            "agent": "bp",
            "num_tasks": 50,  # Reduced for CIFAR-100 (100 classes / 2 classes per task)
            "num_classes": 2,
            "num_showings": 100, #250
            "mini_batch_size": 100,
            "opt": "sgd",
            "step_size": 0.01,
            "momentum": 0.9,
            "weight_decay": 0,
            "run_idx": 0,
            "data_file": "data/cifar100_bp_jax.pkl",
            "compute_hessian": True,
            "compute_hessian_size": 200,
            "agent_type": "bp",
        }
    repeat_expr(params)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))