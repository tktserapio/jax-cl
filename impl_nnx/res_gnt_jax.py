# from torch.nn import Conv2d, Linear, BatchNorm2d
# from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
from math import sqrt
# import torch
import sys
# from torch.nn.init import calculate_gain

from flax import nnx
from functools import partial
import jax
import jax.numpy as jnp
import optax


def get_layer_bound(layer, init, gain):
    if isinstance(layer, nnx.Conv):
        return sqrt(1 / (layer.in_features * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nnx.Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


def get_layer_std(layer, gain):
    if isinstance(layer, nnx.Conv):
        return gain * sqrt(1 / (layer.in_features * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nnx.Linear):
        return gain * sqrt(1 / layer.in_features)


class ResGnT(object):
    """
    Generate-and-Test algorithm for a simple resnet, assuming only one fully connected layer at the top and that
    there is no pooling at the end
    """
    def __init__(self, net, hidden_activation, decay_rate=0.99, replacement_rate=1e-4, util_type='weight',
                 maturity_threshold=1000, device=jax.devices("cpu")):
        super(ResGnT, self).__init__()

        self.net = net
        self.bn_layers = []
        self.weight_layers = []
        self.get_weight_layers(nn_module=self.net)
        self.num_hidden_layers = len(self.weight_layers) - 1
        self.device = device

        """
        Define the hyper-parameters of the algorithm
        """
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        """
        Utility of all features/neurons
        """
        self.util, self.ages, self.mean_feature_mag = [], [], []
        self.initialized = False  # Flag to track if we've initialized with actual feature shapes

        self.accumulated_num_features_to_replace = []  # Will be initialized later
        # Note: jax.nn.softmax is a function, not a stored operation
        self.m = jax.nn.softmax  # Remove dim parameter as it's not used the same way

        """
        Calculate uniform distribution's bound for random feature initialization
        """
        self.stds = self.compute_std(hidden_activation=hidden_activation)
        """
        Some pre-calculation - will be completed after feature initialization
        """
        self.num_new_features_to_replace = []  # Will be initialized later

    def get_weight_layers(self, nn_module: nnx.Module):
        if isinstance(nn_module, nnx.Conv) or isinstance(nn_module, nnx.Linear):
            self.weight_layers.append(nn_module)
        elif isinstance(nn_module, nnx.BatchNorm):
            self.bn_layers.append(nn_module)
        else:
            # can't just do for m in nn_module.children() as we do in pytorch
            if hasattr(nn_module, '__dict__'):
                for attr_name, attr_value in nn_module.__dict__.items():
                    if isinstance(attr_value, nnx.Module):
                        if hasattr(nn_module, 'downsample') and nn_module.downsample == attr_value:
                            continue
                        self.get_weight_layers(nn_module=attr_value)
                    elif isinstance(attr_value, list):
                        for item in attr_value:
                            if isinstance(item, nnx.Module):
                                self.get_weight_layers(nn_module=item)

    def compute_std(self, hidden_activation):
        stds = []
        # we implement gain calculation manually
        if hidden_activation == 'relu':
            gain = jnp.sqrt(2.0)
        elif hidden_activation == 'leaky_relu':
            gain = jnp.sqrt(2.0 / (1 + 0.01**2))
        elif hidden_activation in ['tanh', 'sigmoid']:
            gain = 1.0
        elif hidden_activation == 'linear':
            gain = 1.0
        else:
            gain = 1.0  # Default gain
            
        for i in range(self.num_hidden_layers):
            stds.append(get_layer_std(layer=self.weight_layers[i], gain=gain))
        stds.append(get_layer_std(layer=self.weight_layers[-1], gain=1))
        return stds

    def initialize_from_features(self, features):
        """Initialize ResGnT arrays based on actual feature shapes from the network"""
        if self.initialized:
            return
            
        self.util, self.ages, self.mean_feature_mag = [], [], []
        
        # Initialize based on actual feature shapes
        for i, feat in enumerate(features):
            if i >= self.num_hidden_layers:
                break
                
            # Get the channel dimension from the actual feature
            if len(feat.shape) == 4:  # Conv features: [batch, H, W, channels]
                num_channels = feat.shape[-1]
            elif len(feat.shape) == 2:  # Linear features: [batch, features]
                num_channels = feat.shape[-1]
            else:
                raise ValueError(f"Unexpected feature shape: {feat.shape}")
                
            self.util.append(jnp.zeros(num_channels, dtype=jnp.float32))
            self.ages.append(jnp.zeros(num_channels, dtype=jnp.float32))
            self.mean_feature_mag.append(jnp.zeros(num_channels, dtype=jnp.float32))
        
        # Update num_hidden_layers to match actual features
        self.num_hidden_layers = len(self.util)
        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]
        
        # Recalculate num_new_features_to_replace based on actual features
        self.num_new_features_to_replace = []
        for i in range(self.num_hidden_layers):
            if len(features[i].shape) == 4:
                out_features = features[i].shape[-1]
            elif len(features[i].shape) == 2:
                out_features = features[i].shape[-1]
            else:
                raise ValueError(f"Unexpected feature shape: {features[i].shape}")
            self.num_new_features_to_replace.append(self.replacement_rate * out_features)
            
        self.initialized = True

    def test_features(self, features):
        """
        Args:
            features: Activation values in the neural network
        Returns:
            Features to replace in each layer, Number of features to replace in each layer
        """
        # Initialize on first call with actual feature shapes
        if not self.initialized:
            self.initialize_from_features(features)
            
        features_to_replace = [jnp.array([], dtype=jnp.int32) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace

        # Only process features that we have tracking arrays for
        num_features_to_process = min(len(features), self.num_hidden_layers)
        
        for i in range(num_features_to_process):
            self.ages[i] = self.ages[i] + 1  # JAX arrays are immutable, create new array
            """
            Update feature stats
            """
            # jax no need for no_grad context: it's opt-in rather than opt-in for torch
            if len(features[i].shape) == 2:
                self.mean_feature_mag[i] = self.mean_feature_mag[i] + (1 - self.decay_rate) * jnp.abs(features[i]).mean(axis=0)
            elif len(features[i].shape) == 4:
                self.mean_feature_mag[i] = self.mean_feature_mag[i] + (1 - self.decay_rate) * jnp.abs(features[i]).mean(axis=(0, 2, 3))
            """
            Find the no. of features to replace
            """
            eligible_feature_indices = jnp.where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0:
                continue
            self.accumulated_num_features_to_replace[i] += self.num_new_features_to_replace[i]

            """
            Case when the number of features to be replaced is between 0 and 1.
            """
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
            self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace

            if num_new_features_to_replace == 0: continue

            """
            Calculate utility
            """
            # JAX doesn't need no_grad context
            next_layer = self.weight_layers[i + 1]
            if isinstance(next_layer, nnx.Linear):
                output_wight_mag = jnp.abs(next_layer.kernel.value).mean(axis=0)
            elif isinstance(next_layer, nnx.Conv):
                output_wight_mag = jnp.abs(next_layer.kernel.value).mean(axis=(0, 2, 3))

            if self.util_type == 'weight':
                self.util[i] = output_wight_mag
            elif self.util_type in ['contribution']:
                self.util[i] = output_wight_mag * self.mean_feature_mag[i]

            """
            Find features to replace in the current layer
            """
            new_features_to_replace = jnp.argsort(-self.util[i][eligible_feature_indices])[:num_new_features_to_replace] # topk
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]

            """
            Initialize utility for new features
            """
            self.util[i] = self.util[i].at[new_features_to_replace].set(0)

            num_features_to_replace[i] = num_new_features_to_replace
            features_to_replace[i] = new_features_to_replace

        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        """
        Generate new features: Reset input and output weights for low utility features
        """
        # removed no_grad
        for i in range(self.num_hidden_layers):
            if num_features_to_replace[i] == 0:
                continue
            current_layer, next_layer = self.weight_layers[i], self.weight_layers[i+1]

            # since jax is functional, we need to create new arrs instead of in-place modif
            key = jax.random.key(0)  # You might want to make this configurable
            new_weights_shape = [num_features_to_replace[i]] + list(current_layer.kernel.value.shape[1:])
            new_weights = jax.random.normal(key, new_weights_shape) * self.stds[i]
            
            current_layer.kernel.value = current_layer.kernel.value.at[features_to_replace[i], :].set(new_weights)
            
            if hasattr(current_layer, 'bias') and current_layer.bias is not None:
                current_layer.bias.value = current_layer.bias.value.at[features_to_replace[i]].set(0.0)
            
            """
            Set the outgoing weights and ages to zero
            """
            if isinstance(next_layer, nnx.Linear):
                next_layer.kernel.value = next_layer.kernel.value.at[:, features_to_replace[i]].set(0)
            elif isinstance(next_layer, nnx.Conv):
                next_layer.kernel.value = next_layer.kernel.value.at[:, features_to_replace[i]].set(0)
                
            self.ages[i] = self.ages[i].at[features_to_replace[i]].set(0)
            
            """
            Reset the corresponding batchnorm layers
            """
            if i < len(self.bn_layers):
                bn_layer = self.bn_layers[i]
                if hasattr(bn_layer, 'bias') and bn_layer.bias is not None:
                    bn_layer.bias.value = bn_layer.bias.value.at[features_to_replace[i]].set(0.0)
                if hasattr(bn_layer, 'scale') and bn_layer.scale is not None:
                    bn_layer.scale.value = bn_layer.scale.value.at[features_to_replace[i]].set(1.0)
                
                # JAX/Flax nnx BatchNorm uses 'mean' and 'var' instead of 'running_mean'/'running_var'
                if hasattr(bn_layer, 'mean') and bn_layer.mean is not None:
                    bn_layer.mean.value = bn_layer.mean.value.at[features_to_replace[i]].set(0.0)
                if hasattr(bn_layer, 'var') and bn_layer.var is not None:
                    bn_layer.var.value = bn_layer.var.value.at[features_to_replace[i]].set(1.0)

    def gen_and_test(self, features):
        """
        Perform generate-and-test
        :param features: activation of hidden units in the neural network
        """
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)