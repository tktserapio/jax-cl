from math import sqrt
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


def get_layer_bound(layer, init, gain):
    """Get initialization bounds for a layer"""
    if hasattr(layer, 'features'):  # Conv or Dense layer
        if hasattr(layer, 'kernel_size'):  # Conv layer
            fan_in = layer.features * layer.kernel_size[0] * layer.kernel_size[1]
        else:  # Dense layer  
            fan_in = layer.features
            
        if init == 'default':
            bound = sqrt(1 / fan_in)
        elif init == 'xavier':
            fan_out = layer.features
            bound = gain * sqrt(6 / (fan_in + fan_out))
        elif init == 'lecun':
            bound = sqrt(3 / fan_in)
        else:
            bound = gain * sqrt(3 / fan_in)
        return bound
    return 0.1  # fallback


def get_layer_std(layer, gain):
    """Get standard deviation for layer initialization"""
    if hasattr(layer, 'features'):  # Conv or Dense layer
        if hasattr(layer, 'kernel_size'):  # Conv layer
            fan_in = layer.features * layer.kernel_size[0] * layer.kernel_size[1]
        else:  # Dense layer
            fan_in = layer.features
        return gain * sqrt(1 / fan_in)
    return 0.1  # fallback


class ResGnT(object):
    """
    Generate-and-Test algorithm for a Linen ResNet, assuming only one fully connected layer at the top and that
    there is no pooling at the end
    """
    def __init__(self, net_fn, params, hidden_activation, decay_rate=0.99, replacement_rate=1e-4, util_type='weight',
                 maturity_threshold=1000, device=None):
        super(ResGnT, self).__init__()

        self.net_fn = net_fn  # The network function (Linen model)
        self.params = params  # Network parameters
        self.device = device or jax.devices("cpu")[0]
        
        # Initialize layer tracking
        self.bn_layers = []
        self.weight_layers = []
        self.get_weight_layers()
        self.num_hidden_layers = len(self.weight_layers) - 1

        # Hyperparameters
        self.hidden_activation = hidden_activation
        self.decay_rate = decay_rate
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type

        # For tracking neuron utility and maturity
        self.utility = []
        self.maturity = []
        self.neuron_count = []
        
        # Initialize utility tracking for each layer
        for i, layer_params in enumerate(self.weight_layers):
            if i < self.num_hidden_layers:  # Skip output layer
                layer_shape = layer_params['kernel'].shape if 'kernel' in layer_params else layer_params['weight'].shape
                num_neurons = layer_shape[-1]  # Output features
                self.utility.append(jnp.zeros(num_neurons))
                self.maturity.append(jnp.zeros(num_neurons))
                self.neuron_count.append(num_neurons)

        # Random number generator
        self.rng = jax.random.PRNGKey(42)

    def get_weight_layers(self):
        """Extract weight layers from Linen model parameters"""
        def extract_layers(params, path=""):
            for key, value in params.items():
                current_path = f"{path}/{key}" if path else key
                if isinstance(value, dict):
                    if 'kernel' in value or 'weight' in value:
                        # This is a layer with weights
                        self.weight_layers.append(value)
                        print(f"Found weight layer at: {current_path}")
                    else:
                        # Recurse into nested dictionaries
                        extract_layers(value, current_path)
        
        extract_layers(self.params)

    def generate_and_test(self, features):
        """
        Generate new neurons and test existing ones based on feature activations
        """
        if len(features) == 0:
            return
            
        # Update utilities based on current features
        self.update_utilities(features)
        
        # Update maturity counters
        for i in range(len(self.maturity)):
            self.maturity[i] = self.maturity[i] + 1
            
        # Perform replacement if mature enough
        for layer_idx in range(self.num_hidden_layers):
            if jnp.min(self.maturity[layer_idx]) > self.maturity_threshold:
                self.replace_neurons(layer_idx, features)

    def gen_and_test(self, features):
        """Alias for generate_and_test to match PyTorch version"""
        self.generate_and_test(features)

    def update_utilities(self, features):
        """Update utility estimates for each neuron based on current activations"""
        for layer_idx, feature_map in enumerate(features):
            if layer_idx >= len(self.utility):
                continue
                
            # Flatten spatial dimensions if this is a conv layer
            if len(feature_map.shape) == 4:  # (batch, height, width, channels)
                activations = jnp.mean(feature_map, axis=(0, 1, 2))  # Average over batch and spatial dims
            elif len(feature_map.shape) == 2:  # (batch, features)
                activations = jnp.mean(feature_map, axis=0)  # Average over batch
            else:
                continue
                
            if self.util_type == 'weight':
                # Use activation magnitude as utility
                current_utility = jnp.abs(activations)
            else:  # contribution
                # Use some measure of contribution (simplified)
                current_utility = activations ** 2
                
            # Update utility with decay
            self.utility[layer_idx] = (self.decay_rate * self.utility[layer_idx] + 
                                     (1 - self.decay_rate) * current_utility)

    def replace_neurons(self, layer_idx, features):
        """Replace low-utility neurons with new random ones"""
        if layer_idx >= len(self.utility):
            return
            
        utilities = self.utility[layer_idx]
        num_to_replace = max(1, int(self.replacement_rate * len(utilities)))
        
        # Find neurons with lowest utility
        lowest_indices = jnp.argsort(utilities)[:num_to_replace]
        
        print(f"Replacing {num_to_replace} neurons in layer {layer_idx}")
        
        # Reset utility and maturity for replaced neurons
        self.utility[layer_idx] = self.utility[layer_idx].at[lowest_indices].set(0.0)
        self.maturity[layer_idx] = self.maturity[layer_idx].at[lowest_indices].set(0)
        
        # In practice, you would also reinitialize the weights for these neurons
        # This is simplified - in a full implementation you'd update the model parameters
        
    def get_stats(self):
        """Get statistics about the current state"""
        stats = {}
        for i, utility in enumerate(self.utility):
            stats[f'layer_{i}_mean_utility'] = float(jnp.mean(utility))
            stats[f'layer_{i}_min_utility'] = float(jnp.min(utility))
            stats[f'layer_{i}_max_utility'] = float(jnp.max(utility))
            stats[f'layer_{i}_mean_maturity'] = float(jnp.mean(self.maturity[i]))
        return stats
