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

@jax.jit
def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)

class Agent:
    def __init__(self, network):
        self.network = network
        
    def predict(self, x):
        return self.network(x)
    
    def loss(self, params, x, y):
        # Reconstruct network from graph definition and parameters
        net_temp = nnx.merge(nnx.graphdef(self.network), params)
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

class ShrinkAndPerturbAgent(Agent):
    def __init__(self, network, shrink_factor: float, perturb_scale: float):
        super().__init__(network)
        self.shrink_factor = shrink_factor
        self.perturb_scale = perturb_scale
    
    def perturb_params(self, params, rng, scale):
        """Add N(0, scale) noise to every parameter tensor in the tree."""
        leaves, treedef = jax.tree_util.tree_flatten(params)
        rngs = jax.random.split(rng, len(leaves))
        
        new_leaves = [
            p + scale * jax.random.normal(r, p.shape, p.dtype)
            for p, r in zip(leaves, rngs)
        ]
        return jax.tree_util.tree_unflatten(treedef, new_leaves), rngs[-1]

    def train_step(self, net_state, opt_state, optimizer, batch, rng):
        # Standard gradient update
        new_net_state, new_opt_state, loss, logits = super().train_step(
            net_state, opt_state, optimizer, batch
        )
        
        # Shrink and perturb
        shrunk_params = jax.tree_util.tree_map(
            lambda p: p * (1.0 - self.shrink_factor), new_net_state
        )
        
        rng, perturb_rng = jax.random.split(rng)
        perturbed_params, _ = self.perturb_params(shrunk_params, perturb_rng, self.perturb_scale)
        
        return perturbed_params, new_opt_state, loss, logits, rng

class EffectiveRankAgent(Agent):
    def __init__(self, network):
        super().__init__(network)
        
    def predict_with_features(self, params, x):
        net_temp = nnx.merge(nnx.graphdef(self.network), params)
        return net_temp(x, return_features=True)  # Assuming your network supports this
    
    def effective_rank(self, features, eps=1e-8):
        sv = jnp.linalg.svdvals(features.T)
        sv = jnp.abs(sv)  
        total = jnp.maximum(sv.sum(), eps)
        p = sv / total
        entropy = -(p * jnp.log(p + eps)).sum()
        return jnp.exp(entropy)
    
    def effective_rank_loss(self, params, x):
        output, features = self.predict_with_features(params, x)
        erank_losses = [self.effective_rank(f) for f in features.values() if f is not None]
        return -jnp.stack(erank_losses).mean()

# Helper function to create the same ResNet18 used in other experiments
def create_resnet18_agent(num_classes=100, random_seed=42, agent_type="base"):
    """Create an agent with the same ResNet18 architecture used in incremental experiments."""
    rngs = nnx.Rngs(random_seed)
    network = build_resnet18(num_classes=num_classes, norm_layer=nnx.BatchNorm, rngs=rngs)
    
    if agent_type == "shrink_perturb":
        return ShrinkAndPerturbAgent(network, shrink_factor=0.1, perturb_scale=0.01)
    elif agent_type == "effective_rank":
        return EffectiveRankAgent(network)
    else:
        return Agent(network)