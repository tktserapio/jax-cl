#!/usr/bin/env python3
"""
Debug script to test data loading and label formats
"""

import sys
import os
sys.path.insert(0, '/users/tserapio/jax_cl/impl_linen')

import numpy as np
import torch
from torch.utils.data import DataLoader
import jax.numpy as jnp

# Import from the experiment
from mlproj_manager.problems import CifarDataSet
import torchvision.transforms
from mlproj_manager.util.data_preprocessing_and_transformations import (
    Normalize, RandomCrop, RandomHorizontalFlip, RandomRotator, ToTensor
)

def test_data_format():
    """Test CIFAR data loading and formatting"""
    print("Testing CIFAR data loading...")
    
    # Load data similar to experiment
    data_path = "/users/tserapio/jax_cl/impl_linen/data"
    
    cifar_data = CifarDataSet(
        root_dir=data_path,
        train=False,  # Use test set for simpler testing
        cifar_type=100,
        device=None,
        image_normalization="max",
        label_preprocessing="one-hot",
        use_torch=True,
    )
    
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    
    transformations = [
        ToTensor(swap_color_axis=True),
        Normalize(mean=mean, std=std),
    ]
    cifar_data.set_transformation(torchvision.transforms.Compose(transformations))
    
    # Create a class order like in the experiment
    all_classes = np.random.permutation(100)
    current_num_classes = 5
    
    # Select first 5 classes
    cifar_data.select_new_partition(all_classes[:current_num_classes])
    
    dataloader = DataLoader(cifar_data, batch_size=10, shuffle=False, num_workers=0)
    
    # Get one batch
    batch = next(iter(dataloader))
    images = batch["image"].numpy()
    labels = batch["label"].numpy()
    
    print(f"Raw image shape: {images.shape}")
    print(f"Raw label shape: {labels.shape}")
    print(f"Image min/max: {images.min():.3f}/{images.max():.3f}")
    
    # Convert to JAX format
    images_jax = jnp.asarray(images)
    labels_jax = jnp.asarray(labels)
    
    # Check if transpose is needed
    if len(images_jax.shape) == 4 and images_jax.shape[1] == 3:
        print("Transposing images from (N,C,H,W) to (N,H,W,C)")
        images_jax = jnp.transpose(images_jax, (0, 2, 3, 1))
    
    print(f"JAX image shape: {images_jax.shape}")
    print(f"JAX label shape: {labels_jax.shape}")
    
    if labels_jax.ndim > 1:
        print("One-hot labels detected")
        print(f"First few label samples (as class indices):")
        for i in range(min(5, labels_jax.shape[0])):
            class_idx = jnp.argmax(labels_jax[i])
            print(f"  Sample {i}: class {class_idx}")
        
        # Test current class selection
        current_labels = labels_jax[:, :current_num_classes]
        print(f"Current labels shape (first {current_num_classes} classes): {current_labels.shape}")
        print(f"Sum of current labels per sample (should be 1.0): {jnp.sum(current_labels, axis=1)[:5]}")
    else:
        print("Integer labels detected")
        print(f"Label range: {labels_jax.min()} to {labels_jax.max()}")
        print(f"First few labels: {labels_jax[:5]}")
    
    return True

if __name__ == "__main__":
    try:
        test_data_format()
        print("\n✓ Data format test completed")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
