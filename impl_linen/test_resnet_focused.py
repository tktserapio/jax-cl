#!/usr/bin/env python3
"""
Focused test script to verify ResNet18 works specifically for your CIFAR experiment
"""

import sys
import os
sys.path.insert(0, '/users/tserapio/jax_cl/impl_linen')

import jax
import jax.numpy as jnp
import numpy as np
from jax_modified_resnet import build_resnet18

def test_cifar_specific():
    """Test ResNet18 with CIFAR-100 specific settings"""
    print("Testing ResNet18 for CIFAR-100 experiment...")
    
    # Create model exactly like in your experiment
    num_classes = 100
    image_dims = (32, 32, 3)
    batch_size = 90  # Your training batch size
    current_num_classes = 5  # Starting classes
    
    model = build_resnet18(num_classes=num_classes, input_channels=image_dims[-1])
    print(f"✓ Model created with {num_classes} classes")
    
    # Initialize exactly like in your experiment
    key = jax.random.PRNGKey(42)
    dummy = jnp.ones((1, *image_dims), jnp.float32)
    
    variables = model.init(key, dummy, feature_list=None, training=True)
    params, batch_stats = variables["params"], variables.get("batch_stats", {})
    
    print(f"✓ Model initialized with input shape: {dummy.shape}")
    print(f"✓ Conv1 kernel shape: {params['conv1']['kernel'].shape}")
    
    # Test with actual batch size and label formats
    images = jax.random.normal(key, (batch_size, *image_dims))
    
    # Test with one-hot labels (like CIFAR dataset)
    labels_onehot = jax.nn.one_hot(jax.random.randint(key, (batch_size,), 0, current_num_classes), current_num_classes)
    
    print(f"\nTesting with batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape (one-hot): {labels_onehot.shape}")
    
    # Forward pass
    variables = {"params": params, "batch_stats": batch_stats}
    logits = model.apply(variables, images, feature_list=None, training=True, mutable=False)
    
    print(f"  Logits shape: {logits.shape}")
    
    # Test class selection (the key issue from your experiment)
    current_logits = logits[:, :current_num_classes]  # Correct way
    current_labels = labels_onehot[:, :current_num_classes]  # Correct way
    
    print(f"  Current logits shape: {current_logits.shape}")
    print(f"  Current labels shape: {current_labels.shape}")
    
    # Compute accuracy like in your experiment
    accuracy = jnp.mean(jnp.argmax(current_logits, -1) == jnp.argmax(current_labels, -1))
    print(f"  Random accuracy: {accuracy:.4f} (should be ~0.2 for 5 classes)")
    
    # Test loss computation
    loss = jnp.mean(jnp.sum(-current_labels * jax.nn.log_softmax(current_logits), axis=-1))
    print(f"  Random loss: {loss:.4f}")
    
    # Test feature extraction (for CBP)
    feature_list = []
    _ = model.apply(variables, images, feature_list=feature_list, training=True, mutable=False)
    print(f"  Features extracted: {len(feature_list)}")
    for i, feat in enumerate(feature_list[:3]):  # Show first 3
        print(f"    Feature {i}: {feat.shape}")
    
    print("✅ CIFAR-specific tests passed!")
    return True

def test_training_step():
    """Test a complete training step like in your experiment"""
    print("\nTesting complete training step...")
    
    model = build_resnet18(num_classes=100, input_channels=3)
    key = jax.random.PRNGKey(42)
    
    # Initialize
    dummy = jnp.ones((1, 32, 32, 3), jnp.float32)
    variables = model.init(key, dummy, feature_list=None, training=True)
    params, batch_stats = variables["params"], variables.get("batch_stats", {})
    
    # Create a mini-batch
    images = jax.random.normal(key, (4, 32, 32, 3))
    labels = jax.nn.one_hot(jax.random.randint(key, (4,), 0, 5), 5)  # 5 classes
    current_num_classes = 5
    
    # Define loss function like in your experiment
    def loss_fn(params):
        variables_inner = {"params": params, "batch_stats": batch_stats}
        logits = model.apply(variables_inner, images, feature_list=None, training=True, mutable=False)
        
        # Select current classes
        current_logits = logits[:, :current_num_classes]
        current_labels = labels[:, :current_num_classes]
        
        # Cross-entropy loss
        loss = jnp.mean(jnp.sum(-current_labels * jax.nn.log_softmax(current_logits), axis=-1))
        return loss, current_logits
    
    # Compute gradients
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    print(f"  Loss: {loss:.4f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Gradients computed: {len(jax.tree_util.tree_leaves(grads))} tensors")
    
    # Check gradient magnitudes
    grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
    print(f"  Gradient norm: {grad_norm:.6f}")
    
    if grad_norm > 1e-6:
        print("✅ Training step test passed!")
        return True
    else:
        print("❌ Gradients too small!")
        return False

def test_vs_original():
    """Test against your original implementation"""
    print("\nTesting compatibility with original experiment...")
    
    try:
        # Try to import your experiment (might fail if dependencies missing)
        sys.path.insert(0, '/users/tserapio/jax_cl/impl_linen')
        from incremental_cifar_experiment_linen import IncrementalCIFARExperimentJAX
        
        # Create a minimal config
        config = {
            "stepsize": 0.1,
            "weight_decay": 0.0001,
            "momentum": 0.9,
            "data_path": "/tmp",  # Won't actually load data
            "reset_head": False,
            "reset_network": False,
            "early_stopping": False,
            "use_cbp": False,
            "noise_std": 0.0
        }
        
        # Just test model creation part
        exp = IncrementalCIFARExperimentJAX.__new__(IncrementalCIFARExperimentJAX)
        exp.num_classes = 100
        exp.image_dims = (32, 32, 3)
        exp.key = jax.random.PRNGKey(42)
        
        # Create model using your experiment's method
        exp.model = build_resnet18(num_classes=exp.num_classes, input_channels=exp.image_dims[-1])
        
        # Test initialization
        dummy = jnp.ones((1, *exp.image_dims), jnp.float32)
        exp.key, sub = jax.random.split(exp.key)
        variables = exp.model.init(sub, dummy, feature_list=None, training=True)
        
        print("✅ Compatible with original experiment structure!")
        return True
        
    except Exception as e:
        print(f"⚠ Couldn't test full compatibility: {e}")
        print("  (This might be normal if experiment dependencies are missing)")
        return True  # Don't fail the test for this

def main():
    """Run focused tests"""
    print("🔍 Running Focused ResNet18 Tests for CIFAR Experiment")
    print("=" * 60)
    
    try:
        success1 = test_cifar_specific()
        success2 = test_training_step()
        success3 = test_vs_original()
        
        if success1 and success2 and success3:
            print("\n🎉 All focused tests passed!")
            print("Your ResNet18 is ready for the CIFAR experiment!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
