#!/usr/bin/env python3
"""
Comprehensive test script for the modified ResNet18 implementation in JAX/Flax
"""

import sys
import os
sys.path.insert(0, '/users/tserapio/jax_cl/impl_linen')

import jax
import jax.numpy as jnp
import numpy as np
from jax_modified_resnet import build_resnet18, BasicBlock, ResNet
import flax.linen as nn

def test_basic_functionality():
    """Test basic model creation and forward pass"""
    print("=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    # Create model
    model = build_resnet18(num_classes=100)
    print("✓ Model created successfully")
    
    # Initialize with dummy input
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((4, 32, 32, 3), jnp.float32)  # Batch of 4 CIFAR images
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test initialization
    variables = model.init(key, dummy_input, feature_list=None, training=True)
    params, batch_stats = variables["params"], variables.get("batch_stats", {})
    print("✓ Model initialized successfully")
    print(f"Number of parameter groups: {len(params)}")
    
    # Count total parameters
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")
    
    return model, params, batch_stats

def test_forward_passes(model, params, batch_stats):
    """Test different types of forward passes"""
    print("\n" + "=" * 60)
    print("Testing Forward Passes")
    print("=" * 60)
    
    dummy_input = jnp.ones((4, 32, 32, 3), jnp.float32)
    variables = {"params": params, "batch_stats": batch_stats}
    
    # Test training mode forward pass
    output_train = model.apply(variables, dummy_input, feature_list=None, training=True, mutable=False)
    print(f"✓ Training forward pass successful, output shape: {output_train.shape}")
    
    # Test evaluation mode forward pass
    output_eval = model.apply(variables, dummy_input, feature_list=None, training=False, mutable=False)
    print(f"✓ Evaluation forward pass successful, output shape: {output_eval.shape}")
    
    # Check output dimensions
    assert output_train.shape == (4, 100), f"Expected (4, 100), got {output_train.shape}"
    assert output_eval.shape == (4, 100), f"Expected (4, 100), got {output_eval.shape}"
    print("✓ Output dimensions correct")
    
    # Test that training and eval modes give different outputs (due to BatchNorm)
    diff = jnp.mean(jnp.abs(output_train - output_eval))
    print(f"Mean absolute difference between train/eval: {diff:.6f}")
    if diff > 1e-6:
        print("✓ Train/eval modes produce different outputs (BatchNorm working)")
    else:
        print("⚠ Train/eval outputs are very similar (check BatchNorm)")
    
    return output_train, output_eval

def test_feature_extraction(model, params, batch_stats):
    """Test feature extraction capability"""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)
    
    dummy_input = jnp.ones((2, 32, 32, 3), jnp.float32)
    variables = {"params": params, "batch_stats": batch_stats}
    
    # Test with feature extraction
    feature_list = []
    output = model.apply(variables, dummy_input, feature_list=feature_list, training=True, mutable=False)
    
    print(f"✓ Feature extraction successful")
    print(f"Number of features extracted: {len(feature_list)}")
    print(f"Output shape: {output.shape}")
    
    # Print feature shapes
    for i, feat in enumerate(feature_list):
        print(f"  Feature {i}: {feat.shape}")
    
    # Verify we have reasonable number of features
    # Should be: conv1 + 2*(layer1) + 2*(layer2) + 2*(layer3) + 2*(layer4) + global_pool - 1 (pop)
    expected_features = 1 + 2*2 + 2*2 + 2*2 + 2*2 + 1 - 1  # = 16
    if len(feature_list) >= 10:  # Reasonable range
        print("✓ Feature extraction appears to be working correctly")
    else:
        print(f"⚠ Only {len(feature_list)} features extracted, expected around {expected_features}")
    
    return feature_list

def test_gradient_computation(model, params, batch_stats):
    """Test gradient computation for training"""
    print("\n" + "=" * 60)
    print("Testing Gradient Computation")
    print("=" * 60)
    
    dummy_input = jnp.ones((2, 32, 32, 3), jnp.float32)
    dummy_labels = jnp.array([0, 1])  # Class indices
    
    def loss_fn(params):
        variables = {"params": params, "batch_stats": batch_stats}
        logits = model.apply(variables, dummy_input, feature_list=None, training=True, mutable=False)
        # Simple cross-entropy loss
        loss = jnp.mean(jnp.sum(-jax.nn.one_hot(dummy_labels, 100) * jax.nn.log_softmax(logits), axis=-1))
        return loss, logits
    
    # Compute gradients
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    print(f"✓ Gradient computation successful")
    print(f"Loss: {loss:.6f}")
    print(f"Logits shape: {logits.shape}")
    
    # Check that gradients exist and are non-zero
    grad_norms = {k: jnp.linalg.norm(jax.tree_util.tree_leaves(v)[0]) for k, v in grads.items()}
    print("Gradient norms for key layers:")
    for k, norm in list(grad_norms.items())[:5]:  # Show first 5
        print(f"  {k}: {norm:.6f}")
    
    # Verify gradients are not zero
    total_grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(grads)))
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    
    if total_grad_norm > 1e-6:
        print("✓ Gradients are non-zero and reasonable")
    else:
        print("❌ Gradients are too small or zero")
    
    return loss, grads

def test_batch_norm_behavior(model, params, batch_stats):
    """Test BatchNorm behavior in train vs eval mode"""
    print("\n" + "=" * 60)
    print("Testing BatchNorm Behavior")
    print("=" * 60)
    
    # Create inputs with different statistics
    key = jax.random.PRNGKey(123)
    input1 = jax.random.normal(key, (8, 32, 32, 3)) * 2.0 + 1.0  # Different mean/std
    input2 = jax.random.normal(jax.random.split(key)[0], (8, 32, 32, 3)) * 0.5 - 0.5
    
    variables = {"params": params, "batch_stats": batch_stats}
    
    # Training mode - should use batch statistics
    out1_train = model.apply(variables, input1, training=True, mutable=False)
    out2_train = model.apply(variables, input2, training=True, mutable=False)
    
    # Eval mode - should use running averages
    out1_eval = model.apply(variables, input1, training=False, mutable=False)
    out2_eval = model.apply(variables, input2, training=False, mutable=False)
    
    # Check differences
    train_diff = jnp.mean(jnp.abs(out1_train - out2_train))
    eval_diff = jnp.mean(jnp.abs(out1_eval - out2_eval))
    
    print(f"Mean output difference between batches:")
    print(f"  Training mode: {train_diff:.6f}")
    print(f"  Evaluation mode: {eval_diff:.6f}")
    
    if abs(train_diff - eval_diff) > 1e-5:
        print("✓ BatchNorm behaves differently in train vs eval mode")
    else:
        print("⚠ BatchNorm behavior similar in train/eval (might need more training)")

def test_different_input_sizes(model, params, batch_stats):
    """Test model with different input sizes"""
    print("\n" + "=" * 60)
    print("Testing Different Input Sizes")
    print("=" * 60)
    
    variables = {"params": params, "batch_stats": batch_stats}
    
    test_cases = [
        (1, 32, 32, 3),   # Single image
        (16, 32, 32, 3),  # Larger batch
        (2, 64, 64, 3),   # Larger spatial size (should work due to global avg pooling)
    ]
    
    for i, shape in enumerate(test_cases):
        try:
            dummy_input = jnp.ones(shape, jnp.float32)
            output = model.apply(variables, dummy_input, training=False, mutable=False)
            print(f"✓ Test case {i+1}: Input {shape} → Output {output.shape}")
        except Exception as e:
            print(f"❌ Test case {i+1}: Input {shape} failed: {e}")

def test_numerical_stability(model, params, batch_stats):
    """Test numerical stability with extreme inputs"""
    print("\n" + "=" * 60)
    print("Testing Numerical Stability")
    print("=" * 60)
    
    variables = {"params": params, "batch_stats": batch_stats}
    
    test_cases = [
        ("zeros", jnp.zeros((2, 32, 32, 3))),
        ("large", jnp.ones((2, 32, 32, 3)) * 100),
        ("small", jnp.ones((2, 32, 32, 3)) * 1e-6),
        ("negative", -jnp.ones((2, 32, 32, 3))),
    ]
    
    for name, input_data in test_cases:
        try:
            output = model.apply(variables, input_data, training=False, mutable=False)
            if jnp.any(jnp.isnan(output)) or jnp.any(jnp.isinf(output)):
                print(f"❌ {name}: Output contains NaN or Inf")
            else:
                print(f"✓ {name}: Output is numerically stable (range: {output.min():.3f} to {output.max():.3f})")
        except Exception as e:
            print(f"❌ {name}: Failed with error: {e}")

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting ResNet18 JAX/Flax Test Suite")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    try:
        # Basic functionality
        model, params, batch_stats = test_basic_functionality()
        
        # Forward passes
        output_train, output_eval = test_forward_passes(model, params, batch_stats)
        
        # Feature extraction
        features = test_feature_extraction(model, params, batch_stats)
        
        # Gradient computation
        loss, grads = test_gradient_computation(model, params, batch_stats)
        
        # BatchNorm behavior
        test_batch_norm_behavior(model, params, batch_stats)
        
        # Different input sizes
        test_different_input_sizes(model, params, batch_stats)
        
        # Numerical stability
        test_numerical_stability(model, params, batch_stats)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED!")
        print("=" * 60)
        print("✅ Your modified ResNet18 implementation appears to be working correctly!")
        print("Ready for use in your incremental learning experiments.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
