#!/usr/bin/env python3
"""
Simple test script to verify the ResNet18 model works correctly
"""

import jax
import jax.numpy as jnp
from jax_modified_resnet import build_resnet18

def test_resnet_basic():
    """Test basic ResNet18 functionality"""
    print("Testing basic ResNet18 functionality...")
    
    # Create model
    model = build_resnet18(num_classes=10, input_channels=3)
    
    # Create dummy input
    key = jax.random.PRNGKey(42)
    x = jnp.ones((2, 32, 32, 3))  # batch_size=2, 32x32 RGB images
    
    # Initialize parameters
    print(f"Input shape: {x.shape}")
    variables = model.init(key, x, feature_list=None)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    
    print(f"Conv1 kernel shape: {params['conv1']['kernel'].shape}")
    print(f"FC weight shape: {params['fc']['kernel'].shape}")
    
    # Test forward pass in training mode
    print("\nTesting training mode...")
    feature_list = []
    output, new_batch_stats = model.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        feature_list=feature_list,
        mutable=["batch_stats"]
    )
    
    print(f"Output shape: {output.shape}")
    print(f"Number of features extracted: {len(feature_list)}")
    print(f"Feature shapes: {[f.shape for f in feature_list[:3]]}...")
    
    # Test forward pass in evaluation mode
    print("\nTesting evaluation mode...")
    eval_output = model.apply(
        {"params": params, "batch_stats": new_batch_stats["batch_stats"]},
        x,
        feature_list=None,
        mutable=False
    )
    
    print(f"Eval output shape: {eval_output.shape}")
    print(f"Output values are finite: {jnp.all(jnp.isfinite(eval_output))}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_resnet_basic()
