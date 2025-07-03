#!/usr/bin/env python3
"""
Quick test script to verify the ResNet18 model initializes correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '/users/tserapio/jax_cl/impl_linen')

import jax
import jax.numpy as jnp
from jax_modified_resnet import build_resnet18

def test_model_initialization():
    """Test that the ResNet18 model initializes without errors"""
    print("Testing ResNet18 model initialization...")
    
    # Create model
    model = build_resnet18(num_classes=100, input_channels=3)
    print("✓ Model created successfully")
    
    # Initialize with dummy input
    key = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 32, 32, 3), jnp.float32)
    
    # Test initialization
    variables = model.init(key, dummy_input, feature_list=None, train=True)
    print("✓ Model initialized successfully")
    
    # Test forward pass
    params, batch_stats = variables["params"], variables.get("batch_stats", {})
    output = model.apply({"params": params, "batch_stats": batch_stats}, 
                        dummy_input, feature_list=None, train=False, mutable=False)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    # Test with feature extraction
    feature_list = []
    output_with_features = model.apply({"params": params, "batch_stats": batch_stats}, 
                                     dummy_input, feature_list=feature_list, train=True, 
                                     mutable=["batch_stats"])
    print(f"✓ Feature extraction successful, {len(feature_list)} features extracted")
    
    return True

if __name__ == "__main__":
    try:
        success = test_model_initialization()
        if success:
            print("\n🎉 All tests passed! The ResNet18 model should work correctly now.")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
