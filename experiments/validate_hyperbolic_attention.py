# experiments/validate_hyperbolic_attention.py

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

# ##############################################################################
# # Validation Script
# ##############################################################################

def validate_attention_module():
    """
    Validates the forward and backward passes of the HyperbolicMultiHeadAttention module.
    """
    print("--- Starting HyperbolicMultiHeadAttention Validation ---")

    # 1. Configuration
    d_model = 128
    num_heads = 8
    batch_size = 4
    seq_len = 10

    # 2. Model Instantiation
    try:
        attention_module = HyperbolicMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        attention_module.train()
        print("✅ Model instantiated successfully.")
    except Exception as e:
        print(f"❌ Model instantiation failed: {e}")
        return

    # 3. Dummy Input
    # Input is a standard Euclidean tensor
    dummy_input = torch.randn(batch_size, seq_len, d_model)
    print(f"Input tensor shape: {dummy_input.shape}")

    # 4. Forward Pass
    try:
        output = attention_module(dummy_input)
        print("✅ Forward pass completed.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # 5. Output Validation
    print(f"Output tensor shape: {output.shape}")
    assert output.shape == dummy_input.shape, f"Output shape mismatch! Expected {dummy_input.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "NaN detected in the output tensor!"
    print("✅ Output shape and content are valid.")

    # 6. Backward Pass
    try:
        # Create a dummy loss and perform backpropagation
        loss = output.sum()
        loss.backward()
        print("✅ Backward pass completed.")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return

    # 7. Gradient Validation
    grad_found = False
    for name, param in attention_module.named_parameters():
        if param.grad is not None:
            grad_found = True
            assert not torch.isnan(param.grad).any(), f"NaN detected in gradient of '{name}'!"

    if grad_found:
        print("✅ Gradients were computed and are not NaN.")
    else:
        print("❌ No gradients were computed.")

    print("\n--- ✅ HyperbolicMultiHeadAttention Validation Successful ---")


if __name__ == '__main__':
    validate_attention_module()
