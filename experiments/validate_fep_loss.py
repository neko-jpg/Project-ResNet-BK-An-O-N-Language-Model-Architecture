# experiments/validate_fep_loss.py

import torch
import torch.nn as nn
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.losses import FreeEnergyLoss

# ##############################################################################
# # Validation Script
# ##############################################################################

def validate_loss_module():
    """
    Validates the forward and backward passes of the FreeEnergyLoss module.
    """
    print("--- Starting FreeEnergyLoss Validation ---")

    # 1. Configuration
    hidden_dim = 128
    vocab_size = 1000
    batch_size = 4
    seq_len = 10

    # 2. Loss Module Instantiation
    try:
        loss_fn = FreeEnergyLoss(hidden_dim=hidden_dim, kl_weight=0.1)
        loss_fn.train()
        print("✅ Loss module instantiated successfully.")
    except Exception as e:
        print(f"❌ Loss module instantiation failed: {e}")
        return

    # 3. Dummy Tensors (simulating model output)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"Logits shape: {logits.shape}")
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Targets shape: {targets.shape}")

    # 4. Forward Pass
    try:
        loss = loss_fn(logits, hidden_states, targets)
        print("✅ Forward pass completed.")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return

    # 5. Output Validation
    print(f"Computed loss: {loss.item()}")
    assert loss.ndim == 0, "Loss must be a scalar value."
    assert not torch.isnan(loss), "NaN detected in the loss!"
    print("✅ Loss is a valid scalar.")

    # 6. Backward Pass
    try:
        loss.backward()
        print("✅ Backward pass completed.")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return

    # 7. Gradient Validation
    # Check if the uncertainty_head has gradients
    uncertainty_head_grad = loss_fn.uncertainty_head.weight.grad
    assert uncertainty_head_grad is not None, "Gradients for uncertainty_head are missing."
    assert not torch.isnan(uncertainty_head_grad).any(), "NaN detected in uncertainty_head gradients!"
    print("✅ Gradients were computed for the uncertainty head and are not NaN.")

    print("\n--- ✅ FreeEnergyLoss Validation Successful ---")


if __name__ == '__main__':
    validate_loss_module()
