"""
Integration Smoke Test for Fused MoE Kernel
=============================================

This test verifies that the `LanguageModel` can be successfully instantiated
and can perform a forward pass without crashing when the new `use_fused_moe_kernel`
flag is enabled.

This is a critical smoke test to ensure that the integration of the new
Triton kernel does not break the basic functionality of the model.
"""

import pytest
import torch
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.resnet_bk import LanguageModel

# --- Test Parameters ---
VOCAB_SIZE = 1000
D_MODEL = 64
N_LAYERS = 2
N_SEQ = 128
NUM_EXPERTS = 4

@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available for this test.")
def test_language_model_instantiation_with_fused_kernel(top_k):
    """
    Tests that the LanguageModel can be instantiated with use_fused_moe_kernel=True
    for both top_k=1 (Triton) and top_k=2 (PyTorch fallback).
    """
    try:
        model = LanguageModel(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_seq=N_SEQ,
            num_experts=NUM_EXPERTS,
            top_k=top_k,
            use_fused_moe_kernel=True
        ).to('cuda')
        assert model is not None, "Model instantiation returned None."
        print(f"LanguageModel instantiated successfully for top_k={top_k}.")
    except Exception as e:
        pytest.fail(f"Failed to instantiate LanguageModel for top_k={top_k}: {e}")

@pytest.mark.parametrize("top_k", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available for this test.")
def test_language_model_forward_pass_with_fused_kernel(top_k):
    """
    Tests that the LanguageModel can perform a forward pass with use_fused_moe_kernel=True
    for both top_k=1 (Triton) and top_k=2 (PyTorch fallback).
    """
    try:
        model = LanguageModel(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_seq=N_SEQ,
            num_experts=NUM_EXPERTS,
            top_k=top_k,
            use_fused_moe_kernel=True
        ).to('cuda')

        # Create a dummy input tensor
        batch_size = 2
        input_tensor = torch.randint(0, VOCAB_SIZE, (batch_size, N_SEQ), device='cuda')

        # Perform a forward pass
        with torch.no_grad():
            logits = model(input_tensor)

        # Check the output shape
        expected_shape = (batch_size, N_SEQ, VOCAB_SIZE)
        assert logits.shape == expected_shape, \
            f"Output shape mismatch. Expected {expected_shape}, got {logits.shape}"

        # Check that the output is finite
        assert torch.isfinite(logits).all(), "Output contains non-finite values (NaN or Inf)."

        print(f"Forward pass completed successfully for top_k={top_k}.")

    except Exception as e:
        pytest.fail(f"Forward pass failed for top_k={top_k}: {e}")

if __name__ == "__main__":
    # To run this test from the command line from the root of the repository:
    # PYTHONPATH=. python tests/test_fused_moe_integration.py
    pytest.main([__file__])
