"""
Unit tests for Prime-Bump Initialization.
"""
import torch
import pytest
from src.utils.prime_init import prime_bump_init_
from src.models.phase7.hybrid_attention import HybridHyperbolicAttention

def test_prime_bump_init_function():
    """
    Tests the prime_bump_init_ function directly.
    """
    tensor = torch.randn(128, 128)
    original_norm = torch.norm(tensor)

    prime_bump_init_(tensor)

    # Check if tensor is modified
    assert torch.norm(tensor) != original_norm

    # Check singular values
    _, s, _ = torch.linalg.svd(tensor)

    # Check if max singular value is close to 1.0
    assert torch.allclose(s.max(), torch.tensor(1.0), atol=1e-5)

    # Check if all singular values are non-negative
    assert torch.all(s >= 0)

from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

def test_prime_bump_init_in_model():
    """
    Tests if the prime-bump initialization is applied in the model.
    """
    model = HyperbolicMultiHeadAttention(d_model=128, num_heads=8)

    # Check W_q weight singular values
    _, s, _ = torch.linalg.svd(model.W_q.weight)

    # Check if max singular value is close to 1.0
    assert torch.allclose(s.max(), torch.tensor(1.0), atol=1e-5)

    # Check if all singular values are non-negative
    assert torch.all(s >= 0)
