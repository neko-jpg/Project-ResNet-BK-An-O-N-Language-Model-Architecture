
import pytest
import torch
import torch.nn as nn
import sys
import os
import warnings
from unittest.mock import MagicMock, patch
import numpy as np

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.models.birman_schwinger_core import BirmanSchwingerCore
from src.models.bk_core import BKCoreFunction, get_tridiagonal_inverse_diagonal
from src.training.gradient_caching import GradientCachingTrainer

# ============================================================================
# Smoke Tests for Critical Components
# ============================================================================

def test_bk_core_fallback_smoke():
    """
    Test that BKCoreFunction falls back to PyTorch when Triton fails or is unavailable.
    """
    B, N = 2, 10
    he_diag = torch.randn(B, N, dtype=torch.float32)
    h0_super = torch.randn(B, N-1, dtype=torch.float32)
    h0_sub = torch.randn(B, N-1, dtype=torch.float32)
    z = 1.0j

    # Force Triton mode but mock the kernel to raise exception
    with patch('src.models.bk_core.BKCoreFunction.USE_TRITON', True):
        with patch.dict('sys.modules', {'src.kernels.bk_scan': MagicMock()}):
            # Mock the triton function to raise an error
            mock_triton = sys.modules['src.kernels.bk_scan']
            mock_triton.bk_scan_triton.side_effect = ImportError("Triton failure simulation")
            mock_triton.is_triton_available.return_value = True

            # This should trigger warning and fallback
            with pytest.warns(UserWarning, match="Triton kernel failed"):
                out = BKCoreFunction.apply(he_diag, h0_super, h0_sub, z, True)

            assert out.shape == (B, N, 2)
            assert torch.isfinite(out).all()

def test_birman_schwinger_stability_logic():
    """
    Test condition number tracking and promotion logic triggers in BirmanSchwingerCore.
    """
    N = 16
    # Create a core with low threshold to trigger promotion
    # Disable semiseparable to test the raw BS path where stability logic lives
    core = BirmanSchwingerCore(n_seq=N, precision_upgrade_threshold=10.0, use_semiseparable=False)

    # Create a potential that causes instability (large values)
    V = torch.ones(1, N) * 100.0
    z = 0.001j # Small imaginary part -> high condition number

    # We need to mock compute_condition_number to control the test
    with patch.object(BirmanSchwingerCore, 'compute_condition_number', return_value=100.0) as mock_cond:
        features, diagnostics = core(V, z)

        # Check diagnostics
        assert 'condition_number' in diagnostics
        assert diagnostics['condition_number'] == 100.0
        # Expect precision upgrades count to increment
        assert diagnostics['precision_upgrades'] > 0

def test_grad_cache_drift_smoke():
    """
    Test GradientCachingTrainer detects drift and falls back.
    """
    model = nn.Linear(10, 1)
    trainer = GradientCachingTrainer(model, cache_size=5)

    x = torch.randn(1, 10)
    y = torch.randn(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 1. Train once to populate cache
    loss1, used_cache = trainer.train_step(x, y, optimizer, criterion)
    assert not used_cache

    # 2. Train again with same input (should use cache)
    # But we force drift by modifying the model weights manually to change loss
    with torch.no_grad():
        model.weight.add_(10.0) # Huge change

    # The trainer checks current loss vs cached loss.
    # Cached loss was small. Current loss (with changed weights) will be huge.
    # Drift check should fail -> fallback to fresh gradients.

    loss2, used_cache2 = trainer.train_step(x, y, optimizer, criterion)

    # Should be FALSE because drift was detected
    assert not used_cache2

def test_config_conversion_smoke():
    """
    Test Config dataclass <-> dict conversion.
    """
    from src.models.phase3.config import Phase3Config
    import dataclasses

    # Added vocab_size which is required
    config = Phase3Config(vocab_size=1000, n_layers=4, d_model=128)
    config_dict = dataclasses.asdict(config)

    # Reconstruct
    config2 = Phase3Config(**config_dict)
    assert config2.n_layers == 4
    assert config2.d_model == 128
    assert config2.vocab_size == 1000

def test_amp_trainer_import_and_init():
    """Test AMP Trainer import and initialization."""
    from src.training.amp_trainer import MixedPrecisionTrainer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    trainer = MixedPrecisionTrainer(model, optimizer, criterion, enabled=False)
    assert trainer is not None
    assert trainer.resource_guard is not None

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
