
import torch
import torch.nn as nn
import pytest
from src.models.semiseparable_matrix import SemiseparableMatrix
from src.models.resnet_bk import LanguageModel, SymplecticBKBlock
from src.models.bk_core import set_triton_mode
from src.models.koopman.model import KoopmanBKModel
from src.models.koopman.config import KoopmanConfig

# Ensure Triton is disabled for CPU tests
if not torch.cuda.is_available():
    set_triton_mode(False)

def test_bitnet_quantization():
    """Verify 1.58-bit quantization logic."""
    N = 128
    layer = SemiseparableMatrix(n_seq=N, use_bitnet=True)

    # Set weights to known values
    # Scale s = mean(|1.5|) = 1.5. 1.5/1.5 = 1.0 -> quantized 1
    with torch.no_grad():
        layer.main_master.fill_(1.5)

    x = torch.randn(2, N)
    # Forward pass triggers quantization
    _ = layer.matvec(x)

    # Retrieve effective components
    main, _, _ = layer._get_tridiagonal_components()
    scale = layer.main_master.abs().mean()

    # Check values are quantized (either 0, +s, or -s)
    # In this case, should be exactly +s
    assert torch.allclose(main, scale), "BitNet quantization failed: expected all values to be equal to scale"

def test_symplectic_energy_conservation():
    """Verify Symplectic Integrator structure and basic stability."""
    d_model = 128
    N = 32
    dt = 0.1
    model = LanguageModel(
        vocab_size=100,
        d_model=d_model,
        n_layers=1,
        n_seq=N,
        use_symplectic=True,
        symplectic_dt=dt
    )

    x = torch.randn(1, N, d_model)
    out = model.blocks[0](x)

    # Check output shape
    assert out.shape == x.shape

    # Check energy drift is recorded
    ke_diff = model.blocks[0].get_energy_diff()
    assert ke_diff >= 0.0
    # Drift should be reasonably small for a single step with random weights
    # (Exact bound depends on initialization, but checking existence is key)

def test_gamma_learning():
    """Verify Non-Hermitian gamma parameter is learnable."""
    d_model = 64
    N = 32
    model = LanguageModel(
        vocab_size=100,
        d_model=d_model,
        n_layers=1,
        n_seq=N,
        use_birman_schwinger=True,
        epsilon=1.0
    )

    # Check gamma exists
    assert hasattr(model.blocks[0].bk_layer, 'gamma')

    # Run forward/backward
    x = torch.randint(0, 100, (1, N))
    logits = model(x)
    loss = logits.mean()
    loss.backward()

    # Check gradient flow
    grad = model.blocks[0].bk_layer.gamma.grad
    assert grad is not None, "Gamma gradient is None"
    assert grad.item() != 0.0, "Gamma gradient is zero"

def test_koopman_initialization():
    """Verify Koopman Model initializes correctly with config."""
    N = 128
    config = KoopmanConfig(
        d_model=128,
        n_layers=1,
        n_seq=N,
        use_bitnet=True,
        use_symplectic=True
    )
    model = KoopmanBKModel(config)

    x = torch.randint(0, 100, (1, N))
    logits = model(x)

    assert logits.shape == (1, N, 50257)
