"""
Tests for the dynamic routing mechanism in HybridHyperbolicAttention.
"""
import torch
import pytest
from src.models.phase7.hybrid_attention import HybridHyperbolicAttention

D_MODEL = 64
NUM_HEADS = 4
BATCH_SIZE = 2
SEQ_LENGTH = 32

@pytest.fixture
def model():
    """
    Returns an instance of the HybridHyperbolicAttention.
    """
    return HybridHyperbolicAttention(d_model=D_MODEL, num_heads=NUM_HEADS, use_triton_kernel=False)

def test_dynamic_gate_forward_pass(model):
    """
    Tests that the forward pass with dynamic gating runs and returns diagnostics.
    """
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL)
    # Mock g_ii, as it's now a required input from the complex BK-Core
    g_ii = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL, dtype=torch.cfloat)
    output, diagnostics = model(x, g_ii, return_diagnostics=True)

    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, D_MODEL)
    assert 'hybrid_gate_mean' in diagnostics
    assert 'scattering_energy_mean' in diagnostics
    assert diagnostics['hybrid_gate_mean'].item() > 0

def test_gate_is_dynamic(model):
    """
    Tests that the gate value changes with different g_ii inputs.
    The gate depends on the imaginary part of g_ii.
    """
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL) # x is still needed as input

    # Low energy g_ii (zero imaginary part)
    g_ii_low_real = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL)
    g_ii_low_imag = torch.zeros(BATCH_SIZE, SEQ_LENGTH, D_MODEL)
    g_ii_low = torch.complex(g_ii_low_real, g_ii_low_imag)
    _, diagnostics_low = model(x, g_ii_low, return_diagnostics=True)
    energy_low = diagnostics_low['scattering_energy_mean'].item()
    gate_low = diagnostics_low['hybrid_gate_mean'].item()

    # High energy g_ii (large imaginary part)
    g_ii_high_real = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_MODEL)
    g_ii_high_imag = torch.ones(BATCH_SIZE, SEQ_LENGTH, D_MODEL) * 10.0
    g_ii_high = torch.complex(g_ii_high_real, g_ii_high_imag)
    _, diagnostics_high = model(x, g_ii_high, return_diagnostics=True)
    energy_high = diagnostics_high['scattering_energy_mean'].item()
    gate_high = diagnostics_high['hybrid_gate_mean'].item()

    # Energies should be different, and gates should be different
    assert energy_low == 0.0, "Energy for zero imaginary part should be zero"
    assert energy_high > 0.0, "Energy for non-zero imaginary part should be positive"
    assert energy_low != energy_high, "Energy for low and high g_ii inputs should be different"
    assert gate_low != gate_high, "Gate value for low and high g_ii inputs should be different"

    # Test that a higher energy (larger g_ii.imag.abs()) leads to a larger gate value (closer to 1)
    # Sigmoid is monotonic, so this should hold.
    assert gate_high > gate_low, "Higher energy input should result in a larger gate value"
