"""
Unit tests for the Phase 7 Integrated Model, updated for config-based instantiation.
"""
import torch
import torch.nn as nn
import pytest

from src.models.phase7.integrated_model import Phase7IntegratedModel, Phase7Config
from src.models.phase1.htt_embedding import HolographicTTEmbedding

# Define model parameters for testing
VOCAB_SIZE = 1000
D_MODEL = 64 # Reduced for faster testing
HTT_RANK = 8
BATCH_SIZE = 4
SEQ_LENGTH = 16
N_LAYERS = 2 # Use a minimal number of layers for testing

@pytest.fixture
def config():
    """
    Returns a Phase7Config instance for testing.
    """
    return Phase7Config(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_seq=SEQ_LENGTH,
        htt_rank=HTT_RANK,
        use_hybrid_attention=True, # Ensure the core feature is enabled
        use_triton_kernel=False,    # Disable Triton for CPU-based testing
    )

@pytest.fixture
def model(config):
    """
    Returns an instance of the Phase7IntegratedModel using the test config.
    """
    return Phase7IntegratedModel(config)

def test_model_instantiation(model):
    """
    Tests that the model can be instantiated without errors and has the correct components.
    """
    assert isinstance(model, Phase7IntegratedModel)
    # After refactoring, the HTT embedding is stored in `htt_embedding`
    # and also assigned to the inner model's `token_embedding`.
    assert isinstance(model.htt_embedding, HolographicTTEmbedding)
    assert model.model.token_embedding is model.htt_embedding

def test_forward_pass(model):
    """
    Tests that the forward pass runs without errors and returns the correct shape.
    """
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    output = model(input_ids)
    # The output of the final lm_head should be (batch, seq_len, vocab_size)
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)

def test_parameter_reduction(model):
    """
    Tests that the HTT embedding layer has significantly fewer parameters
    than a standard nn.Embedding layer.
    """
    # Standard nn.Embedding parameters
    standard_embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
    standard_params = sum(p.numel() for p in standard_embedding.parameters())

    # HTT embedding parameters
    htt_params = model.get_embedding_parameter_count()

    print(f"Standard embedding parameters: {standard_params}")
    print(f"HTT embedding parameters: {htt_params}")

    # Check that HTT embedding has fewer parameters
    assert htt_params < standard_params

    # Check for significant reduction (e.g., at least 90%)
    # This ratio depends on the rank and d_model
    reduction_ratio = htt_params / standard_params
    print(f"Parameter reduction ratio: {reduction_ratio:.4f}")
    assert reduction_ratio < 0.2 # Expecting >80% reduction for these params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton test")
def test_triton_fallback_with_mask():
    """
    Verify that the model falls back to the PyTorch implementation when a mask is
    used with the Triton kernel enabled, instead of crashing.
    
    Note: This test requires CUDA as Triton kernels only work on GPU.
    """
    # This test can only run if Triton is installed and CUDA is available
    config = Phase7Config(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_seq=SEQ_LENGTH,
        htt_rank=HTT_RANK,
        use_hybrid_attention=True,
        use_triton_kernel=True  # Explicitly enable the Triton path
    )
    model = Phase7IntegratedModel(config)
    
    # Move to CUDA for Triton test
    device = torch.device('cuda')
    model = model.to(device)
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), device=device)

    # This forward pass will automatically trigger a causal mask to be created
    # internally. If the fallback logic is correct, it should execute without raising
    # a NotImplementedError, even though the Triton kernel itself doesn't
    # support masking.
    try:
        output = model(input_ids)
        assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
        print("Triton fallback test passed: Model ran without error.")
    except NotImplementedError:
        pytest.fail(
            "Model raised NotImplementedError instead of falling back to PyTorch "
            "implementation when a mask was used with the Triton kernel enabled."
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Phase 7 training")
def test_phase7_cuda_training_ready():
    """
    Phase 7のトレーニング準備が整っているかを確認するテスト。
    CUDA + Tritonが必須。
    """
    # Check Triton is available
    try:
        import triton
        triton_available = True
    except ImportError:
        triton_available = False
    
    assert triton_available, "Triton is required for Phase 7 training"
    
    # Check Triton kernel can be loaded
    try:
        from src.kernels.hyperbolic_attention_fast import fast_hyperbolic_attention
        kernel_loaded = True
    except Exception:
        kernel_loaded = False
    
    assert kernel_loaded, "Hyperbolic attention Triton kernel must be loadable"
    
    # Create model on CUDA
    config = Phase7Config(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_seq=SEQ_LENGTH,
        htt_rank=HTT_RANK,
        use_hybrid_attention=True,
        use_triton_kernel=True,
    )
    
    device = torch.device('cuda')
    model = Phase7IntegratedModel(config).to(device)
    
    # Test forward pass with Triton
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), device=device)
    output = model(input_ids)
    
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
    print(f"✓ Phase 7 CUDA training ready: {torch.cuda.get_device_name(0)}")


def test_forward_with_diagnostics(model):
    """
    Tests that the forward pass with diagnostics returns both output and diagnostics dict.
    """
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
    output, diagnostics = model(input_ids, return_diagnostics=True)
    
    assert output.shape == (BATCH_SIZE, SEQ_LENGTH, VOCAB_SIZE)
    assert isinstance(diagnostics, dict)
    print(f"Diagnostics keys: {list(diagnostics.keys())}")
