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
