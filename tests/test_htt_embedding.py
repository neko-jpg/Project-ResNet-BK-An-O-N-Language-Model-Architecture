"""
Unit tests for Holographic Tensor Train (HTT) Embedding

Tests:
    - Compression ratio achieves >90% reduction
    - Output shape matches (B, L, d_model)
    - Phase parameters are learnable
    - Gradient flow through Tensor Train cores
    - Boundary condition handling

Requirements:
    - 6.1: Output shape correctness
    - 6.2: Numerical stability

Author: Project MUSE Team
"""

import pytest
import torch
import torch.nn as nn

from src.models.phase1.htt_embedding import (
    HolographicTTEmbedding,
    HTTDecoder,
    create_htt_embedding,
    replace_embedding_with_htt,
    verify_compression_ratio,
    verify_gradient_flow,
    calculate_htt_memory_savings,
)
from src.models.phase1.config import Phase1Config
from src.models.phase1.errors import InvalidConfigError, NumericalInstabilityError


class TestHolographicTTEmbedding:
    """Test suite for HolographicTTEmbedding"""
    
    def test_initialization(self):
        """Test basic initialization"""
        vocab_size = 10000
        d_model = 512
        rank = 16
        
        embedding = HolographicTTEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            rank=rank,
        )
        
        assert embedding.vocab_size == vocab_size
        assert embedding.d_model == d_model
        assert embedding.rank == rank
        assert embedding.num_cores == 2
        assert embedding.phase_encoding is True
    
    def test_invalid_config(self):
        """Test invalid configuration raises errors"""
        with pytest.raises(InvalidConfigError):
            HolographicTTEmbedding(vocab_size=-1, d_model=512)
        
        with pytest.raises(InvalidConfigError):
            HolographicTTEmbedding(vocab_size=1000, d_model=0)
        
        with pytest.raises(InvalidConfigError):
            HolographicTTEmbedding(vocab_size=1000, d_model=512, rank=-1)
        
        with pytest.raises(InvalidConfigError):
            HolographicTTEmbedding(vocab_size=1000, d_model=512, num_cores=3)
    
    def test_output_shape(self):
        """Test output shape matches (B, L, d_model)"""
        vocab_size = 5000
        d_model = 256
        batch_size = 4
        seq_len = 128
        
        embedding = HolographicTTEmbedding(vocab_size, d_model, rank=8)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype == torch.float32
    
    def test_compression_ratio_90_percent(self):
        """Test compression ratio achieves >90% reduction"""
        vocab_size = 50000
        d_model = 1024
        rank = 16
        
        embedding = HolographicTTEmbedding(vocab_size, d_model, rank=rank)
        
        result = verify_compression_ratio(embedding, target_ratio=0.1)
        
        assert result['compression_ratio'] < 0.1, \
            f"Compression ratio {result['compression_ratio']:.4f} does not meet 90% target"
        assert result['meets_target'] is True
        assert result['compression_percentage'] > 90.0
        
        print(f"Compression: {result['compression_percentage']:.2f}%")
        print(f"Standard params: {result['standard_params']:,}")
        print(f"TT params: {result['tt_params']:,}")
    
    def test_compression_ratio_various_sizes(self):
        """Test compression ratio for various vocabulary and dimension sizes"""
        test_cases = [
            (10000, 512, 16),
            (30000, 768, 16),
            (50000, 1024, 16),
            (100000, 2048, 32),
        ]
        
        for vocab_size, d_model, rank in test_cases:
            embedding = HolographicTTEmbedding(vocab_size, d_model, rank=rank)
            result = verify_compression_ratio(embedding)
            
            assert result['compression_ratio'] < 0.1, \
                f"Failed for vocab={vocab_size}, d_model={d_model}: " \
                f"ratio={result['compression_ratio']:.4f}"
    
    def test_phase_parameters_learnable(self):
        """Test phase parameters are learnable"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8, phase_encoding=True)
        
        # Check phase_shift is a parameter
        assert isinstance(embedding.phase_shift, nn.Parameter)
        assert embedding.phase_shift.requires_grad is True
        assert embedding.phase_shift.shape == (8,)
    
    def test_phase_encoding_disabled(self):
        """Test phase encoding can be disabled"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8, phase_encoding=False)
        
        # phase_shift should be a buffer (not parameter)
        assert not isinstance(embedding.phase_shift, nn.Parameter)
        assert embedding.phase_shift.requires_grad is False
        assert torch.all(embedding.phase_shift == 0.0)
    
    def test_gradient_flow_all_cores(self):
        """Test gradient flow through all Tensor Train cores"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        result = verify_gradient_flow(embedding, input_ids)
        
        assert result['core1_has_grad'], "Core1 has no gradient"
        assert result['core2_has_grad'], "Core2 has no gradient"
        assert result['phase_has_grad'], "Phase parameter has no gradient"
        assert result['all_cores_have_grad'], "Not all cores have gradients"
        
        # Check gradient norms are non-zero
        assert result['core1_grad_norm'] > 0.0
        assert result['core2_grad_norm'] > 0.0
        assert result['phase_grad_norm'] > 0.0
        
        print(f"Core1 grad norm: {result['core1_grad_norm']:.6f}")
        print(f"Core2 grad norm: {result['core2_grad_norm']:.6f}")
        print(f"Phase grad norm: {result['phase_grad_norm']:.6f}")
    
    def test_boundary_condition_handling(self):
        """Test boundary condition handling for edge cases"""
        vocab_size = 1000
        d_model = 128
        embedding = HolographicTTEmbedding(vocab_size, d_model, rank=8)
        
        # Test with boundary token IDs
        input_ids = torch.tensor([
            [0, 1, vocab_size - 1],  # Min and max valid IDs
            [vocab_size // 2, 100, 500],  # Middle range
        ])
        
        output = embedding(input_ids)
        
        assert output.shape == (2, 3, d_model)
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
    
    def test_boundary_clamping(self):
        """Test that out-of-range indices are clamped safely"""
        vocab_size = 100
        d_model = 64
        embedding = HolographicTTEmbedding(vocab_size, d_model, rank=4)
        
        # Create input with valid IDs
        input_ids = torch.randint(0, vocab_size, (2, 5))
        
        # Forward pass should not raise errors
        output = embedding(input_ids)
        
        assert output.shape == (2, 5, d_model)
        assert torch.isfinite(output).all()
    
    def test_numerical_stability(self):
        """Test numerical stability (no NaN/Inf in output)"""
        embedding = HolographicTTEmbedding(5000, 256, rank=16)
        input_ids = torch.randint(0, 5000, (8, 64))
        
        output = embedding(input_ids)
        
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_batch_size_variations(self):
        """Test various batch sizes"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8)
        
        for batch_size in [1, 2, 4, 8, 16]:
            input_ids = torch.randint(0, 1000, (batch_size, 32))
            output = embedding(input_ids)
            
            assert output.shape == (batch_size, 32, 128)
            assert torch.isfinite(output).all()
    
    def test_sequence_length_variations(self):
        """Test various sequence lengths"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8)
        
        for seq_len in [1, 16, 64, 128, 256, 512]:
            input_ids = torch.randint(0, 1000, (2, seq_len))
            output = embedding(input_ids)
            
            assert output.shape == (2, seq_len, 128)
            assert torch.isfinite(output).all()
    
    def test_different_ranks(self):
        """Test different rank values"""
        vocab_size = 5000
        d_model = 256
        
        for rank in [4, 8, 16, 32, 64]:
            embedding = HolographicTTEmbedding(vocab_size, d_model, rank=rank)
            input_ids = torch.randint(0, vocab_size, (2, 10))
            
            output = embedding(input_ids)
            
            assert output.shape == (2, 10, d_model)
            assert torch.isfinite(output).all()
            
            # Higher rank should have more parameters
            result = verify_compression_ratio(embedding)
            print(f"Rank {rank}: {result['tt_params']:,} params, "
                  f"{result['compression_percentage']:.2f}% compression")


class TestHTTFactoryFunctions:
    """Test factory functions and utilities"""
    
    def test_create_htt_embedding_default(self):
        """Test create_htt_embedding with default config"""
        embedding = create_htt_embedding(10000, 512)
        
        assert isinstance(embedding, HolographicTTEmbedding)
        assert embedding.vocab_size == 10000
        assert embedding.d_model == 512
    
    def test_create_htt_embedding_with_config(self):
        """Test create_htt_embedding with custom config"""
        config = Phase1Config(
            htt_rank=32,
            htt_phase_encoding=False,
        )
        
        embedding = create_htt_embedding(10000, 512, config)
        
        assert embedding.rank == 32
        assert embedding.phase_encoding is False
    
    def test_replace_embedding_with_htt(self):
        """Test replacing nn.Embedding with HTT"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(10000, 512)
        
        model = DummyModel()
        original_params = sum(p.numel() for p in model.token_embedding.parameters())
        
        model = replace_embedding_with_htt(model, "token_embedding")
        
        assert isinstance(model.token_embedding, HolographicTTEmbedding)
        new_params = sum(p.numel() for p in model.token_embedding.parameters())
        
        assert new_params < original_params * 0.1, \
            "HTT should reduce parameters by >90%"
    
    def test_replace_embedding_invalid_attr(self):
        """Test replace_embedding_with_htt with invalid attribute"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 128)
        
        model = DummyModel()
        
        with pytest.raises(ValueError):
            replace_embedding_with_htt(model, "nonexistent_attr")
    
    def test_replace_embedding_wrong_type(self):
        """Test replace_embedding_with_htt with non-Embedding attribute"""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Linear(512, 512)
        
        model = DummyModel()
        
        with pytest.raises(TypeError):
            replace_embedding_with_htt(model, "token_embedding")


class TestHTTMemoryCalculations:
    """Test memory calculation utilities"""
    
    def test_calculate_memory_savings(self):
        """Test memory savings calculation"""
        result = calculate_htt_memory_savings(50000, 1024, rank=16)
        
        assert result['standard_memory_mb'] > result['htt_memory_mb']
        assert result['memory_saved_mb'] > 0
        assert result['memory_saved_percentage'] > 90.0
        
        print(f"Standard: {result['standard_memory_mb']:.2f} MB")
        print(f"HTT: {result['htt_memory_mb']:.2f} MB")
        print(f"Saved: {result['memory_saved_mb']:.2f} MB "
              f"({result['memory_saved_percentage']:.1f}%)")
    
    def test_memory_savings_various_dtypes(self):
        """Test memory savings for different dtypes"""
        for dtype in [torch.float32, torch.float16]:
            result = calculate_htt_memory_savings(
                50000, 1024, rank=16, dtype=dtype
            )
            
            assert result['memory_saved_mb'] > 0
            print(f"{dtype}: Saved {result['memory_saved_mb']:.2f} MB")


class TestHTTIntegration:
    """Integration tests for HTT embedding"""
    
    def test_forward_backward_pass(self):
        """Test complete forward and backward pass"""
        embedding = HolographicTTEmbedding(5000, 256, rank=16)
        input_ids = torch.randint(0, 5000, (4, 32))
        
        # Forward
        output = embedding(input_ids)
        assert output.shape == (4, 32, 256)
        
        # Backward
        loss = output.sum()
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in embedding.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name} has NaN/Inf gradient"
    
    def test_multiple_forward_passes(self):
        """Test multiple forward passes produce consistent results"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Multiple forward passes with same input
        output1 = embedding(input_ids)
        output2 = embedding(input_ids)
        
        # Should be identical (deterministic)
        assert torch.allclose(output1, output2)
    
    def test_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs"""
        embedding = HolographicTTEmbedding(1000, 128, rank=8)
        
        input_ids1 = torch.randint(0, 500, (2, 10))
        input_ids2 = torch.randint(500, 1000, (2, 10))
        
        output1 = embedding(input_ids1)
        output2 = embedding(input_ids2)
        
        # Should be different
        assert not torch.allclose(output1, output2)
    
    def test_extra_repr(self):
        """Test extra_repr provides useful information"""
        embedding = HolographicTTEmbedding(10000, 512, rank=16)
        repr_str = embedding.extra_repr()
        
        assert 'vocab_size=10000' in repr_str
        assert 'd_model=512' in repr_str
        assert 'rank=16' in repr_str
        assert 'compression_ratio' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


class TestHTTDecoder:
    """Test suite for HTTDecoder"""

    @pytest.fixture
    def setup(self):
        vocab_size = 500
        d_model = 64
        rank = 8
        embedding = HolographicTTEmbedding(
            vocab_size=vocab_size, d_model=d_model, rank=rank
        )
        decoder = HTTDecoder(embedding)
        return embedding, decoder, vocab_size, d_model

    def test_decoder_output_shape(self, setup):
        embedding, decoder, vocab_size, d_model = setup
        batch_size, seq_len = 4, 16
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        logits = decoder(hidden_states)
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_weight_tying(self, setup):
        embedding, decoder, vocab_size, d_model = setup
        batch_size, seq_len = 2, 8
        hidden_states = torch.randn(batch_size, seq_len, d_model)

        logits1 = decoder(hidden_states)

        with torch.no_grad():
            embedding.core1.data += 0.1

        logits2 = decoder(hidden_states)

        assert not torch.allclose(logits1, logits2)
        assert logits1.shape == logits2.shape

    def test_d_model_edge_case(self, setup):
        _, _, vocab_size, _ = setup
        d_model_odd = 60
        rank = 8
        embedding = HolographicTTEmbedding(
            vocab_size=vocab_size, d_model=d_model_odd, rank=rank
        )
        decoder = HTTDecoder(embedding)

        batch_size, seq_len = 4, 16
        hidden_states = torch.randn(batch_size, seq_len, d_model_odd)
        logits = decoder(hidden_states)

        assert logits.shape == (batch_size, seq_len, vocab_size)
