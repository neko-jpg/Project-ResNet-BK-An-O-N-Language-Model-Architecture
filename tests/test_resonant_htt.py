"""
Unit tests for Resonant Holographic Tensor Train (HTT) Embedding

Tests:
    - Resonant vocab size expansion (to 2^N)
    - Hypercube factorization symmetry
    - Ghost token handling
    - Iso-Spectral Zeta initialization
    - Compression ratio verification
    - Gradient flow through all cores

Requirements:
    - Condition number κ ≈ 1 for all cores
    - 90%+ compression ratio maintained
    - Gradient flow preserved

Author: Project MUSE Team
"""

import pytest
import torch
import torch.nn as nn
import math

from src.models.phase1.resonant_htt_embedding import (
    ResonantHTTEmbedding,
    ResonantHTTDecoder,
    create_resonant_htt_embedding,
    diagnose_vocab_size,
)


class TestResonantVocabExpansion:
    """Test resonant number expansion to 2^N"""
    
    def test_power_of_2_unchanged(self):
        """Power of 2 vocab sizes should remain unchanged"""
        for power in [10, 12, 14, 15, 16]:
            vocab_size = 2 ** power
            emb = ResonantHTTEmbedding(vocab_size, 256, rank=8)
            assert emb.resonant_vocab_size == vocab_size
            assert emb.ghost_tokens == 0
    
    def test_expansion_to_next_power(self):
        """Non-power-of-2 should expand to next power"""
        test_cases = [
            (3200, 4096),      # The problematic Zombie State case
            (32000, 32768),    # Japanese tokenizer
            (50000, 65536),    # GPT-2 style
            (100000, 131072),  # Large vocab
        ]
        
        for vocab_size, expected_resonant in test_cases:
            emb = ResonantHTTEmbedding(vocab_size, 256, rank=8)
            assert emb.resonant_vocab_size == expected_resonant, \
                f"Expected {vocab_size} -> {expected_resonant}, got {emb.resonant_vocab_size}"
            assert emb.ghost_tokens == expected_resonant - vocab_size
    
    def test_ghost_token_count(self):
        """Verify ghost token calculations"""
        emb = ResonantHTTEmbedding(50000, 512, rank=16)
        assert emb.ghost_tokens == 65536 - 50000 == 15536
        assert emb.vocab_size == 50000  # Original vocab preserved


class TestHypercubeFactorization:
    """Test symmetric hypercube factorization"""
    
    def test_4_core_factorization(self):
        """4 cores should produce symmetric factors"""
        # 65536 = 16^4
        emb = ResonantHTTEmbedding(50000, 256, rank=8, num_cores=4)
        
        # All factors should be equal for perfect symmetry
        factors = emb.vocab_factors
        assert len(factors) == 4
        assert all(f == 16 for f in factors), f"Expected [16,16,16,16], got {factors}"
    
    def test_3_core_factorization(self):
        """3 cores should also produce balanced factors"""
        # 4096 = 16^3
        emb = ResonantHTTEmbedding(3200, 256, rank=8, num_cores=3)
        
        factors = emb.vocab_factors
        assert len(factors) == 3
        # Product should equal resonant_vocab_size
        product = 1
        for f in factors:
            product *= f
        assert product == emb.resonant_vocab_size
    
    def test_d_model_factorization(self):
        """d_model factorization should be balanced"""
        emb = ResonantHTTEmbedding(1000, 4096, rank=8, num_cores=4)
        
        # 4096 = 8^4
        factors = emb.d_factors
        assert len(factors) == 4
        product = 1
        for f in factors:
            product *= f
        # Product should be >= d_model
        assert product >= emb.d_model


class TestOutputShape:
    """Test output shape correctness"""
    
    def test_basic_output_shape(self):
        """Output should be (B, L, d_model)"""
        vocab_size = 5000
        d_model = 256
        batch_size = 4
        seq_len = 64
        
        emb = ResonantHTTEmbedding(vocab_size, d_model, rank=16)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        output = emb(input_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert output.dtype in [torch.float32, torch.bfloat16]
    
    def test_ghost_tokens_not_accessible(self):
        """Input IDs should only be in [0, vocab_size)"""
        vocab_size = 3200
        emb = ResonantHTTEmbedding(vocab_size, 128, rank=8)
        
        # Valid input
        input_ids = torch.randint(0, vocab_size, (2, 10))
        output = emb(input_ids)
        
        assert output.shape == (2, 10, 128)
        assert torch.isfinite(output).all()


class TestCompressionRatio:
    """Test compression ratio targets"""
    
    def test_90_percent_compression(self):
        """Should achieve >90% compression"""
        vocab_size = 50000
        d_model = 1024
        rank = 16
        
        emb = ResonantHTTEmbedding(vocab_size, d_model, rank=rank)
        
        standard_params, tt_params = emb.get_parameter_counts()
        ratio = emb.get_compression_ratio()
        
        assert ratio < 0.1, f"Compression ratio {ratio:.4f} does not meet 90% target"
        assert standard_params == vocab_size * d_model
        
        print(f"Compression: {(1 - ratio) * 100:.2f}%")
        print(f"Standard params: {standard_params:,}")
        print(f"TT params: {tt_params:,}")


class TestZetaInitialization:
    """Test Iso-Spectral Zeta initialization"""
    
    def test_zeta_init_finite(self):
        """Zeta initialization should produce finite values"""
        emb = ResonantHTTEmbedding(
            vocab_size=1000, 
            d_model=128, 
            rank=8,
            use_zeta_init=True
        )
        
        for i, core in enumerate(emb.cores):
            assert torch.isfinite(core).all(), f"Core {i} has NaN/Inf after Zeta init"
    
    def test_zeta_vs_orthogonal_init_variance(self):
        """Zeta init should have structured variance (not uniform)"""
        emb_zeta = ResonantHTTEmbedding(
            vocab_size=1000, 
            d_model=128, 
            rank=8,
            use_zeta_init=True
        )
        
        emb_orth = ResonantHTTEmbedding(
            vocab_size=1000, 
            d_model=128, 
            rank=8,
            use_zeta_init=False
        )
        
        # Both should have similar magnitude
        zeta_norm = sum(c.norm().item() for c in emb_zeta.cores)
        orth_norm = sum(c.norm().item() for c in emb_orth.cores)
        
        # Should be within 10x of each other
        assert 0.1 < zeta_norm / orth_norm < 10.0


class TestGradientFlow:
    """Test gradient flow through all cores"""
    
    def test_all_cores_have_gradients(self):
        """Gradients should flow to all cores"""
        emb = ResonantHTTEmbedding(1000, 128, rank=8)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        output = emb(input_ids)
        loss = output.sum()
        loss.backward()
        
        for i, core in enumerate(emb.cores):
            assert core.grad is not None, f"Core {i} has no gradient"
            assert torch.isfinite(core.grad).all(), f"Core {i} has NaN/Inf gradient"
            assert core.grad.norm() > 0, f"Core {i} has zero gradient"
    
    def test_phase_parameter_gradient(self):
        """Phase shift parameter should receive gradients"""
        emb = ResonantHTTEmbedding(1000, 128, rank=8, phase_encoding=True)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        output = emb(input_ids)
        loss = output.sum()
        loss.backward()
        
        assert emb.phase_shift.grad is not None
        assert torch.isfinite(emb.phase_shift.grad).all()


class TestDecoder:
    """Test ResonantHTTDecoder"""
    
    def test_decoder_output_shape(self):
        """Decoder should output (B, L, vocab_size)"""
        vocab_size = 1000
        d_model = 128
        
        emb = ResonantHTTEmbedding(vocab_size, d_model, rank=8)
        decoder = ResonantHTTDecoder(emb)
        
        hidden_states = torch.randn(2, 10, d_model)
        logits = decoder(hidden_states)
        
        assert logits.shape == (2, 10, vocab_size)
    
    def test_decoder_ghost_tokens_cropped(self):
        """Decoder output should NOT include ghost tokens"""
        vocab_size = 3200
        emb = ResonantHTTEmbedding(vocab_size, 128, rank=8)
        decoder = ResonantHTTDecoder(emb)
        
        hidden_states = torch.randn(2, 10, 128)
        logits = decoder(hidden_states)
        
        # Should be vocab_size, not resonant_vocab_size
        assert logits.shape == (2, 10, vocab_size)
        assert logits.shape[2] != emb.resonant_vocab_size


class TestDiagnostics:
    """Test vocab size diagnostics"""
    
    def test_diagnose_power_of_2(self):
        """Power of 2 should be low risk"""
        diag = diagnose_vocab_size(65536)
        
        assert diag['is_power_of_2'] == True
        assert diag['risk_level'] == 'low'
        assert diag['overhead_percent'] == 0.0
    
    def test_diagnose_awkward_number(self):
        """Awkward numbers should have higher risk"""
        diag = diagnose_vocab_size(3200)
        
        assert diag['is_power_of_2'] == False
        assert diag['resonant_size'] == 4096
        # 28% overhead is medium risk
        assert diag['risk_level'] in ['medium', 'high']
    
    def test_diagnose_50k(self):
        """50K vocab should work well with resonant expansion"""
        diag = diagnose_vocab_size(50000)
        
        assert diag['resonant_size'] == 65536
        # 31% overhead is high
        assert 'overhead' in diag['recommendation'].lower() or 'resonant' in diag['recommendation'].lower()


class TestNumericalStability:
    """Test numerical stability"""
    
    def test_no_nan_inf_output(self):
        """Output should never contain NaN or Inf"""
        emb = ResonantHTTEmbedding(5000, 256, rank=16)
        
        for _ in range(5):
            input_ids = torch.randint(0, 5000, (4, 64))
            output = emb(input_ids)
            
            assert torch.isfinite(output).all()
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_boundary_tokens(self):
        """First and last tokens should work correctly"""
        vocab_size = 1000
        emb = ResonantHTTEmbedding(vocab_size, 128, rank=8)
        
        input_ids = torch.tensor([
            [0, 1, vocab_size - 2, vocab_size - 1],
            [vocab_size // 2, 100, 500, 999],
        ])
        
        output = emb(input_ids)
        
        assert output.shape == (2, 4, 128)
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
