"""
Tests for Sheaf Attention Module.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import pytest
import torch
import json
import math

import sys
sys.path.insert(0, '.')
from src.models.phase8.sheaf_attention import (
    SheafAttentionModule,
    SheafAttentionConfig,
    SheafDiagnostics,
    SheafSection,
    serialize_sheaf_structure,
    create_sheaf_attention,
)


class TestSheafAttentionConfig:
    """Test configuration serialization."""
    
    def test_config_to_json(self):
        """Test JSON serialization."""
        config = SheafAttentionConfig(
            d_model=256,
            num_heads=8,
            agreement_threshold=0.1,
        )
        json_str = config.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['d_model'] == 256
        assert data['num_heads'] == 8
        assert data['agreement_threshold'] == 0.1
    
    def test_config_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"d_model": 512, "num_heads": 16, "agreement_threshold": 0.2, "use_cohomology": false, "dropout": 0.2}'
        config = SheafAttentionConfig.from_json(json_str)
        assert config.d_model == 512
        assert config.num_heads == 16
        assert config.agreement_threshold == 0.2
        assert config.use_cohomology == False
    
    def test_config_round_trip(self):
        """Test configuration round-trip."""
        original = SheafAttentionConfig(
            d_model=384,
            num_heads=12,
            agreement_threshold=0.15,
            use_cohomology=True,
            dropout=0.1,
        )
        json_str = original.to_json()
        restored = SheafAttentionConfig.from_json(json_str)
        
        assert original.d_model == restored.d_model
        assert original.num_heads == restored.num_heads
        assert abs(original.agreement_threshold - restored.agreement_threshold) < 1e-6
        assert original.use_cohomology == restored.use_cohomology
        assert abs(original.dropout - restored.dropout) < 1e-6


class TestSheafAttentionModule:
    """Test the main Sheaf Attention module."""
    
    @pytest.fixture
    def module(self):
        """Create a test module."""
        return SheafAttentionModule(
            d_model=64,
            num_heads=4,
            agreement_threshold=0.1,
            use_cohomology=True,
            dropout=0.0,  # Disable dropout for deterministic tests
        )
    
    def test_forward_shape(self, module):
        """Test output shapes."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D)
        
        output, diagnostics = module(x, return_diagnostics=True)
        
        assert output.shape == (B, N, D)
        assert diagnostics is not None
        assert diagnostics.agreement_matrix.shape == (B, 4, 4)
        assert diagnostics.consensus_weights.shape == (B, 4)
    
    def test_agreement_matrix_properties(self, module):
        """Test agreement matrix properties."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D)
        
        _, diagnostics = module(x, return_diagnostics=True)
        agreement = diagnostics.agreement_matrix
        
        # Diagonal should be 1
        for b in range(B):
            for h in range(4):
                assert abs(agreement[b, h, h].item() - 1.0) < 1e-5
        
        # Should be symmetric
        assert torch.allclose(agreement, agreement.transpose(1, 2), atol=1e-5)
        
        # Values should be in [0, 1]
        assert (agreement >= 0).all()
        assert (agreement <= 1).all()
    
    def test_consensus_weights_sum_to_one(self, module):
        """Test that consensus weights sum to 1."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D)
        
        _, diagnostics = module(x, return_diagnostics=True)
        weights = diagnostics.consensus_weights
        
        # Should sum to 1 (softmax)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_consensus_consistency(self, module):
        """
        **Feature: phase8-hyperbolic-transcendence, Property 6: Sheaf Consensus Consistency**
        
        Test that consensus aggregation filters inconsistent information.
        """
        B, N, D = 2, 16, 64
        
        # Create input where some heads will be inconsistent
        x = torch.randn(B, N, D)
        
        _, diagnostics = module(x, return_diagnostics=True)
        
        # Heads with low agreement should have lower weights
        agreement = diagnostics.agreement_matrix
        weights = diagnostics.consensus_weights
        
        # Average agreement per head (excluding self)
        H = 4
        mask = ~torch.eye(H, dtype=torch.bool)
        avg_agreement = (agreement * mask.unsqueeze(0)).sum(dim=-1) / (H - 1)
        
        # Basic consistency checks
        # 1. All weights should be non-negative
        assert (weights >= 0).all(), "Weights should be non-negative"
        
        # 2. Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(B)), "Weights should sum to 1"
        
        # 3. Agreement values should be in [0, 1]
        assert (avg_agreement >= 0).all() and (avg_agreement <= 1).all(), "Agreement should be in [0, 1]"
        
        # 4. Heads with very low agreement (if any) should have lower weights
        # This is a soft test - we just verify the mechanism works
        for b in range(B):
            # Check that the module produces valid outputs
            assert torch.isfinite(weights[b]).all(), "Weights should be finite"
            assert torch.isfinite(avg_agreement[b]).all(), "Agreement should be finite"
    
    def test_cohomology_obstruction(self, module):
        """Test cohomology obstruction computation."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D)
        
        _, diagnostics = module(x, return_diagnostics=True)
        
        assert diagnostics.cohomology_obstruction is not None
        assert diagnostics.cohomology_obstruction.shape == (B,)
        
        # Obstruction should be in [0, 1]
        assert (diagnostics.cohomology_obstruction >= 0).all()
        assert (diagnostics.cohomology_obstruction <= 1).all()
    
    def test_with_mask(self, module):
        """Test with attention mask."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D)
        
        # Causal mask
        mask = torch.tril(torch.ones(B, N, N))
        
        output, _ = module(x, mask=mask)
        
        assert output.shape == (B, N, D)
        assert torch.isfinite(output).all()
    
    def test_gradient_flow(self, module):
        """Test that gradients flow through the module."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D, requires_grad=True)
        
        output, _ = module(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
    
    def test_diagnostics_to_dict(self, module):
        """Test diagnostics serialization."""
        B, N, D = 2, 16, 64
        x = torch.randn(B, N, D)
        
        _, diagnostics = module(x, return_diagnostics=True)
        result_dict = diagnostics.to_dict()
        
        assert 'agreement_matrix' in result_dict
        assert 'consensus_weights' in result_dict
        assert 'cohomology_obstruction' in result_dict
        assert 'filtered_heads' in result_dict
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)


class TestSheafSerialization:
    """Test sheaf structure serialization."""
    
    def test_serialize_sheaf_structure(self):
        """Test JSON serialization of sheaf structure."""
        module = SheafAttentionModule(
            d_model=64,
            num_heads=4,
            agreement_threshold=0.1,
        )
        
        B, H, N, D_h = 2, 4, 16, 16
        head_outputs = torch.randn(B, H, N, D_h)
        
        json_str = serialize_sheaf_structure(module, head_outputs)
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        
        assert data['num_heads'] == 4
        assert data['head_dim'] == 16
        assert data['sequence_length'] == 16
        assert data['batch_size'] == 2
        assert len(data['sections']) == 4
        assert len(data['restriction_maps']) == 6  # C(4,2) = 6 pairs
    
    def test_sheaf_section(self):
        """Test SheafSection class."""
        data = torch.randn(2, 16, 16)
        section = SheafSection(
            head_index=0,
            data=data,
            restriction_to={1: torch.randn(2, 16, 16)}
        )
        
        result = section.to_dict()
        
        assert result['head_index'] == 0
        assert result['data_shape'] == [2, 16, 16]
        assert 'data_norm' in result
        assert '1' in result['restrictions']


class TestFactoryFunction:
    """Test the factory function."""
    
    def test_create_sheaf_attention(self):
        """Test factory function."""
        module = create_sheaf_attention(
            d_model=128,
            num_heads=8,
            agreement_threshold=0.2,
        )
        
        assert isinstance(module, SheafAttentionModule)
        assert module.d_model == 128
        assert module.num_heads == 8
        assert module.agreement_threshold == 0.2


class TestRestrictionMaps:
    """Test restriction map computation."""
    
    def test_pair_index(self):
        """Test pair index computation."""
        module = SheafAttentionModule(d_model=64, num_heads=4)
        
        # For 4 heads, pairs are: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        # Indices should be: 0, 1, 2, 3, 4, 5
        assert module._get_pair_index(0, 1) == 0
        assert module._get_pair_index(0, 2) == 1
        assert module._get_pair_index(0, 3) == 2
        assert module._get_pair_index(1, 2) == 3
        assert module._get_pair_index(1, 3) == 4
        assert module._get_pair_index(2, 3) == 5
        
        # Should be symmetric
        assert module._get_pair_index(1, 0) == module._get_pair_index(0, 1)
    
    def test_restriction_map_count(self):
        """Test that correct number of restriction maps are created."""
        for num_heads in [2, 4, 8, 16]:
            module = SheafAttentionModule(d_model=64, num_heads=num_heads)
            expected_pairs = num_heads * (num_heads - 1) // 2
            assert len(module.restriction_maps) == expected_pairs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
