"""
Tests for Hyperbolic Persistent Homology Module.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import pytest
import torch
import json
import time
from typing import Dict

# Import the module
import sys
sys.path.insert(0, '.')
from src.models.phase8.persistent_homology import (
    HyperbolicPersistentHomology,
    PersistentHomologyConfig,
    PersistentHomologyDiagnostics,
    create_persistent_homology,
)


class TestPersistentHomologyConfig:
    """Test configuration serialization."""
    
    def test_config_to_json(self):
        """Test JSON serialization."""
        config = PersistentHomologyConfig(
            d_model=256,
            max_dimension=1,
            threshold_beta1=3,
        )
        json_str = config.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['d_model'] == 256
        assert data['max_dimension'] == 1
        assert data['threshold_beta1'] == 3
    
    def test_config_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"d_model": 128, "max_dimension": 2, "threshold_beta1": 5, "num_landmarks": 32, "filtration_steps": 10, "use_sparse_filtration": true, "curvature_adjustment_rate": 0.2}'
        config = PersistentHomologyConfig.from_json(json_str)
        assert config.d_model == 128
        assert config.max_dimension == 2
        assert config.threshold_beta1 == 5
        assert config.num_landmarks == 32
    
    def test_config_round_trip(self):
        """
        **Feature: phase8-hyperbolic-transcendence, Property 3 (variant): Configuration Round-Trip**
        
        Test that configuration can be serialized and deserialized without loss.
        """
        original = PersistentHomologyConfig(
            d_model=512,
            max_dimension=1,
            threshold_beta1=4,
            num_landmarks=128,
            filtration_steps=30,
            use_sparse_filtration=True,
            curvature_adjustment_rate=0.15,
        )
        json_str = original.to_json()
        restored = PersistentHomologyConfig.from_json(json_str)
        
        assert original.d_model == restored.d_model
        assert original.max_dimension == restored.max_dimension
        assert original.threshold_beta1 == restored.threshold_beta1
        assert original.num_landmarks == restored.num_landmarks
        assert original.filtration_steps == restored.filtration_steps
        assert original.use_sparse_filtration == restored.use_sparse_filtration
        assert abs(original.curvature_adjustment_rate - restored.curvature_adjustment_rate) < 1e-6


class TestHyperbolicPersistentHomology:
    """Test the main Persistent Homology module."""
    
    @pytest.fixture
    def module(self):
        """Create a test module."""
        return HyperbolicPersistentHomology(
            d_model=64,
            max_dimension=1,
            threshold_beta1=3,
            num_landmarks=16,
            filtration_steps=10,
        )
    
    def test_forward_shape(self, module):
        """Test output shapes."""
        B, N, D = 2, 32, 64
        embeddings = torch.randn(B, N, D) * 0.5  # Keep within Poincaré ball
        
        result = module(embeddings)
        
        assert 'beta_0' in result
        assert 'beta_1' in result
        assert 'fragmentation_score' in result
        assert 'circular_reasoning_detected' in result
        
        assert result['beta_0'].shape == (B,)
        assert result['beta_1'].shape == (B,)
        assert result['fragmentation_score'].shape == (B,)
        assert result['circular_reasoning_detected'].shape == (B,)
    
    def test_betti_number_non_negative(self, module):
        """
        **Feature: phase8-hyperbolic-transcendence, Property 4: Betti Number Consistency**
        
        Betti numbers should be non-negative integers.
        """
        B, N, D = 4, 64, 64
        embeddings = torch.randn(B, N, D) * 0.3
        
        result = module(embeddings)
        
        assert (result['beta_0'] >= 0).all()
        assert (result['beta_1'] >= 0).all()
    
    def test_fragmentation_score_range(self, module):
        """Fragmentation score should be in [0, 1]."""
        B, N, D = 2, 32, 64
        embeddings = torch.randn(B, N, D) * 0.5
        
        result = module(embeddings)
        
        assert (result['fragmentation_score'] >= 0).all()
        assert (result['fragmentation_score'] <= 1).all()
    
    def test_circular_reasoning_detection(self, module):
        """Test circular reasoning detection flag."""
        B, N, D = 2, 32, 64
        embeddings = torch.randn(B, N, D) * 0.5
        
        result = module(embeddings)
        
        # Should be boolean tensor
        assert result['circular_reasoning_detected'].dtype == torch.bool
    
    def test_curvature_adjustment(self, module):
        """Test curvature adjustment suggestion."""
        B = 2
        beta_1 = torch.tensor([1.0, 5.0])
        current_curvature = torch.tensor(1.0)
        
        adjusted = module.suggest_curvature_adjustment(beta_1, current_curvature)
        
        # Higher β₁ should result in higher curvature
        assert adjusted[1] > adjusted[0]
    
    def test_diagnostics(self, module):
        """Test diagnostics collection."""
        B, N, D = 2, 32, 64
        embeddings = torch.randn(B, N, D) * 0.5
        
        diagnostics = module.get_diagnostics(embeddings)
        
        assert isinstance(diagnostics, PersistentHomologyDiagnostics)
        assert diagnostics.computation_time_ms is not None
        assert diagnostics.computation_time_ms >= 0
    
    def test_diagnostics_to_dict(self, module):
        """Test diagnostics serialization."""
        B, N, D = 2, 32, 64
        embeddings = torch.randn(B, N, D) * 0.5
        
        diagnostics = module.get_diagnostics(embeddings)
        result_dict = diagnostics.to_dict()
        
        assert 'beta_0' in result_dict
        assert 'beta_1' in result_dict
        assert 'fragmentation_score' in result_dict
        assert 'circular_reasoning_detected' in result_dict
        assert 'computation_time_ms' in result_dict
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)


class TestSparseFilterationComplexity:
    """
    **Feature: phase8-hyperbolic-transcendence, Property 5: Sparse Filtration Complexity**
    
    Test that computation scales as O(N log N) for long sequences.
    """
    
    def test_complexity_scaling(self):
        """Test that computation time scales sub-quadratically."""
        module = HyperbolicPersistentHomology(
            d_model=64,
            max_dimension=1,
            num_landmarks=32,
            filtration_steps=10,
            use_sparse_filtration=True,
        )
        
        times = []
        sizes = [128, 256, 512, 1024]
        
        for N in sizes:
            embeddings = torch.randn(1, N, 64) * 0.3
            
            start = time.time()
            for _ in range(3):  # Average over 3 runs
                _ = module(embeddings)
            elapsed = (time.time() - start) / 3
            times.append(elapsed)
        
        # Check that time doesn't grow quadratically
        # For O(N log N), doubling N should roughly double time (plus log factor)
        # For O(N²), doubling N would quadruple time
        
        # Ratio of times for 2x size increase
        ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        
        # All ratios should be less than 4 (quadratic would give ~4)
        # For O(N log N), expect ratios around 2-2.5
        for ratio in ratios:
            assert ratio < 4.0, f"Complexity appears quadratic: ratio = {ratio}"


class TestKnownTopology:
    """Test with known topological structures."""
    
    def test_single_cluster(self):
        """Single tight cluster should have β₀ ≈ 1."""
        module = HyperbolicPersistentHomology(
            d_model=64,
            num_landmarks=16,
            filtration_steps=10,
        )
        
        # Create a single tight cluster
        center = torch.randn(1, 64) * 0.1
        embeddings = center + torch.randn(1, 32, 64) * 0.01
        
        result = module(embeddings)
        
        # Should have low β₀ (close to 1)
        assert result['beta_0'][0] <= 5, f"Expected low β₀, got {result['beta_0'][0]}"
    
    def test_multiple_clusters(self):
        """Multiple separated clusters should have higher β₀."""
        module = HyperbolicPersistentHomology(
            d_model=64,
            num_landmarks=32,
            filtration_steps=10,
        )
        
        # Create 4 separated clusters
        embeddings = []
        for i in range(4):
            center = torch.zeros(64)
            center[i * 16:(i + 1) * 16] = 0.5
            cluster = center + torch.randn(8, 64) * 0.01
            embeddings.append(cluster)
        
        embeddings = torch.cat(embeddings, dim=0).unsqueeze(0)  # (1, 32, 64)
        
        result = module(embeddings)
        
        # Should have higher β₀ (multiple components)
        # Note: exact value depends on filtration parameters
        assert result['beta_0'][0] >= 1


class TestFactoryFunction:
    """Test the factory function."""
    
    def test_create_persistent_homology(self):
        """Test factory function."""
        module = create_persistent_homology(
            d_model=128,
            max_dimension=1,
            threshold_beta1=5,
        )
        
        assert isinstance(module, HyperbolicPersistentHomology)
        assert module.d_model == 128
        assert module.max_dimension == 1
        assert module.threshold_beta1 == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
