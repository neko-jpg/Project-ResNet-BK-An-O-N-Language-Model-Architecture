import pytest
import torch
from src.models.phase4.adscft_core.bulk_generator import BulkSpaceGenerator
from src.models.phase4.adscft_core.geodesic_search import fast_marching_method_cpu

class TestHolographicDual:
    def test_bulk_generator_shape(self):
        d_model = 32
        bulk_dim = 4
        model = BulkSpaceGenerator(d_model, bulk_dim=bulk_dim)

        batch = 2
        seq = 10
        tokens = torch.randn(batch, seq, d_model)

        features, diag = model(tokens)

        assert features.shape == (batch, seq, d_model)
        assert 'geodesic_sample' in diag

    def test_fast_marching_cpu(self):
        batch = 1
        seq = 5
        bulk_dim = 4
        d = 10

        coords = torch.randn(batch, seq, bulk_dim, d)

        geodesics = fast_marching_method_cpu(coords, ads_radius=1.0)

        assert geodesics.shape == coords.shape
        # Check that geodesics are modified (not identical to coords)
        # Except at z=0 which is fixed boundary (initially 0 cost, but coords not changed)
        # In FMM, z=0 coords are reference.
        # Our FMM copies z=0.
        assert torch.allclose(geodesics[:, :, 0], coords[:, :, 0])
        # At z>0, should differ if coords are random because of the update rule
        # update rule: prev + (curr-prev)*ratio. ratio < 1 usually.
        # So it won't be equal to curr.
        assert not torch.allclose(geodesics[:, :, 1:], coords[:, :, 1:])
