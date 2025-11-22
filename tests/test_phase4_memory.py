import pytest
import torch
import shutil
import tempfile
import os
from unittest.mock import MagicMock, patch

from src.models.phase4.memory_monitor import MemoryMonitor
from src.models.phase4.adscft_core.bulk_generator import BulkSpaceGenerator
from src.models.phase4.topological_memory.sparse_tensor_rep import SparseKnotRepresentation

class TestPhase4Memory:

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_memory_monitor_mock(self):
        """Test that MemoryMonitor mock mode works."""
        monitor = MemoryMonitor(mock_limit_gb=10.0, mode='mock')
        assert monitor.get_total_memory() == int(10.0 * 1024**3)
        assert monitor.get_free_memory() == int(10.0 * 1024**3)

        monitor.set_mock_usage(2.0)
        assert monitor.get_free_memory() == int(8.0 * 1024**3)

        stats = monitor.get_memory_stats()
        assert stats['percent_used'] == 20.0

    def test_bulk_dynamic_adjustment(self):
        """Test that BulkSpaceGenerator reduces dimensions under memory pressure."""
        monitor = MemoryMonitor(mock_limit_gb=8.0, mode='mock')
        d_model = 64
        bulk_gen = BulkSpaceGenerator(d_model=d_model, monitor=monitor)

        # Normal operation (Free > 2GB)
        monitor.set_mock_usage(1.0) # 7GB free
        x = torch.randn(1, 10, d_model)
        _, diag = bulk_gen(x)

        assert diag['active_bulk_dim'] == bulk_gen.bulk_dim
        assert diag['low_memory_mode'] is False

        # Low memory operation (Free < 2GB)
        # 8GB total. Use 6.5GB -> 1.5GB free
        monitor.set_mock_usage(6.5)
        _, diag = bulk_gen(x)

        expected_dim = max(2, bulk_gen.bulk_dim // 2)
        assert diag['active_bulk_dim'] == expected_dim
        assert diag['low_memory_mode'] is True

    @patch('src.models.phase4.topological_memory.sparse_tensor_rep.HAS_ZARR', True)
    def test_knot_lru_cache(self):
        """Test LRU cache eviction in SparseKnotRepresentation."""

        # Mock Zarr store to avoid real disk I/O and dependency issues
        with patch('src.models.phase4.topological_memory.sparse_tensor_rep.zarr') as mock_zarr:
            # Setup mock store
            mock_store = MagicMock()
            mock_zarr.open.return_value = mock_store

            # Mock array data return
            def get_item(key):
                # Return random numpy array
                return torch.randn(10, 3).numpy()
            mock_store.__getitem__.side_effect = get_item
            mock_store.__contains__.side_effect = lambda k: True

            knot_rep = SparseKnotRepresentation(
                d_model=32,
                storage_path=os.path.join(self.temp_dir, 'test.zarr'),
                cache_capacity=2
            )

            # Manually inject into indices so we can request them
            knot_rep.knot_indices = [0, 1, 2]

            # Request knot 0 -> Load and Cache
            k0 = knot_rep.get_knot(0)
            assert k0 is not None
            assert 0 in knot_rep.cache
            assert len(knot_rep.cache) == 1

            # Request knot 1 -> Load and Cache
            k1 = knot_rep.get_knot(1)
            assert 1 in knot_rep.cache
            assert len(knot_rep.cache) == 2

            # Request knot 2 -> Load and Cache, Evict 0 (LRU)
            k2 = knot_rep.get_knot(2)
            assert 2 in knot_rep.cache
            assert 0 not in knot_rep.cache # 0 should be evicted
            assert 1 in knot_rep.cache # 1 should remain
            assert len(knot_rep.cache) == 2

            # Request knot 1 again -> Move to MRU
            _ = knot_rep.get_knot(1)

            # Request knot 0 -> Evict 2 (LRU)
            _ = knot_rep.get_knot(0)
            assert 0 in knot_rep.cache
            assert 2 not in knot_rep.cache
            assert 1 in knot_rep.cache # 1 was accessed recently, so it stays
