import torch
import unittest
from src.models.phase8.topology import TopologicalNorm

class TestTopologicalNorm(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.norm = TopologicalNorm(self.d_model)

    def test_forward_shape(self):
        x = torch.randn(2, 32, self.d_model)
        out = self.norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_metric_sensitivity(self):
        """Test that clustered data yields higher distance variance than uniform data."""
        # 1. Clustered Data: 2 distinct far clusters
        c1 = torch.zeros(1, 15, self.d_model) # Cluster at origin
        c2 = torch.ones(1, 15, self.d_model) * 10.0 # Cluster at (10,10...)
        x_clustered = torch.cat([c1, c2], dim=1) # (1, 30, D)

        # 2. Uniform Data: Random points spread in the same range [0, 10]
        x_uniform = torch.rand(1, 30, self.d_model) * 10.0

        metric_clustered = self.norm._approximate_persistence(x_clustered)
        metric_uniform = self.norm._approximate_persistence(x_uniform)

        # Clustered data has distances of ~0 (within c1), ~0 (within c2), and ~sqrt(D)*10 (between).
        # This bimodal distribution has HIGH variance.

        # Uniform data has a "bell curve" of distances. Lower variance.

        print(f"Clustered Metric: {metric_clustered.item()}")
        print(f"Uniform Metric: {metric_uniform.item()}")

        self.assertGreater(metric_clustered.item(), metric_uniform.item())

if __name__ == '__main__':
    unittest.main()
