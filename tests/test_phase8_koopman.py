import torch
import unittest
from src.models.phase8.koopman_bridge import KoopmanBridge

class TestKoopmanBridge(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.bridge = KoopmanBridge(self.d_model)

    def test_mapping_range(self):
        """Test that outputs are always within Poincare ball."""
        # Random eigenfunctions
        efuncs = torch.randn(2, 10, self.d_model) * 10.0 # Large values
        evals = torch.randn(self.d_model)

        out = self.bridge(efuncs, evals)

        norms = out.norm(dim=-1)
        max_norm = norms.max().item()

        print(f"Max Norm: {max_norm}")
        self.assertLess(max_norm, 1.0)

    def test_amplitude_mapping(self):
        """Test that larger amplitude eigenfunctions map closer to boundary."""
        small = torch.randn(1, 1, self.d_model) * 0.1
        large = torch.randn(1, 1, self.d_model) * 10.0

        out_small = self.bridge(small, None)
        out_large = self.bridge(large, None)

        self.assertGreater(out_large.norm().item(), out_small.norm().item())

if __name__ == '__main__':
    unittest.main()
