import torch
import unittest
from src.models.phase8.adaptive import AdaptiveComputation

class TestAdaptiveComputation(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.adaptive = AdaptiveComputation(self.d_model)

    def test_origin_vs_boundary(self):
        """Test that tokens near origin have higher exit probability than tokens near boundary."""
        # 1. Near Origin (Norm ~ 0)
        x_origin = torch.randn(1, 10, self.d_model) * 0.01

        # 2. Near Boundary (Norm ~ 0.9)
        x_boundary = torch.randn(1, 10, self.d_model)
        x_boundary = x_boundary / x_boundary.norm(dim=-1, keepdim=True) * 0.95

        _, p_origin = self.adaptive(x_origin, 0, 10)
        _, p_boundary = self.adaptive(x_boundary, 0, 10)

        print(f"P(Origin Exit): {p_origin.mean().item()}")
        print(f"P(Boundary Exit): {p_boundary.mean().item()}")

        self.assertGreater(p_origin.mean().item(), p_boundary.mean().item())

    def test_thresholding(self):
        # Create input that should definitely exit (origin)
        x = torch.zeros(1, 1, self.d_model)
        should_exit, _ = self.adaptive(x, 0, 10)

        # With high bias for origin, it should exit
        self.assertTrue(should_exit.item())

if __name__ == '__main__':
    unittest.main()
