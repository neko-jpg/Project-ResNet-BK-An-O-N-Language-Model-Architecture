import torch
import unittest
from src.models.phase8.guard import NumericalGuard

class TestNumericalGuard(unittest.TestCase):
    def setUp(self):
        self.guard = NumericalGuard(max_norm=0.9)

    def test_boundary_collapse_prevention(self):
        """Test that vectors outside the ball are pulled back."""
        # Vector with norm 2.0
        x = torch.randn(2, 5)
        x = x / x.norm(dim=-1, keepdim=True) * 2.0

        out = self.guard(x)
        norms = out.norm(dim=-1)

        self.assertTrue(torch.all(norms <= 0.9 + 1e-4), f"Norms exceeded max: {norms}")
        self.assertGreater(self.guard.collapse_count, 0)

    def test_safe_vectors_untouched(self):
        """Test that vectors inside the ball are not modified."""
        x = torch.randn(2, 5)
        x = x / x.norm(dim=-1, keepdim=True) * 0.5

        out = self.guard(x)
        self.assertTrue(torch.allclose(x, out), "Safe vectors were modified")

if __name__ == '__main__':
    unittest.main()
