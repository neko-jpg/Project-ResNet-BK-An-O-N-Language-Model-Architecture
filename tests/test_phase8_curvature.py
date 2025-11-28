import torch
import torch.nn.functional as F
import unittest
from src.models.phase8.curvature import CurvatureAdapter

class TestCurvatureAdapter(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.adapter = CurvatureAdapter(self.d_model)

    def test_hierarchy_detection(self):
        """Test that hierarchical data yields higher CV score than flat data."""
        # 1. Flat Data (All points on a shell)
        x_flat = torch.randn(1, 32, self.d_model)
        x_flat = F.normalize(x_flat, dim=-1) * 0.5 # All norm 0.5

        # 2. Hierarchical Data (Points at 0.1, 0.5, 0.9)
        # 10 points at 0.1, 10 at 0.5, 10 at 0.9
        r1 = torch.randn(1, 10, self.d_model); r1 = F.normalize(r1, dim=-1) * 0.1
        r2 = torch.randn(1, 10, self.d_model); r2 = F.normalize(r2, dim=-1) * 0.5
        r3 = torch.randn(1, 10, self.d_model); r3 = F.normalize(r3, dim=-1) * 0.9
        x_hier = torch.cat([r1, r2, r3], dim=1)

        score_flat = self.adapter._estimate_hierarchy(x_flat)
        score_hier = self.adapter._estimate_hierarchy(x_hier)

        print(f"Flat Score: {score_flat.mean().item()}")
        print(f"Hier Score: {score_hier.mean().item()}")

        # Hierarchy (high variance in norms) should have higher score
        self.assertGreater(score_hier.mean().item(), score_flat.mean().item())

    def test_c_range(self):
        x = torch.randn(1, 10, self.d_model)
        c = self.adapter(x)
        self.assertGreaterEqual(c.item(), self.adapter.c_min)
        self.assertLessEqual(c.item(), self.adapter.c_max)

if __name__ == '__main__':
    unittest.main()
