import torch
import unittest
from src.models.phase8.entailment import EntailmentCone

class TestEntailmentCone(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.cone = EntailmentCone(self.d_model)

    def test_entailment_hierarchy(self):
        """Test that vectors further from origin can be entailed by vectors closer."""
        # u is closer to origin (e.g. "animal")
        u = torch.randn(1, self.d_model) * 0.1

        # v is further in same direction (e.g. "dog")
        # u/|u| = v/|v| so angle is 0
        v = u * 5.0

        penalty, _ = self.cone(u, v)
        # Should be 0 penalty as angle is 0 and |u| < |v|
        self.assertTrue(torch.all(penalty < 1e-4), f"Expected low penalty, got {penalty}")

    def test_angle_violation(self):
        """Test that wide angles cause penalty."""
        u = torch.zeros(1, self.d_model)
        u[0, 0] = 0.5 # Point on x-axis

        v = torch.zeros(1, self.d_model)
        v[0, 1] = 0.8 # Point on y-axis (90 deg away)

        penalty, aperture = self.cone(u, v)

        # 90 degrees (1.57 rad) is likely larger than default aperture
        self.assertTrue(torch.all(penalty > 0), "Expected penalty for 90 degree separation")

    def test_order_violation(self):
        """Test that if |u| > |v|, we get penalty (assuming generic -> specific flow)."""
        # u is far
        u = torch.randn(1, self.d_model) * 0.8

        # v is near (same dir)
        v = u * 0.1

        penalty, _ = self.cone(u, v)
        self.assertTrue(torch.all(penalty > 0), "Expected penalty for reverse hierarchy")

if __name__ == '__main__':
    unittest.main()
