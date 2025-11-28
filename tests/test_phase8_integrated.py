import torch
import unittest
from src.models.phase8.integrated_model import Phase8IntegratedModel

class TestPhase8Integrated(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.n_layers = 4
        self.model = Phase8IntegratedModel(self.d_model, self.n_layers)

    def test_full_forward_pass(self):
        """Test that data flows through all enabled components."""
        x = torch.randn(1, 10, self.d_model)

        # Run forward
        out, diagnostics = self.model(x)

        # Check output shape
        self.assertEqual(out.shape, x.shape)

        # Check diagnostics populated
        print("Diagnostics:", diagnostics)
        self.assertIn("curvature_value", diagnostics)
        self.assertIn("persistent_entropy", diagnostics)
        self.assertIn("avg_layers_executed", diagnostics)

        # Adaptive computation check
        # With random data, it might exit early or not, but layers should be > 0
        self.assertGreater(diagnostics['avg_layers_executed'], 0)

    def test_disable_components(self):
        """Test that model runs even with components disabled."""
        from src.models.phase8.config import Phase8Config
        config = Phase8Config(enable_adaptive_computation=False)
        model = Phase8IntegratedModel(self.d_model, self.n_layers, config=config)

        x = torch.randn(1, 10, self.d_model)
        out, diag = model(x)

        # Should run full layers
        self.assertEqual(diag['avg_layers_executed'], self.n_layers)

if __name__ == '__main__':
    unittest.main()
