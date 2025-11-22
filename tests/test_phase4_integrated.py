import unittest
import torch
import torch.nn as nn
from src.models.phase4.integrated_model import Phase4IntegratedModel
from src.models.phase3.config import Phase3Config
from src.models.phase3.integrated_model import Phase3IntegratedModel

class TestPhase4Integrated(unittest.TestCase):

    def setUp(self):
        # Minimal config for speed
        self.config = Phase3Config(
            vocab_size=100,
            d_model=32,
            n_layers=2,
            max_seq_len=64
        )
        self.phase3_model = Phase3IntegratedModel(self.config)
        self.model = Phase4IntegratedModel(
            self.phase3_model,
            enable_emotion=True,
            enable_dream=True,
            enable_holographic=True,
            enable_quantum=True,
            enable_topological=True,
            enable_ethics=True
        )

    def test_forward_pass(self):
        """Test standard forward pass with diagnostics."""
        input_ids = torch.randint(0, 100, (1, 16))

        # Standard forward
        output = self.model(input_ids)
        self.assertIn('logits', output)
        self.assertEqual(output['logits'].shape, (1, 16, 100))

        # With diagnostics
        output_diag = self.model(input_ids, return_diagnostics=True)
        self.assertIn('diagnostics', output_diag)
        diag = output_diag['diagnostics']

        # Check components
        if self.model.enable_emotion:
            self.assertIn('emotion', diag)
        if self.model.enable_holographic:
            self.assertIn('bulk', diag)
        if self.model.enable_quantum:
            self.assertIn('quantum', diag)

    def test_backward_pass(self):
        """Test gradient flow."""
        input_ids = torch.randint(0, 100, (1, 16))
        labels = torch.randint(0, 100, (1, 16))

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        output = self.model(input_ids, labels=labels)
        loss = output['loss']

        self.assertIsNotNone(loss)
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad)

    def test_idle_mode(self):
        """Test entering and exiting idle mode."""
        # Just check it doesn't crash immediately
        self.model.enter_idle_mode(interval=0.1)
        self.model.exit_idle_mode()

    def test_phase3_compatibility(self):
        """Test strict Phase 3 mode."""
        model_p3 = Phase4IntegratedModel(
            self.phase3_model,
            enable_emotion=False,
            enable_dream=False,
            enable_holographic=False,
            enable_quantum=False,
            enable_topological=False,
            enable_ethics=False
        )
        self.assertTrue(model_p3.is_phase3_only)

        input_ids = torch.randint(0, 100, (1, 16))
        output = model_p3(input_ids, return_diagnostics=True)
        # Should rely on Phase 3 output format
        self.assertIn('logits', output)

if __name__ == '__main__':
    unittest.main()
