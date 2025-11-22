import torch
import torch.nn as nn
import unittest
from src.models.bk_core import get_tridiagonal_inverse_diagonal
from src.models.resnet_bk import MoEResNetBKLayer
from src.models.koopman_layer import KoopmanResNetBKLayer
from src.models.physics_informed_layer import PhysicsInformedLanguageModel
from src.training.physics_informed_trainer import PhysicsInformedTrainer
import tempfile
import os

class TestModifications(unittest.TestCase):
    def test_bk_core_stability(self):
        print("\nTesting BK-Core Stability with Large N...")
        N = 4096
        device = 'cpu'
        a = torch.randn(N, device=device)
        b = torch.randn(N-1, device=device) * 0.5
        c = torch.randn(N-1, device=device) * 0.5
        z = torch.complex(torch.tensor(0.1), torch.tensor(0.1)).to(device)

        # This used to overflow. Now it should work.
        diag = get_tridiagonal_inverse_diagonal(a, b, c, z)

        self.assertTrue(torch.isfinite(diag).all())
        print(f"BK-Core result finite: {diag.isfinite().all()}")

    def test_resnet_bk_scale(self):
        print("\nTesting ResNet-BK Scale Shape...")
        d_model = 64
        layer = MoEResNetBKLayer(d_model, n_seq=128)
        self.assertEqual(layer.bk_scale.shape, (d_model,))
        print(f"bk_scale shape: {layer.bk_scale.shape}")

    def test_koopman_update(self):
        print("\nTesting Koopman Update Interval...")
        layer = KoopmanResNetBKLayer(d_model=32, n_seq=10, update_interval=2)

        x_curr = torch.randn(2, 10, 32)
        x_next = torch.randn(2, 10, 32)

        # Step 1: Counter=1. No update (1 % 2 != 0)
        layer.update_koopman_operator(x_curr, x_next)
        # We can't easily check internal state without mocking, but we check it doesn't crash
        self.assertEqual(layer.update_counter, 1)

        # Step 2: Counter=2. Update triggered.
        layer.update_koopman_operator(x_curr, x_next)
        self.assertEqual(layer.update_counter, 2)
        print("Koopman update ran without error.")

    def test_trainer_energy_conservation(self):
        print("\nTesting Trainer Energy Conservation Logic...")
        vocab_size = 100
        d_model = 16
        n_seq = 10
        model = PhysicsInformedLanguageModel(vocab_size, d_model=d_model, n_seq=n_seq)
        trainer = PhysicsInformedTrainer(model, torch.optim.Adam(model.parameters()), nn.CrossEntropyLoss(), physics_start_epoch=0)
        trainer.physics_enabled = True

        x_batch = torch.randint(0, vocab_size, (4, n_seq))
        y_batch = torch.randint(0, vocab_size, (4, n_seq))

        # Run a step
        metrics = trainer.train_step(x_batch, y_batch)
        print(f"Trainer metrics: {metrics}")
        self.assertIn('loss_energy', metrics)

if __name__ == '__main__':
    unittest.main()
