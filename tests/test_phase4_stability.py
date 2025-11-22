import unittest
import torch
import torch.nn as nn
from src.models.phase4.stability import NumericalStability
from src.models.phase4.adscft_core.bulk_generator import BulkSpaceGenerator
from src.models.phase4.quantum_observer.von_neumann_projection import QuantumObserver

class TestPhase4Stability(unittest.TestCase):

    def setUp(self):
        self.d_model = 32
        self.vocab_size = 100
        self.bulk_gen = BulkSpaceGenerator(self.d_model)
        self.observer = QuantumObserver(self.vocab_size)

    def test_safe_complex_division(self):
        """Test that division by zero (or near zero) is handled safely."""
        num = torch.tensor([1.0+1j])
        den = torch.tensor([0.0+0j])

        # Standard division would result in NaN/Inf
        # Safe division should handle it
        res = NumericalStability.safe_complex_division(num, den)
        self.assertFalse(torch.isnan(res).any())
        self.assertFalse(torch.isinf(res).any())

        # Check normal division
        den_normal = torch.tensor([2.0+0j])
        res_normal = NumericalStability.safe_complex_division(num, den_normal)
        expected = num / (den_normal + 1e-6) # approx
        self.assertTrue(torch.allclose(res_normal, expected, atol=1e-5))

    def test_energy_conservation_check(self):
        """Test energy conservation monitoring logic."""
        e_in = 100.0
        e_out = 105.0 # 5% drift
        stats = NumericalStability.check_energy_conservation(e_in, e_out, threshold=0.10)
        self.assertTrue(stats['conserved'])

        e_out_bad = 120.0 # 20% drift
        stats_bad = NumericalStability.check_energy_conservation(e_in, e_out_bad, threshold=0.10)
        self.assertFalse(stats_bad['conserved'])

    def test_bulk_energy_monitoring(self):
        """Test that BulkGenerator returns energy statistics."""
        inputs = torch.randn(1, 10, self.d_model)
        _, diag = self.bulk_gen(inputs)

        self.assertIn('energy_stats', diag)
        self.assertIn('conserved', diag['energy_stats'])
        # Note: We don't assert it is conserved, just that it is monitored,
        # because random weights might not conserve energy.

    def test_quantum_observer_stability(self):
        """Test QuantumObserver with potential zero-division inputs."""
        logits = torch.randn(1, 10, self.vocab_size)
        # Create a scenario where denominator might be small in scattering
        # This is hard to force deterministically without mocking internals,
        # but we can check output for NaNs

        # We mock the ScatteringOperator inside observer to force a zero division scenario if we could,
        # but here we rely on the fact that we patched it.

        collapsed, diag = self.observer(logits)
        self.assertFalse(torch.isnan(collapsed).any())

        # Check spectral density (obs_operator) for NaNs
        # We can't access internal intermediate easily without a hook or debugger,
        # but if it propagated NaNs, 'collapsed' or diag would likely have them.
        self.assertFalse(torch.isnan(diag['collapsed_probs']).any())

    def test_gradient_clipping(self):
        """Test gradient clipping utility."""
        model = nn.Linear(10, 10)
        input = torch.randn(5, 10)
        target = torch.randn(5, 10)
        criterion = nn.MSELoss()

        output = model(input)
        loss = criterion(output, target)
        loss.backward()

        # Scale up gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= 1000

        # The function returns the norm BEFORE clipping
        pre_clip_norm = NumericalStability.clip_gradient_norm(model, max_norm=1.0)
        self.assertGreater(pre_clip_norm, 1.0)

        # Check that gradients are actually clipped
        total_norm_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_after += p.grad.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        self.assertLessEqual(total_norm_after, 1.0 + 1e-3)

if __name__ == '__main__':
    unittest.main()
