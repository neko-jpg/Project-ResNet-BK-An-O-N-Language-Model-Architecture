
import unittest
import sys
import os
from unittest.mock import MagicMock

# Add scripts to path to import AutoTuner
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.configure_recipe import AutoTuner

class MockCalibrator:
    def __init__(self):
        self.vram_total = 24000  # 24GB VRAM
        self.memory_coeffs = {'base': 1000, 'per_complex': 0.0001} # Dummy coeffs

    def predict(self, batch, seq_len, d_model, layers):
        # Simple linear model for testing
        # complexity = B * S * D * L
        complexity = batch * seq_len * d_model * layers
        mem = self.memory_coeffs['base'] + (complexity * self.memory_coeffs['per_complex'])
        return mem, 0.1

class TestAutoTuner(unittest.TestCase):
    def setUp(self):
        self.cal = MockCalibrator()
        self.tuner = AutoTuner(self.cal, goal="2") # Benchmark mode
        self.base_config = {
            'd_model': 512,
            'n_layers': 10,
            'batch_size': 4,
            'n_seq': 1024
        }

    def _predict_config(self, config):
        """Helper to call predict with config dict keys mapping to args."""
        return self.cal.predict(
            batch=config['batch_size'],
            seq_len=config['n_seq'],
            d_model=config['d_model'],
            layers=config['n_layers']
        )

    def test_reduction_priority(self):
        """Test that d_model is reduced first when over limit."""
        # Set target very low to force reduction
        target_ratio = 0.1 # 2400 MB target

        # Calculate initial memory to ensure we are over
        initial_mem, _ = self._predict_config(self.base_config)
        self.assertTrue(initial_mem > 24000 * 0.1)

        new_config, status = self.tuner.tune(self.base_config, {}, target_ratio)

        # Check that d_model was reduced
        self.assertLess(new_config['d_model'], self.base_config['d_model'])

        print(f"\nReduction Test: {self.base_config} -> {new_config} (Status: {status})")

    def test_locked_parameter(self):
        """Test that locked d_model forces n_layers to reduce."""
        target_ratio = 0.1
        locked = {'d_model': True}

        new_config, status = self.tuner.tune(self.base_config, locked, target_ratio)

        # d_model should NOT change
        self.assertEqual(new_config['d_model'], self.base_config['d_model'])
        # n_layers SHOULD change (reduce)
        self.assertLess(new_config['n_layers'], self.base_config['n_layers'])

        print(f"\nLocked Test: {self.base_config} -> {new_config}")

    def test_expansion(self):
        """Test that parameters expand when under limit."""
        target_ratio = 0.9 # ~21GB
        small_config = {
            'd_model': 128,
            'n_layers': 2,
            'batch_size': 1,
            'n_seq': 128
        }

        new_config, status = self.tuner.tune(small_config, {}, target_ratio)

        # Should have expanded d_model first
        self.assertGreater(new_config['d_model'], small_config['d_model'])
        print(f"\nExpansion Test: {small_config} -> {new_config}")

    def test_soft_caps(self):
        """Test that expansion respects max limits."""
        self.tuner.limits['d_model']['max'] = 600 # Artificially low max
        target_ratio = 0.99 # Huge target

        new_config, status = self.tuner.tune(self.base_config, {}, target_ratio)

        # Check result
        self.assertTrue(new_config['d_model'] <= 600 + 64, f"Value {new_config['d_model']} exceeded limit + step")

        print(f"\nSoft Cap Test: Max d_model=600. Result: {new_config['d_model']}")

    def test_auto_correction_empty_enter(self):
        """Simulate user hitting Enter (auto-tune) on an invalid config."""
        # Config over 100%
        huge_config = {
            'd_model': 4096,
            'n_layers': 100,
            'batch_size': 10,
            'n_seq': 4096
        }
        target_ratio = 0.9

        new_config, status = self.tuner.tune(huge_config, {}, target_ratio)
        mem, _ = self._predict_config(new_config)

        # Should be close to target (within margin of error due to step sizes)
        # Since we use large steps, it might be slightly under or slightly over if we accept overshoots?
        # Logic says: if est_mem > target, reduce.
        # So it should end up <= target.
        self.assertLessEqual(mem, self.cal.vram_total * target_ratio)
        print(f"\nAuto-Correct Test: Huge -> {new_config} (Mem: {mem:.0f}/{self.cal.vram_total * target_ratio:.0f})")

if __name__ == '__main__':
    unittest.main()
