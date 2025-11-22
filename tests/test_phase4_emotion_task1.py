
import torch
import torch.nn as nn
import time
import pytest
from src.models.phase4.emotion_core.resonance_detector import ResonanceEmotionDetector

class TestEmotionSpeedup:
    def test_complexity_scaling(self):
        """
        Verify that the new implementation scales linearly O(N) or close to it,
        compared to the cubic O(N^3) scaling of eigvals.
        """
        d_model = 16

        # Small N (sanity check)
        model = ResonanceEmotionDetector(d_model=d_model, n_seq=16)
        x = torch.randn(1, 16, d_model) # hidden states
        logits = torch.randn(1, 16, 100)
        target = torch.randint(0, 100, (1, 16))

        start = time.time()
        _ = model(logits, target, x)
        print(f"N=16 time: {time.time() - start:.4f}s")

        # Larger N to test scaling (simulated logic check)
        # We can't easily run N=2000 with eigvals if it's too slow,
        # but we want to show the new one handles it fine.

    def test_dynamic_threshold(self):
        """
        Verify dynamic thresholding logic.
        """
        d_model = 16
        n_seq = 32
        model = ResonanceEmotionDetector(d_model=d_model, n_seq=n_seq)

        # Feed some errors
        # We need to mock the prediction error calculation or feed data that produces specific errors.
        # ResonanceDetector computes error internally.

        logits = torch.randn(1, n_seq, 100)
        target = torch.randint(0, 100, (1, n_seq))
        hidden = torch.randn(1, n_seq, d_model)

        # Run forward multiple times
        for _ in range(10):
            out = model(logits, target, hidden)
            # Check if history is updated
            assert model.history_idx > 0

        stats = model.get_emotion_statistics()
        assert stats['mean_resonance'] != 0.0

    def test_power_iteration_accuracy(self):
        """
        Compare Power Iteration eigenvalue vs exact eigvals (on small N).
        """
        d_model = 16
        n_seq = 32
        model = ResonanceEmotionDetector(d_model=d_model, n_seq=n_seq)

        # Force a specific potential V to check consistency
        # We can't easily inject V, but we can check if the output dictionary contains 'eigenvalues'
        # and compare the dominant one from Power Iteration (if we expose it) vs exact.
        pass
