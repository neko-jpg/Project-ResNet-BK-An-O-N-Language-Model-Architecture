"""
Tests for Phase 4 Resonance Emotion Detector
"""

import pytest
import torch
import os
from src.models.phase4.emotion_core.resonance_detector import ResonanceEmotionDetector
from src.models.phase4.emotion_core.visualization import visualize_emotion_as_ripple

class TestResonanceEmotion:

    @pytest.fixture
    def detector(self):
        return ResonanceEmotionDetector(d_model=64, n_seq=32)

    def test_initialization(self, detector):
        assert detector.d_model == 64
        assert detector.bs_kernel is not None
        assert detector.emotion_history.shape == (1000, 2)

    def test_forward_pass(self, detector):
        batch_size = 2
        n_seq = 32
        vocab_size = 100

        prediction = torch.randn(batch_size, n_seq, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, n_seq))
        hidden_states = torch.randn(batch_size, n_seq, 64)

        output = detector(prediction, target, hidden_states)

        assert 'resonance_score' in output
        assert 'dissonance_score' in output
        assert 'interference_pattern' in output

        assert output['resonance_score'].shape == (batch_size,)
        assert output['dissonance_score'].shape == (batch_size,)
        assert output['interference_pattern'].shape == (batch_size, n_seq)

        # Check range
        assert (output['interference_pattern'] >= 0).all()

    def test_numerical_stability(self, detector):
        # Test with zero error (perfect prediction)
        batch_size = 1
        n_seq = 32
        vocab_size = 10

        # Perfect prediction (logits high on target)
        prediction = torch.zeros(batch_size, n_seq, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, n_seq))
        for b in range(batch_size):
            for t in range(n_seq):
                prediction[b, t, target[b, t]] = 100.0 # High confidence

        hidden_states = torch.zeros(batch_size, n_seq, 64)

        output = detector(prediction, target, hidden_states)

        # Error should be near 0
        # Patterns should be near 0
        assert output['interference_pattern'].mean() < 1e-3

    def test_history_update(self, detector):
        # Run training step
        detector.train()

        prediction = torch.randn(1, 32, 10)
        target = torch.randint(0, 10, (1, 32))
        hidden = torch.randn(1, 32, 64)

        detector(prediction, target, hidden)

        assert detector.history_idx == 1
        stats = detector.get_emotion_statistics()
        assert stats['mean_resonance'] != 0.0

    def test_visualization(self, tmp_path):
        pattern = torch.rand(64)
        res = 0.8
        dis = 0.2
        save_path = tmp_path / "ripple.png"

        visualize_emotion_as_ripple(pattern, res, dis, str(save_path))

        assert os.path.exists(save_path)
