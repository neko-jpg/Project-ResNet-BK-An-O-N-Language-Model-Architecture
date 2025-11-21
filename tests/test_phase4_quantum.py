import pytest
import torch
import os
from src.models.phase4.quantum_observer.von_neumann_projection import QuantumObserver
from src.models.phase4.quantum_observer.visualization import visualize_wave_function_collapse

class TestQuantumObserver:
    def test_forward_collapse(self):
        vocab = 100
        observer = QuantumObserver(vocab_size=vocab, n_candidates=3)

        batch = 2
        seq = 5
        logits = torch.randn(batch, seq, vocab)

        collapsed, diag = observer(logits)

        assert collapsed.shape == (batch, seq)
        assert 'entropy_reduction' in diag
        assert diag['collapsed_probs'].shape == (batch, seq, 3)

    def test_entropy_reduction(self):
        vocab = 100
        observer = QuantumObserver(vocab_size=vocab)
        logits = torch.randn(1, 1, vocab) * 10

        collapsed, diag = observer(logits)
        reduction = diag['entropy_reduction']
        # Check shape
        assert reduction.shape == (1, 1)

    def test_visualization(self):
        save_path = "test_collapse.gif"
        candidates = ["A", "B", "C"]
        probs = torch.tensor([0.6, 0.3, 0.1])

        try:
            visualize_wave_function_collapse(candidates, probs, "A", save_path)
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
