
import pytest
import torch
import torch.nn as nn
from src.models.phase1.htt_embedding import HolographicTTEmbedding
from src.models.phase8.quantized_htt import QuantizedHolographicTTEmbedding
from src.models.phase8.quantization import QuantizationConfig

class TestQuantizedHolographicTTEmbedding:

    @pytest.fixture
    def vocab_size(self):
        return 1000

    @pytest.fixture
    def d_model(self):
        return 64

    @pytest.fixture
    def rank(self):
        return 8

    def test_initialization(self, vocab_size, d_model, rank):
        model = QuantizedHolographicTTEmbedding(vocab_size, d_model, rank=rank)
        assert model.vocab_size == vocab_size
        assert model.d_model == d_model
        assert model.rank == rank
        assert model.core1_q.dtype == torch.int8
        assert model.core2_q.dtype == torch.int8

    def test_forward_shape(self, vocab_size, d_model, rank):
        model = QuantizedHolographicTTEmbedding(vocab_size, d_model, rank=rank)
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = model(input_ids)
        assert output.shape == (batch_size, seq_len, d_model)

    def test_from_htt_conversion(self, vocab_size, d_model, rank):
        # Train a small HTT model (randomly initialized)
        htt = HolographicTTEmbedding(vocab_size, d_model, rank=rank)

        # Convert
        qhtt = QuantizedHolographicTTEmbedding.from_htt(htt, bits=8)

        assert qhtt.is_quantized
        assert qhtt.core1_q.dtype == torch.int8

        # Check output similarity
        input_ids = torch.randint(0, vocab_size, (2, 5))
        with torch.no_grad():
            out_htt = htt(input_ids)
            out_qhtt = qhtt(input_ids)

        # Error should be small but not zero
        diff = (out_htt - out_qhtt).abs().mean()
        assert diff < 0.1 # Loose bound for random weights

    def test_gradients_phase_only(self, vocab_size, d_model, rank):
        """Verify that only phase_shift receives gradients."""
        model = QuantizedHolographicTTEmbedding(vocab_size, d_model, rank=rank, phase_encoding=True)
        input_ids = torch.randint(0, vocab_size, (2, 5))

        out = model(input_ids)
        loss = out.mean()
        loss.backward()

        # Phase shift should have grad
        assert model.phase_shift.grad is not None

        # Cores are buffers, no grad
        assert model.core1_q.grad is None
        assert model.core1_scale.grad is None

    def test_memory_stats(self, vocab_size, d_model, rank):
        model = QuantizedHolographicTTEmbedding(vocab_size, d_model, rank=rank, bits=8)
        stats = model.get_compression_stats()

        assert "qhtt_mb" in stats
        assert "compression_ratio" in stats
        assert stats["qhtt_mb"] < stats["standard_mb"]

    def test_custom_quantization_config(self):
        config = QuantizationConfig(bits=4, boundary_factor=1.5)
        model = QuantizedHolographicTTEmbedding(100, 16, rank=4, quantization_config=config)

        assert model.bits == 4
        assert model.quantizer.bits == 4
        # Check that per_channel is forced to False
        assert model.quantizer.per_channel is False
