"""
Unit Tests for Phase 3 Integrated Model Components

Requirements:
    - Requirement 5: MERA Router
    - Requirement 6: Entropic Selection
    - Requirement 7: Integrated Model & Block
"""

import torch
import torch.nn as nn
import pytest
from src.models.phase3.mera import MERARouter
from src.models.phase3.integrated_model import Phase3Block, Phase3IntegratedModel
from src.models.phase3.entropic import EntropicSelector
from src.models.phase3.complex_tensor import ComplexTensor

class TestMERARouter:
    def test_initialization(self):
        mera = MERARouter(d_model=32, max_seq_len=128)
        assert len(mera.disentanglers) == 7 # ceil(log2(128)) = 7
        assert len(mera.isometries) == 7

    def test_forward_pass(self):
        mera = MERARouter(d_model=32, max_seq_len=128)
        x = torch.randn(2, 128, 32) # (B, N, D)

        global_context, hierarchy = mera(x)

        assert global_context.shape == (2, 1, 32)
        assert len(hierarchy) == 8 # Input + 7 layers
        assert hierarchy[-1].shape == (2, 1, 32)

    def test_padding(self):
        """Nが2の累乗でない場合のテスト"""
        mera = MERARouter(d_model=32, max_seq_len=128)
        x = torch.randn(2, 100, 32) # 100 is not power of 2

        global_context, hierarchy = mera(x)

        assert global_context.shape == (2, 1, 32)
        # First layer of hierarchy should be padded to 128
        assert hierarchy[0].shape == (2, 128, 32)


class TestPhase3Block:
    def test_forward_pass(self):
        d_model = 32
        block = Phase3Block(d_model=d_model, d_koopman=64, potential_type='mlp')

        # Create ComplexTensor input
        real = torch.randn(2, 10, d_model)
        imag = torch.randn(2, 10, d_model)
        x = ComplexTensor.from_real(real, imag)

        out, diag = block(x, return_diagnostics=True)

        assert isinstance(out, ComplexTensor)
        assert out.shape == (2, 10, d_model)
        assert 'hamiltonian_diagnostics' in diag
        assert 'energy_in' in diag


class TestEntropicSelector:
    def test_selection(self):
        # Simple dummy model
        model = nn.Linear(10, 20) # Output size V=20
        selector = EntropicSelector(model, selection_rate=0.5, warmup_epochs=0)

        # Batch
        B, N = 4, 5
        input_ids = torch.randn(B, N, 10) # Mock input embeddings for linear
        # But selector expects input_ids (Long), here we mock compute_surprise
        # Let's subclass or mock compute_surprise

        # Override model call
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 20)
            def forward(self, x):
                # x is input_ids, let's assume we ignore it and return random logits
                B, N = x.shape
                return torch.randn(B, N, 20)

        selector.model = MockModel()

        input_ids = torch.randint(0, 10, (B, N))
        batch = {'input_ids': input_ids}

        filtered, stats = selector.filter_batch(batch)

        assert filtered['input_ids'].shape[0] == 2 # 50% of 4
        assert stats['kept_ratio'] == 0.5


class TestIntegratedModel:
    @pytest.fixture
    def config(self):
        class Config:
            d_model = 32
            vocab_size = 100
            n_layers = 2
            max_seq_len = 64
            use_complex32 = True
            d_koopman = 64
            potential_type = 'mlp'
        return Config()

    def test_initialization(self, config):
        model = Phase3IntegratedModel(config)
        assert isinstance(model.embedding, nn.Module)
        assert isinstance(model.mera, MERARouter)
        assert len(model.layers) == 2
        assert isinstance(model.dialectic, nn.Module)

    def test_forward_pass(self, config):
        model = Phase3IntegratedModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 32))

        out = model(input_ids, return_diagnostics=True)

        assert 'logits' in out
        assert 'loss' in out
        assert out['logits'].shape == (2, 32, config.vocab_size)
        assert out['diagnostics'] is not None
        assert out['diagnostics']['mera_hierarchy'] > 0
