"""
Unit Tests for Phase 3 Model Factory

Requirements:
    - Requirement 7.8: create_phase3_model Function
    - Requirement 7.9: convert_phase2_to_phase3 Function
"""

import torch
import torch.nn as nn
import pytest
from src.models.phase3.factory import create_phase3_model, convert_phase2_to_phase3, get_preset_config
from src.models.phase3.config import Phase3Config
from src.models.phase3.integrated_model import Phase3IntegratedModel

class TestModelFactory:
    def test_create_phase3_model_from_config_obj(self):
        config = Phase3Config(vocab_size=1000, d_model=64, n_layers=2)
        model = create_phase3_model(config)
        assert isinstance(model, Phase3IntegratedModel)
        assert model.config.d_model == 64

    def test_create_phase3_model_from_dict(self):
        config_dict = {
            'vocab_size': 1000,
            'd_model': 64,
            'n_layers': 2
        }
        model = create_phase3_model(config_dict)
        assert isinstance(model, Phase3IntegratedModel)
        assert model.config.d_model == 64

    def test_get_preset_config(self):
        config = get_preset_config('small')
        assert isinstance(config, Phase3Config)
        assert config.d_model == 256

        with pytest.raises(ValueError):
            get_preset_config('invalid')

    def test_convert_phase2_to_phase3(self):
        # Mock Phase 2 Model
        class Phase2Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'vocab_size': 100, 'd_model': 32, 'n_layers': 2, 'max_seq_len': 64})()
                self.embedding = nn.Embedding(100, 32)
                self.head = nn.Linear(32, 100)

        p2_model = Phase2Model()
        # Set some weights to check copy
        with torch.no_grad():
            p2_model.embedding.weight.fill_(1.0)
            p2_model.head.weight.fill_(2.0)

        p3_model = convert_phase2_to_phase3(p2_model)

        assert isinstance(p3_model, Phase3IntegratedModel)
        assert p3_model.config.d_model == 32

        # Check embedding weight copy (Real part should match, cast to float16)
        assert torch.allclose(
            p3_model.embedding.token_embedding_real.weight.float(),
            p2_model.embedding.weight,
            atol=1e-3
        )

        # Check head weight copy
        assert torch.allclose(
            p3_model.dialectic.generator_head.weight,
            p2_model.head.weight
        )
