"""
Tests for Phase 2 Model Factory

Requirements: 6.1, 6.2
"""

import pytest
import torch
import torch.nn as nn

from src.models.phase2 import (
    Phase2Config,
    create_phase2_model,
    convert_phase1_to_phase2,
    get_phase2_preset,
    Phase2IntegratedModel,
)


class TestPhase2Config:
    """Phase2Configのテスト"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = Phase2Config()
        
        assert config.vocab_size == 50257
        assert config.d_model == 512
        assert config.n_layers == 6
        assert config.base_decay == 0.01
        assert config.hebbian_eta == 0.1
        assert config.use_triton_bk is True
        assert config.resonance_enabled is True
    
    def test_config_validation_valid(self):
        """正常な設定の検証"""
        config = Phase2Config(
            vocab_size=30000,
            d_model=768,
            n_layers=8,
        )
        
        # 例外が発生しないことを確認
        config.validate()
    
    def test_config_validation_invalid_base_decay(self):
        """無効なbase_decayの検証"""
        config = Phase2Config(base_decay=-0.01)
        
        with pytest.raises(ValueError, match="base_decay must be > 0"):
            config.validate()
    
    def test_config_validation_invalid_hebbian_eta(self):
        """無効なhebbian_etaの検証"""
        config = Phase2Config(hebbian_eta=0.0)
        
        with pytest.raises(ValueError, match="hebbian_eta must be > 0"):
            config.validate()
    
    def test_config_validation_invalid_d_model(self):
        """無効なd_modelの検証"""
        config = Phase2Config(d_model=0)
        
        with pytest.raises(ValueError, match="d_model must be > 0"):
            config.validate()
    
    def test_config_to_dict(self):
        """辞書変換のテスト"""
        config = Phase2Config(
            vocab_size=30000,
            d_model=768,
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['vocab_size'] == 30000
        assert config_dict['d_model'] == 768
    
    def test_config_from_dict(self):
        """辞書からの生成テスト"""
        config_dict = {
            'vocab_size': 30000,
            'd_model': 768,
            'n_layers': 8,
        }
        
        config = Phase2Config.from_dict(config_dict)
        
        assert config.vocab_size == 30000
        assert config.d_model == 768
        assert config.n_layers == 8


class TestPresets:
    """プリセット設定のテスト"""
    
    def test_get_small_preset(self):
        """smallプリセットのテスト"""
        config = get_phase2_preset("small")
        
        assert config.d_model == 256
        assert config.n_layers == 4
        assert config.n_seq == 512
        assert config.target_vram_gb == 4.0
    
    def test_get_base_preset(self):
        """baseプリセットのテスト"""
        config = get_phase2_preset("base")
        
        assert config.d_model == 512
        assert config.n_layers == 6
        assert config.n_seq == 1024
        assert config.target_vram_gb == 8.0
    
    def test_get_large_preset(self):
        """largeプリセットのテスト"""
        config = get_phase2_preset("large")
        
        assert config.d_model == 1024
        assert config.n_layers == 12
        assert config.n_seq == 2048
        assert config.target_vram_gb == 16.0
    
    def test_get_invalid_preset(self):
        """無効なプリセット名のテスト"""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_phase2_preset("invalid_preset")


class TestCreatePhase2Model:
    """create_phase2_model関数のテスト"""
    
    def test_create_with_default_config(self):
        """デフォルト設定でのモデル作成"""
        model = create_phase2_model(device=torch.device("cpu"))
        
        assert isinstance(model, Phase2IntegratedModel)
        assert model.d_model == 512
        assert model.n_layers == 6
    
    def test_create_with_preset(self):
        """プリセットでのモデル作成"""
        model = create_phase2_model(preset="small", device=torch.device("cpu"))
        
        assert isinstance(model, Phase2IntegratedModel)
        assert model.d_model == 256
        assert model.n_layers == 4
    
    def test_create_with_custom_config(self):
        """カスタム設定でのモデル作成"""
        config = Phase2Config(
            vocab_size=30000,
            d_model=768,
            n_layers=8,
        )
        
        model = create_phase2_model(config=config, device=torch.device("cpu"))
        
        assert isinstance(model, Phase2IntegratedModel)
        assert model.d_model == 768
        assert model.n_layers == 8
    
    def test_create_with_direct_params(self):
        """パラメータ直接指定でのモデル作成"""
        model = create_phase2_model(
            vocab_size=25000,
            d_model=512,
            n_layers=4,
            device=torch.device("cpu")
        )
        
        assert isinstance(model, Phase2IntegratedModel)
        assert model.d_model == 512
        assert model.n_layers == 4
    
    def test_model_has_all_components(self):
        """モデルが全コンポーネントを持つことを確認"""
        model = create_phase2_model(preset="small", device=torch.device("cpu"))
        
        # Token embedding
        assert hasattr(model, 'token_embedding')
        
        # Position embedding (Zeta)
        assert hasattr(model, 'position_embedding')
        
        # Blocks
        assert hasattr(model, 'blocks')
        assert len(model.blocks) == 4  # small preset
        
        # Output layers
        assert hasattr(model, 'ln_f')
        assert hasattr(model, 'lm_head')
    
    def test_model_forward_pass(self):
        """Forward passのテスト"""
        model = create_phase2_model(preset="small", device=torch.device("cpu"))
        model.eval()
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model(input_ids)
        
        assert output.shape == (batch_size, seq_len, 50257)
        assert output.dtype == torch.float32
    
    def test_model_forward_with_diagnostics(self):
        """診断情報付きforward passのテスト"""
        model = create_phase2_model(preset="small", device=torch.device("cpu"))
        model.eval()
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        with torch.no_grad():
            output, diagnostics = model(input_ids, return_diagnostics=True)
        
        assert output.shape == (batch_size, seq_len, 50257)
        assert isinstance(diagnostics, dict)
        assert 'layer_outputs' in diagnostics
        assert 'gamma_values' in diagnostics


class TestConvertPhase1ToPhase2:
    """convert_phase1_to_phase2関数のテスト"""
    
    def test_convert_simple_model(self):
        """シンプルなモデルの変換"""
        # ダミーPhase 1モデル
        class DummyPhase1Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50257, 512)
                self.linear = nn.Linear(512, 512)
                self.lm_head = nn.Linear(512, 50257)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.linear(x)
                return self.lm_head(x)
        
        phase1_model = DummyPhase1Model()
        
        # Phase 2に変換
        phase2_config = Phase2Config(
            vocab_size=50257,
            d_model=512,
            n_layers=4,
        )
        
        phase2_model = convert_phase1_to_phase2(
            phase1_model,
            phase2_config=phase2_config,
            copy_compatible_weights=True,
        )
        
        assert isinstance(phase2_model, Phase2IntegratedModel)
        assert phase2_model.d_model == 512
        assert phase2_model.n_layers == 4
    
    def test_convert_with_weight_copying(self):
        """重みコピー付き変換のテスト"""
        # ダミーPhase 1モデル
        class DummyPhase1Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50257, 512)
                self.lm_head = nn.Linear(512, 50257)
            
            def forward(self, x):
                return self.lm_head(self.embedding(x))
        
        phase1_model = DummyPhase1Model()
        
        # 元の重みを保存
        original_emb_weight = phase1_model.embedding.weight.data.clone()
        
        # Phase 2に変換
        phase2_config = Phase2Config(
            vocab_size=50257,
            d_model=512,
            n_layers=2,
        )
        
        phase2_model = convert_phase1_to_phase2(
            phase1_model,
            phase2_config=phase2_config,
            copy_compatible_weights=True,
        )
        
        # 重みがコピーされたことを確認
        assert torch.allclose(
            phase2_model.token_embedding.weight.data,
            original_emb_weight,
            atol=1e-6
        )
    
    def test_convert_with_frozen_weights(self):
        """重み凍結付き変換のテスト"""
        # ダミーPhase 1モデル
        class DummyPhase1Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50257, 512)
            
            def forward(self, x):
                return self.embedding(x)
        
        phase1_model = DummyPhase1Model()
        
        # Phase 2に変換（重みを凍結）
        phase2_config = Phase2Config(
            vocab_size=50257,
            d_model=512,
            n_layers=2,
        )
        
        phase2_model = convert_phase1_to_phase2(
            phase1_model,
            phase2_config=phase2_config,
            copy_compatible_weights=True,
            freeze_phase1_weights=True,
        )
        
        # Token embeddingが凍結されていることを確認
        assert phase2_model.token_embedding.weight.requires_grad is False
    
    def test_convert_without_weight_copying(self):
        """重みコピーなし変換のテスト"""
        # ダミーPhase 1モデル
        class DummyPhase1Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(50257, 512)
            
            def forward(self, x):
                return self.embedding(x)
        
        phase1_model = DummyPhase1Model()
        
        # Phase 2に変換（重みコピーなし）
        phase2_config = Phase2Config(
            vocab_size=50257,
            d_model=512,
            n_layers=2,
        )
        
        phase2_model = convert_phase1_to_phase2(
            phase1_model,
            phase2_config=phase2_config,
            copy_compatible_weights=False,
        )
        
        # モデルが作成されることを確認
        assert isinstance(phase2_model, Phase2IntegratedModel)


class TestIntegration:
    """統合テスト"""
    
    def test_create_and_train_step(self):
        """モデル作成と学習ステップのテスト"""
        model = create_phase2_model(preset="small", device=torch.device("cpu"))
        model.train()
        
        # ダミーデータ
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        target_ids = torch.randint(0, 50257, (batch_size, seq_len))
        
        # Forward pass
        output = model(input_ids)
        
        # Loss計算
        loss = nn.functional.cross_entropy(
            output.view(-1, 50257),
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # 勾配が計算されたことを確認
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_preset_models_parameter_count(self):
        """プリセットモデルのパラメータ数確認"""
        presets = ["small", "base", "large"]
        
        for preset_name in presets:
            model = create_phase2_model(preset=preset_name, device=torch.device("cpu"))
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # パラメータ数が正の値であることを確認
            assert total_params > 0
            assert trainable_params > 0
            assert trainable_params <= total_params
            
            print(f"{preset_name}: {total_params:,} total, {trainable_params:,} trainable")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
