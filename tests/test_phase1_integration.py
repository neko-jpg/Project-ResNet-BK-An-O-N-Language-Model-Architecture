"""
Phase 1 Integration Tests

Phase 1の全コンポーネント（AR-SSM, HTT, LNS, Stability Monitor）の
統合テストを実施します。

Requirements:
    - 6.5: 統合テスト
    - Task 9.4: Write integration tests

Author: Project MUSE Team
"""

import pytest
import torch
import torch.nn as nn

from src.models.phase1 import (
    Phase1Config,
    Phase1IntegratedModel,
    create_phase1_model,
    AdaptiveRankSemiseparableLayer,
    HolographicTTEmbedding,
    BKStabilityMonitor,
    convert_all_embeddings_to_htt,
    initialize_htt_from_embedding,
    get_conversion_summary,
    get_preset_8gb,
    get_preset_10gb,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """シンプルなテストモデル"""
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000, d_model=128, n_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(n_layers)
            ])
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = torch.relu(layer(x))
            return self.output(x)
    
    return SimpleModel()


@pytest.fixture
def phase1_config():
    """テスト用Phase1Config"""
    return Phase1Config(
        ar_ssm_enabled=True,
        ar_ssm_max_rank=16,
        ar_ssm_min_rank=4,
        htt_enabled=True,
        htt_rank=8,
        htt_compression_target=0.1,
        lns_enabled=False,  # Skip LNS for CPU testing
        stability_monitoring_enabled=True,
        use_gradient_checkpointing=False,  # Disable for testing
    )


@pytest.fixture
def test_input():
    """テスト用入力"""
    return torch.randint(0, 1000, (2, 10))


# ============================================================================
# Test: AR-SSM with BK-Core Integration
# ============================================================================

class TestARSSMWithBKCore:
    """AR-SSMとBK-Coreの統合テスト"""
    
    def test_ar_ssm_accepts_bk_features(self):
        """AR-SSMがBK-Core出力を受け入れることを確認"""
        # Create AR-SSM layer
        ar_ssm = AdaptiveRankSemiseparableLayer(d_model=64, max_rank=16)
        
        # Simulate BK-Core output
        bk_features = torch.randn(2, 128, 64)
        
        # Forward pass
        output, diagnostics = ar_ssm(bk_features)
        
        # Verify output shape
        assert output.shape == bk_features.shape
        assert torch.isfinite(output).all()
    
    def test_ar_ssm_integrate_with_bk_core_method(self):
        """AR-SSMのintegrate_with_bk_coreメソッドをテスト"""
        ar_ssm = AdaptiveRankSemiseparableLayer(d_model=64, max_rank=16)
        bk_features = torch.randn(2, 128, 64)
        
        # Use integration method
        output = ar_ssm.integrate_with_bk_core(bk_features)
        
        assert output.shape == bk_features.shape
        assert torch.isfinite(output).all()
    
    def test_ar_ssm_stability_checks(self):
        """AR-SSMの安定性チェックが動作することを確認"""
        stability_monitor = BKStabilityMonitor()
        ar_ssm = AdaptiveRankSemiseparableLayer(
            d_model=64,
            max_rank=16,
            stability_monitor=stability_monitor,
            enable_stability_checks=True,
        )
        
        x = torch.randn(2, 128, 64)
        output, diagnostics = ar_ssm(x)
        
        # Check that stability diagnostics are present
        assert 'condition_number' in diagnostics
        assert 'is_singular' in diagnostics


# ============================================================================
# Test: HTT with Full Language Model
# ============================================================================

class TestHTTWithLanguageModel:
    """HTT Embeddingと言語モデルの統合テスト"""
    
    def test_htt_in_simple_model(self, simple_model, phase1_config, test_input):
        """シンプルなモデルでHTTが動作することを確認"""
        # Convert embedding to HTT
        model, info = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=False,
        )
        
        # Verify conversion
        assert info['num_converted'] == 1
        assert info['overall_compression_ratio'] < 0.2
        
        # Forward pass
        output = model(test_input)
        
        # Verify output
        assert output.shape == (2, 10, 1000)
        assert torch.isfinite(output).all()
    
    def test_htt_gradient_flow(self, simple_model, phase1_config, test_input):
        """HTT Embeddingで勾配が流れることを確認"""
        # Convert embedding to HTT
        model, _ = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=False,
        )
        
        # Forward pass
        output = model(test_input)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Invalid gradient for {name}"
    
    def test_htt_initialization_from_weights(self, simple_model, phase1_config, test_input):
        """既存の重みからHTTを初期化できることを確認"""
        # Train simple model a bit
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        for _ in range(5):
            output = simple_model(test_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Convert with weight initialization
        model, _ = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=True,
            initialization_method="svd",
        )
        
        # Forward pass should work
        output = model(test_input)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: All Phase 1 Components Together
# ============================================================================

class TestFullPhase1Stack:
    """Phase 1全コンポーネントの統合テスト"""
    
    def test_create_phase1_model_from_scratch(self, phase1_config):
        """Phase 1モデルをゼロから作成できることを確認"""
        # This test requires actual model implementation
        # Skip if models are not available
        try:
            from src.models.resnet_bk import LanguageModel
        except ImportError:
            pytest.skip("LanguageModel not available")
        
        # Create Phase 1 model with n_seq matching input
        model = create_phase1_model(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_seq=10,  # Match input sequence length
            config=phase1_config,
            model_type="resnet_bk",
        )
        
        # Verify it's a Phase1IntegratedModel
        assert isinstance(model, Phase1IntegratedModel)
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 10))
        output, diagnostics = model(input_ids, return_diagnostics=True)
        
        # Verify output
        assert torch.isfinite(output).all()
        assert diagnostics is not None
    
    def test_phase1_integrated_model_wrapper(self, simple_model, phase1_config, test_input):
        """Phase1IntegratedModelラッパーが動作することを確認"""
        # Wrap simple model
        integrated_model = Phase1IntegratedModel(
            base_model=simple_model,
            config=phase1_config,
            replace_embeddings=True,
            replace_linears=False,
            enable_stability_monitoring=True,
            enable_gradient_monitoring=True,
        )
        
        # Forward pass
        output, diagnostics = integrated_model(test_input, return_diagnostics=True)
        
        # Verify output
        assert output.shape == (2, 10, 1000)
        assert torch.isfinite(output).all()
        
        # Verify diagnostics
        assert diagnostics.htt_compression_ratio > 0
        assert diagnostics.forward_time_ms > 0
    
    def test_phase1_rank_scheduling(self, simple_model, phase1_config):
        """ランクスケジューリングが動作することを確認"""
        integrated_model = Phase1IntegratedModel(
            base_model=simple_model,
            config=phase1_config,
            replace_embeddings=False,  # Skip for this test
        )
        
        # Initial rank
        initial_rank = integrated_model.training_state.current_max_rank
        
        # Update rank schedule
        for _ in range(10):
            integrated_model.update_rank_schedule()
        
        # Rank should have increased
        assert integrated_model.training_state.current_max_rank >= initial_rank
    
    def test_phase1_checkpointing(self, simple_model, phase1_config):
        """Gradient checkpointingが動作することを確認"""
        integrated_model = Phase1IntegratedModel(
            base_model=simple_model,
            config=phase1_config,
            replace_embeddings=False,
        )
        
        # Enable checkpointing
        integrated_model.enable_checkpointing()
        
        # Disable checkpointing
        integrated_model.disable_checkpointing()
        
        # Should not raise errors


# ============================================================================
# Test: Model Conversion from Baseline
# ============================================================================

class TestModelConversion:
    """ベースラインモデルからの変換テスト"""
    
    def test_conversion_summary(self, simple_model, phase1_config):
        """変換サマリーが正しく生成されることを確認"""
        # Before conversion
        summary_before = get_conversion_summary(simple_model)
        assert summary_before['num_standard_embeddings'] == 1
        assert summary_before['num_htt_embeddings'] == 0
        
        # Convert
        model, _ = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=False,
        )
        
        # After conversion
        summary_after = get_conversion_summary(model)
        assert summary_after['num_standard_embeddings'] == 0
        assert summary_after['num_htt_embeddings'] == 1
        assert summary_after['embedding_conversion_percentage'] == 100.0
    
    def test_conversion_preserves_functionality(self, simple_model, phase1_config, test_input):
        """変換後もモデルが動作することを確認"""
        # Original output
        simple_model.eval()
        with torch.no_grad():
            original_output = simple_model(test_input)
        
        # Convert
        model, _ = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=False,
        )
        
        # Converted output
        model.eval()
        with torch.no_grad():
            converted_output = model(test_input)
        
        # Shapes should match
        assert converted_output.shape == original_output.shape
        assert torch.isfinite(converted_output).all()
    
    def test_conversion_with_initialization(self, simple_model, phase1_config, test_input):
        """重み初期化付き変換が動作することを確認"""
        # Train a bit
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        for _ in range(5):
            output = simple_model(test_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Convert with initialization
        model, info = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=True,
            initialization_method="svd",
        )
        
        # Should have converted
        assert info['num_converted'] == 1
        
        # Should work
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        assert torch.isfinite(output).all()


# ============================================================================
# Test: Configuration Presets
# ============================================================================

class TestConfigurationPresets:
    """設定プリセットのテスト"""
    
    def test_preset_8gb(self):
        """8GBプリセットが正しく設定されることを確認"""
        config = get_preset_8gb()
        
        assert config.target_vram_gb == 8.0
        assert config.ar_ssm_max_rank == 16
        assert config.htt_rank == 8
        assert config.htt_compression_target == 0.05
        assert config.use_gradient_checkpointing is True
    
    def test_preset_10gb(self):
        """10GBプリセットが正しく設定されることを確認"""
        config = get_preset_10gb()
        
        assert config.target_vram_gb == 10.0
        assert config.ar_ssm_max_rank == 32
        assert config.htt_rank == 16
        assert config.htt_compression_target == 0.1
        assert config.use_gradient_checkpointing is True
    
    def test_presets_validate(self):
        """すべてのプリセットがvalidateを通過することを確認"""
        from src.models.phase1.presets import PRESET_REGISTRY
        
        for name, config in PRESET_REGISTRY.items():
            try:
                config.validate()
            except Exception as e:
                pytest.fail(f"Preset '{name}' failed validation: {e}")


# ============================================================================
# Test: Memory and Performance
# ============================================================================

class TestMemoryAndPerformance:
    """メモリとパフォーマンスのテスト"""
    
    def test_phase1_reduces_parameters(self, simple_model, phase1_config):
        """Phase 1変換がパラメータ数を削減することを確認"""
        # Count original parameters
        original_params = sum(p.numel() for p in simple_model.parameters())
        
        # Convert
        model, info = convert_all_embeddings_to_htt(
            simple_model,
            phase1_config,
            initialize_from_weights=False,
        )
        
        # Count converted parameters
        converted_params = sum(p.numel() for p in model.parameters())
        
        # Should have fewer parameters
        assert converted_params < original_params
        
        # Verify compression info
        assert info['total_params_after'] < info['total_params_before']
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_phase1_fits_in_memory(self, phase1_config):
        """Phase 1モデルがメモリ制約内に収まることを確認"""
        try:
            from src.models.resnet_bk import LanguageModel
        except ImportError:
            pytest.skip("LanguageModel not available")
        
        # Create small model with n_seq matching input
        model = create_phase1_model(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_seq=64,  # Match input sequence length
            config=phase1_config,
            model_type="resnet_bk",
        )
        
        model = model.cuda()
        
        # Forward pass
        input_ids = torch.randint(0, 1000, (2, 64), device='cuda')
        
        torch.cuda.reset_peak_memory_stats()
        output, diagnostics = model(input_ids, return_diagnostics=True)
        
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        # Should be reasonable
        assert peak_memory_mb < 1000  # Less than 1GB for this small model
        assert diagnostics.peak_vram_mb > 0


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """エラーハンドリングのテスト"""
    
    def test_invalid_config_raises_error(self):
        """無効な設定がエラーを発生させることを確認"""
        with pytest.raises(ValueError):
            config = Phase1Config(
                ar_ssm_max_rank=4,
                ar_ssm_min_rank=16,  # Invalid: min > max
            )
            config.validate()
    
    def test_conversion_with_no_embeddings(self, phase1_config):
        """Embeddingがないモデルの変換が適切に処理されることを確認"""
        class NoEmbeddingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        model = NoEmbeddingModel()
        
        # Should not crash, just return 0 conversions
        converted_model, info = convert_all_embeddings_to_htt(
            model,
            phase1_config,
            initialize_from_weights=False,
        )
        
        assert info['num_converted'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
