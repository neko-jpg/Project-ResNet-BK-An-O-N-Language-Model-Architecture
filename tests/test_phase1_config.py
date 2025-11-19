"""
Unit tests for Phase 1 Configuration System

このテストモジュールは、Phase1Config、Phase1Diagnostics、Phase1TrainingStateの
機能を検証します。

Requirements:
    - 4.2: テストファイルの作成
    - 6.1: 出力形状の正確性
    - 6.2: 数値安定性テスト
"""

import pytest
import warnings
from dataclasses import asdict

from src.models.phase1 import (
    Phase1Config,
    Phase1Diagnostics,
    Phase1TrainingState,
)


class TestPhase1Config:
    """Phase1Configのテストクラス"""
    
    def test_default_config_is_valid(self):
        """デフォルト設定が有効であることを確認"""
        config = Phase1Config()
        config.validate()  # Should not raise
    
    def test_config_validation_rank_constraints(self):
        """ランク制約の検証"""
        # max_rank < min_rank should fail
        config = Phase1Config(ar_ssm_max_rank=2, ar_ssm_min_rank=4)
        with pytest.raises(ValueError, match="must be >="):
            config.validate()
    
    def test_config_validation_compression_target(self):
        """圧縮率の検証"""
        # compression_target must be in (0, 1)
        config = Phase1Config(htt_compression_target=1.5)
        with pytest.raises(ValueError, match="must be in range"):
            config.validate()
        
        config = Phase1Config(htt_compression_target=0.0)
        with pytest.raises(ValueError, match="must be in range"):
            config.validate()
    
    def test_config_validation_positive_values(self):
        """正の値の検証"""
        # stability_threshold must be > 0
        config = Phase1Config(stability_threshold=-0.1)
        with pytest.raises(ValueError, match="must be > 0"):
            config.validate()
        
        # target_vram_gb must be > 0
        config = Phase1Config(target_vram_gb=0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            config.validate()
    
    def test_config_lns_warning(self):
        """LNS有効化時の警告"""
        config = Phase1Config(lns_enabled=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.validate()
            assert len(w) >= 1
            assert "experimental" in str(w[0].message).lower()
    
    def test_config_to_dict(self):
        """辞書変換のテスト"""
        config = Phase1Config(ar_ssm_max_rank=64)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['ar_ssm_max_rank'] == 64
        assert 'ar_ssm_enabled' in config_dict
    
    def test_config_from_dict(self):
        """辞書からの復元テスト"""
        original = Phase1Config(ar_ssm_max_rank=64, htt_rank=32)
        config_dict = original.to_dict()
        restored = Phase1Config.from_dict(config_dict)
        
        assert restored.ar_ssm_max_rank == 64
        assert restored.htt_rank == 32
    
    def test_config_for_hardware_8gb(self):
        """8GB VRAM用プリセット"""
        config = Phase1Config.for_hardware(vram_gb=8.0)
        config.validate()
        
        assert config.target_vram_gb == 8.0
        assert config.ar_ssm_max_rank <= 16  # Maximum compression
        assert config.htt_compression_target <= 0.1
        assert config.use_gradient_checkpointing is True
    
    def test_config_for_hardware_10gb(self):
        """10GB VRAM用プリセット"""
        config = Phase1Config.for_hardware(vram_gb=10.0)
        config.validate()
        
        assert config.target_vram_gb == 10.0
        assert config.ar_ssm_max_rank == 32  # Balanced
    
    def test_config_for_hardware_24gb(self):
        """24GB VRAM用プリセット"""
        config = Phase1Config.for_hardware(vram_gb=24.0)
        config.validate()
        
        assert config.target_vram_gb == 24.0
        assert config.ar_ssm_max_rank >= 64  # Quality-focused
        assert config.use_gradient_checkpointing is False
    
    def test_config_for_inference(self):
        """推論用プリセット"""
        config = Phase1Config.for_inference()
        config.validate()
        
        assert config.lns_enabled is True
        assert config.use_gradient_checkpointing is False
    
    def test_config_for_maximum_quality(self):
        """品質優先プリセット"""
        config = Phase1Config.for_maximum_quality()
        config.validate()
        
        assert config.ar_ssm_max_rank >= 128
        assert config.htt_compression_target >= 0.3
        assert config.target_ppl_degradation <= 0.01
    
    def test_config_for_maximum_efficiency(self):
        """効率優先プリセット"""
        config = Phase1Config.for_maximum_efficiency()
        config.validate()
        
        assert config.ar_ssm_max_rank <= 16
        assert config.htt_compression_target <= 0.05
        assert config.target_vram_gb <= 6.0


class TestPhase1Diagnostics:
    """Phase1Diagnosticsのテストクラス"""
    
    def test_default_diagnostics(self):
        """デフォルト診断情報の作成"""
        diag = Phase1Diagnostics()
        
        assert diag.ar_ssm_effective_rank == 0.0
        assert diag.bk_det_condition == 1.0
        assert len(diag.stability_warnings) == 0
    
    def test_diagnostics_to_dict(self):
        """辞書変換のテスト"""
        diag = Phase1Diagnostics(
            ar_ssm_effective_rank=16.5,
            peak_vram_mb=7500.0
        )
        diag_dict = diag.to_dict()
        
        assert isinstance(diag_dict, dict)
        assert diag_dict['ar_ssm_effective_rank'] == 16.5
        assert diag_dict['peak_vram_mb'] == 7500.0
    
    def test_diagnostics_is_healthy_all_good(self):
        """健全性チェック: すべて正常"""
        config = Phase1Config(
            stability_threshold=1e-6,
            schatten_s1_bound=100.0,
            schatten_s2_bound=50.0,
            target_vram_gb=8.0
        )
        
        diag = Phase1Diagnostics(
            bk_det_condition=1e-3,  # > threshold
            bk_schatten_s1=50.0,    # < bound
            bk_schatten_s2=25.0,    # < bound
            peak_vram_mb=7000.0,    # < target
            stability_warnings=[]
        )
        
        assert diag.is_healthy(config) is True
    
    def test_diagnostics_is_healthy_det_violation(self):
        """健全性チェック: det条件違反"""
        config = Phase1Config(stability_threshold=1e-6)
        
        diag = Phase1Diagnostics(
            bk_det_condition=1e-8,  # < threshold (BAD)
        )
        
        assert diag.is_healthy(config) is False
    
    def test_diagnostics_is_healthy_schatten_violation(self):
        """健全性チェック: Schattenノルム違反"""
        config = Phase1Config(schatten_s1_bound=100.0)
        
        diag = Phase1Diagnostics(
            bk_det_condition=1.0,
            bk_schatten_s1=150.0,  # > bound (BAD)
        )
        
        assert diag.is_healthy(config) is False
    
    def test_diagnostics_is_healthy_vram_violation(self):
        """健全性チェック: VRAM超過"""
        config = Phase1Config(target_vram_gb=8.0)
        
        diag = Phase1Diagnostics(
            bk_det_condition=1.0,
            peak_vram_mb=9000.0,  # > 8GB (BAD)
        )
        
        assert diag.is_healthy(config) is False
    
    def test_diagnostics_is_healthy_with_warnings(self):
        """健全性チェック: 警告あり"""
        config = Phase1Config()
        
        diag = Phase1Diagnostics(
            bk_det_condition=1.0,
            stability_warnings=["Test warning"]
        )
        
        assert diag.is_healthy(config) is False
    
    def test_diagnostics_get_summary(self):
        """サマリー生成のテスト"""
        diag = Phase1Diagnostics(
            ar_ssm_effective_rank=16.5,
            ar_ssm_gate_sparsity=0.35,
            htt_compression_ratio=0.1,
            peak_vram_mb=7500.0,
        )
        
        summary = diag.get_summary()
        
        assert "Phase 1 Diagnostics Summary" in summary
        assert "16.5" in summary  # effective rank
        assert "35.00%" in summary  # gate sparsity
        assert "7500.0 MB" in summary  # peak VRAM
    
    def test_diagnostics_get_summary_with_lns(self):
        """LNS有効時のサマリー"""
        diag = Phase1Diagnostics(
            lns_speedup=2.5,
            lns_accuracy_loss=0.001,
        )
        
        summary = diag.get_summary()
        
        assert "LNS:" in summary
        assert "2.5" in summary  # speedup


class TestPhase1TrainingState:
    """Phase1TrainingStateのテストクラス"""
    
    def test_default_training_state(self):
        """デフォルト訓練状態の作成"""
        state = Phase1TrainingState()
        
        assert state.current_max_rank == 4
        assert state.rank_schedule_step == 0
        assert state.stability_violations == 0
        assert state.best_ppl == float('inf')
    
    def test_update_rank_schedule_start(self):
        """ランクスケジュール更新: 開始時"""
        config = Phase1Config(ar_ssm_min_rank=4, ar_ssm_max_rank=32)
        state = Phase1TrainingState(rank_warmup_steps=1000)
        
        # Step 0: should be at min_rank
        state.update_rank_schedule(config)
        assert state.current_max_rank == 4
        assert state.rank_schedule_step == 1
    
    def test_update_rank_schedule_middle(self):
        """ランクスケジュール更新: 中間"""
        config = Phase1Config(ar_ssm_min_rank=4, ar_ssm_max_rank=32)
        state = Phase1TrainingState(
            rank_schedule_step=500,
            rank_warmup_steps=1000
        )
        
        # Step 500/1000: should be at midpoint
        state.update_rank_schedule(config)
        expected = 4 + 0.5 * (32 - 4)  # 18
        assert state.current_max_rank == int(expected)
    
    def test_update_rank_schedule_end(self):
        """ランクスケジュール更新: 終了時"""
        config = Phase1Config(ar_ssm_min_rank=4, ar_ssm_max_rank=32)
        state = Phase1TrainingState(
            rank_schedule_step=1000,
            rank_warmup_steps=1000
        )
        
        # Step 1000/1000: should be at max_rank
        state.update_rank_schedule(config)
        assert state.current_max_rank == 32
    
    def test_update_rank_schedule_beyond_warmup(self):
        """ランクスケジュール更新: ウォームアップ後"""
        config = Phase1Config(ar_ssm_min_rank=4, ar_ssm_max_rank=32)
        state = Phase1TrainingState(
            rank_schedule_step=2000,
            rank_warmup_steps=1000
        )
        
        # Step 2000/1000: should stay at max_rank
        state.update_rank_schedule(config)
        assert state.current_max_rank == 32
    
    def test_record_stability_violation(self):
        """安定性違反の記録"""
        state = Phase1TrainingState()
        
        assert state.stability_violations == 0
        assert state.last_stable_checkpoint is None
        
        state.record_stability_violation(checkpoint_path="checkpoint_100.pt")
        
        assert state.stability_violations == 1
        assert state.last_stable_checkpoint == "checkpoint_100.pt"
        
        # 2回目の違反ではチェックポイントは更新されない
        state.record_stability_violation(checkpoint_path="checkpoint_200.pt")
        assert state.stability_violations == 2
        assert state.last_stable_checkpoint == "checkpoint_100.pt"
    
    def test_update_best_metrics_ppl(self):
        """最良メトリクス更新: PPL"""
        state = Phase1TrainingState()
        
        # 初回更新
        updated = state.update_best_metrics(ppl=10.5)
        assert updated['ppl'] is True
        assert state.best_ppl == 10.5
        
        # より良い値で更新
        updated = state.update_best_metrics(ppl=9.0)
        assert updated['ppl'] is True
        assert state.best_ppl == 9.0
        
        # より悪い値では更新されない
        updated = state.update_best_metrics(ppl=11.0)
        assert updated['ppl'] is False
        assert state.best_ppl == 9.0
    
    def test_update_best_metrics_vram(self):
        """最良メトリクス更新: VRAM"""
        state = Phase1TrainingState()
        
        # 初回更新
        updated = state.update_best_metrics(vram_mb=8000.0)
        assert updated['vram'] is True
        assert state.best_vram_mb == 8000.0
        
        # より良い値で更新
        updated = state.update_best_metrics(vram_mb=7500.0)
        assert updated['vram'] is True
        assert state.best_vram_mb == 7500.0
    
    def test_update_best_metrics_both(self):
        """最良メトリクス更新: 両方"""
        state = Phase1TrainingState()
        
        updated = state.update_best_metrics(ppl=10.0, vram_mb=7500.0)
        assert updated['ppl'] is True
        assert updated['vram'] is True
    
    def test_training_state_to_dict(self):
        """辞書変換のテスト"""
        state = Phase1TrainingState(
            current_max_rank=16,
            rank_schedule_step=500
        )
        state_dict = state.to_dict()
        
        assert isinstance(state_dict, dict)
        assert state_dict['current_max_rank'] == 16
        assert state_dict['rank_schedule_step'] == 500
    
    def test_training_state_from_dict(self):
        """辞書からの復元テスト"""
        original = Phase1TrainingState(
            current_max_rank=16,
            rank_schedule_step=500,
            best_ppl=10.5
        )
        state_dict = original.to_dict()
        restored = Phase1TrainingState.from_dict(state_dict)
        
        assert restored.current_max_rank == 16
        assert restored.rank_schedule_step == 500
        assert restored.best_ppl == 10.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
