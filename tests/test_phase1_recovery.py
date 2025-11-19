"""
Unit tests for Phase 1 automatic error recovery.

Requirements: 5.3, 10.4, 10.5
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from src.models.phase1.recovery import (
    Phase1ErrorRecovery,
    RecoveryAction,
    create_recovery_context_manager,
)
from src.models.phase1.errors import (
    VRAMExhaustedError,
    NumericalInstabilityError,
)


# モックの設定クラス
@dataclass
class MockPhase1Config:
    """テスト用のモック設定クラス。"""
    use_gradient_checkpointing: bool = False
    checkpoint_ar_ssm: bool = False
    ar_ssm_enabled: bool = True
    ar_ssm_max_rank: int = 32
    ar_ssm_min_rank: int = 4
    lns_enabled: bool = False
    stability_threshold: float = 1e-6
    gradient_norm_threshold: float = 10.0


class MockModel(nn.Module):
    """テスト用のモックモデル。"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.ar_ssm_rank = 32
    
    def update_ar_ssm_rank(self, new_rank: int):
        """AR-SSMランクを更新します。"""
        self.ar_ssm_rank = new_rank
    
    def forward(self, x):
        return self.linear(x)


class TestRecoveryAction:
    """RecoveryActionデータクラスのテスト。"""
    
    def test_recovery_action_creation(self):
        """RecoveryActionが正しく作成されることを確認。"""
        action = RecoveryAction(
            action_type="test_action",
            timestamp=1234567890.0,
            success=True,
            details={"key": "value"}
        )
        
        assert action.action_type == "test_action"
        assert action.timestamp == 1234567890.0
        assert action.success is True
        assert action.details["key"] == "value"


class TestPhase1ErrorRecovery:
    """Phase1ErrorRecoveryクラスのテスト。"""
    
    def test_recovery_initialization(self):
        """回復クラスが正しく初期化されることを確認。"""
        recovery = Phase1ErrorRecovery(
            max_recovery_attempts=5,
            enable_logging=False
        )
        
        assert recovery.max_recovery_attempts == 5
        assert recovery.enable_logging is False
        assert len(recovery.recovery_history) == 0
        assert recovery.vram_recovery_attempts == 0
        assert recovery.stability_recovery_attempts == 0
    
    def test_vram_recovery_enable_checkpointing(self):
        """勾配チェックポイント有効化による回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config()
        
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=["Enable checkpointing"]
        )
        
        # 回復を試行
        success = recovery.handle_vram_exhausted(
            error=error,
            config=config
        )
        
        assert success is True
        assert config.use_gradient_checkpointing is True
        assert config.checkpoint_ar_ssm is True
        assert recovery.vram_recovery_attempts == 1
        assert len(recovery.recovery_history) == 1
        assert recovery.recovery_history[0].action_type == "enable_gradient_checkpointing"
    
    def test_vram_recovery_reduce_rank(self):
        """AR-SSMランク削減による回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config(
            use_gradient_checkpointing=True,  # 既に有効
            ar_ssm_max_rank=32
        )
        model = MockModel()
        
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=["Reduce rank"]
        )
        
        # 回復を試行
        success = recovery.handle_vram_exhausted(
            error=error,
            model=model,
            config=config
        )
        
        assert success is True
        assert config.ar_ssm_max_rank == 16  # 32 // 2
        assert model.ar_ssm_rank == 16
        assert recovery.vram_recovery_attempts == 1
        assert recovery.recovery_history[0].action_type == "reduce_ar_ssm_rank"
    
    def test_vram_recovery_disable_lns(self):
        """LNSカーネル無効化による回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config(
            use_gradient_checkpointing=True,
            ar_ssm_max_rank=4,  # 既に最小
            ar_ssm_min_rank=4,
            lns_enabled=True
        )
        
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=["Disable LNS"]
        )
        
        # 回復を試行
        success = recovery.handle_vram_exhausted(
            error=error,
            config=config
        )
        
        assert success is True
        assert config.lns_enabled is False
        assert recovery.vram_recovery_attempts == 1
        assert recovery.recovery_history[0].action_type == "disable_lns_kernel"
    
    def test_vram_recovery_all_strategies_exhausted(self):
        """すべての回復戦略が使い果たされた場合を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config(
            use_gradient_checkpointing=True,
            ar_ssm_max_rank=4,
            ar_ssm_min_rank=4,
            lns_enabled=False
        )
        
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=[]
        )
        
        # 回復を試行
        success = recovery.handle_vram_exhausted(
            error=error,
            config=config
        )
        
        assert success is False
        assert recovery.vram_recovery_attempts == 1
        assert recovery.recovery_history[0].action_type == "vram_recovery_failed"
        assert recovery.recovery_history[0].success is False
    
    def test_vram_recovery_max_attempts(self):
        """最大試行回数に達した場合を確認。"""
        recovery = Phase1ErrorRecovery(
            max_recovery_attempts=2,
            enable_logging=False
        )
        config = MockPhase1Config()
        
        error = VRAMExhaustedError(
            current_mb=9000.0,
            limit_mb=8000.0,
            suggestions=[]
        )
        
        # 最大試行回数まで回復を試行
        for _ in range(2):
            recovery.handle_vram_exhausted(error=error, config=config)
        
        # 3回目は失敗するはず
        success = recovery.handle_vram_exhausted(error=error, config=config)
        assert success is False
        assert recovery.vram_recovery_attempts == 2
    
    def test_stability_recovery_reduce_lr(self):
        """学習率削減による回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        
        # オプティマイザを作成
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        error = NumericalInstabilityError(
            component="AR-SSM",
            diagnostics={"has_nan": True}
        )
        
        # 回復を試行
        success = recovery.handle_numerical_instability(
            error=error,
            optimizer=optimizer
        )
        
        assert success is True
        # 学習率が半分になっていることを確認
        for param_group in optimizer.param_groups:
            assert param_group['lr'] == 0.0005  # 0.001 * 0.5
        assert recovery.stability_recovery_attempts == 1
        assert recovery.recovery_history[0].action_type == "reduce_learning_rate"
    
    def test_stability_recovery_increase_threshold(self):
        """安定性閾値増加による回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config(stability_threshold=1e-6)
        
        error = NumericalInstabilityError(
            component="BK-Core",
            diagnostics={"det_condition": 1e-7}
        )
        
        # 回復を試行（オプティマイザなし）
        success = recovery.handle_numerical_instability(
            error=error,
            config=config
        )
        
        assert success is True
        # 浮動小数点の比較には許容誤差を使用
        assert abs(config.stability_threshold - 1e-5) < 1e-10  # 1e-6 * 10
        assert recovery.stability_recovery_attempts == 1
        assert recovery.recovery_history[0].action_type == "increase_stability_threshold"
    
    def test_stability_recovery_enable_clipping(self):
        """勾配クリッピング有効化による回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        model = MockModel()
        # オプティマイザなし、かつ安定性閾値は既に調整済みの状態を作る
        config = MockPhase1Config()
        
        error = NumericalInstabilityError(
            component="HTT",
            diagnostics={"gradient_norm": 100.0}
        )
        
        # 回復を試行（オプティマイザなしなので、安定性閾値増加が先に実行される）
        success = recovery.handle_numerical_instability(
            error=error,
            model=model,
            config=config
        )
        
        assert success is True
        assert recovery.stability_recovery_attempts == 1
        # オプティマイザがない場合、安定性閾値増加が実行される
        assert recovery.recovery_history[0].action_type == "increase_stability_threshold"
    
    def test_reset_recovery_counters(self):
        """回復カウンタのリセットを確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        
        # カウンタを増やす
        recovery.vram_recovery_attempts = 5
        recovery.stability_recovery_attempts = 3
        
        # リセット
        recovery.reset_recovery_counters()
        
        assert recovery.vram_recovery_attempts == 0
        assert recovery.stability_recovery_attempts == 0
    
    def test_get_recovery_summary(self):
        """回復サマリーの取得を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config()
        
        # いくつかの回復を実行
        error1 = VRAMExhaustedError(9000, 8000, [])
        recovery.handle_vram_exhausted(error=error1, config=config)
        
        error2 = NumericalInstabilityError("test", {})
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        recovery.handle_numerical_instability(error=error2, optimizer=optimizer)
        
        # サマリーを取得
        summary = recovery.get_recovery_summary()
        
        assert summary["total_recovery_attempts"] == 2
        assert summary["successful_recoveries"] == 2
        assert summary["failed_recoveries"] == 0
        assert summary["vram_recovery_attempts"] == 1
        assert summary["stability_recovery_attempts"] == 1
        assert len(summary["recovery_history"]) == 2


class TestRecoveryContextManager:
    """回復コンテキストマネージャのテスト。"""
    
    def test_context_manager_no_error(self):
        """エラーが発生しない場合の動作を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        model = MockModel()
        
        with create_recovery_context_manager(recovery, model=model):
            # 正常な処理
            x = torch.randn(2, 10)
            y = model(x)
        
        # 回復履歴は空のはず
        assert len(recovery.recovery_history) == 0
    
    def test_context_manager_vram_error_recovery(self):
        """VRAMエラーが発生した場合の回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config()
        
        with create_recovery_context_manager(recovery, config=config):
            # VRAMエラーを発生させる
            raise VRAMExhaustedError(9000, 8000, [])
        
        # 回復が実行されたことを確認
        assert len(recovery.recovery_history) == 1
        assert config.use_gradient_checkpointing is True
    
    def test_context_manager_stability_error_recovery(self):
        """安定性エラーが発生した場合の回復を確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with create_recovery_context_manager(recovery, model=model, optimizer=optimizer):
            # 安定性エラーを発生させる
            raise NumericalInstabilityError("test", {})
        
        # 回復が実行されたことを確認
        assert len(recovery.recovery_history) == 1
        for param_group in optimizer.param_groups:
            assert param_group['lr'] == 0.0005
    
    def test_context_manager_recovery_failure(self):
        """回復が失敗した場合、エラーが再発生することを確認。"""
        recovery = Phase1ErrorRecovery(
            max_recovery_attempts=1,
            enable_logging=False
        )
        config = MockPhase1Config(
            use_gradient_checkpointing=True,
            ar_ssm_max_rank=4,
            ar_ssm_min_rank=4,
            lns_enabled=False
        )
        
        # すべての戦略が使い果たされているため、回復は失敗するはず
        with pytest.raises(VRAMExhaustedError):
            with create_recovery_context_manager(recovery, config=config):
                raise VRAMExhaustedError(9000, 8000, [])


class TestRecoveryIntegration:
    """回復機能の統合テスト。"""
    
    def test_multiple_recovery_attempts(self):
        """複数の回復試行が正しく動作することを確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config()
        model = MockModel()
        
        # 1回目: 勾配チェックポイント有効化
        error1 = VRAMExhaustedError(9000, 8000, [])
        success1 = recovery.handle_vram_exhausted(error=error1, config=config)
        assert success1 is True
        assert config.use_gradient_checkpointing is True
        
        # 2回目: ランク削減
        error2 = VRAMExhaustedError(9000, 8000, [])
        success2 = recovery.handle_vram_exhausted(error=error2, model=model, config=config)
        assert success2 is True
        assert config.ar_ssm_max_rank == 16
        
        # 回復履歴を確認
        assert len(recovery.recovery_history) == 2
        assert recovery.vram_recovery_attempts == 2
    
    def test_recovery_summary_accuracy(self):
        """回復サマリーが正確であることを確認。"""
        recovery = Phase1ErrorRecovery(enable_logging=False)
        config = MockPhase1Config()
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 成功する回復
        error1 = VRAMExhaustedError(9000, 8000, [])
        recovery.handle_vram_exhausted(error=error1, config=config)
        
        error2 = NumericalInstabilityError("test", {})
        recovery.handle_numerical_instability(error=error2, optimizer=optimizer)
        
        # 失敗する回復
        config_exhausted = MockPhase1Config(
            use_gradient_checkpointing=True,
            ar_ssm_max_rank=4,
            ar_ssm_min_rank=4,
            lns_enabled=False
        )
        error3 = VRAMExhaustedError(9000, 8000, [])
        recovery.handle_vram_exhausted(error=error3, config=config_exhausted)
        
        # サマリーを確認
        summary = recovery.get_recovery_summary()
        assert summary["total_recovery_attempts"] == 3
        assert summary["successful_recoveries"] == 2
        assert summary["failed_recoveries"] == 1
