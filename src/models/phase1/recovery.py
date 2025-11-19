"""
Phase 1 Efficiency Engine - Automatic Error Recovery

このモジュールは、Phase 1コンポーネントで発生するエラーに対する
自動回復戦略を実装します。

物理的直観: システムが不安定状態に近づいた際、自動的に安全な
パラメータ空間に戻すことで、学習の継続性を保証します。

Requirements: 5.3, 10.4, 10.5
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .errors import (
    VRAMExhaustedError,
    NumericalInstabilityError,
)


logger = logging.getLogger(__name__)


@dataclass
class RecoveryAction:
    """
    回復アクションの記録。
    
    Attributes:
        action_type: アクションの種類
        timestamp: アクション実行時刻
        success: アクションが成功したかどうか
        details: アクションの詳細情報
    """
    action_type: str
    timestamp: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)


class Phase1ErrorRecovery:
    """
    Phase 1コンポーネントの自動エラー回復クラス。
    
    このクラスは、VRAM不足や数値的不安定性などのエラーに対して
    自動的に回復戦略を適用します。
    
    回復戦略の優先順位:
    1. 最も影響の少ない変更から開始（例: 勾配チェックポイント有効化）
    2. より大きな変更に進む（例: ランク削減、学習率低減）
    3. すべての戦略が失敗した場合、エラーを再発生
    
    Requirements: 5.3, 10.4, 10.5
    """
    
    def __init__(
        self,
        max_recovery_attempts: int = 3,
        enable_logging: bool = True
    ):
        """
        Phase1ErrorRecoveryを初期化します。
        
        Args:
            max_recovery_attempts: 最大回復試行回数
            enable_logging: ログ記録を有効にするかどうか
        """
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_logging = enable_logging
        
        # 回復履歴
        self.recovery_history: List[RecoveryAction] = []
        self.vram_recovery_attempts = 0
        self.stability_recovery_attempts = 0
    
    def handle_vram_exhausted(
        self,
        error: VRAMExhaustedError,
        model: Optional[nn.Module] = None,
        config: Optional[Any] = None,
        training_state: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        VRAM不足エラーからの自動回復を試みます。
        
        回復戦略（優先順位順）:
        1. 勾配チェックポイントの有効化
        2. AR-SSMランクの削減
        3. LNSカーネルの無効化
        
        Args:
            error: VRAMExhaustedError例外
            model: モデルインスタンス
            config: Phase1Config設定
            training_state: 学習状態の辞書
        
        Returns:
            回復が成功した場合True、すべての戦略が失敗した場合False
        """
        import time
        
        if self.vram_recovery_attempts >= self.max_recovery_attempts:
            self._log_error("Maximum VRAM recovery attempts reached")
            return False
        
        self.vram_recovery_attempts += 1
        timestamp = time.time()
        
        # 戦略1: 勾配チェックポイントの有効化
        if config and not config.use_gradient_checkpointing:
            self._log_warning("Attempting recovery: Enabling gradient checkpointing")
            config.use_gradient_checkpointing = True
            config.checkpoint_ar_ssm = True
            
            self.recovery_history.append(RecoveryAction(
                action_type="enable_gradient_checkpointing",
                timestamp=timestamp,
                success=True,
                details={
                    "vram_before_mb": error.current_mb,
                    "vram_limit_mb": error.limit_mb,
                }
            ))
            
            self._log_info("Recovery successful: Gradient checkpointing enabled")
            return True
        
        # 戦略2: AR-SSMランクの削減
        if config and config.ar_ssm_enabled and config.ar_ssm_max_rank > config.ar_ssm_min_rank:
            old_rank = config.ar_ssm_max_rank
            new_rank = max(config.ar_ssm_min_rank, config.ar_ssm_max_rank // 2)
            
            self._log_warning(
                f"Attempting recovery: Reducing AR-SSM max rank from {old_rank} to {new_rank}"
            )
            config.ar_ssm_max_rank = new_rank
            
            # モデルのランクも更新（存在する場合）
            if model and hasattr(model, 'update_ar_ssm_rank'):
                model.update_ar_ssm_rank(new_rank)
            
            self.recovery_history.append(RecoveryAction(
                action_type="reduce_ar_ssm_rank",
                timestamp=timestamp,
                success=True,
                details={
                    "old_rank": old_rank,
                    "new_rank": new_rank,
                    "vram_before_mb": error.current_mb,
                }
            ))
            
            self._log_info(f"Recovery successful: AR-SSM rank reduced to {new_rank}")
            return True
        
        # 戦略3: LNSカーネルの無効化
        if config and config.lns_enabled:
            self._log_warning("Attempting recovery: Disabling LNS kernel")
            config.lns_enabled = False
            
            self.recovery_history.append(RecoveryAction(
                action_type="disable_lns_kernel",
                timestamp=timestamp,
                success=True,
                details={
                    "vram_before_mb": error.current_mb,
                }
            ))
            
            self._log_info("Recovery successful: LNS kernel disabled")
            return True
        
        # すべての戦略が失敗
        self._log_error("All VRAM recovery strategies exhausted")
        self.recovery_history.append(RecoveryAction(
            action_type="vram_recovery_failed",
            timestamp=timestamp,
            success=False,
            details={
                "vram_current_mb": error.current_mb,
                "vram_limit_mb": error.limit_mb,
            }
        ))
        
        return False
    
    def handle_numerical_instability(
        self,
        error: NumericalInstabilityError,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[Any] = None,
        model: Optional[nn.Module] = None
    ) -> bool:
        """
        数値的不安定性エラーからの自動回復を試みます。
        
        回復戦略（優先順位順）:
        1. 学習率の削減
        2. 安定性閾値の増加
        3. 勾配クリッピングの有効化
        
        Args:
            error: NumericalInstabilityError例外
            optimizer: オプティマイザインスタンス
            config: Phase1Config設定
            model: モデルインスタンス
        
        Returns:
            回復が成功した場合True、すべての戦略が失敗した場合False
        """
        import time
        
        if self.stability_recovery_attempts >= self.max_recovery_attempts:
            self._log_error("Maximum stability recovery attempts reached")
            return False
        
        self.stability_recovery_attempts += 1
        timestamp = time.time()
        
        # 戦略1: 学習率の削減
        if optimizer:
            old_lrs = []
            new_lrs = []
            
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * 0.5
                param_group['lr'] = new_lr
                
                old_lrs.append(old_lr)
                new_lrs.append(new_lr)
            
            self._log_warning(
                f"Attempting recovery: Reducing learning rate by 50% "
                f"(avg: {sum(old_lrs)/len(old_lrs):.2e} → {sum(new_lrs)/len(new_lrs):.2e})"
            )
            
            self.recovery_history.append(RecoveryAction(
                action_type="reduce_learning_rate",
                timestamp=timestamp,
                success=True,
                details={
                    "old_lrs": old_lrs,
                    "new_lrs": new_lrs,
                    "component": error.component,
                    "diagnostics": error.diagnostics,
                }
            ))
            
            self._log_info("Recovery successful: Learning rate reduced")
            return True
        
        # 戦略2: 安定性閾値の増加
        if config and hasattr(config, 'stability_threshold'):
            old_threshold = config.stability_threshold
            new_threshold = old_threshold * 10.0
            
            self._log_warning(
                f"Attempting recovery: Increasing stability threshold "
                f"from {old_threshold:.2e} to {new_threshold:.2e}"
            )
            config.stability_threshold = new_threshold
            
            self.recovery_history.append(RecoveryAction(
                action_type="increase_stability_threshold",
                timestamp=timestamp,
                success=True,
                details={
                    "old_threshold": old_threshold,
                    "new_threshold": new_threshold,
                    "component": error.component,
                }
            ))
            
            self._log_info(f"Recovery successful: Stability threshold increased to {new_threshold:.2e}")
            return True
        
        # 戦略3: 勾配クリッピングの有効化（モデルパラメータに適用）
        if model and config:
            max_grad_norm = 1.0
            
            self._log_warning(
                f"Attempting recovery: Enabling gradient clipping (max_norm={max_grad_norm})"
            )
            
            # 勾配クリッピングを適用
            if hasattr(model, 'parameters'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # 設定に勾配クリッピングフラグを追加
            if hasattr(config, 'gradient_norm_threshold'):
                config.gradient_norm_threshold = max_grad_norm
            
            self.recovery_history.append(RecoveryAction(
                action_type="enable_gradient_clipping",
                timestamp=timestamp,
                success=True,
                details={
                    "max_grad_norm": max_grad_norm,
                    "component": error.component,
                }
            ))
            
            self._log_info("Recovery successful: Gradient clipping enabled")
            return True
        
        # すべての戦略が失敗
        self._log_error("All stability recovery strategies exhausted")
        self.recovery_history.append(RecoveryAction(
            action_type="stability_recovery_failed",
            timestamp=timestamp,
            success=False,
            details={
                "component": error.component,
                "diagnostics": error.diagnostics,
            }
        ))
        
        return False
    
    def reset_recovery_counters(self) -> None:
        """回復試行カウンタをリセットします。"""
        self.vram_recovery_attempts = 0
        self.stability_recovery_attempts = 0
        self._log_info("Recovery counters reset")
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """
        回復履歴のサマリーを取得します。
        
        Returns:
            回復履歴のサマリー辞書
        """
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for action in self.recovery_history if action.success)
        
        action_counts = {}
        for action in self.recovery_history:
            action_type = action.action_type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        return {
            "total_recovery_attempts": total_attempts,
            "successful_recoveries": successful_attempts,
            "failed_recoveries": total_attempts - successful_attempts,
            "vram_recovery_attempts": self.vram_recovery_attempts,
            "stability_recovery_attempts": self.stability_recovery_attempts,
            "action_counts": action_counts,
            "recovery_history": [
                {
                    "action_type": action.action_type,
                    "timestamp": action.timestamp,
                    "success": action.success,
                    "details": action.details,
                }
                for action in self.recovery_history
            ],
        }
    
    def _log_info(self, message: str) -> None:
        """情報ログを出力します。"""
        if self.enable_logging:
            logger.info(f"[Phase1Recovery] {message}")
    
    def _log_warning(self, message: str) -> None:
        """警告ログを出力します。"""
        if self.enable_logging:
            logger.warning(f"[Phase1Recovery] {message}")
    
    def _log_error(self, message: str) -> None:
        """エラーログを出力します。"""
        if self.enable_logging:
            logger.error(f"[Phase1Recovery] {message}")


# 便利な関数

def create_recovery_context_manager(
    recovery: Phase1ErrorRecovery,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Any] = None,
    training_state: Optional[Dict[str, Any]] = None,
):
    """
    自動回復機能を持つコンテキストマネージャを作成します。
    
    使用例:
        recovery = Phase1ErrorRecovery()
        with create_recovery_context_manager(recovery, model, optimizer, config):
            # 学習ループ
            output = model(batch)
            loss.backward()
            optimizer.step()
    
    Args:
        recovery: Phase1ErrorRecoveryインスタンス
        model: モデルインスタンス
        optimizer: オプティマイザインスタンス
        config: Phase1Config設定
        training_state: 学習状態の辞書
    
    Returns:
        コンテキストマネージャ
    """
    from contextlib import contextmanager
    
    @contextmanager
    def recovery_context():
        try:
            yield
        except VRAMExhaustedError as e:
            logger.warning(f"VRAM exhausted: {e}")
            success = recovery.handle_vram_exhausted(
                error=e,
                model=model,
                config=config,
                training_state=training_state
            )
            if not success:
                logger.error("VRAM recovery failed, re-raising exception")
                raise
            logger.info("VRAM recovery successful, continuing execution")
        except NumericalInstabilityError as e:
            logger.warning(f"Numerical instability detected: {e}")
            success = recovery.handle_numerical_instability(
                error=e,
                optimizer=optimizer,
                config=config,
                model=model
            )
            if not success:
                logger.error("Stability recovery failed, re-raising exception")
                raise
            logger.info("Stability recovery successful, continuing execution")
    
    return recovery_context()
