"""
Automatic Recovery System for Mamba-Killer ResNet-BK

Implements automatic failure detection and recovery strategies.
Based on Requirement 12: 失敗モード分析と自動リカバリ
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import gc
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .stability_monitor import StabilityMonitor, StabilityMetrics

logger = logging.getLogger(__name__)


@dataclass
class RecoveryState:
    """State information for recovery operations."""
    
    # Recovery attempts
    rollback_count: int = 0
    lr_reduction_count: int = 0
    epsilon_increase_count: int = 0
    batch_size_reduction_count: int = 0
    precision_upgrade_count: int = 0
    
    # Recovery history
    recovery_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Last successful recovery
    last_recovery_step: int = 0
    last_recovery_action: str = ""
    
    # Failure tracking
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3
    
    # Checkpoint tracking
    checkpoint_queue: List[str] = field(default_factory=list)
    max_checkpoints: int = 5


class AutoRecovery:
    """
    Automatic failure detection and recovery system.
    
    Implements:
    - Rollback to last stable checkpoint (Requirement 12.2)
    - Learning rate reduction (Requirement 12.4)
    - Epsilon adjustment (Requirement 12.6)
    - Batch size reduction (Requirement 12.7, 12.8)
    - Automatic hyperparameter adjustment (Requirement 12.19)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_retries: int = 3,
        enable_auto_adjustment: bool = True,
        min_lr: float = 1e-7,
        max_epsilon: float = 1.0,
        min_batch_size: int = 1
    ):
        """Initialize automatic recovery system."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.enable_auto_adjustment = enable_auto_adjustment
        self.min_lr = min_lr
        self.max_epsilon = max_epsilon
        self.min_batch_size = min_batch_size
        
        self.state = RecoveryState()
        
        logger.info(f"✓ AutoRecovery initialized (checkpoint_dir={checkpoint_dir}, max_retries={max_retries})")
    
    def detect_failure(self, metrics: StabilityMetrics) -> Optional[str]:
        """Detect failure mode from stability metrics."""
        if metrics.has_nan:
            return "nan_detected"
        if metrics.has_inf:
            return "inf_detected"
        if metrics.gradient_explosion:
            return "gradient_explosion"
        if metrics.loss_divergence:
            return "loss_divergence"
        for name, cond_num in metrics.condition_numbers.items():
            if cond_num > 1e6:
                return "condition_number_high"
        return None

    def recover(
        self,
        failure_type: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        current_step: int = 0,
        current_epoch: int = 0,
        **kwargs
    ) -> Tuple[bool, str]:
        """Attempt recovery from failure."""
        if self.state.consecutive_failures >= self.max_retries:
            logger.error(f"❌ Max retries ({self.max_retries}) exceeded. Halting training.")
            return False, "max_retries_exceeded"
        
        action = self._determine_recovery_action(failure_type, kwargs)
        logger.info(f"🔧 Attempting recovery: {action} (failure: {failure_type})")
        
        success = False
        try:
            if action == "rollback_checkpoint":
                success = self._rollback_checkpoint(model, optimizer, scheduler)
                self.state.rollback_count += 1
            elif action == "reduce_lr_10x":
                success = self._reduce_learning_rate(optimizer, factor=0.1)
                self.state.lr_reduction_count += 1
            elif action == "increase_epsilon":
                success = self._increase_epsilon(model, factor=1.5)
                self.state.epsilon_increase_count += 1
            elif action == "reduce_batch_size":
                success = self._reduce_batch_size(kwargs.get('dataloader'), factor=0.5)
                self.state.batch_size_reduction_count += 1
            elif action == "upgrade_precision":
                success = self._upgrade_precision(model)
                self.state.precision_upgrade_count += 1
            elif action == "clear_cuda_cache":
                success = self._clear_cuda_cache()
            else:
                logger.warning(f"Unknown recovery action: {action}")
                success = False
            
            if success:
                self.state.consecutive_failures = 0
                self.state.last_recovery_step = current_step
                self.state.last_recovery_action = action
                logger.info(f"✓ Recovery successful: {action}")
            else:
                self.state.consecutive_failures += 1
                logger.warning(f"⚠️ Recovery failed: {action} (attempt {self.state.consecutive_failures}/{self.max_retries})")
            
            self.state.recovery_history.append({
                'step': current_step,
                'epoch': current_epoch,
                'failure_type': failure_type,
                'action': action,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"❌ Recovery exception: {e}")
            self.state.consecutive_failures += 1
            success = False
        
        return success, action
    
    def _determine_recovery_action(self, failure_type: str, kwargs: Dict[str, Any]) -> str:
        """Determine appropriate recovery action based on failure type."""
        action_map = {
            "nan_detected": "rollback_checkpoint",
            "inf_detected": "rollback_checkpoint",
            "gradient_explosion": "reduce_lr_10x",
            "loss_divergence": "increase_epsilon",
            "oom": "reduce_batch_size",
            "condition_number_high": "upgrade_precision",
        }
        return action_map.get(failure_type, "rollback_checkpoint")

    def _rollback_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any] = None) -> bool:
        """Rollback to last stable checkpoint."""
        if not self.state.checkpoint_queue:
            logger.warning("⚠️ No checkpoints available for rollback")
            return False
        
        checkpoint_path = self.state.checkpoint_queue[-1]
        if not Path(checkpoint_path).exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"✓ Rolled back to checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def _reduce_learning_rate(self, optimizer: torch.optim.Optimizer, factor: float = 0.1) -> bool:
        """Reduce learning rate by factor."""
        try:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * factor, self.min_lr)
                param_group['lr'] = new_lr
            logger.info(f"✓ Learning rate reduced by {factor}×")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reduce learning rate: {e}")
            return False
    
    def _increase_epsilon(self, model: nn.Module, factor: float = 1.5) -> bool:
        """Increase epsilon parameter for stability."""
        try:
            adjusted_count = 0
            for name, module in model.named_modules():
                if hasattr(module, 'epsilon'):
                    old_epsilon = module.epsilon
                    new_epsilon = min(old_epsilon * factor, self.max_epsilon)
                    module.epsilon = new_epsilon
                    adjusted_count += 1
            if adjusted_count > 0:
                logger.info(f"✓ Epsilon increased in {adjusted_count} modules")
                return True
            else:
                logger.warning("⚠️ No epsilon parameters found in model")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to increase epsilon: {e}")
            return False
    
    def _reduce_batch_size(self, dataloader: Optional[Any], factor: float = 0.5) -> bool:
        """Reduce batch size to handle OOM."""
        if dataloader is None:
            logger.warning("⚠️ No dataloader provided for batch size reduction")
            return False
        try:
            old_batch_size = dataloader.batch_size
            new_batch_size = max(int(old_batch_size * factor), self.min_batch_size)
            logger.info(f"✓ Batch size reduction recommended: {old_batch_size} → {new_batch_size}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to reduce batch size: {e}")
            return False
    
    def _upgrade_precision(self, model: nn.Module) -> bool:
        """Upgrade precision for numerical stability."""
        try:
            upgraded_count = 0
            for name, module in model.named_modules():
                if hasattr(module, 'use_complex128'):
                    if not module.use_complex128:
                        module.use_complex128 = True
                        upgraded_count += 1
            if upgraded_count > 0:
                logger.info(f"✓ Precision upgraded in {upgraded_count} modules")
                return True
            else:
                logger.warning("⚠️ No precision upgradeable modules found")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to upgrade precision: {e}")
            return False
    
    def _clear_cuda_cache(self) -> bool:
        """Clear CUDA cache to free memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("✓ CUDA cache cleared")
                return True
            else:
                logger.warning("⚠️ CUDA not available")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to clear CUDA cache: {e}")
            return False

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], 
                       step: int, epoch: int, metrics: Dict[str, Any], is_stable: bool = True) -> str:
        """Save checkpoint with full training state."""
        checkpoint_name = f"checkpoint_step{step}_epoch{epoch}.pt"
        if not is_stable:
            checkpoint_name = f"emergency_{checkpoint_name}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'is_stable': is_stable,
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        try:
            torch.save(checkpoint, checkpoint_path)
            self.state.checkpoint_queue.append(str(checkpoint_path))
            
            if len(self.state.checkpoint_queue) > self.state.max_checkpoints:
                old_checkpoint = self.state.checkpoint_queue.pop(0)
                if Path(old_checkpoint).exists():
                    Path(old_checkpoint).unlink()
            
            logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return ""
    
    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """Verify checkpoint integrity."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'step', 'epoch']
            for key in required_keys:
                if key not in checkpoint:
                    logger.error(f"❌ Checkpoint missing key: {key}")
                    return False
            if not checkpoint['model_state_dict']:
                logger.error("❌ Checkpoint has empty model state")
                return False
            return True
        except Exception as e:
            logger.error(f"❌ Checkpoint verification failed: {e}")
            return False
    
    def get_recovery_report(self) -> Dict[str, Any]:
        """Get comprehensive recovery report."""
        return {
            'total_recoveries': len(self.state.recovery_history),
            'rollback_count': self.state.rollback_count,
            'lr_reduction_count': self.state.lr_reduction_count,
            'epsilon_increase_count': self.state.epsilon_increase_count,
            'batch_size_reduction_count': self.state.batch_size_reduction_count,
            'precision_upgrade_count': self.state.precision_upgrade_count,
            'consecutive_failures': self.state.consecutive_failures,
            'last_recovery_step': self.state.last_recovery_step,
            'last_recovery_action': self.state.last_recovery_action,
            'recovery_history': self.state.recovery_history,
            'checkpoint_count': len(self.state.checkpoint_queue),
        }
    
    def reset(self):
        """Reset recovery state."""
        self.state = RecoveryState()
        logger.info("✓ Recovery state reset")
