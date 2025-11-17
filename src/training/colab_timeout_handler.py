"""
Colab Timeout Handler for Mamba-Killer ResNet-BK

Implements automatic timeout detection and emergency checkpoint saving for Google Colab.
Based on Requirement 12: 失敗モード分析と自動リカバリ (12.11-12.14)
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime, timedelta
import time
import os
import json

logger = logging.getLogger(__name__)


class ColabTimeoutHandler:
    """
    Handle Google Colab timeout detection and emergency checkpointing.
    
    Implements:
    - Timeout detection (<30 min remaining) (Requirement 12.11)
    - Emergency checkpoint saving (Requirement 12.12)
    - Automatic resume from checkpoint (Requirement 12.13, 12.14)
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        session_duration_hours: float = 12.0,
        warning_threshold_minutes: int = 30,
        check_interval_seconds: int = 60,
        enable_auto_resume: bool = True
    ):
        """
        Initialize Colab timeout handler.
        
        Args:
            checkpoint_dir: Directory for emergency checkpoints
            session_duration_hours: Expected Colab session duration (default: 12 hours)
            warning_threshold_minutes: Save checkpoint when this much time remains
            check_interval_seconds: How often to check remaining time
            enable_auto_resume: Automatically resume from checkpoint if found
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_duration = timedelta(hours=session_duration_hours)
        self.warning_threshold = timedelta(minutes=warning_threshold_minutes)
        self.check_interval = check_interval_seconds
        self.enable_auto_resume = enable_auto_resume
        
        # Session tracking
        self.session_start_time = datetime.now()
        self.last_check_time = datetime.now()
        self.timeout_warning_issued = False
        self.emergency_checkpoint_saved = False
        
        # State file for resume detection
        self.state_file = self.checkpoint_dir / "colab_session_state.json"
        
        # Check if we're in Colab
        self.is_colab = self._detect_colab()
        
        if self.is_colab:
            logger.info("✓ Running in Google Colab environment")
        else:
            logger.info("ℹ️ Not running in Colab (timeout handling disabled)")
        
        logger.info(f"✓ ColabTimeoutHandler initialized")
        logger.info(f"  Session duration: {session_duration_hours} hours")
        logger.info(f"  Warning threshold: {warning_threshold_minutes} minutes")
        logger.info(f"  Checkpoint dir: {checkpoint_dir}")
    
    def _detect_colab(self) -> bool:
        """
        Detect if running in Google Colab.
        
        Returns:
            True if in Colab environment
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def get_elapsed_time(self) -> timedelta:
        """
        Get elapsed time since session start.
        
        Returns:
            Elapsed time as timedelta
        """
        return datetime.now() - self.session_start_time
    
    def get_remaining_time(self) -> timedelta:
        """
        Get estimated remaining time in session.
        
        Returns:
            Remaining time as timedelta
        """
        elapsed = self.get_elapsed_time()
        remaining = self.session_duration - elapsed
        return max(remaining, timedelta(0))
    
    def should_save_emergency_checkpoint(self) -> bool:
        """
        Check if emergency checkpoint should be saved.
        
        Implements Requirement 12.11: Detect when <30 min remaining
        
        Returns:
            True if emergency checkpoint should be saved
        """
        if not self.is_colab:
            return False
        
        if self.emergency_checkpoint_saved:
            return False
        
        remaining = self.get_remaining_time()
        
        if remaining <= self.warning_threshold:
            logger.warning(f"⚠️ Timeout warning: {remaining.total_seconds()/60:.1f} minutes remaining")
            return True
        
        return False
    
    def check_timeout(self) -> Dict[str, Any]:
        """
        Check timeout status and return information.
        
        Returns:
            Dictionary with timeout information
        """
        now = datetime.now()
        
        # Only check at specified intervals
        if (now - self.last_check_time).total_seconds() < self.check_interval:
            return {
                'should_save': False,
                'remaining_minutes': self.get_remaining_time().total_seconds() / 60,
                'elapsed_minutes': self.get_elapsed_time().total_seconds() / 60
            }
        
        self.last_check_time = now
        
        remaining = self.get_remaining_time()
        elapsed = self.get_elapsed_time()
        should_save = self.should_save_emergency_checkpoint()
        
        if should_save and not self.timeout_warning_issued:
            self.timeout_warning_issued = True
            logger.warning("="*60)
            logger.warning("⚠️ COLAB TIMEOUT WARNING")
            logger.warning("="*60)
            logger.warning(f"Elapsed time: {elapsed.total_seconds()/3600:.2f} hours")
            logger.warning(f"Remaining time: {remaining.total_seconds()/60:.1f} minutes")
            logger.warning("Emergency checkpoint will be saved!")
            logger.warning("="*60)
        
        return {
            'should_save': should_save,
            'remaining_minutes': remaining.total_seconds() / 60,
            'elapsed_minutes': elapsed.total_seconds() / 60,
            'warning_issued': self.timeout_warning_issued,
            'is_colab': self.is_colab
        }
    
    def save_emergency_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        step: int,
        epoch: int,
        metrics: Dict[str, Any],
        random_state: Optional[Dict[str, Any]] = None,
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save emergency checkpoint with full training state.
        
        Implements Requirement 12.12: Save model, optimizer, scheduler, training state, metrics
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler (optional)
            step: Current training step
            epoch: Current training epoch
            metrics: Training metrics history
            random_state: Random number generator states (optional)
            additional_state: Any additional state to save (optional)
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"emergency_checkpoint_step{step}_epoch{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint
        checkpoint = {
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'session_info': {
                'elapsed_hours': self.get_elapsed_time().total_seconds() / 3600,
                'remaining_minutes': self.get_remaining_time().total_seconds() / 60,
            }
        }
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add random states for reproducibility
        if random_state is None:
            random_state = self._capture_random_state()
        checkpoint['random_state'] = random_state
        
        # Add any additional state
        if additional_state is not None:
            checkpoint['additional_state'] = additional_state
        
        try:
            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Save session state for resume detection
            self._save_session_state(checkpoint_path, step, epoch)
            
            self.emergency_checkpoint_saved = True
            
            logger.info("="*60)
            logger.info("✓ EMERGENCY CHECKPOINT SAVED")
            logger.info("="*60)
            logger.info(f"Path: {checkpoint_path}")
            logger.info(f"Step: {step}, Epoch: {epoch}")
            logger.info(f"Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
            logger.info("="*60)
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save emergency checkpoint: {e}")
            return ""
    
    def _capture_random_state(self) -> Dict[str, Any]:
        """
        Capture random number generator states.
        
        Returns:
            Dictionary with random states
        """
        import random
        import numpy as np
        
        random_state = {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
        }
        
        if torch.cuda.is_available():
            random_state['torch_cuda_random'] = torch.cuda.get_rng_state_all()
        
        return random_state
    
    def _restore_random_state(self, random_state: Dict[str, Any]):
        """
        Restore random number generator states.
        
        Args:
            random_state: Dictionary with random states
        """
        import random
        import numpy as np
        
        if 'python_random' in random_state:
            random.setstate(random_state['python_random'])
        
        if 'numpy_random' in random_state:
            np.random.set_state(random_state['numpy_random'])
        
        if 'torch_random' in random_state:
            torch.set_rng_state(random_state['torch_random'])
        
        if 'torch_cuda_random' in random_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(random_state['torch_cuda_random'])
    
    def _save_session_state(self, checkpoint_path: str, step: int, epoch: int):
        """
        Save session state for resume detection.
        
        Args:
            checkpoint_path: Path to checkpoint
            step: Current step
            epoch: Current epoch
        """
        state = {
            'checkpoint_path': str(checkpoint_path),
            'step': step,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'session_start': self.session_start_time.isoformat(),
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def detect_incomplete_training(self) -> Optional[Dict[str, Any]]:
        """
        Detect if there's an incomplete training session.
        
        Implements Requirement 12.13: Detect incomplete training
        
        Returns:
            Session state if incomplete training detected, None otherwise
        """
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            checkpoint_path = Path(state['checkpoint_path'])
            
            if checkpoint_path.exists():
                logger.info("="*60)
                logger.info("🔄 INCOMPLETE TRAINING DETECTED")
                logger.info("="*60)
                logger.info(f"Checkpoint: {checkpoint_path}")
                logger.info(f"Step: {state['step']}, Epoch: {state['epoch']}")
                logger.info(f"Timestamp: {state['timestamp']}")
                logger.info("="*60)
                return state
            else:
                logger.warning(f"⚠️ Checkpoint not found: {checkpoint_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to read session state: {e}")
            return None
    
    def resume_from_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Resume training from checkpoint.
        
        Implements Requirement 12.14: Verify epoch, step, random state, optimizer state
        
        Args:
            checkpoint_path: Path to checkpoint
            model: Model to restore
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore (optional)
        
        Returns:
            Dictionary with restored state information
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Restore model
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✓ Model state restored")
            
            # Restore optimizer
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Optimizer state restored")
            
            # Restore scheduler
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✓ Scheduler state restored")
            
            # Restore random states
            if 'random_state' in checkpoint:
                self._restore_random_state(checkpoint['random_state'])
                logger.info("✓ Random states restored")
            
            # Extract resume information
            resume_info = {
                'step': checkpoint['step'],
                'epoch': checkpoint['epoch'],
                'metrics': checkpoint.get('metrics', {}),
                'additional_state': checkpoint.get('additional_state', {}),
                'timestamp': checkpoint.get('timestamp', 'unknown'),
            }
            
            logger.info("="*60)
            logger.info("✓ TRAINING RESUMED")
            logger.info("="*60)
            logger.info(f"Resuming from step: {resume_info['step']}")
            logger.info(f"Resuming from epoch: {resume_info['epoch']}")
            logger.info(f"Checkpoint timestamp: {resume_info['timestamp']}")
            logger.info("="*60)
            
            return resume_info
            
        except Exception as e:
            logger.error(f"❌ Failed to resume from checkpoint: {e}")
            raise
    
    def auto_resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically detect and resume from incomplete training.
        
        Args:
            model: Model to restore
            optimizer: Optimizer to restore
            scheduler: Scheduler to restore (optional)
        
        Returns:
            Resume info if resumed, None otherwise
        """
        if not self.enable_auto_resume:
            return None
        
        state = self.detect_incomplete_training()
        
        if state is None:
            logger.info("ℹ️ No incomplete training detected")
            return None
        
        checkpoint_path = state['checkpoint_path']
        
        try:
            resume_info = self.resume_from_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            return resume_info
        except Exception as e:
            logger.error(f"❌ Auto-resume failed: {e}")
            return None
    
    def clear_session_state(self):
        """Clear session state file."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("✓ Session state cleared")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current timeout handler status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'is_colab': self.is_colab,
            'elapsed_hours': self.get_elapsed_time().total_seconds() / 3600,
            'remaining_minutes': self.get_remaining_time().total_seconds() / 60,
            'warning_issued': self.timeout_warning_issued,
            'emergency_checkpoint_saved': self.emergency_checkpoint_saved,
            'session_start': self.session_start_time.isoformat(),
            'checkpoint_dir': str(self.checkpoint_dir),
        }


if __name__ == '__main__':
    # Test Colab timeout handler
    import tempfile
    
    print("\n" + "="*60)
    print("Testing ColabTimeoutHandler")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create handler with short duration for testing
        handler = ColabTimeoutHandler(
            checkpoint_dir=tmpdir,
            session_duration_hours=0.1,  # 6 minutes
            warning_threshold_minutes=5,
            check_interval_seconds=1
        )
        
        # Create dummy model and optimizer
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        optimizer = torch.optim.Adam(model.parameters())
        
        # Check status
        status = handler.get_status()
        print(f"\nInitial Status:")
        print(f"  Is Colab: {status['is_colab']}")
        print(f"  Elapsed: {status['elapsed_hours']:.2f} hours")
        print(f"  Remaining: {status['remaining_minutes']:.1f} minutes")
        
        # Check timeout
        timeout_info = handler.check_timeout()
        print(f"\nTimeout Check:")
        print(f"  Should save: {timeout_info['should_save']}")
        print(f"  Remaining: {timeout_info['remaining_minutes']:.1f} minutes")
        
        # Test emergency checkpoint
        checkpoint_path = handler.save_emergency_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            epoch=5,
            metrics={'loss': 0.5, 'ppl': 30.0}
        )
        print(f"\n✓ Emergency checkpoint saved: {checkpoint_path}")
        
        # Test resume detection
        state = handler.detect_incomplete_training()
        if state:
            print(f"\n✓ Incomplete training detected")
            print(f"  Step: {state['step']}, Epoch: {state['epoch']}")
        
        # Test auto-resume
        resume_info = handler.auto_resume(model, optimizer)
        if resume_info:
            print(f"\n✓ Auto-resume successful")
            print(f"  Resumed from step: {resume_info['step']}")
            print(f"  Resumed from epoch: {resume_info['epoch']}")
        
        # Clear state
        handler.clear_session_state()
        print(f"\n✓ Session state cleared")
    
    print("\n✓ ColabTimeoutHandler test passed")
