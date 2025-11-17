"""
Checkpoint Manager for Mamba-Killer ResNet-BK

Handles saving, loading, and managing model checkpoints with automatic recovery.
"""

import os
import torch
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic recovery and cleanup."""
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        keep_last_n: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            save_optimizer: Save optimizer state
            save_scheduler: Save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.checkpoints = []
        self._load_checkpoint_list()
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Scheduler to save
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            config: Model configuration
            is_best: Whether this is the best checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'timestamp': timestamp,
            'metrics': metrics or {},
            'config': config or {}
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        if self.save_scheduler and scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Update checkpoint list
            self.checkpoints.append({
                'path': str(checkpoint_path),
                'epoch': epoch,
                'step': step,
                'timestamp': timestamp,
                'metrics': metrics or {},
                'is_best': is_best
            })
            
            # Save as latest
            latest_path = self.checkpoint_dir / "latest.pt"
            shutil.copy(checkpoint_path, latest_path)
            
            # Save as best if applicable
            if is_best:
                best_path = self.checkpoint_dir / "best.pt"
                shutil.copy(checkpoint_path, best_path)
                logger.info(f"✓ Best checkpoint updated: {best_path}")
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            # Save checkpoint list
            self._save_checkpoint_list()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (uses latest if None)
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            device: Device to load checkpoint to
        
        Returns:
            Checkpoint data dictionary
        """
        # Use latest checkpoint if path not specified
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.pt"
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            logger.info(f"✓ Checkpoint loaded: {checkpoint_path}")
            
            # Load model state
            if model is not None and 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'])
                logger.info("✓ Model state loaded")
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                logger.info("✓ Optimizer state loaded")
            
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
                logger.info("✓ Scheduler state loaded")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def load_best(
        self,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cuda"
    ) -> Dict[str, Any]:
        """Load best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        return self.load(str(best_path), model, optimizer, scheduler, device)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return str(latest_path)
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        if best_path.exists():
            return str(best_path)
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints."""
        return self.checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        if len(self.checkpoints) <= self.keep_last_n:
            return
        
        # Sort by step (most recent first)
        sorted_checkpoints = sorted(
            self.checkpoints,
            key=lambda x: x['step'],
            reverse=True
        )
        
        # Keep recent checkpoints and best checkpoint
        to_keep = set()
        for i, ckpt in enumerate(sorted_checkpoints):
            if i < self.keep_last_n or ckpt.get('is_best', False):
                to_keep.add(ckpt['path'])
        
        # Remove old checkpoints
        for ckpt in self.checkpoints[:]:
            if ckpt['path'] not in to_keep:
                try:
                    Path(ckpt['path']).unlink()
                    self.checkpoints.remove(ckpt)
                    logger.info(f"Removed old checkpoint: {ckpt['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {ckpt['path']}: {e}")
    
    def _save_checkpoint_list(self):
        """Save checkpoint list to JSON."""
        list_path = self.checkpoint_dir / "checkpoints.json"
        with open(list_path, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
    
    def _load_checkpoint_list(self):
        """Load checkpoint list from JSON."""
        list_path = self.checkpoint_dir / "checkpoints.json"
        if list_path.exists():
            with open(list_path, 'r') as f:
                self.checkpoints = json.load(f)
    
    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Verify checkpoint integrity.
        
        Args:
            checkpoint_path: Path to checkpoint
        
        Returns:
            True if checkpoint is valid
        """
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            required_keys = ['epoch', 'step', 'model_state_dict']
            return all(key in checkpoint_data for key in required_keys)
        except Exception as e:
            logger.error(f"Checkpoint verification failed: {e}")
            return False


def create_checkpoint_manager(config: Dict[str, Any]) -> CheckpointManager:
    """
    Create checkpoint manager from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        CheckpointManager instance
    """
    checkpoint_config = config.get('checkpoint', {})
    
    return CheckpointManager(
        checkpoint_dir=checkpoint_config.get('save_dir', './checkpoints'),
        keep_last_n=checkpoint_config.get('keep_last_n', 5),
        save_optimizer=checkpoint_config.get('save_optimizer', True),
        save_scheduler=checkpoint_config.get('save_scheduler', True)
    )


if __name__ == '__main__':
    # Test checkpoint manager
    import torch.nn as nn
    
    # Create dummy model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create manager
    manager = CheckpointManager(checkpoint_dir='./test_checkpoints')
    
    # Save checkpoint
    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=1,
        step=100,
        metrics={'loss': 0.5, 'accuracy': 0.9}
    )
    
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint_data = manager.load(checkpoint_path, model, optimizer)
    print(f"✓ Loaded checkpoint: epoch={checkpoint_data['epoch']}, step={checkpoint_data['step']}")
    
    # Cleanup
    shutil.rmtree('./test_checkpoints')
    print("✓ Test passed")
