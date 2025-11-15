"""
Dynamic Learning Rate Scheduler

Implements dynamic learning rate scheduling that adapts based on training progress.

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Optional, List
import numpy as np


class DynamicLRScheduler(_LRScheduler):
    """
    Dynamic learning rate scheduler with adaptive adjustments.
    
    Implements Requirements 7.15, 7.16:
    - Increase LR when loss decreases steadily
    - Decrease LR when loss plateaus
    - Implement warm restarts
    """
    
    def __init__(
        self,
        optimizer,
        patience: int = 5,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        min_delta: float = 0.01,
        warmup_steps: int = 0,
        restart_period: Optional[int] = None,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            patience: Number of epochs to wait before adjusting LR
            increase_factor: Factor to increase LR when improving
            decrease_factor: Factor to decrease LR when plateauing
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            min_delta: Minimum change in loss to consider improvement
            warmup_steps: Number of warmup steps
            restart_period: Period for warm restarts (None = no restarts)
            last_epoch: Last epoch number
        """
        self.patience = patience
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_delta = min_delta
        self.warmup_steps = warmup_steps
        self.restart_period = restart_period
        
        # State tracking
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.epochs_with_improvement = 0
        self.current_step = 0
        self.restart_count = 0
        
        # Loss history
        self.loss_history = []
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        lrs = []
        
        for base_lr in self.base_lrs:
            # Warmup
            if self.current_step < self.warmup_steps:
                lr = base_lr * (self.current_step + 1) / self.warmup_steps
            else:
                lr = base_lr
            
            # Clamp to valid range
            lr = max(self.min_lr, min(self.max_lr, lr))
            
            lrs.append(lr)
        
        return lrs
    
    def step(self, loss: Optional[float] = None, epoch: Optional[int] = None):
        """
        Update learning rate based on loss.
        
        Args:
            loss: Current training/validation loss
            epoch: Current epoch (optional)
        """
        self.current_step += 1
        
        if loss is not None:
            self.loss_history.append(loss)
            
            # Check for improvement
            if loss < self.best_loss - self.min_delta:
                # Improvement detected
                self.best_loss = loss
                self.epochs_with_improvement += 1
                self.epochs_without_improvement = 0
                
                # Increase LR if consistently improving
                if self.epochs_with_improvement >= self.patience:
                    self._increase_lr()
                    self.epochs_with_improvement = 0
            
            else:
                # No improvement
                self.epochs_without_improvement += 1
                self.epochs_with_improvement = 0
                
                # Decrease LR if plateauing
                if self.epochs_without_improvement >= self.patience:
                    self._decrease_lr()
                    self.epochs_without_improvement = 0
            
            # Warm restart
            if self.restart_period is not None:
                if self.current_step % self.restart_period == 0:
                    self._warm_restart()
        
        # Update optimizer learning rates
        super().step(epoch)
    
    def _increase_lr(self):
        """Increase learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = min(old_lr * self.increase_factor, self.max_lr)
            param_group['lr'] = new_lr
            self.base_lrs[i] = new_lr
            
            if new_lr != old_lr:
                print(f"Increasing LR: {old_lr:.6f} → {new_lr:.6f}")
    
    def _decrease_lr(self):
        """Decrease learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.decrease_factor, self.min_lr)
            param_group['lr'] = new_lr
            self.base_lrs[i] = new_lr
            
            if new_lr != old_lr:
                print(f"Decreasing LR: {old_lr:.6f} → {new_lr:.6f}")
    
    def _warm_restart(self):
        """Perform warm restart."""
        self.restart_count += 1
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = self.base_lrs[i] * 2.0  # Reset to higher LR
            new_lr = min(new_lr, self.max_lr)
            param_group['lr'] = new_lr
            
            print(f"Warm restart #{self.restart_count}: {old_lr:.6f} → {new_lr:.6f}")
        
        # Reset state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.epochs_with_improvement = 0
    
    def get_last_lr(self):
        """Return last computed learning rate."""
        return [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).
    
    Learning rate follows cosine curve and periodically restarts.
    """
    
    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            T_0: Number of iterations for first restart
            T_mult: Factor to increase T_i after restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate using cosine annealing."""
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            
            if self.T_cur >= self.T_i:
                # Restart
                self.T_cur = 0
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Epoch must be non-negative")
            
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class OneCycleLR(_LRScheduler):
    """
    One-cycle learning rate policy.
    
    Increases LR from initial to max, then decreases to min.
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            max_lr: Maximum learning rate
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing LR
            anneal_strategy: 'cos' or 'linear'
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = max_lr / final_div_factor
            last_epoch: Last epoch number
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        step_num = self.last_epoch
        
        if step_num < self.step_size_up:
            # Increasing phase
            pct = step_num / self.step_size_up
            
            if self.anneal_strategy == 'cos':
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (1 - math.cos(math.pi * pct)) / 2
            else:  # linear
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        
        else:
            # Decreasing phase
            pct = (step_num - self.step_size_up) / self.step_size_down
            
            if self.anneal_strategy == 'cos':
                lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + math.cos(math.pi * pct)) / 2
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.final_lr) * pct
        
        return [lr for _ in self.base_lrs]


def create_dynamic_scheduler(
    optimizer,
    scheduler_type: str = 'dynamic',
    **kwargs
) -> _LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
            - 'dynamic': DynamicLRScheduler (adaptive)
            - 'cosine_restart': CosineAnnealingWarmRestarts
            - 'one_cycle': OneCycleLR
        **kwargs: Scheduler-specific arguments
    
    Returns:
        scheduler: Learning rate scheduler
    """
    if scheduler_type == 'dynamic':
        scheduler = DynamicLRScheduler(
            optimizer,
            patience=kwargs.get('patience', 5),
            increase_factor=kwargs.get('increase_factor', 1.1),
            decrease_factor=kwargs.get('decrease_factor', 0.5),
            min_lr=kwargs.get('min_lr', 1e-6),
            max_lr=kwargs.get('max_lr', 1e-2),
            warmup_steps=kwargs.get('warmup_steps', 0),
            restart_period=kwargs.get('restart_period', None)
        )
        print(f"Created DynamicLRScheduler:")
        print(f"  Patience: {kwargs.get('patience', 5)}")
        print(f"  Increase factor: {kwargs.get('increase_factor', 1.1)}")
        print(f"  Decrease factor: {kwargs.get('decrease_factor', 0.5)}")
    
    elif scheduler_type == 'cosine_restart':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 0)
        )
        print(f"Created CosineAnnealingWarmRestarts:")
        print(f"  T_0: {kwargs.get('T_0', 10)}")
        print(f"  T_mult: {kwargs.get('T_mult', 2)}")
    
    elif scheduler_type == 'one_cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=kwargs.get('total_steps', 1000),
            pct_start=kwargs.get('pct_start', 0.3)
        )
        print(f"Created OneCycleLR:")
        print(f"  Max LR: {kwargs.get('max_lr', 1e-3)}")
        print(f"  Total steps: {kwargs.get('total_steps', 1000)}")
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def train_with_dynamic_lr(
    model: nn.Module,
    train_dataset,
    optimizer,
    criterion,
    num_epochs: int = 10,
    batch_size: int = 32,
    scheduler_type: str = 'dynamic',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **scheduler_kwargs
):
    """
    Train model with dynamic learning rate scheduling.
    
    Args:
        model: Model to train
        train_dataset: Training dataset
        optimizer: Optimizer
        criterion: Loss criterion
        num_epochs: Number of epochs
        batch_size: Batch size
        scheduler_type: Type of scheduler
        device: Device for computation
        **scheduler_kwargs: Scheduler-specific arguments
    
    Returns:
        metrics: Training metrics
    """
    from torch.utils.data import DataLoader
    
    model = model.to(device)
    
    # Create scheduler
    if scheduler_type == 'one_cycle':
        total_steps = len(train_dataset) // batch_size * num_epochs
        scheduler_kwargs['total_steps'] = total_steps
    
    scheduler = create_dynamic_scheduler(optimizer, scheduler_type, **scheduler_kwargs)
    
    # Create dataloader
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )
    
    print(f"\nTraining with {scheduler_type} scheduler:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print()
    
    lr_history = []
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_batches = 0
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'forward') and 'ponder_cost' in str(model.forward.__code__.co_varnames):
                logits, ponder_cost = model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                loss = loss + 0.01 * ponder_cost
            else:
                logits = model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Step scheduler (for one-cycle)
            if scheduler_type == 'one_cycle':
                scheduler.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Epoch metrics
        avg_epoch_loss = epoch_loss / epoch_batches
        loss_history.append(avg_epoch_loss)
        
        # Step scheduler (for dynamic and cosine_restart)
        if scheduler_type in ['dynamic', 'cosine_restart']:
            if scheduler_type == 'dynamic':
                scheduler.step(loss=avg_epoch_loss)
            else:
                scheduler.step()
        
        # Record LR
        current_lr = scheduler.get_last_lr()[0]
        lr_history.append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_epoch_loss:.4f}, LR = {current_lr:.6f}")
    
    return {
        'loss_history': loss_history,
        'lr_history': lr_history
    }
