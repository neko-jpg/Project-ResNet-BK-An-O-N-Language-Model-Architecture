"""
Koopman Loss Weight Scheduler
Implements scheduling for Koopman auxiliary loss weight during training.
"""

import torch


class KoopmanLossScheduler:
    """
    Scheduler for Koopman auxiliary loss weight.
    
    Strategy: Start with low weight, gradually increase to encourage
    Koopman operator learning without disrupting initial training.
    
    Schedule types:
    - 'linear': Linear increase from min_weight to max_weight
    - 'exponential': Exponential increase
    - 'step': Step-wise increase at specified epochs
    """
    
    def __init__(
        self,
        min_weight=0.0,
        max_weight=0.5,
        warmup_epochs=2,
        total_epochs=10,
        schedule_type='linear'
    ):
        """
        Initialize Koopman loss scheduler.
        
        Args:
            min_weight: initial Koopman loss weight (typically 0.0)
            max_weight: maximum Koopman loss weight (typically 0.1-0.5)
            warmup_epochs: number of epochs before Koopman loss starts
            total_epochs: total number of training epochs
            schedule_type: 'linear', 'exponential', or 'step'
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule_type = schedule_type
        
        self.current_epoch = 0
        self.current_weight = min_weight
    
    def step(self, epoch=None):
        """
        Update Koopman loss weight for the current epoch.
        
        Args:
            epoch: current epoch (if None, increment internal counter)
        
        Returns:
            current_weight: Koopman loss weight for this epoch
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        # No Koopman loss during warmup
        if self.current_epoch < self.warmup_epochs:
            self.current_weight = self.min_weight
            return self.current_weight
        
        # Compute progress after warmup
        progress = (self.current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)  # Clamp to [0, 1]
        
        # Apply schedule
        if self.schedule_type == 'linear':
            self.current_weight = self.min_weight + progress * (self.max_weight - self.min_weight)
        
        elif self.schedule_type == 'exponential':
            # Exponential growth: weight = min_weight * (max_weight/min_weight)^progress
            if self.min_weight > 0:
                ratio = self.max_weight / self.min_weight
                self.current_weight = self.min_weight * (ratio ** progress)
            else:
                # If min_weight is 0, use linear for first part
                self.current_weight = progress * self.max_weight
        
        elif self.schedule_type == 'step':
            # Step-wise increase at 25%, 50%, 75% of training
            if progress < 0.25:
                self.current_weight = self.min_weight
            elif progress < 0.5:
                self.current_weight = self.min_weight + 0.33 * (self.max_weight - self.min_weight)
            elif progress < 0.75:
                self.current_weight = self.min_weight + 0.67 * (self.max_weight - self.min_weight)
            else:
                self.current_weight = self.max_weight
        
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
        
        return self.current_weight
    
    def get_weight(self):
        """Get current Koopman loss weight."""
        return self.current_weight
    
    def state_dict(self):
        """Get scheduler state for checkpointing."""
        return {
            'current_epoch': self.current_epoch,
            'current_weight': self.current_weight,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'schedule_type': self.schedule_type,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self.current_epoch = state_dict['current_epoch']
        self.current_weight = state_dict['current_weight']
        self.min_weight = state_dict['min_weight']
        self.max_weight = state_dict['max_weight']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.schedule_type = state_dict['schedule_type']
