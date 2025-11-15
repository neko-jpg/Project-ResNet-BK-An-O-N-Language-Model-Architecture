"""
Curriculum Learning Scheduler

Implements curriculum learning by ordering training examples by difficulty
and gradually increasing the difficulty threshold during training.

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Optional, Literal


class CurriculumLearningScheduler:
    """
    Order training examples by difficulty, gradually increase difficulty.
    
    Implements Requirements 7.1, 7.2, 7.3:
    - Compute difficulty scores using pretrained model
    - Order examples by difficulty
    - Gradually increase threshold during training
    """
    
    def __init__(
        self,
        dataset,
        model: nn.Module,
        difficulty_metric: Literal['perplexity', 'loss', 'entropy'] = 'perplexity',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            dataset: PyTorch dataset with (input, target) pairs
            model: Pretrained model for computing difficulty scores
            difficulty_metric: Metric for measuring difficulty
            device: Device for computation
        """
        self.dataset = dataset
        self.model = model.to(device)
        self.device = device
        self.difficulty_metric = difficulty_metric
        self.difficulties = None
        self.sorted_indices = None
    
    def compute_difficulties(self, batch_size: int = 32) -> torch.Tensor:
        """
        Compute difficulty score for each example using pretrained model.
        
        Returns:
            difficulties: (num_examples,) tensor of difficulty scores
        """
        self.model.eval()
        difficulties = []
        
        # Create dataloader for efficient batched processing
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        print(f"Computing difficulty scores for {len(self.dataset)} examples...")
        
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    # ACT model returns (logits, ponder_cost)
                    logits, _ = self.model(x_batch)
                else:
                    logits = self.model(x_batch)
                
                # Compute difficulty based on metric
                if self.difficulty_metric == 'perplexity':
                    # Perplexity = exp(cross_entropy_loss)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y_batch.view(-1),
                        reduction='none'
                    )
                    # Average loss per example
                    loss_per_example = loss.view(x_batch.size(0), -1).mean(dim=1)
                    difficulty = torch.exp(loss_per_example)
                
                elif self.difficulty_metric == 'loss':
                    # Raw cross-entropy loss
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        y_batch.view(-1),
                        reduction='none'
                    )
                    difficulty = loss.view(x_batch.size(0), -1).mean(dim=1)
                
                elif self.difficulty_metric == 'entropy':
                    # Entropy of output distribution
                    probs = F.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    difficulty = entropy.mean(dim=1)  # Average over sequence
                
                else:
                    raise ValueError(f"Unknown difficulty metric: {self.difficulty_metric}")
                
                difficulties.append(difficulty.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {(batch_idx + 1) * batch_size}/{len(self.dataset)} examples")
        
        self.difficulties = torch.cat(difficulties)
        
        # Sort indices by difficulty (easy to hard)
        self.sorted_indices = torch.argsort(self.difficulties)
        
        print(f"Difficulty scores computed:")
        print(f"  Min: {self.difficulties.min().item():.4f}")
        print(f"  Max: {self.difficulties.max().item():.4f}")
        print(f"  Mean: {self.difficulties.mean().item():.4f}")
        print(f"  Median: {self.difficulties.median().item():.4f}")
        
        return self.difficulties
    
    def get_curriculum_dataloader(
        self,
        epoch: int,
        total_epochs: int,
        batch_size: int,
        shuffle: bool = True,
        strategy: Literal['linear', 'exponential', 'root'] = 'linear'
    ) -> DataLoader:
        """
        Return dataloader with examples ordered by difficulty.
        
        Early epochs: easy examples only
        Later epochs: gradually add harder examples
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of training epochs
            batch_size: Batch size for dataloader
            shuffle: Whether to shuffle selected examples
            strategy: Curriculum pacing strategy
                - 'linear': Linear increase in difficulty
                - 'exponential': Exponential increase (slow start, fast end)
                - 'root': Square root increase (fast start, slow end)
        
        Returns:
            dataloader: DataLoader with curriculum-selected examples
        """
        if self.difficulties is None:
            raise ValueError("Must call compute_difficulties() first")
        
        # Compute difficulty percentile threshold
        progress = (epoch + 1) / total_epochs  # 0 → 1
        
        if strategy == 'linear':
            percentile = progress * 100  # 0% → 100%
        elif strategy == 'exponential':
            percentile = (progress ** 2) * 100  # Slower start
        elif strategy == 'root':
            percentile = (progress ** 0.5) * 100  # Faster start
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Ensure at least 10% of data in first epoch
        percentile = max(percentile, 10.0)
        
        # Get threshold difficulty
        num_examples = int(len(self.difficulties) * percentile / 100.0)
        num_examples = max(num_examples, 1)  # At least 1 example
        
        # Select easiest examples up to threshold
        selected_indices = self.sorted_indices[:num_examples].tolist()
        
        print(f"Epoch {epoch + 1}/{total_epochs}: Using {len(selected_indices)}/{len(self.dataset)} examples ({percentile:.1f}%)")
        print(f"  Difficulty range: {self.difficulties[selected_indices[0]].item():.4f} - {self.difficulties[selected_indices[-1]].item():.4f}")
        
        # Create subset
        subset = Subset(self.dataset, selected_indices)
        
        # Create dataloader
        dataloader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        
        return dataloader
    
    def get_difficulty_statistics(self) -> dict:
        """
        Get statistics about difficulty distribution.
        
        Returns:
            stats: Dictionary with difficulty statistics
        """
        if self.difficulties is None:
            raise ValueError("Must call compute_difficulties() first")
        
        return {
            'min': self.difficulties.min().item(),
            'max': self.difficulties.max().item(),
            'mean': self.difficulties.mean().item(),
            'median': self.difficulties.median().item(),
            'std': self.difficulties.std().item(),
            'q25': torch.quantile(self.difficulties, 0.25).item(),
            'q75': torch.quantile(self.difficulties, 0.75).item(),
        }
    
    def save_difficulties(self, path: str):
        """Save computed difficulties to file."""
        if self.difficulties is None:
            raise ValueError("Must call compute_difficulties() first")
        
        torch.save({
            'difficulties': self.difficulties,
            'sorted_indices': self.sorted_indices,
            'metric': self.difficulty_metric
        }, path)
        print(f"Saved difficulties to {path}")
    
    def load_difficulties(self, path: str):
        """Load precomputed difficulties from file."""
        data = torch.load(path)
        self.difficulties = data['difficulties']
        self.sorted_indices = data['sorted_indices']
        self.difficulty_metric = data['metric']
        print(f"Loaded difficulties from {path}")
        print(f"  Metric: {self.difficulty_metric}")
        print(f"  Num examples: {len(self.difficulties)}")


class DynamicDifficultyAdjuster:
    """
    Monitor validation loss and adjust curriculum pacing dynamically.
    
    Implements Requirement 7.3:
    - Monitor validation loss plateau
    - Accelerate difficulty increase if learning stalls
    """
    
    def __init__(
        self,
        patience: int = 3,
        acceleration_factor: float = 1.5,
        min_delta: float = 0.01
    ):
        """
        Args:
            patience: Number of epochs to wait before accelerating
            acceleration_factor: How much to accelerate (>1.0)
            min_delta: Minimum change in loss to consider improvement
        """
        self.patience = patience
        self.acceleration_factor = acceleration_factor
        self.min_delta = min_delta
        
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_acceleration = 1.0
    
    def update(self, val_loss: float) -> float:
        """
        Update based on validation loss.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            acceleration: Current acceleration factor
        """
        # Check if loss improved
        if val_loss < self.best_loss - self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.epochs_without_improvement = 0
            # Reset acceleration
            self.current_acceleration = 1.0
        else:
            # No improvement
            self.epochs_without_improvement += 1
            
            # Accelerate if plateau detected
            if self.epochs_without_improvement >= self.patience:
                self.current_acceleration *= self.acceleration_factor
                self.epochs_without_improvement = 0  # Reset counter
                print(f"Validation loss plateau detected. Accelerating curriculum: {self.current_acceleration:.2f}x")
        
        return self.current_acceleration
    
    def get_adjusted_epoch(self, epoch: int) -> int:
        """
        Get adjusted epoch number based on acceleration.
        
        Args:
            epoch: Actual epoch number
        
        Returns:
            adjusted_epoch: Accelerated epoch number
        """
        return int(epoch * self.current_acceleration)


def create_curriculum_trainer(
    model: nn.Module,
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    total_epochs: int = 10,
    difficulty_metric: str = 'perplexity',
    strategy: str = 'linear',
    use_dynamic_adjustment: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Create curriculum learning trainer with all components.
    
    Args:
        model: Model for training (also used for difficulty computation)
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        total_epochs: Total training epochs
        difficulty_metric: Metric for difficulty
        strategy: Curriculum pacing strategy
        use_dynamic_adjustment: Whether to use dynamic difficulty adjustment
        device: Device for computation
    
    Returns:
        scheduler: CurriculumLearningScheduler
        adjuster: DynamicDifficultyAdjuster (if enabled)
    """
    # Create scheduler
    scheduler = CurriculumLearningScheduler(
        train_dataset,
        model,
        difficulty_metric=difficulty_metric,
        device=device
    )
    
    # Compute difficulties
    scheduler.compute_difficulties(batch_size=batch_size)
    
    # Create dynamic adjuster if enabled
    adjuster = None
    if use_dynamic_adjustment:
        adjuster = DynamicDifficultyAdjuster(
            patience=3,
            acceleration_factor=1.5,
            min_delta=0.01
        )
    
    return scheduler, adjuster
