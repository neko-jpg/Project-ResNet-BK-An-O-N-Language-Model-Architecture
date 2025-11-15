"""
Transfer Learning Pipeline

Implements transfer learning by pretraining on large corpus and finetuning
on target dataset to reduce training cost.

Based on Step 7 design for achieving 10× cost reduction through data efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from typing import Optional, Dict, Callable
import time


class TransferLearningPipeline:
    """
    Transfer learning pipeline: pretrain → finetune.
    
    Implements Requirements 7.9, 7.10:
    - Pretrain on large corpus (C4)
    - Finetune on target dataset (WikiText-2)
    - Measure training cost reduction
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: Model to train
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.pretrain_metrics = {}
        self.finetune_metrics = {}
    
    def pretrain(
        self,
        pretrain_dataset,
        optimizer,
        criterion,
        num_epochs: int = 5,
        batch_size: int = 32,
        save_path: Optional[str] = None,
        log_interval: int = 100
    ) -> Dict:
        """
        Pretrain model on large corpus.
        
        Args:
            pretrain_dataset: Large pretraining dataset (e.g., C4)
            optimizer: Optimizer for pretraining
            criterion: Loss criterion
            num_epochs: Number of pretraining epochs
            batch_size: Batch size
            save_path: Path to save pretrained checkpoint
            log_interval: Logging interval (batches)
        
        Returns:
            metrics: Dictionary with pretraining metrics
        """
        print("=" * 60)
        print("PRETRAINING PHASE")
        print("=" * 60)
        print(f"Dataset size: {len(pretrain_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print()
        
        dataloader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.model.train()
        
        total_loss = 0
        total_batches = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    logits, ponder_cost = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    loss = loss + 0.01 * ponder_cost
                else:
                    logits = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                epoch_loss += loss.item()
                total_batches += 1
                epoch_batches += 1
                
                # Logging
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / epoch_batches
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"\nEpoch {epoch+1} completed: Avg Loss = {avg_epoch_loss:.4f}")
            print("-" * 60)
        
        # Compute metrics
        pretrain_time = time.time() - start_time
        avg_loss = total_loss / total_batches
        
        self.pretrain_metrics = {
            'avg_loss': avg_loss,
            'total_batches': total_batches,
            'total_time': pretrain_time,
            'time_per_batch': pretrain_time / total_batches
        }
        
        print(f"\nPretraining completed:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Total time: {pretrain_time:.2f}s")
        print(f"  Time per batch: {pretrain_time/total_batches:.4f}s")
        
        # Save checkpoint
        if save_path is not None:
            self.save_checkpoint(save_path, 'pretrain')
            print(f"  Saved checkpoint: {save_path}")
        
        return self.pretrain_metrics
    
    def finetune(
        self,
        finetune_dataset,
        optimizer,
        criterion,
        num_epochs: int = 3,
        batch_size: int = 32,
        learning_rate: Optional[float] = None,
        freeze_layers: Optional[int] = None,
        save_path: Optional[str] = None,
        log_interval: int = 50
    ) -> Dict:
        """
        Finetune pretrained model on target dataset.
        
        Args:
            finetune_dataset: Target dataset (e.g., WikiText-2)
            optimizer: Optimizer for finetuning
            criterion: Loss criterion
            num_epochs: Number of finetuning epochs
            batch_size: Batch size
            learning_rate: Learning rate (if different from pretrain)
            freeze_layers: Number of initial layers to freeze
            save_path: Path to save finetuned checkpoint
            log_interval: Logging interval (batches)
        
        Returns:
            metrics: Dictionary with finetuning metrics
        """
        print("\n" + "=" * 60)
        print("FINETUNING PHASE")
        print("=" * 60)
        print(f"Dataset size: {len(finetune_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        
        # Optionally freeze layers
        if freeze_layers is not None and freeze_layers > 0:
            print(f"Freezing first {freeze_layers} layers")
            self._freeze_layers(freeze_layers)
        
        # Optionally adjust learning rate
        if learning_rate is not None:
            print(f"Adjusting learning rate to {learning_rate}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        print()
        
        dataloader = DataLoader(
            finetune_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        self.model.train()
        
        total_loss = 0
        total_batches = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'forward') and 'ponder_cost' in str(self.model.forward.__code__.co_varnames):
                    logits, ponder_cost = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                    loss = loss + 0.01 * ponder_cost
                else:
                    logits = self.model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                epoch_loss += loss.item()
                total_batches += 1
                epoch_batches += 1
                
                # Logging
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / epoch_batches
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"\nEpoch {epoch+1} completed: Avg Loss = {avg_epoch_loss:.4f}")
            print("-" * 60)
        
        # Compute metrics
        finetune_time = time.time() - start_time
        avg_loss = total_loss / total_batches
        
        self.finetune_metrics = {
            'avg_loss': avg_loss,
            'total_batches': total_batches,
            'total_time': finetune_time,
            'time_per_batch': finetune_time / total_batches
        }
        
        print(f"\nFinetuning completed:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Total time: {finetune_time:.2f}s")
        print(f"  Time per batch: {finetune_time/total_batches:.4f}s")
        
        # Save checkpoint
        if save_path is not None:
            self.save_checkpoint(save_path, 'finetune')
            print(f"  Saved checkpoint: {save_path}")
        
        return self.finetune_metrics
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first num_layers layers."""
        if hasattr(self.model, 'blocks'):
            for i, block in enumerate(self.model.blocks):
                if i < num_layers:
                    for param in block.parameters():
                        param.requires_grad = False
    
    def compute_cost_reduction(self, baseline_time: float) -> Dict:
        """
        Compute training cost reduction compared to baseline.
        
        Args:
            baseline_time: Time to train from scratch on target dataset
        
        Returns:
            metrics: Dictionary with cost reduction metrics
        """
        if not self.finetune_metrics:
            raise ValueError("Must run finetune() first")
        
        # Total transfer learning time
        total_time = self.pretrain_metrics.get('total_time', 0) + self.finetune_metrics['total_time']
        
        # Cost reduction
        cost_reduction = baseline_time / total_time if total_time > 0 else 0
        
        # Finetune-only reduction (assuming pretrain is amortized)
        finetune_reduction = baseline_time / self.finetune_metrics['total_time']
        
        metrics = {
            'baseline_time': baseline_time,
            'pretrain_time': self.pretrain_metrics.get('total_time', 0),
            'finetune_time': self.finetune_metrics['total_time'],
            'total_time': total_time,
            'cost_reduction_total': cost_reduction,
            'cost_reduction_finetune_only': finetune_reduction
        }
        
        print("\n" + "=" * 60)
        print("COST REDUCTION ANALYSIS")
        print("=" * 60)
        print(f"Baseline training time: {baseline_time:.2f}s")
        print(f"Pretrain time: {self.pretrain_metrics.get('total_time', 0):.2f}s")
        print(f"Finetune time: {self.finetune_metrics['total_time']:.2f}s")
        print(f"Total transfer learning time: {total_time:.2f}s")
        print(f"\nCost reduction (total): {cost_reduction:.2f}×")
        print(f"Cost reduction (finetune only): {finetune_reduction:.2f}×")
        
        return metrics
    
    def save_checkpoint(self, path: str, phase: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'phase': phase,
            'pretrain_metrics': self.pretrain_metrics,
            'finetune_metrics': self.finetune_metrics
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pretrain_metrics = checkpoint.get('pretrain_metrics', {})
        self.finetune_metrics = checkpoint.get('finetune_metrics', {})
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Phase: {checkpoint.get('phase', 'unknown')}")


def create_transfer_learning_pipeline(
    model: nn.Module,
    pretrain_dataset,
    finetune_dataset,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 3,
    batch_size: int = 32,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    freeze_layers: Optional[int] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> TransferLearningPipeline:
    """
    Create and run complete transfer learning pipeline.
    
    Args:
        model: Model to train
        pretrain_dataset: Pretraining dataset
        finetune_dataset: Finetuning dataset
        pretrain_epochs: Pretraining epochs
        finetune_epochs: Finetuning epochs
        batch_size: Batch size
        pretrain_lr: Pretraining learning rate
        finetune_lr: Finetuning learning rate
        freeze_layers: Number of layers to freeze during finetuning
        device: Device for computation
    
    Returns:
        pipeline: TransferLearningPipeline instance
    """
    pipeline = TransferLearningPipeline(model, device=device)
    
    # Pretrain
    pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=pretrain_lr)
    pretrain_criterion = nn.CrossEntropyLoss()
    
    pipeline.pretrain(
        pretrain_dataset,
        pretrain_optimizer,
        pretrain_criterion,
        num_epochs=pretrain_epochs,
        batch_size=batch_size,
        save_path='checkpoints/pretrained_model.pt'
    )
    
    # Finetune
    finetune_optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)
    finetune_criterion = nn.CrossEntropyLoss()
    
    pipeline.finetune(
        finetune_dataset,
        finetune_optimizer,
        finetune_criterion,
        num_epochs=finetune_epochs,
        batch_size=batch_size,
        freeze_layers=freeze_layers,
        save_path='checkpoints/finetuned_model.pt'
    )
    
    return pipeline


class DomainAdaptationPipeline(TransferLearningPipeline):
    """
    Domain adaptation: adapt model from source domain to target domain.
    
    Extends transfer learning with domain-specific techniques.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(model, device)
    
    def gradual_unfreezing(
        self,
        finetune_dataset,
        optimizer,
        criterion,
        num_phases: int = 3,
        epochs_per_phase: int = 1,
        batch_size: int = 32
    ) -> Dict:
        """
        Gradual unfreezing: progressively unfreeze layers during finetuning.
        
        Phase 1: Freeze all but last layer
        Phase 2: Unfreeze last 2 layers
        Phase 3: Unfreeze all layers
        
        Args:
            finetune_dataset: Target dataset
            optimizer: Optimizer
            criterion: Loss criterion
            num_phases: Number of unfreezing phases
            epochs_per_phase: Epochs per phase
            batch_size: Batch size
        
        Returns:
            metrics: Dictionary with metrics
        """
        print("\n" + "=" * 60)
        print("GRADUAL UNFREEZING")
        print("=" * 60)
        
        if not hasattr(self.model, 'blocks'):
            print("Warning: Model doesn't have 'blocks' attribute. Using standard finetuning.")
            return self.finetune(finetune_dataset, optimizer, criterion, num_phases * epochs_per_phase, batch_size)
        
        num_layers = len(self.model.blocks)
        
        all_metrics = []
        
        for phase in range(num_phases):
            # Determine which layers to freeze
            layers_to_unfreeze = max(1, (phase + 1) * num_layers // num_phases)
            layers_to_freeze = num_layers - layers_to_unfreeze
            
            print(f"\nPhase {phase + 1}/{num_phases}: Unfreezing last {layers_to_unfreeze} layers")
            
            # Freeze/unfreeze layers
            for i, block in enumerate(self.model.blocks):
                freeze = i < layers_to_freeze
                for param in block.parameters():
                    param.requires_grad = not freeze
            
            # Train for this phase
            phase_metrics = self.finetune(
                finetune_dataset,
                optimizer,
                criterion,
                num_epochs=epochs_per_phase,
                batch_size=batch_size,
                save_path=f'checkpoints/gradual_unfreeze_phase{phase+1}.pt'
            )
            
            all_metrics.append(phase_metrics)
        
        # Combine metrics
        combined_metrics = {
            'phases': all_metrics,
            'total_time': sum(m['total_time'] for m in all_metrics),
            'final_loss': all_metrics[-1]['avg_loss']
        }
        
        return combined_metrics
