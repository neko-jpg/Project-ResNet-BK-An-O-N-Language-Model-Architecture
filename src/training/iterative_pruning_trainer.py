"""
Iterative Pruning Trainer with Retraining

This module implements the complete iterative magnitude-based pruning workflow:
1. Train model to convergence
2. Prune weights by magnitude
3. Retrain to recover accuracy
4. Repeat until target sparsity reached

Focuses on output_proj and fc layers as specified in requirements.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path

from ..models.pruned_moe import IterativeMagnitudePruner


class IterativePruningTrainer:
    """
    Trainer for iterative magnitude-based pruning with retraining.
    
    Implements the prune-retrain cycle to achieve high sparsity with minimal
    accuracy degradation.
    """
    
    def __init__(self, model: nn.Module, 
                 initial_sparsity: float = 0.2,
                 final_sparsity: float = 0.8,
                 num_iterations: int = 5,
                 prune_layers: Optional[List[str]] = None,
                 device: str = 'cuda'):
        """
        Args:
            model: Model to prune
            initial_sparsity: Starting sparsity (e.g., 0.2 = 20%)
            final_sparsity: Target final sparsity (e.g., 0.8 = 80%)
            num_iterations: Number of prune-retrain cycles
            prune_layers: Layer patterns to prune (e.g., ['output_proj', 'fc'])
            device: Device to train on
        """
        self.model = model.to(device)
        self.device = device
        
        # Create iterative pruner
        self.pruner = IterativeMagnitudePruner(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            num_iterations=num_iterations,
            prune_layers=prune_layers
        )
        
        # Training history
        self.training_history = []
        
        print(f"\n=== Iterative Pruning Trainer Initialized ===")
        print(f"Initial sparsity: {initial_sparsity:.1%}")
        print(f"Final sparsity: {final_sparsity:.1%}")
        print(f"Iterations: {num_iterations}")
        print(f"Prune layers: {prune_layers if prune_layers else 'all linear layers'}")
    
    def train_epoch(self, train_loader, optimizer, criterion, 
                   apply_mask: bool = False) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss criterion
            apply_mask: If True, apply pruning masks after each step
        
        Returns:
            Average training loss
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Apply pruning masks to maintain sparsity
            if apply_mask:
                self.pruner.train_step_with_mask(self.model)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def evaluate(self, val_loader, criterion) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss criterion
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                
                total_loss += loss.item() * y_batch.size(0)
                total_tokens += y_batch.size(0)
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {'loss': avg_loss, 'perplexity': perplexity}
    
    def run_iterative_pruning(self, train_loader, val_loader,
                             retrain_epochs: int = 3,
                             learning_rate: float = 1e-4,
                             save_dir: Optional[str] = None) -> Dict:
        """
        Execute complete iterative pruning workflow.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            retrain_epochs: Epochs to retrain after each pruning step
            learning_rate: Learning rate for retraining
            save_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with final metrics
        """
        print(f"\n{'='*60}")
        print(f"ITERATIVE PRUNING WORKFLOW")
        print(f"{'='*60}")
        
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate baseline
        print(f"\nEvaluating baseline model...")
        baseline_metrics = self.evaluate(val_loader, criterion)
        print(f"Baseline perplexity: {baseline_metrics['perplexity']:.2f}")
        
        # Iterative pruning loop
        for iteration in range(self.pruner.num_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{self.pruner.num_iterations}")
            print(f"{'='*60}")
            
            # Step 1: Prune
            print(f"\nStep 1: Pruning...")
            prune_stats = self.pruner.prune_step(self.model, verbose=True)
            
            # Evaluate after pruning (before retraining)
            print(f"\nEvaluating after pruning (before retraining)...")
            post_prune_metrics = self.evaluate(val_loader, criterion)
            print(f"Post-prune perplexity: {post_prune_metrics['perplexity']:.2f}")
            print(f"Perplexity increase: {post_prune_metrics['perplexity'] / baseline_metrics['perplexity']:.2f}×")
            
            # Step 2: Retrain
            print(f"\nStep 2: Retraining for {retrain_epochs} epochs...")
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
            
            retrain_losses = []
            for epoch in range(retrain_epochs):
                train_loss = self.train_epoch(
                    train_loader, optimizer, criterion, apply_mask=True
                )
                val_metrics = self.evaluate(val_loader, criterion)
                
                retrain_losses.append(train_loss)
                print(f"  Epoch {epoch + 1}/{retrain_epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val PPL = {val_metrics['perplexity']:.2f}")
            
            # Evaluate after retraining
            print(f"\nEvaluating after retraining...")
            post_retrain_metrics = self.evaluate(val_loader, criterion)
            print(f"Post-retrain perplexity: {post_retrain_metrics['perplexity']:.2f}")
            print(f"Recovery: {post_prune_metrics['perplexity'] / post_retrain_metrics['perplexity']:.2f}×")
            
            # Record iteration
            iteration_record = {
                'iteration': iteration,
                'prune_stats': prune_stats,
                'baseline_perplexity': baseline_metrics['perplexity'],
                'post_prune_perplexity': post_prune_metrics['perplexity'],
                'post_retrain_perplexity': post_retrain_metrics['perplexity'],
                'retrain_losses': retrain_losses,
                'sparsity': prune_stats.get('avg_sparsity', 0.0)
            }
            self.training_history.append(iteration_record)
            
            # Save checkpoint
            if save_dir:
                checkpoint_path = Path(save_dir) / f'pruned_iter_{iteration + 1}.pt'
                self._save_checkpoint(checkpoint_path, iteration_record)
        
        # Final summary
        final_metrics = self._compute_final_metrics(baseline_metrics)
        self._print_summary(final_metrics)
        
        return final_metrics
    
    def _compute_final_metrics(self, baseline_metrics: Dict) -> Dict:
        """
        Compute final metrics after all iterations.
        
        Args:
            baseline_metrics: Baseline evaluation metrics
        
        Returns:
            Dictionary with final metrics
        """
        if not self.training_history:
            return {}
        
        final_iteration = self.training_history[-1]
        pruning_summary = self.pruner.get_pruning_summary()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
        actual_sparsity = zero_params / total_params
        
        metrics = {
            'baseline_perplexity': baseline_metrics['perplexity'],
            'final_perplexity': final_iteration['post_retrain_perplexity'],
            'perplexity_degradation': final_iteration['post_retrain_perplexity'] / baseline_metrics['perplexity'],
            'target_sparsity': self.pruner.final_sparsity,
            'achieved_sparsity': actual_sparsity,
            'total_params': total_params,
            'zero_params': zero_params,
            'active_params': total_params - zero_params,
            'compression_ratio': total_params / (total_params - zero_params),
            'num_iterations': len(self.training_history),
            'pruning_summary': pruning_summary,
            'iteration_history': self.training_history
        }
        
        return metrics
    
    def _print_summary(self, metrics: Dict):
        """
        Print final summary.
        
        Args:
            metrics: Final metrics dictionary
        """
        print(f"\n{'='*60}")
        print(f"ITERATIVE PRUNING SUMMARY")
        print(f"{'='*60}")
        print(f"Baseline perplexity: {metrics['baseline_perplexity']:.2f}")
        print(f"Final perplexity: {metrics['final_perplexity']:.2f}")
        print(f"Perplexity degradation: {metrics['perplexity_degradation']:.2%}")
        print(f"\nSparsity:")
        print(f"  Target: {metrics['target_sparsity']:.1%}")
        print(f"  Achieved: {metrics['achieved_sparsity']:.1%}")
        print(f"\nParameters:")
        print(f"  Total: {metrics['total_params']:,}")
        print(f"  Zero: {metrics['zero_params']:,}")
        print(f"  Active: {metrics['active_params']:,}")
        print(f"  Compression ratio: {metrics['compression_ratio']:.2f}×")
        print(f"\nIterations: {metrics['num_iterations']}")
        print(f"{'='*60}")
    
    def _save_checkpoint(self, path: Path, metrics: Dict):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            metrics: Metrics to save with checkpoint
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pruning_masks': self.pruner.pruner.masks,
            'metrics': metrics
        }, path)
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.pruner.pruner.masks = checkpoint['pruning_masks']
        print(f"Loaded checkpoint: {path}")
        return checkpoint.get('metrics', {})


def create_iterative_pruning_trainer(model: nn.Module,
                                     initial_sparsity: float = 0.2,
                                     final_sparsity: float = 0.8,
                                     num_iterations: int = 5,
                                     target_layers: Optional[List[str]] = None,
                                     device: str = 'cuda') -> IterativePruningTrainer:
    """
    Factory function to create iterative pruning trainer.
    
    Args:
        model: Model to prune
        initial_sparsity: Starting sparsity
        final_sparsity: Target final sparsity
        num_iterations: Number of prune-retrain cycles
        target_layers: Layer patterns to prune (default: ['output_proj', 'fc'])
        device: Device to train on
    
    Returns:
        Configured IterativePruningTrainer
    """
    # Default to output_proj and fc layers as per requirements
    if target_layers is None:
        target_layers = ['output_proj', 'fc']
    
    trainer = IterativePruningTrainer(
        model=model,
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        num_iterations=num_iterations,
        prune_layers=target_layers,
        device=device
    )
    
    return trainer
