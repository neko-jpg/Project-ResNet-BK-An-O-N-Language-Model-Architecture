"""
GRAD_BLEND Grid Search Optimizer
Finds optimal blending coefficient for hybrid analytic gradient.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from ..models.bk_core import BKCoreFunction
from ..utils.metrics import TrainingMetrics


@dataclass
class GradBlendResult:
    """Results for a single GRAD_BLEND value."""
    alpha: float
    final_perplexity: float
    convergence_speed: float  # epochs to reach 90% of final performance
    gradient_variance: float
    training_time: float
    loss_curve: List[float]
    perplexity_curve: List[float]
    
    def to_dict(self):
        return asdict(self)


class GradBlendOptimizer:
    """
    Grid search optimizer for GRAD_BLEND parameter.
    
    Searches over α ∈ [0.0, 0.1, ..., 1.0] to find optimal blending
    between theoretical gradient (dG/dv = -G²) and hypothesis-7 gradient
    (dL/dv ~ -dL/dG / G²).
    
    Args:
        model: ResNet-BK model
        train_loader: training data loader
        val_loader: validation data loader
        alpha_values: list of alpha values to search (default: [0.0, 0.1, ..., 1.0])
        epochs_per_trial: number of epochs to train for each alpha
        device: torch device
        save_dir: directory to save results
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        alpha_values: Optional[List[float]] = None,
        epochs_per_trial: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = 'results/grad_blend_search'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.alpha_values = alpha_values if alpha_values is not None else [i * 0.1 for i in range(11)]
        self.epochs_per_trial = epochs_per_trial
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[GradBlendResult] = []
        self.best_alpha: Optional[float] = None
        self.best_perplexity: float = float('inf')
    
    def run_grid_search(self) -> Dict:
        """
        Run grid search over all alpha values.
        
        Returns:
            summary: dictionary with search results and best alpha
        """
        print("=" * 80)
        print("GRAD_BLEND Grid Search")
        print("=" * 80)
        print(f"Alpha values: {self.alpha_values}")
        print(f"Epochs per trial: {self.epochs_per_trial}")
        print(f"Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        print("=" * 80)
        
        for alpha in self.alpha_values:
            print(f"\n{'='*80}")
            print(f"Testing GRAD_BLEND = {alpha:.1f}")
            print(f"{'='*80}")
            
            result = self._train_with_alpha(alpha)
            self.results.append(result)
            
            # Update best
            if result.final_perplexity < self.best_perplexity:
                self.best_perplexity = result.final_perplexity
                self.best_alpha = alpha
            
            # Save intermediate results
            self._save_results()
            
            print(f"\nResult: PPL={result.final_perplexity:.2f}, "
                  f"Convergence={result.convergence_speed:.2f} epochs, "
                  f"GradVar={result.gradient_variance:.4f}, "
                  f"Time={result.training_time:.1f}s")
        
        # Generate summary
        summary = self._generate_summary()
        
        print("\n" + "=" * 80)
        print("Grid Search Complete!")
        print("=" * 80)
        print(f"Best GRAD_BLEND: {self.best_alpha:.1f}")
        print(f"Best Perplexity: {self.best_perplexity:.2f}")
        print(f"Results saved to: {self.save_dir}")
        print("=" * 80)
        
        return summary
    
    def _train_with_alpha(self, alpha: float) -> GradBlendResult:
        """
        Train model with specific GRAD_BLEND value.
        
        Args:
            alpha: GRAD_BLEND value to test
        
        Returns:
            result: GradBlendResult with metrics
        """
        # Reset model to initial state
        self._reset_model()
        
        # Set GRAD_BLEND
        original_grad_blend = BKCoreFunction.GRAD_BLEND
        BKCoreFunction.GRAD_BLEND = alpha
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        loss_curve = []
        perplexity_curve = []
        gradient_variances = []
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.epochs_per_trial):
            # Train epoch
            train_loss, train_ppl, grad_var = self._train_epoch(
                optimizer, criterion
            )
            
            # Validation
            val_loss, val_ppl = self._validate(criterion)
            
            loss_curve.append(val_loss)
            perplexity_curve.append(val_ppl)
            gradient_variances.append(grad_var)
            
            print(f"Epoch {epoch+1}/{self.epochs_per_trial}: "
                  f"Train Loss={train_loss:.4f}, Train PPL={train_ppl:.2f}, "
                  f"Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}, "
                  f"GradVar={grad_var:.4f}")
        
        training_time = time.time() - start_time
        
        # Restore original GRAD_BLEND
        BKCoreFunction.GRAD_BLEND = original_grad_blend
        
        # Compute metrics
        final_perplexity = perplexity_curve[-1]
        convergence_speed = self._compute_convergence_speed(perplexity_curve)
        gradient_variance = np.mean(gradient_variances)
        
        result = GradBlendResult(
            alpha=alpha,
            final_perplexity=final_perplexity,
            convergence_speed=convergence_speed,
            gradient_variance=gradient_variance,
            training_time=training_time,
            loss_curve=loss_curve,
            perplexity_curve=perplexity_curve
        )
        
        return result
    
    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Returns:
            avg_loss: average training loss
            avg_ppl: average perplexity
            grad_var: gradient variance
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        gradient_norms = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(self.train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Track gradient norms
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            gradient_norms.append(grad_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            optimizer.step()
            
            total_loss += loss.item() * y_batch.numel()
            total_tokens += y_batch.numel()
        
        avg_loss = total_loss / total_tokens
        avg_ppl = np.exp(avg_loss)
        grad_var = np.var(gradient_norms)
        
        return avg_loss, avg_ppl, grad_var
    
    def _validate(self, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            avg_loss: average validation loss
            avg_ppl: average perplexity
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits = self.model(x_batch)
                loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                
                total_loss += loss.item() * y_batch.numel()
                total_tokens += y_batch.numel()
        
        avg_loss = total_loss / total_tokens
        avg_ppl = np.exp(avg_loss)
        
        return avg_loss, avg_ppl
    
    def _compute_convergence_speed(self, perplexity_curve: List[float]) -> float:
        """
        Compute convergence speed: epochs to reach 90% of final performance.
        
        Args:
            perplexity_curve: list of perplexity values per epoch
        
        Returns:
            convergence_speed: number of epochs to reach 90% performance
        """
        if len(perplexity_curve) < 2:
            return float(len(perplexity_curve))
        
        initial_ppl = perplexity_curve[0]
        final_ppl = perplexity_curve[-1]
        
        # Target: 90% improvement from initial to final
        target_ppl = initial_ppl - 0.9 * (initial_ppl - final_ppl)
        
        # Find first epoch that reaches target
        for epoch, ppl in enumerate(perplexity_curve):
            if ppl <= target_ppl:
                return float(epoch + 1)
        
        # If never reached, return total epochs
        return float(len(perplexity_curve))
    
    def _reset_model(self):
        """Reset model to initial state."""
        # Re-initialize model parameters
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        self.model.apply(init_weights)
        self.model.to(self.device)
    
    def _save_results(self):
        """Save results to disk."""
        # Save individual results
        results_dict = {
            'alpha_values': self.alpha_values,
            'epochs_per_trial': self.epochs_per_trial,
            'results': [r.to_dict() for r in self.results],
            'best_alpha': self.best_alpha,
            'best_perplexity': self.best_perplexity
        }
        
        with open(self.save_dir / 'grad_blend_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save training curves
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Perplexity curves
        ax = axes[0, 0]
        for result in self.results:
            ax.plot(result.perplexity_curve, label=f'α={result.alpha:.1f}', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Perplexity')
        ax.set_title('Perplexity vs Epoch for Different GRAD_BLEND')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Final perplexity vs alpha
        ax = axes[0, 1]
        alphas = [r.alpha for r in self.results]
        final_ppls = [r.final_perplexity for r in self.results]
        ax.plot(alphas, final_ppls, marker='o', linewidth=2)
        ax.axvline(self.best_alpha, color='r', linestyle='--', label=f'Best α={self.best_alpha:.1f}')
        ax.set_xlabel('GRAD_BLEND (α)')
        ax.set_ylabel('Final Validation Perplexity')
        ax.set_title('Final Perplexity vs GRAD_BLEND')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: Convergence speed vs alpha
        ax = axes[1, 0]
        conv_speeds = [r.convergence_speed for r in self.results]
        ax.plot(alphas, conv_speeds, marker='o', linewidth=2, color='green')
        ax.set_xlabel('GRAD_BLEND (α)')
        ax.set_ylabel('Convergence Speed (epochs)')
        ax.set_title('Convergence Speed vs GRAD_BLEND')
        ax.grid(True)
        
        # Plot 4: Gradient variance vs alpha
        ax = axes[1, 1]
        grad_vars = [r.gradient_variance for r in self.results]
        ax.plot(alphas, grad_vars, marker='o', linewidth=2, color='orange')
        ax.set_xlabel('GRAD_BLEND (α)')
        ax.set_ylabel('Gradient Variance')
        ax.set_title('Gradient Variance vs GRAD_BLEND')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'grad_blend_analysis.png', dpi=150)
        plt.close()
        
        print(f"Results saved to {self.save_dir}")
    
    def _generate_summary(self) -> Dict:
        """Generate summary of grid search results."""
        summary = {
            'best_alpha': self.best_alpha,
            'best_perplexity': self.best_perplexity,
            'num_trials': len(self.results),
            'alpha_range': [min(self.alpha_values), max(self.alpha_values)],
            'results_summary': []
        }
        
        for result in self.results:
            summary['results_summary'].append({
                'alpha': result.alpha,
                'final_perplexity': result.final_perplexity,
                'convergence_speed': result.convergence_speed,
                'gradient_variance': result.gradient_variance,
                'training_time': result.training_time
            })
        
        # Save summary
        with open(self.save_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def load_results(self, results_path: str):
        """Load previously saved results."""
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        self.alpha_values = data['alpha_values']
        self.epochs_per_trial = data['epochs_per_trial']
        self.best_alpha = data['best_alpha']
        self.best_perplexity = data['best_perplexity']
        
        self.results = []
        for r in data['results']:
            self.results.append(GradBlendResult(**r))
        
        print(f"Loaded {len(self.results)} results from {results_path}")
        print(f"Best alpha: {self.best_alpha:.1f}, Best perplexity: {self.best_perplexity:.2f}")
