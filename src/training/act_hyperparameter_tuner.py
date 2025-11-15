"""
ACT Hyperparameter Tuning
Grid search over halting threshold and lambda_act to find optimal configuration.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
import time
from pathlib import Path

from ..models.adaptive_computation import ACTLanguageModel, ACTTrainer
from ..utils.data_utils import get_wikitext2_dataloaders


class ACTHyperparameterTuner:
    """
    Grid search tuner for ACT hyperparameters.
    
    Searches over:
    - act_threshold: halting probability threshold (e.g., [0.5, 0.8, 0.9, 0.95, 0.99])
    - act_lambda: ponder cost weight (e.g., [0.001, 0.005, 0.01, 0.05, 0.1])
    
    Metrics tracked:
    - Validation perplexity
    - Average layers executed
    - Training time
    - Convergence speed
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_layers: int = 4,
        n_seq: int = 128,
        num_experts: int = 4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_seq = n_seq
        self.num_experts = num_experts
        self.device = device
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_score = float('inf')
    
    def create_model(self, act_threshold: float, act_lambda: float) -> ACTLanguageModel:
        """Create ACT model with specified hyperparameters."""
        model = ACTLanguageModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_seq=self.n_seq,
            num_experts=self.num_experts,
            act_threshold=act_threshold,
            act_lambda=act_lambda
        ).to(self.device)
        return model
    
    def train_and_evaluate(
        self,
        model: ACTLanguageModel,
        train_loader,
        val_loader,
        num_epochs: int = 3,
        lr: float = 1e-3
    ) -> Dict:
        """
        Train model and evaluate performance.
        
        Returns:
            metrics: dict with validation perplexity, avg layers, training time
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = ACTTrainer(model, optimizer, device=self.device)
        
        start_time = time.time()
        
        # Training loop
        train_losses = []
        val_perplexities = []
        avg_layers_history = []
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            epoch_layers = 0.0
            num_batches = 0
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                metrics = trainer.train_step(x_batch, y_batch)
                epoch_loss += metrics['total_loss']
                epoch_layers += metrics['avg_layers_executed']
                num_batches += 1
                
                # Limit batches for faster tuning
                if batch_idx >= 50:  # Only use 50 batches per epoch for tuning
                    break
            
            avg_train_loss = epoch_loss / num_batches
            avg_layers = epoch_layers / num_batches
            train_losses.append(avg_train_loss)
            avg_layers_history.append(avg_layers)
            
            # Validation
            val_perplexity = self.evaluate(model, val_loader)
            val_perplexities.append(val_perplexity)
        
        training_time = time.time() - start_time
        
        # Compute convergence speed (improvement from epoch 0 to final)
        convergence_speed = val_perplexities[0] - val_perplexities[-1] if len(val_perplexities) > 1 else 0.0
        
        return {
            'final_val_perplexity': val_perplexities[-1],
            'avg_layers_executed': avg_layers_history[-1],
            'training_time': training_time,
            'convergence_speed': convergence_speed,
            'val_perplexity_history': val_perplexities,
            'avg_layers_history': avg_layers_history,
            'train_loss_history': train_losses
        }
    
    def evaluate(self, model: ACTLanguageModel, val_loader) -> float:
        """
        Evaluate model on validation set.
        
        Returns:
            perplexity: validation perplexity
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (x_batch, y_batch) in enumerate(val_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                logits, ponder_cost = model(x_batch, return_ponder_cost=True)
                _, ce_loss, _ = model.compute_loss(logits, y_batch, ponder_cost)
                
                total_loss += ce_loss.item()
                num_batches += 1
                
                # Limit batches for faster evaluation
                if batch_idx >= 20:  # Only use 20 batches for validation
                    break
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def grid_search(
        self,
        train_loader,
        val_loader,
        threshold_values: List[float] = [0.5, 0.8, 0.9, 0.95, 0.99],
        lambda_values: List[float] = [0.001, 0.005, 0.01, 0.05, 0.1],
        num_epochs: int = 3,
        lr: float = 1e-3,
        score_metric: str = 'balanced'  # 'perplexity', 'layers', or 'balanced'
    ) -> Dict:
        """
        Perform grid search over ACT hyperparameters.
        
        Args:
            train_loader: training data loader
            val_loader: validation data loader
            threshold_values: list of halting thresholds to try
            lambda_values: list of ponder cost weights to try
            num_epochs: number of training epochs per configuration
            lr: learning rate
            score_metric: how to score configurations
                - 'perplexity': minimize validation perplexity
                - 'layers': minimize average layers executed
                - 'balanced': balance perplexity and computational cost
        
        Returns:
            results: dict with best configuration and all results
        """
        print("=" * 70)
        print("ACT Hyperparameter Grid Search")
        print("=" * 70)
        print(f"Threshold values: {threshold_values}")
        print(f"Lambda values: {lambda_values}")
        print(f"Total configurations: {len(threshold_values) * len(lambda_values)}")
        print(f"Epochs per config: {num_epochs}")
        print(f"Score metric: {score_metric}")
        print("=" * 70)
        
        total_configs = len(threshold_values) * len(lambda_values)
        config_idx = 0
        
        for threshold in threshold_values:
            for lambda_val in lambda_values:
                config_idx += 1
                print(f"\n[{config_idx}/{total_configs}] Testing threshold={threshold}, lambda={lambda_val}")
                
                # Create and train model
                model = self.create_model(threshold, lambda_val)
                
                try:
                    metrics = self.train_and_evaluate(
                        model, train_loader, val_loader, num_epochs, lr
                    )
                    
                    # Compute score based on metric
                    if score_metric == 'perplexity':
                        score = metrics['final_val_perplexity']
                    elif score_metric == 'layers':
                        score = metrics['avg_layers_executed']
                    else:  # balanced
                        # Normalize and combine: lower is better
                        # Score = perplexity + 10 * avg_layers (weight layers less)
                        score = metrics['final_val_perplexity'] + 10.0 * metrics['avg_layers_executed']
                    
                    # Store results
                    result = {
                        'threshold': threshold,
                        'lambda': lambda_val,
                        'score': score,
                        **metrics
                    }
                    self.results.append(result)
                    
                    # Update best configuration
                    if score < self.best_score:
                        self.best_score = score
                        self.best_config = {
                            'threshold': threshold,
                            'lambda': lambda_val
                        }
                        print(f"  ✓ New best configuration! Score: {score:.4f}")
                    
                    # Print metrics
                    print(f"  Validation Perplexity: {metrics['final_val_perplexity']:.2f}")
                    print(f"  Avg Layers Executed: {metrics['avg_layers_executed']:.2f} / {self.n_layers}")
                    print(f"  Training Time: {metrics['training_time']:.1f}s")
                    print(f"  Score: {score:.4f}")
                    
                except Exception as e:
                    print(f"  ✗ Configuration failed: {e}")
                    continue
        
        # Summary
        print("\n" + "=" * 70)
        print("Grid Search Complete!")
        print("=" * 70)
        print(f"Best Configuration:")
        print(f"  Threshold: {self.best_config['threshold']}")
        print(f"  Lambda: {self.best_config['lambda']}")
        print(f"  Score: {self.best_score:.4f}")
        
        # Find best result details
        best_result = next(r for r in self.results if 
                          r['threshold'] == self.best_config['threshold'] and 
                          r['lambda'] == self.best_config['lambda'])
        
        print(f"\nBest Configuration Metrics:")
        print(f"  Validation Perplexity: {best_result['final_val_perplexity']:.2f}")
        print(f"  Avg Layers Executed: {best_result['avg_layers_executed']:.2f} / {self.n_layers}")
        print(f"  Speedup Potential: {self.n_layers / best_result['avg_layers_executed']:.2f}x")
        print(f"  Training Time: {best_result['training_time']:.1f}s")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'best_result': best_result,
            'all_results': self.results
        }
    
    def save_results(self, filepath: str):
        """Save grid search results to JSON file."""
        results_dict = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': self.results,
            'model_config': {
                'vocab_size': self.vocab_size,
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'n_seq': self.n_seq,
                'num_experts': self.num_experts
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot grid search results as heatmap.
        
        Args:
            save_path: if provided, save plot to this path
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not available for plotting")
            return
        
        # Extract unique threshold and lambda values
        thresholds = sorted(list(set(r['threshold'] for r in self.results)))
        lambdas = sorted(list(set(r['lambda'] for r in self.results)))
        
        # Create matrices for different metrics
        perplexity_matrix = np.zeros((len(thresholds), len(lambdas)))
        layers_matrix = np.zeros((len(thresholds), len(lambdas)))
        score_matrix = np.zeros((len(thresholds), len(lambdas)))
        
        for result in self.results:
            i = thresholds.index(result['threshold'])
            j = lambdas.index(result['lambda'])
            perplexity_matrix[i, j] = result['final_val_perplexity']
            layers_matrix[i, j] = result['avg_layers_executed']
            score_matrix[i, j] = result['score']
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Validation Perplexity
        im1 = axes[0].imshow(perplexity_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[0].set_xticks(range(len(lambdas)))
        axes[0].set_yticks(range(len(thresholds)))
        axes[0].set_xticklabels([f'{l:.3f}' for l in lambdas])
        axes[0].set_yticklabels([f'{t:.2f}' for t in thresholds])
        axes[0].set_xlabel('Lambda (ponder cost weight)')
        axes[0].set_ylabel('Threshold')
        axes[0].set_title('Validation Perplexity (lower is better)')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Average Layers Executed
        im2 = axes[1].imshow(layers_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[1].set_xticks(range(len(lambdas)))
        axes[1].set_yticks(range(len(thresholds)))
        axes[1].set_xticklabels([f'{l:.3f}' for l in lambdas])
        axes[1].set_yticklabels([f'{t:.2f}' for t in thresholds])
        axes[1].set_xlabel('Lambda (ponder cost weight)')
        axes[1].set_ylabel('Threshold')
        axes[1].set_title('Avg Layers Executed (lower is better)')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot 3: Combined Score
        im3 = axes[2].imshow(score_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[2].set_xticks(range(len(lambdas)))
        axes[2].set_yticks(range(len(thresholds)))
        axes[2].set_xticklabels([f'{l:.3f}' for l in lambdas])
        axes[2].set_yticklabels([f'{t:.2f}' for t in thresholds])
        axes[2].set_xlabel('Lambda (ponder cost weight)')
        axes[2].set_ylabel('Threshold')
        axes[2].set_title('Combined Score (lower is better)')
        plt.colorbar(im3, ax=axes[2])
        
        # Mark best configuration
        best_i = thresholds.index(self.best_config['threshold'])
        best_j = lambdas.index(self.best_config['lambda'])
        for ax in axes:
            ax.plot(best_j, best_i, 'w*', markersize=20, markeredgecolor='black', markeredgewidth=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


def run_act_hyperparameter_tuning(
    data_path: str = 'data',
    batch_size: int = 32,
    num_epochs: int = 3,
    output_dir: str = 'results/act_tuning'
):
    """
    Run ACT hyperparameter tuning on WikiText-2.
    
    Args:
        data_path: path to data directory
        batch_size: batch size for training
        num_epochs: number of epochs per configuration
        output_dir: directory to save results
    """
    print("Loading WikiText-2 dataset...")
    train_loader, val_loader, test_loader, vocab_size = get_wikitext2_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        n_seq=128
    )
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create tuner
    tuner = ACTHyperparameterTuner(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=4,
        n_seq=128,
        num_experts=4
    )
    
    # Run grid search
    results = tuner.grid_search(
        train_loader=train_loader,
        val_loader=val_loader,
        threshold_values=[0.5, 0.8, 0.9, 0.95, 0.99],
        lambda_values=[0.001, 0.005, 0.01, 0.05, 0.1],
        num_epochs=num_epochs,
        score_metric='balanced'
    )
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tuner.save_results(f'{output_dir}/act_tuning_results.json')
    tuner.plot_results(save_path=f'{output_dir}/act_tuning_heatmap.png')
    
    return results


if __name__ == '__main__':
    # Run hyperparameter tuning
    results = run_act_hyperparameter_tuning(
        data_path='data',
        batch_size=32,
        num_epochs=3,
        output_dir='results/act_tuning'
    )
    
    print("\n" + "=" * 70)
    print("ACT Hyperparameter Tuning Complete!")
    print("=" * 70)
    print(f"\nRecommended Configuration:")
    print(f"  act_threshold = {results['best_config']['threshold']}")
    print(f"  act_lambda = {results['best_config']['lambda']}")
