"""
Model Size Scaling Experiments
Train ResNet-BK models with different d_model and n_layers configurations.

This script implements task 9.7:
- Train models with d_model ∈ {64, 128, 256, 512}
- Train models with n_layers ∈ {4, 8, 12, 16}
- Measure scaling laws
- Analyze perplexity vs model size relationships
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple, Optional
import itertools

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.configurable_resnet_bk import ConfigurableResNetBK, FULL_CONFIG
from src.utils import get_data_loader
from src.benchmarks.flops_counter import FLOPsCounter


@dataclass
class ScalingConfig:
    """Configuration for scaling experiment."""
    d_model: int
    n_layers: int
    n_seq: int = 128
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 0.5
    device: str = 'cuda'
    seed: int = 42
    
    # Optimization flags (use baseline for fair comparison)
    use_analytic_gradient: bool = True
    use_mixed_precision: bool = False
    
    def get_model_name(self) -> str:
        """Generate model name from configuration."""
        return f"d{self.d_model}_l{self.n_layers}"
    
    def get_num_params(self, vocab_size: int) -> int:
        """Estimate number of parameters."""
        # Token embedding: vocab_size * d_model
        # Position embedding: n_seq * d_model
        # Per layer: ~4 * d_model^2 (MoE + BK-Core + projections)
        # LM head: d_model * vocab_size
        
        token_emb = vocab_size * self.d_model
        pos_emb = self.n_seq * self.d_model
        layers = self.n_layers * 4 * self.d_model * self.d_model
        lm_head = self.d_model * vocab_size
        
        return token_emb + pos_emb + layers + lm_head


@dataclass
class ScalingResults:
    """Results from a scaling experiment."""
    d_model: int
    n_layers: int
    num_params: int
    
    # Training metrics
    final_loss: float
    final_perplexity: float
    best_perplexity: float
    training_time: float
    
    # FLOPs metrics
    total_flops_per_step: int
    total_training_flops: int
    
    # Memory metrics
    peak_memory_mb: float
    model_size_mb: float
    
    # Per-epoch metrics
    epoch_perplexities: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ScalingExperiments:
    """
    Scaling experiments for ResNet-BK.
    
    Trains models with different d_model and n_layers configurations
    to measure scaling laws.
    """
    
    def __init__(self, output_dir: str = "benchmark_results/scaling"):
        """
        Initialize scaling experiments.
        
        Args:
            output_dir: directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[ScalingResults] = []
    
    def run_experiment(self, config: ScalingConfig) -> ScalingResults:
        """
        Run a single scaling experiment.
        
        Args:
            config: scaling configuration
        
        Returns:
            ScalingResults object
        """
        model_name = config.get_model_name()
        print("=" * 80)
        print(f"Scaling Experiment: {model_name}")
        print(f"  d_model={config.d_model}, n_layers={config.n_layers}")
        print("=" * 80)
        
        # Set random seed
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # Set device
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Load data
        print("\nLoading WikiText-2 dataset...")
        train_data, vocab, get_batch = get_data_loader(
            batch_size=config.batch_size,
            n_seq=config.n_seq,
            dataset_name='wikitext-2',
            data_limit=None
        )
        
        if train_data is None:
            raise ValueError("Failed to load WikiText-2 dataset")
        
        vocab_size = vocab['vocab_size']
        print(f"Vocabulary size: {vocab_size}")
        
        # Create model
        print("\nCreating model...")
        model = self._create_model(config, vocab_size)
        model = model.to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Count FLOPs
        print("\nCounting FLOPs...")
        try:
            flops_counter = FLOPsCounter(model, config.batch_size, config.n_seq)
            flops_per_step = flops_counter.count_total_flops()
            total_flops_per_step = flops_per_step.total
            print(f"FLOPs per step: {total_flops_per_step/1e9:.2f} GFLOPs")
        except Exception as e:
            print(f"Warning: FLOPs counting failed: {e}")
            total_flops_per_step = 0
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        num_steps_per_epoch = train_data.size(0) // config.n_seq
        num_total_steps = num_steps_per_epoch * config.epochs
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_total_steps,
            eta_min=config.lr / 10
        )
        
        print(f"\nTotal training steps: {num_total_steps}")
        
        # Training loop
        print("\nStarting training...")
        model.train()
        
        epoch_perplexities = []
        best_perplexity = float('inf')
        
        # Track memory
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        
        training_start = time.time()
        
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, train_data.size(0) - 1, config.n_seq):
                x_batch, y_batch = get_batch(train_data, i)
                x_batch = x_batch.t().contiguous()
                
                if x_batch.size(1) != config.n_seq:
                    continue
                
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                if config.use_mixed_precision and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        logits = model(x_batch)
                        loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                else:
                    logits = model(x_batch)
                    loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
                
                # Skip if loss is NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss at epoch {epoch}, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / max(1, num_batches)
            perplexity = math.exp(min(avg_loss, 20))
            
            epoch_perplexities.append(perplexity)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
            
            print(f"Epoch {epoch}/{config.epochs}: "
                  f"Loss={avg_loss:.4f}, PPL={perplexity:.2f}, "
                  f"Time={epoch_time:.1f}s")
        
        training_time = time.time() - training_start
        
        # Get memory stats
        if device.type == 'cuda':
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        else:
            peak_memory_mb = 0.0
        
        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        # Calculate total training FLOPs
        total_training_flops = total_flops_per_step * num_total_steps
        
        # Create results
        results = ScalingResults(
            d_model=config.d_model,
            n_layers=config.n_layers,
            num_params=num_params,
            final_loss=avg_loss,
            final_perplexity=epoch_perplexities[-1],
            best_perplexity=best_perplexity,
            training_time=training_time,
            total_flops_per_step=total_flops_per_step,
            total_training_flops=total_training_flops,
            peak_memory_mb=peak_memory_mb,
            model_size_mb=model_size_mb,
            epoch_perplexities=epoch_perplexities,
        )
        
        self.results.append(results)
        
        # Save individual result
        result_file = self.output_dir / f"{model_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        print("\n" + "=" * 80)
        print(f"Experiment Complete: {model_name}")
        print(f"Parameters: {num_params:,}")
        print(f"Final Perplexity: {results.final_perplexity:.2f}")
        print(f"Best Perplexity: {results.best_perplexity:.2f}")
        print(f"Training Time: {results.training_time:.1f}s")
        print("=" * 80 + "\n")
        
        return results
    
    def _create_model(self, config: ScalingConfig, vocab_size: int) -> nn.Module:
        """
        Create ResNet-BK model with specified configuration.
        
        Args:
            config: scaling configuration
            vocab_size: vocabulary size
        
        Returns:
            model instance
        """
        # Use dataclasses.replace to create a modified copy
        model_config = replace(
            FULL_CONFIG,
            vocab_size=vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_seq=config.n_seq,
            # Use baseline configuration for fair comparison
            use_analytic_gradient=config.use_analytic_gradient,
            use_mixed_precision=config.use_mixed_precision,
            use_koopman=False,
            use_physics_informed=False,
            use_quantization=False,
            use_pruning=False,
            use_adaptive_computation=False,
            use_multi_scale=False,
            use_learned_sparsity=False,
        )
        
        return ConfigurableResNetBK(model_config)
    
    def run_all_experiments(
        self,
        d_model_values: List[int] = [64, 128, 256, 512],
        n_layers_values: List[int] = [4, 8, 12, 16],
        n_seq: int = 128,
        batch_size: int = 32,
        epochs: int = 5,
        device: str = 'cuda'
    ):
        """
        Run scaling experiments for all combinations of d_model and n_layers.
        
        Args:
            d_model_values: list of d_model values to test
            n_layers_values: list of n_layers values to test
            n_seq: sequence length
            batch_size: batch size
            epochs: number of training epochs
            device: device to use
        """
        print("=" * 80)
        print("SCALING EXPERIMENTS")
        print("=" * 80)
        print(f"d_model values: {d_model_values}")
        print(f"n_layers values: {n_layers_values}")
        print(f"Total experiments: {len(d_model_values) * len(n_layers_values)}")
        print("=" * 80 + "\n")
        
        # Run experiments for all combinations
        experiment_num = 0
        total_experiments = len(d_model_values) * len(n_layers_values)
        
        for d_model in d_model_values:
            for n_layers in n_layers_values:
                experiment_num += 1
                print(f"\n[{experiment_num}/{total_experiments}] Running experiment: "
                      f"d_model={d_model}, n_layers={n_layers}")
                
                config = ScalingConfig(
                    d_model=d_model,
                    n_layers=n_layers,
                    n_seq=n_seq,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=device,
                )
                
                try:
                    self.run_experiment(config)
                except Exception as e:
                    print(f"Error in experiment: {e}")
                    print("Continuing with next experiment...")
                    continue
        
        # Save all results
        self.save_all_results()
        
        # Analyze scaling laws
        self.analyze_scaling_laws()
        
        # Generate plots
        if HAS_MATPLOTLIB:
            self.plot_scaling_laws()
    
    def save_all_results(self):
        """Save all results to a single JSON file."""
        all_results = [r.to_dict() for r in self.results]
        
        results_file = self.output_dir / "all_scaling_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nAll results saved to {results_file}")
    
    def analyze_scaling_laws(self):
        """
        Analyze scaling laws: perplexity vs model size.
        
        Fits power law: perplexity = a * (num_params)^b
        """
        if len(self.results) == 0:
            print("No results to analyze")
            return
        
        print("\n" + "=" * 80)
        print("SCALING LAW ANALYSIS")
        print("=" * 80)
        
        # Extract data
        num_params = np.array([r.num_params for r in self.results])
        perplexities = np.array([r.best_perplexity for r in self.results])
        
        # Sort by num_params
        sort_idx = np.argsort(num_params)
        num_params = num_params[sort_idx]
        perplexities = perplexities[sort_idx]
        
        # Print table
        print("\nModel Size vs Perplexity:")
        print(f"{'Parameters':<15} {'d_model':<10} {'n_layers':<10} {'Perplexity':<12}")
        print("-" * 50)
        for r in sorted(self.results, key=lambda x: x.num_params):
            print(f"{r.num_params:<15,} {r.d_model:<10} {r.n_layers:<10} {r.best_perplexity:<12.2f}")
        
        # Fit power law if we have enough data points
        if len(num_params) >= 3:
            try:
                # Power law: perplexity = a * (num_params)^b
                def power_law(x, a, b):
                    return a * np.power(x, b)
                
                # Fit
                popt, pcov = curve_fit(
                    power_law,
                    num_params,
                    perplexities,
                    p0=[1000, -0.1],
                    maxfev=10000
                )
                
                a, b = popt
                
                print(f"\nScaling Law Fit:")
                print(f"  perplexity = {a:.2f} * (num_params)^{b:.4f}")
                
                # Compute R²
                predictions = power_law(num_params, a, b)
                ss_res = np.sum((perplexities - predictions) ** 2)
                ss_tot = np.sum((perplexities - np.mean(perplexities)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(f"  R² = {r_squared:.4f}")
                
                # Save scaling law
                scaling_law = {
                    'formula': f'perplexity = {a:.2f} * (num_params)^{b:.4f}',
                    'a': float(a),
                    'b': float(b),
                    'r_squared': float(r_squared),
                    'num_data_points': len(num_params),
                }
                
                scaling_law_file = self.output_dir / "scaling_law.json"
                with open(scaling_law_file, 'w') as f:
                    json.dump(scaling_law, f, indent=2)
                
                print(f"\nScaling law saved to {scaling_law_file}")
                
            except Exception as e:
                print(f"\nWarning: Could not fit scaling law: {e}")
        
        print("=" * 80 + "\n")
    
    def plot_scaling_laws(self):
        """Generate plots for scaling law analysis."""
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        if len(self.results) == 0:
            print("No results to plot")
            return
        
        # Extract data
        num_params = np.array([r.num_params for r in self.results])
        perplexities = np.array([r.best_perplexity for r in self.results])
        d_models = np.array([r.d_model for r in self.results])
        n_layers_list = np.array([r.n_layers for r in self.results])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Perplexity vs Model Size (log-log)
        ax = axes[0, 0]
        ax.scatter(num_params, perplexities, s=100, alpha=0.6)
        
        # Add labels for each point
        for r in self.results:
            ax.annotate(
                f'd{r.d_model}l{r.n_layers}',
                (r.num_params, r.best_perplexity),
                fontsize=8,
                alpha=0.7
            )
        
        # Fit power law
        if len(num_params) >= 3:
            try:
                def power_law(x, a, b):
                    return a * np.power(x, b)
                
                popt, _ = curve_fit(power_law, num_params, perplexities, p0=[1000, -0.1], maxfev=10000)
                a, b = popt
                
                x_fit = np.logspace(np.log10(num_params.min()), np.log10(num_params.max()), 100)
                y_fit = power_law(x_fit, a, b)
                ax.plot(x_fit, y_fit, 'r--', label=f'Fit: {a:.0f}*N^{b:.3f}', linewidth=2)
                ax.legend()
            except:
                pass
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Perplexity')
        ax.set_title('Scaling Law: Perplexity vs Model Size')
        ax.grid(True, alpha=0.3)
        
        # 2. Perplexity vs d_model (for each n_layers)
        ax = axes[0, 1]
        unique_n_layers = sorted(set(n_layers_list))
        for n_layers in unique_n_layers:
            mask = n_layers_list == n_layers
            d_model_subset = d_models[mask]
            ppl_subset = perplexities[mask]
            
            # Sort by d_model
            sort_idx = np.argsort(d_model_subset)
            d_model_subset = d_model_subset[sort_idx]
            ppl_subset = ppl_subset[sort_idx]
            
            ax.plot(d_model_subset, ppl_subset, marker='o', label=f'n_layers={n_layers}')
        
        ax.set_xlabel('d_model')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity vs d_model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. Perplexity vs n_layers (for each d_model)
        ax = axes[1, 0]
        unique_d_models = sorted(set(d_models))
        for d_model in unique_d_models:
            mask = d_models == d_model
            n_layers_subset = n_layers_list[mask]
            ppl_subset = perplexities[mask]
            
            # Sort by n_layers
            sort_idx = np.argsort(n_layers_subset)
            n_layers_subset = n_layers_subset[sort_idx]
            ppl_subset = ppl_subset[sort_idx]
            
            ax.plot(n_layers_subset, ppl_subset, marker='o', label=f'd_model={d_model}')
        
        ax.set_xlabel('n_layers')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity vs n_layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. Training Time vs Model Size
        ax = axes[1, 1]
        training_times = np.array([r.training_time for r in self.results])
        ax.scatter(num_params, training_times, s=100, alpha=0.6)
        
        # Add labels
        for r in self.results:
            ax.annotate(
                f'd{r.d_model}l{r.n_layers}',
                (r.num_params, r.training_time),
                fontsize=8,
                alpha=0.7
            )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Parameters')
        ax.set_ylabel('Training Time (seconds)')
        ax.set_title('Training Time vs Model Size')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "scaling_laws.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Scaling law plots saved to {plot_file}")
        plt.close()


def main():
    """Run scaling experiments."""
    print("Model Size Scaling Experiments")
    print("=" * 80)
    
    # Create experiments
    experiments = ScalingExperiments(output_dir="benchmark_results/scaling")
    
    # Run all experiments
    # Note: For quick testing, use smaller subsets
    # Full experiments: d_model=[64, 128, 256, 512], n_layers=[4, 8, 12, 16]
    
    # Quick test (comment out for full run)
    # experiments.run_all_experiments(
    #     d_model_values=[64, 128],
    #     n_layers_values=[4, 8],
    #     epochs=3,
    #     device='cuda'
    # )
    
    # Full experiments
    experiments.run_all_experiments(
        d_model_values=[64, 128, 256, 512],
        n_layers_values=[4, 8, 12, 16],
        epochs=5,
        device='cuda'
    )
    
    print("\n" + "=" * 80)
    print("ALL SCALING EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {experiments.output_dir}")


if __name__ == '__main__':
    main()
