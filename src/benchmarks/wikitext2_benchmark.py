"""
WikiText-2 Comprehensive Benchmark
Train ResNet-BK with all optimizations enabled and compare to Transformer baseline.

This script implements task 9.2:
- Train with all optimizations enabled
- Measure final perplexity
- Compare to Transformer baseline
- Track FLOPs, wall-clock time, memory usage
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import math
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.configurable_resnet_bk import ConfigurableResNetBK, FULL_CONFIG
from src.utils import get_data_loader, TrainingMetrics, MetricsLogger
from src.benchmarks.flops_counter import FLOPsCounter


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    model_name: str
    d_model: int
    n_layers: int
    n_seq: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    device: str
    seed: int
    
    # Optimization flags
    use_analytic_gradient: bool = True
    use_koopman: bool = False
    use_physics_informed: bool = False
    use_quantization: bool = False
    use_pruning: bool = False
    use_mixed_precision: bool = False
    use_act: bool = False
    use_multi_scale: bool = False
    use_sparse_bk: bool = False
    use_early_exit: bool = False
    use_curriculum: bool = False
    use_active_learning: bool = False


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""
    model_name: str
    config: Dict
    
    # Training metrics
    final_loss: float
    final_perplexity: float
    best_perplexity: float
    training_time: float  # seconds
    
    # FLOPs metrics
    forward_flops: int
    backward_flops: int
    optimizer_flops: int
    total_flops_per_step: int
    total_training_flops: int
    
    # Memory metrics
    peak_memory_mb: float
    model_size_mb: float
    
    # Per-epoch metrics
    epoch_losses: List[float]
    epoch_perplexities: List[float]
    epoch_times: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {filepath}")


class WikiText2Benchmark:
    """
    Comprehensive benchmark for WikiText-2 dataset.
    """
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize benchmark.
        
        Args:
            output_dir: directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, BenchmarkResults] = {}
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResults:
        """
        Run a single benchmark with given configuration.
        
        Args:
            config: benchmark configuration
        
        Returns:
            BenchmarkResults object
        """
        print("=" * 80)
        print(f"Running Benchmark: {config.model_name}")
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
        
        print(f"Vocabulary size: {vocab['vocab_size']}")
        print(f"Training tokens: {train_data.numel()}")
        
        # Create model
        print("\nCreating model...")
        model = self._create_model(config, vocab['vocab_size'])
        model = model.to(device)
        
        # Print model summary
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Count FLOPs
        print("\nCounting FLOPs...")
        flops_counter = FLOPsCounter(model, config.batch_size, config.n_seq)
        flops_per_step = flops_counter.count_total_flops()
        flops_counter.print_summary()
        
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
        print(f"Steps per epoch: {num_steps_per_epoch}")
        
        # Training loop
        print("\nStarting training...")
        model.train()
        
        epoch_losses = []
        epoch_perplexities = []
        epoch_times = []
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
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.grad_clip
                )
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / max(1, num_batches)
            perplexity = math.exp(min(avg_loss, 20))
            
            epoch_losses.append(avg_loss)
            epoch_perplexities.append(perplexity)
            epoch_times.append(epoch_time)
            
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
        total_training_flops = flops_per_step.total * num_total_steps
        
        # Create results
        results = BenchmarkResults(
            model_name=config.model_name,
            config=asdict(config),
            final_loss=epoch_losses[-1],
            final_perplexity=epoch_perplexities[-1],
            best_perplexity=best_perplexity,
            training_time=training_time,
            forward_flops=flops_per_step.forward,
            backward_flops=flops_per_step.backward,
            optimizer_flops=flops_per_step.optimizer,
            total_flops_per_step=flops_per_step.total,
            total_training_flops=total_training_flops,
            peak_memory_mb=peak_memory_mb,
            model_size_mb=model_size_mb,
            epoch_losses=epoch_losses,
            epoch_perplexities=epoch_perplexities,
            epoch_times=epoch_times,
        )
        
        # Save results
        results_file = self.output_dir / f"{config.model_name}_results.json"
        results.save_json(str(results_file))
        
        self.results[config.model_name] = results
        
        print("\n" + "=" * 80)
        print(f"Benchmark Complete: {config.model_name}")
        print(f"Final Perplexity: {results.final_perplexity:.2f}")
        print(f"Best Perplexity: {results.best_perplexity:.2f}")
        print(f"Training Time: {results.training_time:.1f}s")
        print(f"Total FLOPs: {results.total_training_flops/1e12:.2f} TFLOPs")
        print("=" * 80 + "\n")
        
        return results
    
    def _create_model(self, config: BenchmarkConfig, vocab_size: int) -> nn.Module:
        """
        Create model based on configuration.
        
        Args:
            config: benchmark configuration
            vocab_size: vocabulary size
        
        Returns:
            model instance
        """
        if config.model_name.startswith('transformer'):
            # Create Transformer baseline
            return self._create_transformer_baseline(config, vocab_size)
        else:
            # Create ResNet-BK model
            model_config = FULL_CONFIG.copy()
            model_config.vocab_size = vocab_size
            model_config.d_model = config.d_model
            model_config.n_layers = config.n_layers
            model_config.n_seq = config.n_seq
            
            # Set optimization flags
            model_config.use_analytic_gradient = config.use_analytic_gradient
            model_config.use_koopman = config.use_koopman
            model_config.use_physics_informed = config.use_physics_informed
            model_config.use_quantization = config.use_quantization
            model_config.use_pruning = config.use_pruning
            model_config.use_mixed_precision = config.use_mixed_precision
            model_config.use_act = config.use_act
            model_config.use_multi_scale = config.use_multi_scale
            model_config.use_sparse_bk = config.use_sparse_bk
            model_config.use_early_exit = config.use_early_exit
            model_config.use_curriculum = config.use_curriculum
            model_config.use_active_learning = config.use_active_learning
            
            return ConfigurableResNetBK(model_config)
    
    def _create_transformer_baseline(self, config: BenchmarkConfig, vocab_size: int) -> nn.Module:
        """
        Create Transformer baseline for comparison.
        
        Args:
            config: benchmark configuration
            vocab_size: vocabulary size
        
        Returns:
            Transformer model
        """
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        class TransformerLM(nn.Module):
            def __init__(self, vocab_size, d_model, n_layers, n_seq, nhead=4):
                super().__init__()
                self.d_model = d_model
                self.n_seq = n_seq
                
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(n_seq, d_model)
                
                encoder_layer = TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = TransformerEncoder(encoder_layer, num_layers=n_layers)
                
                self.lm_head = nn.Linear(d_model, vocab_size)
            
            def forward(self, x):
                B, N = x.shape
                
                # Embeddings
                token_emb = self.token_embedding(x)
                pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
                pos_emb = self.position_embedding(pos_ids)
                
                h = token_emb + pos_emb
                
                # Transformer
                # Create causal mask
                mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
                h = self.transformer(h, mask=mask, is_causal=True)
                
                # LM head
                logits = self.lm_head(h)
                
                return logits
        
        return TransformerLM(vocab_size, config.d_model, config.n_layers, config.n_seq)
    
    def compare_results(self, model1_name: str, model2_name: str):
        """
        Compare results between two models.
        
        Args:
            model1_name: first model name
            model2_name: second model name (baseline)
        """
        if model1_name not in self.results or model2_name not in self.results:
            print("Error: Both models must be benchmarked first")
            return
        
        results1 = self.results[model1_name]
        results2 = self.results[model2_name]
        
        print("\n" + "=" * 80)
        print(f"Comparison: {model1_name} vs {model2_name} (Baseline)")
        print("=" * 80)
        
        # Perplexity comparison
        ppl_improvement = (results2.final_perplexity - results1.final_perplexity) / results2.final_perplexity * 100
        print(f"\nPerplexity:")
        print(f"  {model1_name}: {results1.final_perplexity:.2f}")
        print(f"  {model2_name}: {results2.final_perplexity:.2f}")
        print(f"  Improvement: {ppl_improvement:+.1f}%")
        
        # FLOPs comparison
        flops_speedup = results2.total_training_flops / results1.total_training_flops
        print(f"\nTotal Training FLOPs:")
        print(f"  {model1_name}: {results1.total_training_flops/1e12:.2f} TFLOPs")
        print(f"  {model2_name}: {results2.total_training_flops/1e12:.2f} TFLOPs")
        print(f"  Speedup: {flops_speedup:.2f}×")
        
        # Time comparison
        time_speedup = results2.training_time / results1.training_time
        print(f"\nTraining Time:")
        print(f"  {model1_name}: {results1.training_time:.1f}s")
        print(f"  {model2_name}: {results2.training_time:.1f}s")
        print(f"  Speedup: {time_speedup:.2f}×")
        
        # Memory comparison
        memory_reduction = (results2.peak_memory_mb - results1.peak_memory_mb) / results2.peak_memory_mb * 100
        print(f"\nPeak Memory:")
        print(f"  {model1_name}: {results1.peak_memory_mb:.1f} MB")
        print(f"  {model2_name}: {results2.peak_memory_mb:.1f} MB")
        print(f"  Reduction: {memory_reduction:+.1f}%")
        
        # Model size comparison
        size_reduction = (results2.model_size_mb - results1.model_size_mb) / results2.model_size_mb * 100
        print(f"\nModel Size:")
        print(f"  {model1_name}: {results1.model_size_mb:.2f} MB")
        print(f"  {model2_name}: {results2.model_size_mb:.2f} MB")
        print(f"  Reduction: {size_reduction:+.1f}%")
        
        print("=" * 80 + "\n")
        
        # Save comparison
        comparison = {
            'model1': model1_name,
            'model2': model2_name,
            'perplexity': {
                'model1': results1.final_perplexity,
                'model2': results2.final_perplexity,
                'improvement_pct': ppl_improvement,
            },
            'flops': {
                'model1': results1.total_training_flops,
                'model2': results2.total_training_flops,
                'speedup': flops_speedup,
            },
            'time': {
                'model1': results1.training_time,
                'model2': results2.training_time,
                'speedup': time_speedup,
            },
            'memory': {
                'model1': results1.peak_memory_mb,
                'model2': results2.peak_memory_mb,
                'reduction_pct': memory_reduction,
            },
            'model_size': {
                'model1': results1.model_size_mb,
                'model2': results2.model_size_mb,
                'reduction_pct': size_reduction,
            },
        }
        
        comparison_file = self.output_dir / f"comparison_{model1_name}_vs_{model2_name}.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison saved to {comparison_file}")
    
    def plot_training_curves(self, model_names: Optional[List[str]] = None):
        """
        Plot training curves for comparison.
        
        Args:
            model_names: list of model names to plot (None = all)
        """
        if not HAS_MATPLOTLIB:
            print("Warning: matplotlib not available, skipping plot generation")
            return
        
        if model_names is None:
            model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        ax = axes[0, 0]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                epochs = range(1, len(results.epoch_losses) + 1)
                ax.plot(epochs, results.epoch_losses, marker='o', label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Perplexity curves
        ax = axes[0, 1]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                epochs = range(1, len(results.epoch_perplexities) + 1)
                ax.plot(epochs, results.epoch_perplexities, marker='o', label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Time per epoch
        ax = axes[1, 0]
        for name in model_names:
            if name in self.results:
                results = self.results[name]
                epochs = range(1, len(results.epoch_times) + 1)
                ax.plot(epochs, results.epoch_times, marker='o', label=name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Time per Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # FLOPs comparison (bar chart)
        ax = axes[1, 1]
        names = []
        flops_values = []
        for name in model_names:
            if name in self.results:
                names.append(name)
                flops_values.append(self.results[name].total_training_flops / 1e12)
        ax.bar(names, flops_values)
        ax.set_ylabel('Total Training FLOPs (TFLOPs)')
        ax.set_title('Total Training FLOPs')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_file}")
        plt.close()


def main():
    """Run comprehensive WikiText-2 benchmark."""
    print("WikiText-2 Comprehensive Benchmark")
    print("=" * 80)
    
    # Create benchmark
    benchmark = WikiText2Benchmark(output_dir="benchmark_results/wikitext2")
    
    # Common configuration
    common_config = {
        'd_model': 64,
        'n_layers': 4,
        'n_seq': 128,
        'batch_size': 32,
        'epochs': 5,
        'lr': 1e-3,
        'weight_decay': 0.01,
        'grad_clip': 0.5,
        'device': 'cuda',
        'seed': 42,
    }
    
    # 1. Transformer Baseline
    print("\n[1/3] Running Transformer Baseline...")
    transformer_config = BenchmarkConfig(
        model_name='transformer_baseline',
        **common_config,
        use_analytic_gradient=False,
    )
    benchmark.run_benchmark(transformer_config)
    
    # 2. ResNet-BK Baseline (no optimizations)
    print("\n[2/3] Running ResNet-BK Baseline...")
    resnet_baseline_config = BenchmarkConfig(
        model_name='resnet_bk_baseline',
        **common_config,
        use_analytic_gradient=False,
    )
    benchmark.run_benchmark(resnet_baseline_config)
    
    # 3. ResNet-BK with All Optimizations
    print("\n[3/3] Running ResNet-BK with All Optimizations...")
    resnet_full_config = BenchmarkConfig(
        model_name='resnet_bk_full',
        **common_config,
        use_analytic_gradient=True,
        use_mixed_precision=True,
        use_act=True,
        use_multi_scale=True,
        use_sparse_bk=True,
    )
    benchmark.run_benchmark(resnet_full_config)
    
    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISONS")
    print("=" * 80)
    
    benchmark.compare_results('resnet_bk_baseline', 'transformer_baseline')
    benchmark.compare_results('resnet_bk_full', 'transformer_baseline')
    benchmark.compare_results('resnet_bk_full', 'resnet_bk_baseline')
    
    # Plot training curves
    benchmark.plot_training_curves()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {benchmark.output_dir}")


if __name__ == '__main__':
    main()
