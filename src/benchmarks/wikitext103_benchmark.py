"""
WikiText-103 Comprehensive Benchmark
Train ResNet-BK on WikiText-103 (10× larger than WikiText-2) and measure performance.

This script implements task 9.3:
- Scale to larger dataset (10× WikiText-2)
- Measure perplexity and training time
- Compare to WikiText-2 results
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
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional
from collections import Counter
from datasets import load_dataset

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from src.models.configurable_resnet_bk import ConfigurableResNetBK, FULL_CONFIG
from src.utils import TrainingMetrics, MetricsLogger
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
    
    # Data configuration
    data_limit: Optional[int] = None  # None = use all data
    
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
    dataset_name: str
    config: Dict
    
    # Training metrics
    final_loss: float
    final_perplexity: float
    best_perplexity: float
    training_time: float  # seconds
    
    # Dataset info
    total_tokens: int
    vocab_size: int
    
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


def load_wikitext103_data(batch_size, n_seq, data_limit=None, vocab_size_limit=30000):
    """
    Load WikiText-103 dataset.
    
    Args:
        batch_size: batch size
        n_seq: sequence length
        data_limit: maximum number of tokens (None = use all)
        vocab_size_limit: maximum vocabulary size
    
    Returns:
        train_data: (seq_len_total, batch_size) LongTensor
        vocab: dict with stoi, itos, vocab_size
        get_batch: function (source, i) -> (data, target)
    """
    print("Loading WikiText-103 dataset...")
    
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
        train_texts = dataset["train"]["text"]
    except Exception as e:
        print(f"Failed to load WikiText-103: {e}")
        print("Please check network connection.")
        return None, None, None
    
    print(f"Loaded {len(train_texts)} text lines")
    
    # Build vocabulary
    print("Building vocabulary...")
    counter = Counter()
    for line in train_texts:
        tokens = line.strip().split()
        if tokens:
            counter.update(tokens)
    
    special_tokens = ["<unk>"]
    stoi = {}
    itos = []
    
    for sp in special_tokens:
        stoi[sp] = len(itos)
        itos.append(sp)
    
    # Limit vocabulary size
    for tok, freq in counter.most_common(vocab_size_limit - len(special_tokens)):
        if tok not in stoi:
            stoi[tok] = len(itos)
            itos.append(tok)
    
    vocab_size = len(itos)
    unk_id = stoi["<unk>"]
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Encode texts
    print("Encoding texts...")
    def encode_texts(texts):
        ids = []
        for line in texts:
            for tok in line.strip().split():
                ids.append(stoi.get(tok, unk_id))
        return torch.tensor(ids, dtype=torch.long)
    
    train_ids = encode_texts(train_texts)
    
    print(f"Total tokens: {train_ids.numel():,}")
    
    # Limit tokens if specified
    if data_limit is not None and train_ids.numel() > data_limit:
        print(f"Limiting to {data_limit:,} tokens")
        train_ids = train_ids[:data_limit]
    
    # Batchify
    def batchify(data, bsz):
        seq_len = data.size(0) // bsz
        data = data.narrow(0, 0, seq_len * bsz)
        data = data.view(bsz, seq_len).t().contiguous()  # (seq_len, batch_size)
        return data
    
    train_data = batchify(train_ids, batch_size)
    
    print(f"Batched data shape: {train_data.shape}")
    
    def get_batch(source, i):
        seq_len = min(n_seq, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)
        return data, target
    
    vocab = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size,
    }
    
    return train_data, vocab, get_batch


class WikiText103Benchmark:
    """
    Comprehensive benchmark for WikiText-103 dataset.
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
        train_data, vocab, get_batch = load_wikitext103_data(
            batch_size=config.batch_size,
            n_seq=config.n_seq,
            data_limit=config.data_limit
        )
        
        if train_data is None:
            raise ValueError("Failed to load WikiText-103 dataset")
        
        total_tokens = train_data.numel()
        print(f"Total training tokens: {total_tokens:,}")
        
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
        
        print(f"\nTotal training steps: {num_total_steps:,}")
        print(f"Steps per epoch: {num_steps_per_epoch:,}")
        
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
            
            # Progress tracking
            print_interval = max(1, num_steps_per_epoch // 10)
            
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
                    print(f"Warning: NaN/Inf loss at epoch {epoch}, batch {num_batches}, skipping")
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
                
                # Print progress
                if num_batches % print_interval == 0:
                    current_loss = total_loss / num_batches
                    current_ppl = math.exp(min(current_loss, 20))
                    elapsed = time.time() - epoch_start
                    print(f"  Epoch {epoch}, Batch {num_batches}/{num_steps_per_epoch}: "
                          f"Loss={current_loss:.4f}, PPL={current_ppl:.2f}, "
                          f"Time={elapsed:.1f}s")
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / max(1, num_batches)
            perplexity = math.exp(min(avg_loss, 20))
            
            epoch_losses.append(avg_loss)
            epoch_perplexities.append(perplexity)
            epoch_times.append(epoch_time)
            
            if perplexity < best_perplexity:
                best_perplexity = perplexity
            
            print(f"\nEpoch {epoch}/{config.epochs} Complete: "
                  f"Loss={avg_loss:.4f}, PPL={perplexity:.2f}, "
                  f"Time={epoch_time:.1f}s, Best PPL={best_perplexity:.2f}\n")
        
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
            dataset_name='wikitext-103',
            config=asdict(config),
            final_loss=epoch_losses[-1],
            final_perplexity=epoch_perplexities[-1],
            best_perplexity=best_perplexity,
            training_time=training_time,
            total_tokens=total_tokens,
            vocab_size=vocab['vocab_size'],
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
        results_file = self.output_dir / f"{config.model_name}_wikitext103_results.json"
        results.save_json(str(results_file))
        
        self.results[config.model_name] = results
        
        print("\n" + "=" * 80)
        print(f"Benchmark Complete: {config.model_name}")
        print(f"Dataset: WikiText-103")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Final Perplexity: {results.final_perplexity:.2f}")
        print(f"Best Perplexity: {results.best_perplexity:.2f}")
        print(f"Training Time: {results.training_time:.1f}s ({results.training_time/60:.1f}min)")
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
            # Create ResNet-BK model using dataclasses.replace
            model_config = replace(
                FULL_CONFIG,
                vocab_size=vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                n_seq=config.n_seq,
                use_analytic_gradient=config.use_analytic_gradient,
                use_koopman=config.use_koopman,
                use_physics_informed=config.use_physics_informed,
                use_quantization=config.use_quantization,
                use_pruning=config.use_pruning,
                use_mixed_precision=config.use_mixed_precision,
                use_adaptive_computation=config.use_act,
                use_multi_scale=config.use_multi_scale,
                use_learned_sparsity=config.use_sparse_bk,
                use_curriculum_learning=config.use_curriculum,
                use_active_learning=config.use_active_learning,
            )
            
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
    
    def compare_to_wikitext2(self, wikitext2_results_path: str):
        """
        Compare WikiText-103 results to WikiText-2 results.
        
        Args:
            wikitext2_results_path: path to WikiText-2 results JSON
        """
        if not Path(wikitext2_results_path).exists():
            print(f"Error: WikiText-2 results not found at {wikitext2_results_path}")
            return
        
        # Load WikiText-2 results
        with open(wikitext2_results_path, 'r') as f:
            wt2_results = json.load(f)
        
        print("\n" + "=" * 80)
        print("WikiText-103 vs WikiText-2 Comparison")
        print("=" * 80)
        
        for model_name, wt103_results in self.results.items():
            print(f"\nModel: {model_name}")
            print("-" * 80)
            
            # Dataset size comparison
            wt2_tokens = wt2_results.get('total_tokens', 0)
            wt103_tokens = wt103_results.total_tokens
            size_ratio = wt103_tokens / wt2_tokens if wt2_tokens > 0 else 0
            
            print(f"\nDataset Size:")
            print(f"  WikiText-2:   {wt2_tokens:,} tokens")
            print(f"  WikiText-103: {wt103_tokens:,} tokens")
            print(f"  Ratio: {size_ratio:.1f}×")
            
            # Perplexity comparison
            wt2_ppl = wt2_results.get('final_perplexity', 0)
            wt103_ppl = wt103_results.final_perplexity
            ppl_change = ((wt103_ppl - wt2_ppl) / wt2_ppl * 100) if wt2_ppl > 0 else 0
            
            print(f"\nPerplexity:")
            print(f"  WikiText-2:   {wt2_ppl:.2f}")
            print(f"  WikiText-103: {wt103_ppl:.2f}")
            print(f"  Change: {ppl_change:+.1f}%")
            
            # Training time comparison
            wt2_time = wt2_results.get('training_time', 0)
            wt103_time = wt103_results.training_time
            time_ratio = wt103_time / wt2_time if wt2_time > 0 else 0
            
            print(f"\nTraining Time:")
            print(f"  WikiText-2:   {wt2_time:.1f}s ({wt2_time/60:.1f}min)")
            print(f"  WikiText-103: {wt103_time:.1f}s ({wt103_time/60:.1f}min)")
            print(f"  Ratio: {time_ratio:.1f}×")
            
            # FLOPs comparison
            wt2_flops = wt2_results.get('total_training_flops', 0)
            wt103_flops = wt103_results.total_training_flops
            flops_ratio = wt103_flops / wt2_flops if wt2_flops > 0 else 0
            
            print(f"\nTotal Training FLOPs:")
            print(f"  WikiText-2:   {wt2_flops/1e12:.2f} TFLOPs")
            print(f"  WikiText-103: {wt103_flops/1e12:.2f} TFLOPs")
            print(f"  Ratio: {flops_ratio:.1f}×")
        
        print("\n" + "=" * 80 + "\n")
    
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
        ax.set_title('Training Loss (WikiText-103)')
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
        ax.set_title('Perplexity (WikiText-103)')
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
        ax.set_title('Time per Epoch (WikiText-103)')
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
        ax.set_title('Total Training FLOPs (WikiText-103)')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "wikitext103_training_curves.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {plot_file}")
        plt.close()


def main():
    """Run comprehensive WikiText-103 benchmark."""
    print("WikiText-103 Comprehensive Benchmark")
    print("=" * 80)
    print("Note: WikiText-103 is ~10× larger than WikiText-2")
    print("      This benchmark may take significantly longer to run")
    print("=" * 80 + "\n")
    
    # Create benchmark
    benchmark = WikiText103Benchmark(output_dir="benchmark_results/wikitext103")
    
    # Common configuration
    common_config = {
        'd_model': 64,
        'n_layers': 4,
        'n_seq': 128,
        'batch_size': 32,
        'epochs': 3,  # Fewer epochs due to larger dataset
        'lr': 1e-3,
        'weight_decay': 0.01,
        'grad_clip': 0.5,
        'device': 'cuda',
        'seed': 42,
        'data_limit': None,  # Use all data
    }
    
    # 1. ResNet-BK Baseline (no optimizations)
    print("\n[1/2] Running ResNet-BK Baseline on WikiText-103...")
    resnet_baseline_config = BenchmarkConfig(
        model_name='resnet_bk_baseline',
        **common_config,
        use_analytic_gradient=False,
    )
    benchmark.run_benchmark(resnet_baseline_config)
    
    # 2. ResNet-BK with All Optimizations
    print("\n[2/2] Running ResNet-BK with All Optimizations on WikiText-103...")
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
    
    # Compare to WikiText-2 results if available
    wt2_results_path = "benchmark_results/wikitext2/resnet_bk_full_results.json"
    if Path(wt2_results_path).exists():
        print("\nComparing to WikiText-2 results...")
        benchmark.compare_to_wikitext2(wt2_results_path)
    else:
        print(f"\nNote: WikiText-2 results not found at {wt2_results_path}")
        print("      Run WikiText-2 benchmark first for comparison")
    
    # Plot training curves
    benchmark.plot_training_curves()
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {benchmark.output_dir}")
    print("\nKey Findings:")
    for name, results in benchmark.results.items():
        print(f"  {name}:")
        print(f"    - Final Perplexity: {results.final_perplexity:.2f}")
        print(f"    - Training Time: {results.training_time/60:.1f} minutes")
        print(f"    - Total Tokens: {results.total_tokens:,}")


if __name__ == '__main__':
    main()
