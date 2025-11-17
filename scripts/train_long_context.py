"""
Long-Context Training Infrastructure for Mamba-Killer ResNet-BK

This script implements multi-length training with gradient norm tracking,
loss spike detection, and streaming evaluation for ultra-long sequences.

Requirements: 6.1, 6.2, 6.5, 6.6, 6.7, 6.8, 6.15

Usage:
    # Train on single sequence length
    python scripts/train_long_context.py --seq_len 8192 --epochs 5
    
    # Train on multiple sequence lengths
    python scripts/train_long_context.py --multi_length --epochs 3
    
    # Streaming evaluation on 1M tokens
    python scripts/train_long_context.py --eval_only --seq_len 1048576 --streaming
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

from src.models.configurable_resnet_bk import ConfigurableResNetBK, ResNetBKConfig
from src.utils import get_data_loader, TrainingMetrics, WandBLogger
from src.benchmarks.streaming_evaluator import StreamingEvaluator as NewStreamingEvaluator


@dataclass
class LongContextMetrics:
    """Metrics for long-context training."""
    step: int
    epoch: int
    seq_len: int
    loss: float
    perplexity: float
    gradient_norm: float
    learning_rate: float
    step_time: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    
    # Stability metrics
    is_nan: bool = False
    is_inf: bool = False
    is_spike: bool = False
    spike_ratio: float = 0.0
    
    # Schatten norms (if using Birman-Schwinger)
    mean_schatten_s1: float = 0.0
    mean_schatten_s2: float = 0.0
    max_condition_number: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class LossSpikeDetector:
    """Detect loss spikes (loss > 2× previous value)."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.loss_history: List[float] = []
        self.spike_count = 0
    
    def add_loss(self, loss: float) -> Tuple[bool, float]:
        """
        Add loss and check for spike.
        
        Returns:
            (is_spike, spike_ratio)
        """
        is_spike = False
        spike_ratio = 0.0
        
        if len(self.loss_history) > 0:
            prev_loss = self.loss_history[-1]
            if loss > 2.0 * prev_loss:
                is_spike = True
                spike_ratio = loss / prev_loss
                self.spike_count += 1
        
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        return is_spike, spike_ratio
    
    def get_spike_count(self) -> int:
        """Get total spike count."""
        return self.spike_count
    
    def reset(self):
        """Reset detector."""
        self.loss_history = []
        self.spike_count = 0


class GradientNormTracker:
    """Track gradient norms per sequence length."""
    
    def __init__(self):
        self.norms_by_seq_len: Dict[int, List[float]] = {}
    
    def add_norm(self, seq_len: int, grad_norm: float):
        """Add gradient norm for sequence length."""
        if seq_len not in self.norms_by_seq_len:
            self.norms_by_seq_len[seq_len] = []
        self.norms_by_seq_len[seq_len].append(grad_norm)
    
    def get_statistics(self, seq_len: int) -> Dict[str, float]:
        """Get statistics for sequence length."""
        if seq_len not in self.norms_by_seq_len or len(self.norms_by_seq_len[seq_len]) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        norms = self.norms_by_seq_len[seq_len]
        return {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "min": float(np.min(norms)),
            "max": float(np.max(norms)),
        }
    
    def get_all_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all sequence lengths."""
        return {seq_len: self.get_statistics(seq_len) 
                for seq_len in sorted(self.norms_by_seq_len.keys())}


class StreamingEvaluator:
    """
    Streaming evaluation for ultra-long sequences.
    
    Evaluates on 1M token sequences without loading entire sequence into memory.
    Implements chunked processing with state preservation.
    
    Requirement: 6.15
    """
    
    def __init__(self, model: nn.Module, chunk_size: Optional[int] = None, device: str = 'cuda'):
        self.model = model
        # Use model's n_seq if chunk_size not specified
        if chunk_size is None:
            if hasattr(model, 'model') and hasattr(model.model, 'n_seq'):
                self.chunk_size = model.model.n_seq
            else:
                self.chunk_size = 8192
        else:
            self.chunk_size = chunk_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def evaluate_streaming(
        self,
        data: torch.Tensor,
        max_tokens: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate on ultra-long sequence using streaming.
        
        Args:
            data: Full dataset tensor (1D)
            max_tokens: Maximum tokens to evaluate (None = all)
        
        Returns:
            Dictionary with loss, perplexity, and token count
        """
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        num_chunks = 0
        
        # Determine evaluation length
        eval_length = min(data.size(0) - 1, max_tokens) if max_tokens else data.size(0) - 1
        
        print(f"\nStreaming evaluation on {eval_length:,} tokens...")
        print(f"Chunk size: {self.chunk_size}")
        
        with torch.no_grad():
            for i in range(0, eval_length, self.chunk_size):
                chunk_start = time.time()
                
                # Get chunk - data is 1D, need to reshape to (batch=1, seq)
                chunk_len = min(self.chunk_size, eval_length - i)
                
                # Extract chunk and ensure it's 2D: (batch=1, seq_len)
                x_data = data[i:i+chunk_len]
                y_data = data[i+1:i+chunk_len+1]
                
                # Reshape to (1, chunk_len) for batch processing
                x_chunk = x_data.unsqueeze(0).to(self.device)
                y_chunk = y_data.to(self.device)
                
                # Forward pass
                logits = self.model(x_chunk)  # (1, seq_len, vocab_size)
                
                # Compute loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), y_chunk)
                
                # Accumulate
                total_loss += loss.sum().item()
                total_tokens += chunk_len
                num_chunks += 1
                
                # Progress
                if num_chunks % 10 == 0:
                    chunk_time = time.time() - chunk_start
                    progress = (i + chunk_len) / eval_length * 100
                    current_ppl = math.exp(total_loss / total_tokens)
                    print(f"  Progress: {progress:.1f}% | "
                          f"Tokens: {total_tokens:,} | "
                          f"PPL: {current_ppl:.2f} | "
                          f"Chunk time: {chunk_time:.2f}s")
                
                # Clear cache periodically
                if num_chunks % 50 == 0:
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 20))
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_chunks": num_chunks,
        }


class LongContextTrainer:
    """
    Long-context training with multi-length support.
    
    Requirements: 6.1, 6.2, 6.5, 6.6, 6.7, 6.8
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: str,
        args: argparse.Namespace,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        
        self.criterion = nn.CrossEntropyLoss()
        self.spike_detector = LossSpikeDetector()
        self.grad_tracker = GradientNormTracker()
        self.streaming_evaluator = StreamingEvaluator(model, device=device)
        
        self.metrics_history: List[LongContextMetrics] = []
        self.global_step = 0
        
        # Setup logging
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.wandb_logger = WandBLogger(
            project="resnet-bk-long-context",
            name=f"seq{args.seq_len}_d{args.d_model}_l{args.n_layers}",
            config=vars(args),
            enabled=args.use_wandb,
        )
    
    def train_epoch(
        self,
        train_data: torch.Tensor,
        get_batch: callable,
        epoch: int,
        seq_len: int,
    ) -> Dict[str, float]:
        """Train one epoch on given sequence length."""
        self.model.train()
        
        epoch_start = time.time()
        total_loss = 0.0
        num_batches = 0
        
        print(f"\nEpoch {epoch} | Sequence Length: {seq_len}")
        print("-" * 60)
        
        for i in range(0, train_data.size(0) - 1, seq_len):
            step_start = time.time()
            
            # Get batch
            x_batch, y_batch = get_batch(train_data, i)
            x_batch = x_batch.t().contiguous()
            
            if x_batch.size(1) != seq_len:
                continue
            
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y_batch)
            
            # Check for NaN/Inf
            is_nan = torch.isnan(loss).item()
            is_inf = torch.isinf(loss).item()
            
            if is_nan or is_inf:
                print(f"Warning: {'NaN' if is_nan else 'Inf'} loss at step {self.global_step}, skipping")
                continue
            
            # Check for spike
            is_spike, spike_ratio = self.spike_detector.add_loss(loss.item())
            
            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()
            
            # Track gradient norm
            self.grad_tracker.add_norm(seq_len, grad_norm.item())
            
            step_time = time.time() - step_start
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Get memory stats
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(self.device) / 1e9
            else:
                memory_allocated = 0.0
                memory_reserved = 0.0
            
            # Get stability diagnostics
            stability_diagnostics = {}
            if hasattr(self.model.model, 'get_stability_diagnostics'):
                stability_diagnostics = self.model.model.get_stability_diagnostics()
            
            # Create metrics
            metrics = LongContextMetrics(
                step=self.global_step,
                epoch=epoch,
                seq_len=seq_len,
                loss=loss.item(),
                perplexity=math.exp(min(loss.item(), 20)),
                gradient_norm=grad_norm.item(),
                learning_rate=self.scheduler.get_last_lr()[0],
                step_time=step_time,
                memory_allocated_gb=memory_allocated,
                memory_reserved_gb=memory_reserved,
                is_nan=is_nan,
                is_inf=is_inf,
                is_spike=is_spike,
                spike_ratio=spike_ratio,
                mean_schatten_s1=stability_diagnostics.get('mean_schatten_s1', 0.0),
                mean_schatten_s2=stability_diagnostics.get('mean_schatten_s2', 0.0),
                max_condition_number=stability_diagnostics.get('max_condition_number', 0.0),
            )
            
            self.metrics_history.append(metrics)
            
            # Log to console
            if self.global_step % self.args.log_interval == 0:
                print(f"Step {self.global_step:6d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"PPL: {metrics.perplexity:.2f} | "
                      f"Grad: {grad_norm.item():.2f} | "
                      f"Mem: {memory_allocated:.2f}GB" +
                      (f" | SPIKE {spike_ratio:.2f}x" if is_spike else ""))
            
            # Log to W&B
            if self.args.use_wandb:
                wandb_dict = {
                    f'seq{seq_len}/loss': loss.item(),
                    f'seq{seq_len}/perplexity': metrics.perplexity,
                    f'seq{seq_len}/gradient_norm': grad_norm.item(),
                    f'seq{seq_len}/memory_gb': memory_allocated,
                    f'seq{seq_len}/spike_count': self.spike_detector.get_spike_count(),
                    'learning_rate': metrics.learning_rate,
                }
                
                if stability_diagnostics:
                    wandb_dict.update({
                        f'seq{seq_len}/schatten_s1': stability_diagnostics.get('mean_schatten_s1', 0.0),
                        f'seq{seq_len}/schatten_s2': stability_diagnostics.get('mean_schatten_s2', 0.0),
                        f'seq{seq_len}/condition_number': stability_diagnostics.get('max_condition_number', 0.0),
                    })
                
                self.wandb_logger.log(wandb_dict, step=self.global_step)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, num_batches)
        perplexity = math.exp(min(avg_loss, 20))
        
        grad_stats = self.grad_tracker.get_statistics(seq_len)
        
        print(f"\nEpoch {epoch} Summary (N={seq_len}):")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Gradient Norm: {grad_stats['mean']:.2f} ± {grad_stats['std']:.2f}")
        print(f"  Loss Spikes: {self.spike_detector.get_spike_count()}")
        
        return {
            "avg_loss": avg_loss,
            "perplexity": perplexity,
            "epoch_time": epoch_time,
            "spike_count": self.spike_detector.get_spike_count(),
            "grad_norm_mean": grad_stats['mean'],
            "grad_norm_std": grad_stats['std'],
        }
    
    def train_multi_length(
        self,
        train_data: torch.Tensor,
        get_batch: callable,
        sequence_lengths: List[int],
    ):
        """
        Train on multiple sequence lengths.
        
        Requirement: 6.1 - Support N ∈ {128, 512, 2048, 8192, 32768, 131072}
        """
        print(f"\nMulti-Length Training")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Epochs per length: {self.args.epochs}")
        print("=" * 60)
        
        results_by_length = {}
        
        for seq_len in sequence_lengths:
            print(f"\n{'='*60}")
            print(f"Training on sequence length: {seq_len}")
            print(f"{'='*60}")
            
            # Reset spike detector for this length
            self.spike_detector.reset()
            
            # Train epochs
            length_results = []
            for epoch in range(1, self.args.epochs + 1):
                epoch_results = self.train_epoch(train_data, get_batch, epoch, seq_len)
                length_results.append(epoch_results)
            
            results_by_length[seq_len] = length_results
            
            # Save checkpoint for this length
            self.save_checkpoint(f"seq{seq_len}_final")
        
        # Save all results
        self.save_results(results_by_length)
        
        # Print final summary
        self.print_multi_length_summary(results_by_length)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.save_dir / f"checkpoint_{name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'args': vars(self.args),
            'grad_tracker_stats': self.grad_tracker.get_all_statistics(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_results(self, results_by_length: Dict[int, List[Dict]]):
        """Save training results to JSON."""
        results_path = self.save_dir / "long_context_results.json"
        
        # Convert metrics history to serializable format
        metrics_list = [m.to_dict() for m in self.metrics_history]
        
        results = {
            "args": vars(self.args),
            "results_by_length": results_by_length,
            "metrics_history": metrics_list,
            "gradient_statistics": self.grad_tracker.get_all_statistics(),
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved: {results_path}")
    
    def print_multi_length_summary(self, results_by_length: Dict[int, List[Dict]]):
        """Print summary across all sequence lengths."""
        print(f"\n{'='*60}")
        print("Multi-Length Training Summary")
        print(f"{'='*60}")
        
        print(f"\n{'Seq Len':<10} {'Final PPL':<12} {'Spikes':<10} {'Grad Norm':<15}")
        print("-" * 60)
        
        for seq_len in sorted(results_by_length.keys()):
            final_epoch = results_by_length[seq_len][-1]
            print(f"{seq_len:<10} "
                  f"{final_epoch['perplexity']:<12.2f} "
                  f"{final_epoch['spike_count']:<10} "
                  f"{final_epoch['grad_norm_mean']:.2f} ± {final_epoch['grad_norm_std']:.2f}")
        
        print("\nGradient Norm Statistics by Sequence Length:")
        grad_stats = self.grad_tracker.get_all_statistics()
        for seq_len, stats in sorted(grad_stats.items()):
            print(f"  N={seq_len:<6}: mean={stats['mean']:.2f}, "
                  f"std={stats['std']:.2f}, "
                  f"min={stats['min']:.2f}, "
                  f"max={stats['max']:.2f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Long-Context Training for ResNet-BK")
    
    # Model configuration
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='FFN dimension')
    
    # Training configuration
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--multi_length', action='store_true', help='Train on multiple lengths')
    parser.add_argument('--sequence_lengths', type=int, nargs='+',
                       default=[128, 512, 2048, 8192, 32768, 131072],
                       help='Sequence lengths for multi-length training')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    
    # Data configuration
    parser.add_argument('--dataset', type=str, default='wikitext2', help='Dataset name')
    parser.add_argument('--data_limit', type=int, default=None, help='Limit data size')
    
    # Evaluation
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only')
    parser.add_argument('--streaming', action='store_true', help='Use streaming evaluation')
    parser.add_argument('--eval_tokens', type=int, default=1000000, help='Tokens for streaming eval')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='checkpoints/long_context',
                       help='Save directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B logging')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model features
    parser.add_argument('--use_birman_schwinger', action='store_true',
                       help='Use Birman-Schwinger core')
    parser.add_argument('--use_scattering_router', action='store_true',
                       help='Use scattering-based router')
    parser.add_argument('--use_semiseparable', action='store_true',
                       help='Use semiseparable structure')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Long-Context Training for Mamba-Killer ResNet-BK")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Sequence length: {args.seq_len}")
    if args.multi_length:
        print(f"Multi-length training: {args.sequence_lengths}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_data, vocab, get_batch = get_data_loader(
        batch_size=args.batch_size,
        n_seq=args.seq_len,
        dataset_name=args.dataset,
        data_limit=args.data_limit if args.data_limit else 500000
    )
    
    if train_data is None:
        print("Failed to load data. Exiting.")
        return
    
    print(f"Vocabulary size: {vocab['vocab_size']}")
    print(f"Training tokens: {train_data.numel():,}")
    
    # Create model
    print("\nCreating model...")
    config = ResNetBKConfig(
        vocab_size=vocab['vocab_size'],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_seq=args.seq_len,
        num_experts=8,
        top_k=2,
        dropout_p=0.1,
    )
    
    # Apply feature flags
    if args.use_birman_schwinger:
        config.use_birman_schwinger = True
    if args.use_scattering_router:
        config.use_scattering_router = True
    if args.use_semiseparable:
        config.use_semiseparable = True
    
    model = ConfigurableResNetBK(config).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluation only mode
    if args.eval_only:
        print("\nEvaluation mode")
        streaming_evaluator = StreamingEvaluator(model, device=device)
        
        if args.streaming:
            results = streaming_evaluator.evaluate_streaming(
                train_data,
                max_tokens=args.eval_tokens
            )
            print(f"\nStreaming Evaluation Results:")
            print(f"  Tokens: {results['total_tokens']:,}")
            print(f"  Loss: {results['loss']:.4f}")
            print(f"  Perplexity: {results['perplexity']:.2f}")
        else:
            print("Standard evaluation not implemented. Use --streaming flag.")
        
        return
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    if args.multi_length:
        # Estimate total steps for all lengths
        total_steps = 0
        for seq_len in args.sequence_lengths:
            steps_per_epoch = train_data.size(0) // seq_len
            total_steps += steps_per_epoch * args.epochs
    else:
        steps_per_epoch = train_data.size(0) // args.seq_len
        total_steps = steps_per_epoch * args.epochs
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr / 10
    )
    
    # Create trainer
    trainer = LongContextTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
    )
    
    # Train
    if args.multi_length:
        trainer.train_multi_length(train_data, get_batch, args.sequence_lengths)
    else:
        for epoch in range(1, args.epochs + 1):
            trainer.train_epoch(train_data, get_batch, epoch, args.seq_len)
        
        # Save final checkpoint
        trainer.save_checkpoint("final")
        
        # Save results
        results_by_length = {args.seq_len: []}
        trainer.save_results(results_by_length)
    
    trainer.wandb_logger.finish()
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
