"""
Training Metrics and Logging
"""

import json
import csv
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from pathlib import Path
import math


@dataclass
class TrainingMetrics:
    """
    Comprehensive training metrics for ResNet-BK.
    
    Tracks all performance metrics across training steps.
    """
    # Step information
    step: int = 0
    epoch: int = 0
    
    # Loss metrics
    loss: float = 0.0
    perplexity: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Timing metrics
    step_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0
    
    # Memory metrics (MB)
    gpu_memory_allocated: float = 0.0
    gpu_memory_reserved: float = 0.0
    
    # Gradient metrics
    grad_norm: float = 0.0
    grad_max: float = 0.0
    
    # BK-Core specific metrics
    bk_scale: float = 0.0
    v_mean: float = 0.0
    v_std: float = 0.0
    G_ii_mean_real: float = 0.0
    G_ii_mean_imag: float = 0.0
    
    # MoE metrics
    expert_usage: List[float] = field(default_factory=list)
    routing_entropy: float = 0.0
    
    # Numerical stability metrics
    num_nan_grads: int = 0
    num_inf_grads: int = 0
    
    # Validation metrics
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)
    
    def compute_perplexity(self):
        """Compute perplexity from loss."""
        if self.loss > 0:
            self.perplexity = math.exp(min(self.loss, 20))  # Cap to prevent overflow
        return self.perplexity


class MetricsLogger:
    """
    Logger for training metrics with CSV and JSON export.
    
    Supports real-time logging and periodic saving to disk.
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "resnet_bk"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.metrics_history: List[TrainingMetrics] = []
        
        # Create log files
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.json_path = self.log_dir / f"{experiment_name}_metrics.json"
        
        # Initialize CSV file with headers
        self._init_csv()
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write header
                sample_metrics = TrainingMetrics()
                headers = list(sample_metrics.to_dict().keys())
                writer.writerow(headers)
    
    def log(self, metrics: TrainingMetrics):
        """
        Log metrics for current step.
        
        Args:
            metrics: TrainingMetrics instance
        """
        # Compute derived metrics
        metrics.compute_perplexity()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        # Append to CSV
        self._append_to_csv(metrics)
        
        # Print to console
        self._print_metrics(metrics)
    
    def _append_to_csv(self, metrics: TrainingMetrics):
        """Append metrics to CSV file."""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            metrics_dict = metrics.to_dict()
            # Convert lists to strings for CSV
            row = []
            for key, value in metrics_dict.items():
                if isinstance(value, list):
                    row.append(str(value))
                else:
                    row.append(value)
            writer.writerow(row)
    
    def _print_metrics(self, metrics: TrainingMetrics):
        """Print metrics to console."""
        elapsed = time.time() - self.start_time
        print(f"[Step {metrics.step:5d}] "
              f"Epoch {metrics.epoch} | "
              f"Loss: {metrics.loss:.4f} | "
              f"PPL: {metrics.perplexity:.2f} | "
              f"LR: {metrics.learning_rate:.6f} | "
              f"Time: {metrics.step_time:.3f}s | "
              f"Elapsed: {elapsed/60:.1f}min")
    
    def save_json(self):
        """Save all metrics to JSON file."""
        metrics_list = [m.to_dict() for m in self.metrics_history]
        with open(self.json_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of training."""
        if not self.metrics_history:
            return {}
        
        losses = [m.loss for m in self.metrics_history]
        perplexities = [m.perplexity for m in self.metrics_history]
        step_times = [m.step_time for m in self.metrics_history if m.step_time > 0]
        
        summary = {
            "total_steps": len(self.metrics_history),
            "final_loss": losses[-1] if losses else 0,
            "final_perplexity": perplexities[-1] if perplexities else 0,
            "best_loss": min(losses) if losses else 0,
            "best_perplexity": min(perplexities) if perplexities else 0,
            "avg_step_time": sum(step_times) / len(step_times) if step_times else 0,
            "total_time": time.time() - self.start_time,
        }
        
        return summary
    
    def print_summary(self):
        """Print training summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
            else:
                print(f"{key:20s}: {value}")
        print("="*60 + "\n")
