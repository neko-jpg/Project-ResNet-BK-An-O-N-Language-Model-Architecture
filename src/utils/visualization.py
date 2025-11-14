"""
Real-time Training Dashboard and Visualization
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional
import numpy as np
from pathlib import Path


class TrainingDashboard:
    """
    Real-time training dashboard using matplotlib.
    
    Displays loss, perplexity, learning rate, and other metrics
    in a multi-panel dashboard that updates during training.
    """
    
    def __init__(self, save_dir: str = "plots", experiment_name: str = "resnet_bk"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # Data storage
        self.steps = []
        self.losses = []
        self.perplexities = []
        self.learning_rates = []
        self.grad_norms = []
        self.step_times = []
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.fig.suptitle(f'ResNet-BK Training Dashboard: {experiment_name}', fontsize=14)
        
        # Configure subplots
        self.ax_loss = self.axes[0, 0]
        self.ax_ppl = self.axes[0, 1]
        self.ax_lr = self.axes[0, 2]
        self.ax_grad = self.axes[1, 0]
        self.ax_time = self.axes[1, 1]
        self.ax_memory = self.axes[1, 2]
        
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Step')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_ppl.set_title('Perplexity')
        self.ax_ppl.set_xlabel('Step')
        self.ax_ppl.set_ylabel('PPL')
        self.ax_ppl.grid(True, alpha=0.3)
        
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.set_xlabel('Step')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.grid(True, alpha=0.3)
        
        self.ax_grad.set_title('Gradient Norm')
        self.ax_grad.set_xlabel('Step')
        self.ax_grad.set_ylabel('Grad Norm')
        self.ax_grad.grid(True, alpha=0.3)
        
        self.ax_time.set_title('Step Time')
        self.ax_time.set_xlabel('Step')
        self.ax_time.set_ylabel('Time (s)')
        self.ax_time.grid(True, alpha=0.3)
        
        self.ax_memory.set_title('GPU Memory')
        self.ax_memory.set_xlabel('Step')
        self.ax_memory.set_ylabel('Memory (MB)')
        self.ax_memory.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update(self, metrics):
        """
        Update dashboard with new metrics.
        
        Args:
            metrics: TrainingMetrics instance
        """
        self.steps.append(metrics.step)
        self.losses.append(metrics.loss)
        self.perplexities.append(metrics.perplexity)
        self.learning_rates.append(metrics.learning_rate)
        self.grad_norms.append(metrics.grad_norm)
        self.step_times.append(metrics.step_time)
        
        # Update plots
        self.ax_loss.clear()
        self.ax_loss.plot(self.steps, self.losses, 'b-', linewidth=1)
        self.ax_loss.set_title('Training Loss')
        self.ax_loss.set_xlabel('Step')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        self.ax_ppl.clear()
        self.ax_ppl.plot(self.steps, self.perplexities, 'r-', linewidth=1)
        self.ax_ppl.set_title('Perplexity')
        self.ax_ppl.set_xlabel('Step')
        self.ax_ppl.set_ylabel('PPL')
        self.ax_ppl.grid(True, alpha=0.3)
        
        self.ax_lr.clear()
        self.ax_lr.plot(self.steps, self.learning_rates, 'g-', linewidth=1)
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.set_xlabel('Step')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.grid(True, alpha=0.3)
        
        self.ax_grad.clear()
        self.ax_grad.plot(self.steps, self.grad_norms, 'm-', linewidth=1)
        self.ax_grad.set_title('Gradient Norm')
        self.ax_grad.set_xlabel('Step')
        self.ax_grad.set_ylabel('Grad Norm')
        self.ax_grad.grid(True, alpha=0.3)
        
        self.ax_time.clear()
        self.ax_time.plot(self.steps, self.step_times, 'c-', linewidth=1)
        self.ax_time.set_title('Step Time')
        self.ax_time.set_xlabel('Step')
        self.ax_time.set_ylabel('Time (s)')
        self.ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def save(self, filename: Optional[str] = None):
        """Save current dashboard to file."""
        if filename is None:
            filename = f"{self.experiment_name}_dashboard.png"
        save_path = self.save_dir / filename
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    def close(self):
        """Close the dashboard."""
        plt.close(self.fig)


def plot_training_curves(metrics_history, save_path: Optional[str] = None):
    """
    Plot training curves from metrics history.
    
    Args:
        metrics_history: List of TrainingMetrics
        save_path: Optional path to save plot
    """
    steps = [m.step for m in metrics_history]
    losses = [m.loss for m in metrics_history]
    perplexities = [m.perplexity for m in metrics_history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(steps, losses, 'b-', linewidth=2)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(steps, perplexities, 'r-', linewidth=2)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Perplexity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig
