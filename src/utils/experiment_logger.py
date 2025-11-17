"""
Experiment Logger for Mamba-Killer ResNet-BK

Provides unified logging interface for TensorBoard and Weights & Biases.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """Unified logger for experiments with TensorBoard and W&B support."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        log_dir: str = "./logs",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        project_name: str = "mamba-killer-resnet-bk",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            config: Experiment configuration
            log_dir: Directory for logs
            use_tensorboard: Enable TensorBoard logging
            use_wandb: Enable Weights & Biases logging
            project_name: W&B project name
            experiment_name: Experiment name
        """
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.project_name = project_name
        
        # Initialize TensorBoard
        self.tensorboard_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"✓ TensorBoard logging enabled: {tb_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("⚠ TensorBoard requested but not available. Install: pip install tensorboard")
        
        # Initialize Weights & Biases
        self.wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                dir=str(self.log_dir)
            )
            print(f"✓ Weights & Biases logging enabled: {project_name}/{experiment_name}")
        elif use_wandb and not WANDB_AVAILABLE:
            print("⚠ W&B requested but not available. Install: pip install wandb")
        
        # Save configuration
        config_file = self.log_dir / self.experiment_name / "config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Metrics history
        self.metrics_history = []
        self.step = 0
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step (uses internal counter if None)
        """
        if step is None:
            step = self.step
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, step)
        
        # Weights & Biases
        if self.wandb_run is not None:
            wandb.log({name: value}, step=step)
        
        # History
        self.metrics_history.append({
            'step': step,
            'name': name,
            'value': value,
            'timestamp': time.time()
        })
    
    def log_scalars(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple scalar values.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_histogram(self, name: str, values, step: Optional[int] = None):
        """
        Log a histogram.
        
        Args:
            name: Histogram name
            values: Values to histogram
            step: Training step
        """
        if step is None:
            step = self.step
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_histogram(name, values, step)
        
        # Weights & Biases
        if self.wandb_run is not None:
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None):
        """
        Log text.
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        if step is None:
            step = self.step
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(name, text, step)
        
        # Weights & Biases
        if self.wandb_run is not None:
            wandb.log({name: text}, step=step)
    
    def log_figure(self, name: str, figure, step: Optional[int] = None):
        """
        Log a matplotlib figure.
        
        Args:
            name: Figure name
            figure: Matplotlib figure
            step: Training step
        """
        if step is None:
            step = self.step
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_figure(name, figure, step)
        
        # Weights & Biases
        if self.wandb_run is not None:
            wandb.log({name: wandb.Image(figure)}, step=step)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Weights & Biases
        if self.wandb_run is not None:
            wandb.config.update(config)
    
    def increment_step(self):
        """Increment internal step counter."""
        self.step += 1
    
    def set_step(self, step: int):
        """Set internal step counter."""
        self.step = step
    
    def save_metrics(self):
        """Save metrics history to JSON file."""
        metrics_file = self.log_dir / self.experiment_name / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        """Close logger and save metrics."""
        # Save metrics
        self.save_metrics()
        
        # Close TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        # Close W&B
        if self.wandb_run is not None:
            wandb.finish()
        
        print(f"✓ Experiment logs saved to {self.log_dir / self.experiment_name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_logger(config: Dict[str, Any]) -> ExperimentLogger:
    """
    Create experiment logger from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        ExperimentLogger instance
    """
    monitoring_config = config.get('monitoring', {})
    
    return ExperimentLogger(
        config=config,
        log_dir=config.get('checkpoint', {}).get('save_dir', './logs'),
        use_tensorboard=monitoring_config.get('use_tensorboard', True),
        use_wandb=monitoring_config.get('use_wandb', False),
        project_name=monitoring_config.get('project_name', 'mamba-killer-resnet-bk'),
        experiment_name=monitoring_config.get('experiment_name', None)
    )


if __name__ == '__main__':
    # Test logger
    test_config = {
        'model': {'d_model': 256},
        'training': {'learning_rate': 1e-3},
        'monitoring': {
            'use_tensorboard': True,
            'use_wandb': False,
            'experiment_name': 'test_experiment'
        }
    }
    
    with create_logger(test_config) as logger:
        # Log some test metrics
        for step in range(10):
            logger.log_scalar('loss', 1.0 / (step + 1), step)
            logger.log_scalar('accuracy', step / 10.0, step)
        
        print("✓ Logger test passed")
