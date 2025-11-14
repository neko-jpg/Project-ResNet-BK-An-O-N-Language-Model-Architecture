"""
Weights & Biases Integration (Optional)
"""

from typing import Optional, Dict
import warnings


class WandBLogger:
    """
    Optional Weights & Biases logger for experiment tracking.
    
    Falls back gracefully if wandb is not installed.
    """
    
    def __init__(
        self,
        project: str = "resnet-bk",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.wandb = None
        
        if not enabled:
            return
        
        try:
            import wandb
            self.wandb = wandb
            
            # Initialize wandb
            self.wandb.init(
                project=project,
                name=name,
                config=config,
            )
            print(f"Weights & Biases logging enabled: {project}/{name}")
            
        except ImportError:
            warnings.warn(
                "wandb not installed. Install with: pip install wandb\n"
                "Continuing without W&B logging."
            )
            self.enabled = False
        except Exception as e:
            warnings.warn(f"Failed to initialize wandb: {e}\nContinuing without W&B logging.")
            self.enabled = False
    
    def log(self, metrics: Dict, step: Optional[int] = None):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if not self.enabled or self.wandb is None:
            return
        
        try:
            self.wandb.log(metrics, step=step)
        except Exception as e:
            warnings.warn(f"Failed to log to wandb: {e}")
    
    def log_model(self, model, name: str = "model"):
        """Log model architecture to W&B."""
        if not self.enabled or self.wandb is None:
            return
        
        try:
            self.wandb.watch(model, log="all", log_freq=100)
        except Exception as e:
            warnings.warn(f"Failed to watch model in wandb: {e}")
    
    def finish(self):
        """Finish W&B run."""
        if not self.enabled or self.wandb is None:
            return
        
        try:
            self.wandb.finish()
        except Exception as e:
            warnings.warn(f"Failed to finish wandb run: {e}")
