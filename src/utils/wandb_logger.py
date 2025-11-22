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

    def log_phase4_diagnostics(self, diagnostics: Dict, step: Optional[int] = None):
        """
        Log Phase 4 specific metrics (Ghost in the Shell).
        Flattens the nested diagnostics dictionary for WandB.
        """
        if not self.enabled or self.wandb is None:
            return

        metrics = {}

        # 1. Emotion (Task 1 & 8)
        if 'emotion' in diagnostics:
            e = diagnostics['emotion']
            # Handle if values are tensors
            res = e.get('resonance_score', 0)
            dis = e.get('dissonance_score', 0)
            if isinstance(res, torch.Tensor): res = res.float().mean().item()
            if isinstance(dis, torch.Tensor): dis = dis.float().mean().item()

            metrics['emotion/resonance'] = res
            metrics['emotion/dissonance'] = dis

            # Map state string to integer for plotting
            state_map = {'RESONANCE': 1, 'NEUTRAL': 0, 'DISSONANCE': -1}
            state = e.get('state', 'NEUTRAL')
            metrics['emotion/polarity'] = state_map.get(state, 0)

        # 2. Quantum (Task 5 & 11)
        if 'quantum' in diagnostics:
            q = diagnostics['quantum']
            ent = q.get('entropy_reduction', 0)
            if isinstance(ent, torch.Tensor): ent = ent.float().mean().item()
            metrics['quantum/entropy_reduction'] = ent

            if 'suggested_temperature' in diagnostics:
                metrics['quantum/temperature'] = diagnostics['suggested_temperature']

        # 3. Physics / Unitarity (Task 5)
        # Assuming aggregated unitarity violation is passed in top-level or gathered
        if 'unitarity_violation' in diagnostics:
            metrics['physics/unitarity_violation'] = diagnostics['unitarity_violation']

        # 4. Meta Commentary
        if 'meta_commentary' in diagnostics:
            # Log as text
            # WandB handles text logging differently? Or just Table.
            # We skip logging text every step to avoid spam, or log as alert?
            pass

        self.log(metrics, step=step)
    
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
