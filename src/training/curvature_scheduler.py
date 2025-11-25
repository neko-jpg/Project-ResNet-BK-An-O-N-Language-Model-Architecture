# src/training/curvature_scheduler.py
import torch
import torch.nn.functional as F
import math
from torch.optim import Optimizer

class BaseCurvatureScheduler:
    """
    Base class for curvature schedulers. It finds all 'log_c' parameters in the model
    and applies a curriculum to update them during training.
    """
    def __init__(self, model: torch.nn.Module, warmup_steps: int, target_curvature: float = 1.0):
        self.model = model
        self.warmup_steps = max(1, warmup_steps)  # Avoid division by zero
        self.target_curvature = target_curvature
        self._step = 0
        self.hyperbolic_params = self._find_hyperbolic_params()

        if not self.hyperbolic_params:
            print("Warning: Curvature scheduler was enabled, but no 'log_c' parameters were found in the model.")

    def _find_hyperbolic_params(self) -> list[torch.nn.Parameter]:
        """Finds all parameters named 'log_c' in the model."""
        params = []
        for module in self.model.modules():
            # Conventionally, the learnable curvature parameter is named 'log_c'
            if hasattr(module, 'log_c') and isinstance(getattr(module, 'log_c'), torch.nn.Parameter):
                params.append(module.log_c)
        return params

    def get_curvature(self) -> float:
        """Calculates the curvature value for the current step."""
        raise NotImplementedError("Subclasses must implement this method.")

    def step(self):
        """
        Updates the curvature of the model. This should be called once per training step.
        """
        if not self.hyperbolic_params:
            return

        self._step += 1
        new_c = self.get_curvature()

        # c = softplus(log_c)  => log_c = log(exp(c) - 1)
        # This is the inverse of the softplus function.
        if new_c <= 1e-6:
            # For c -> 0, log_c -> -inf. We use a large negative number for practical purposes.
            # softplus(-18.0) is approx 1.5e-8, which is effectively zero curvature.
            new_log_c_val = -18.0
        else:
            # Use torch.expm1 for better numerical stability with small `new_c`.
            new_log_c_val = torch.log(torch.expm1(torch.tensor(new_c, dtype=torch.float32))).item()

        # Update all found 'log_c' parameters in-place.
        for log_c_param in self.hyperbolic_params:
            log_c_param.data.fill_(new_log_c_val)


class LinearCurvatureScheduler(BaseCurvatureScheduler):
    """
    Linearly increases the curvature from 0 to `target_curvature` over `warmup_steps`.
    """
    def get_curvature(self) -> float:
        if self._step >= self.warmup_steps:
            return self.target_curvature

        # Linear interpolation from 0 to target_curvature
        return (self._step / self.warmup_steps) * self.target_curvature


def create_curvature_scheduler(config: dict, model: torch.nn.Module) -> BaseCurvatureScheduler | None:
    """
    Factory function to create a curvature scheduler based on the training configuration.
    """
    scheduler_config = config.get('curvature_scheduler')
    if not scheduler_config or not scheduler_config.get('enabled', False):
        return None

    scheduler_type = scheduler_config.get('type', 'linear')
    warmup_steps = scheduler_config.get('warmup_steps', 1000)
    target_curvature = scheduler_config.get('target_curvature', 1.0)

    if scheduler_type == 'linear':
        print(f"Initializing LinearCurvatureScheduler with {warmup_steps} warmup steps to target curvature {target_curvature}.")
        return LinearCurvatureScheduler(
            model=model,
            warmup_steps=warmup_steps,
            target_curvature=target_curvature
        )
    else:
        raise ValueError(f"Unknown curvature scheduler type: '{scheduler_type}'")
