import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class HomeostasisController(nn.Module):
    """
    Dynamic Criticality Control (Homeostasis).

    Maintains the model at the "Edge of Chaos" by adjusting the non-Hermitian
    decay rate (gamma) based on spectral feedback.

    Mechanism:
    - Monitors spectral diagnostics (e.g., Unitarity Violation, Condition Number).
    - Adjusts `gamma` to keep the system critical.
      - If system is too chaotic (high unitarity violation/explosion), increase gamma (damping).
      - If system is too ordered (rigid/vanishing), decrease gamma (allow more energy).

    Target Metric: Unitarity Violation (| ||y||/||x|| - 1 |).
    Target Range: [target_min, target_max].
    """

    def __init__(
        self,
        initial_gamma: float = 0.0,
        target_unitarity_min: float = 0.95, # Lower bound for ||y||/||x||
        target_unitarity_max: float = 1.05, # Upper bound for ||y||/||x||
        adjustment_rate: float = 0.001,
        gamma_min: float = -0.1,
        gamma_max: float = 0.5,
    ):
        super().__init__()
        self.target_min = target_unitarity_min
        self.target_max = target_unitarity_max
        self.adjustment_rate = adjustment_rate
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        # Gamma is managed by the layer, but we calculate the delta.
        # Alternatively, we can track internal state if needed.

    def compute_adjustment(self, diagnostics: Dict[str, float]) -> float:
        """
        Compute the adjustment delta for gamma based on diagnostics.

        Args:
            diagnostics: Dictionary containing 'unitarity_violation' or 'spectral_radius'.

        Returns:
            delta_gamma: Value to add to gamma.
        """
        # We look for 'growth_ratio' = ||y|| / ||x||
        # Or 'unitarity_violation' = | ||y||/||x|| - 1 |

        growth_ratio = diagnostics.get('growth_ratio', 1.0)

        # Logic:
        # If growth_ratio > target_max (Exploding/Chaotic) -> Need more damping -> Increase Gamma
        # If growth_ratio < target_min (Vanishing/Ordered) -> Need less damping -> Decrease Gamma

        delta = 0.0

        if growth_ratio > self.target_max:
            # Too chaotic, increase damping
            delta = self.adjustment_rate * (growth_ratio - self.target_max)
        elif growth_ratio < self.target_min:
            # Too damped, decrease damping (or make negative to boost)
            delta = self.adjustment_rate * (growth_ratio - self.target_min) # Negative value

        return delta

    def forward(self, current_gamma: torch.Tensor, diagnostics: Dict[str, float]) -> torch.Tensor:
        """
        Update gamma.

        Args:
            current_gamma: Tensor parameter.
            diagnostics: Dict with 'growth_ratio'.

        Returns:
            updated_gamma: Detached value or new tensor?
                           Usually we update the parameter in-place or return the new value.
                           Here we return the *target* value for the next step.
        """
        delta = self.compute_adjustment(diagnostics)
        new_gamma = current_gamma.item() + delta

        # Clamp
        new_gamma = max(self.gamma_min, min(self.gamma_max, new_gamma))

        return torch.tensor(new_gamma, dtype=current_gamma.dtype, device=current_gamma.device)
