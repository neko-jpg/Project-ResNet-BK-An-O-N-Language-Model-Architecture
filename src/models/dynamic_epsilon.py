import torch
import torch.nn as nn
from typing import Optional, Dict

class DynamicEpsilon(nn.Module):
    """
    Dynamic Epsilon Controller.

    Adjusts the regularization parameter epsilon based on the condition number
    of the previous step to maintain spectral stability.

    Rule:
        epsilon_t = epsilon_base + alpha * log(condition_number)
    """

    def __init__(self, base_epsilon: float = 1.0, alpha: float = 0.1, max_epsilon: float = 10.0):
        super().__init__()
        self.base_epsilon = base_epsilon
        self.alpha = alpha
        self.max_epsilon = max_epsilon
        self.last_condition_number = 1.0

    def forward(self, condition_number: float) -> float:
        """
        Compute dynamic epsilon.
        """
        # Update history (simple lag-1)
        self.last_condition_number = condition_number

        # Guard against inf
        if condition_number == float('inf'):
            cond = 1e6 # Cap
        else:
            cond = max(1.0, condition_number)

        # Logarithmic scaling
        # if cond is 1e4 -> log10 is 4 -> add 0.4
        boost = self.alpha * torch.log10(torch.tensor(cond)).item()

        new_eps = self.base_epsilon + boost
        return min(self.max_epsilon, new_eps)
