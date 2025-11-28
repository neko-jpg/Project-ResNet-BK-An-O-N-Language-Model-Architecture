import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CurvatureAdapter(nn.Module):
    """
    Implements Dynamic Curvature Adaptation (Task 26).
    Adjusts the curvature 'c' of hyperbolic space based on the
    hierarchical structure of the input embeddings.
    """
    def __init__(self, d_model: int, c_min: float = 0.1, c_max: float = 5.0):
        super().__init__()
        self.d_model = d_model
        self.c_min = c_min
        self.c_max = c_max

        # Learnable curvature parameter (unconstrained logic, mapped later)
        self.raw_c = nn.Parameter(torch.tensor([0.0]))

        # Adaptation network: Maps hierarchy score to delta_c
        self.adapter = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh() # Limit update step
        )

    def _estimate_hierarchy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates the "tree-likeness" or hierarchical depth of the batch.
        Proxy: Ratio of norm variance to mean norm.
        In a hierarchy (like a tree embedded in Poincare ball), nodes are distributed
        radially. Root at 0, leaves near boundary.
        High variance in norms => High Hierarchy.
        All norms equal => Flat structure.
        """
        norms = x.norm(dim=-1)
        norm_var = norms.var(dim=-1, keepdim=True) # (B, 1)
        norm_mean = norms.mean(dim=-1, keepdim=True) + 1e-5

        # Coefficient of Variation
        cv = norm_var.sqrt() / norm_mean
        return cv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the adapted curvature 'c' for the current batch.
        """
        # 1. Base curvature
        base_c = F.softplus(self.raw_c) + self.c_min

        # 2. Compute Hierarchy Score
        # No grad for the signal itself, we want to learn HOW to adapt, not change x to fit.
        with torch.no_grad():
            hierarchy_score = self._estimate_hierarchy(x)

        # 3. Compute Adaptation Delta
        # If hierarchy is high, we might want lower curvature (flatter) to fit more levels?
        # Or higher curvature to curve space more?
        # Actually, for deep trees, we need higher negative curvature (larger 'c' in some conventions,
        # but here c usually means -K. So high c = more curved).
        # High curvature allows exponentially more volume -> fits larger trees.
        delta = self.adapter(hierarchy_score.mean().unsqueeze(0).unsqueeze(0))

        # 4. Apply
        # We allow the adapter to shift c by up to +/- 1.0 (tanh) scaled
        c = base_c + delta

        return c.clamp(self.c_min, self.c_max).squeeze()
