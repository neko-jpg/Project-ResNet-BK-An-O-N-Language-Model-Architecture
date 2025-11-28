import torch
import torch.nn as nn
from typing import Optional, Dict

class TopologicalNorm(nn.Module):
    """
    Implements Topological Normalization (Task 6).
    Regularizes embeddings based on persistent homology features.
    """
    def __init__(self, d_model: int, persistence_threshold: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.persistence_threshold = persistence_threshold
        self.topo_bias = nn.Parameter(torch.zeros(d_model))
        self.scale = nn.Parameter(torch.ones(d_model))
        self.last_metric = 0.0

    def _approximate_persistence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Approximates topological complexity using Variance of Pairwise Distances.

        Clustered Data -> High Variance (Small intra-cluster, Large inter-cluster).
        Uniform Data -> Lower Variance (Smoother distribution).
        """
        B, N, D = x.shape
        if N > 64:
            indices = torch.randperm(N)[:64]
            x_sub = x[:, indices, :]
        else:
            x_sub = x

        x_norm = (x_sub ** 2).sum(-1, keepdim=True)
        dist_sq = x_norm + x_norm.transpose(1, 2) - 2 * torch.matmul(x_sub, x_sub.transpose(1, 2))
        dist = dist_sq.clamp(min=1e-6).sqrt()

        # We want the variance of the off-diagonal elements
        # Flatten the last two dimensions
        flat_dist = dist.view(B, -1)

        # Calculate variance
        variance = flat_dist.var(dim=-1)

        # Return mean variance across batch
        return variance.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True)
        x_norm = (x - mu) / (sigma + 1e-5)

        with torch.no_grad():
            topo_metric = self._approximate_persistence(x)
            self.last_metric = topo_metric.item()

        # Scale modulation: If variance is high (clustered), allow more expression (higher scale).
        # We normalize the metric slightly to be around 1.0 roughly for expected inputs
        # But here we just use it raw with a small factor.
        modulation = 1.0 + 0.1 * torch.tanh(topo_metric)

        out = x_norm * self.scale * modulation + self.topo_bias
        return out

    def get_diagnostics(self) -> Dict[str, float]:
        return {"topo_metric": self.last_metric}
