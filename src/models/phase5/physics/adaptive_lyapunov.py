import torch
from src.models.phase4.homeostasis import HomeostasisController

class AdaptiveLyapunovControl(HomeostasisController):
    """
    Adaptive Lyapunov Control (ALC).

    Dynamically adjusts the decay parameter Gamma based on local Lyapunov exponents.

    Rule:
    Gamma_t = Gamma_base + sigma * (lambda_local(t) - lambda_target)

    Where lambda_local is estimated from the divergence of trajectories.
    """

    def __init__(
        self,
        base_gamma: float = 0.01,
        target_lyapunov: float = -0.05, # Slightly negative (stable but close to chaos)
        sigma: float = 0.1, # Sensitivity
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_gamma = base_gamma
        self.target_lyapunov = target_lyapunov
        self.sigma = sigma

    def estimate_local_lyapunov(self, x_t: torch.Tensor, x_next: torch.Tensor) -> float:
        """
        Estimate local Lyapunov exponent from Jacobian norm or proxy.

        lambda ~ log || J ||

        We approximate this by observing the expansion/contraction ratio of the norm.
        """
        norm_t = x_t.norm(p=2, dim=-1).mean()
        norm_next = x_next.norm(p=2, dim=-1).mean()

        # log( ||x_{t+1}|| / ||x_t|| )
        local_lambda = torch.log( (norm_next + 1e-9) / (norm_t + 1e-9) )
        return local_lambda.item()

    def compute_gamma(self, local_lambda: float) -> float:
        """
        Compute the new Gamma value.
        """
        # Linear feedback control
        # If local_lambda > target (Too chaotic), increase Gamma (Damping)
        # If local_lambda < target (Too stable), decrease Gamma

        delta = self.sigma * (local_lambda - self.target_lyapunov)
        new_gamma = self.base_gamma + delta

        # Clamp
        return max(self.gamma_min, min(self.gamma_max, new_gamma))
