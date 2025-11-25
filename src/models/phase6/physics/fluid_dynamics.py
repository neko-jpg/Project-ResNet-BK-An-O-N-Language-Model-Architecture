import torch
import torch.nn as nn

class ViscousFlow(nn.Module):
    """
    Viscous Flow Dynamics for Data Loading.

    Calculates the 'Viscosity' (Importance/Stickiness) of input data.
    High viscosity data 'sticks' to the core (Hyperbolic Origin),
    while low viscosity data flows to the periphery.

    Viscosity $\eta$ is derived from:
    1. Inverse Token Frequency (Rare words = Sticky)
    2. Gradient Norm (High learning signal = Sticky) - during training
    """

    def __init__(self, vocab_size: int, decay: float = 0.99):
        super().__init__()
        self.vocab_size = vocab_size
        self.decay = decay
        # Track running frequency of tokens
        self.register_buffer('token_counts', torch.ones(vocab_size))
        self.register_buffer('total_tokens', torch.tensor(float(vocab_size)))

    def update_counts(self, x: torch.Tensor):
        """Update token frequencies."""
        # Simple counting
        ids, counts = torch.unique(x, return_counts=True)
        self.token_counts[ids] += counts.float()
        self.total_tokens += x.numel()

    def compute_viscosity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute viscosity score for input batch.

        Args:
            x: (B, N) token ids

        Returns:
            viscosity: (B, N) score in [0, 1]
        """
        freq = self.token_counts[x] / self.total_tokens

        # Viscosity ~ 1 / log(freq) (TF-IDF like)
        # Or simply: rare = high viscosity
        # freq is small -> -log(freq) is large
        viscosity = -torch.log(freq + 1e-9)

        # Normalize to [0, 1] range approximately
        viscosity = torch.sigmoid(viscosity - 5.0)

        return viscosity

    def get_loss_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get weights for loss function.
        High viscosity = Higher weight (Must learn).
        """
        v = self.compute_viscosity(x)
        # Scale weight: 1.0 (base) + v * alpha
        return 1.0 + v * 2.0
