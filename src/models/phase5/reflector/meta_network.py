import torch
import torch.nn as nn
import torch.nn.functional as F

class Reflector(nn.Module):
    """
    Reflector (Meta-Network).

    A lightweight network that observes the agent's hidden state and internal logs,
    then updates the physical parameters of the system (Gamma, Bump, Decay, etc.).

    Architecture:
        Input -> MLP (2-3 layers) -> GELU -> Output Params

    Inputs:
        hidden_state: The current hidden state of the model (B, D).
        log_embedding: Aggregated embedding of the inner speech (B, D_log).

    Outputs:
        Physics parameters (gamma, bump_scale, decay_rate, etc.)
    """

    def __init__(
        self,
        d_model: int,
        d_log: int = 0, # If 0, only use hidden_state
        hidden_dim: int = 256,
        output_dim: int = 4, # gamma, bump, decay, non-hermitian_gain
    ):
        super().__init__()

        input_dim = d_model + d_log

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize last layer with small weights to start near neutral
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

        # Output mapping names for clarity
        self.param_names = ['gamma', 'bump_scale', 'decay_rate', 'gain']

    def forward(self, hidden_state: torch.Tensor, log_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Predict new physics parameters.

        Args:
            hidden_state: (B, D)
            log_embedding: (B, D_log) or None

        Returns:
            params: (B, 4) -> [gamma, bump, decay, gain]
        """
        if log_embedding is not None:
            x = torch.cat([hidden_state, log_embedding], dim=-1)
        else:
            x = hidden_state

        out = self.net(x)

        # Apply specific activations if needed to constrain ranges
        # e.g., gamma shouldn't be too negative, bump should be positive
        # But for now, we let the optimizer/controller handle ranges or use raw delta.
        # We can use Tanh to bound the *change*.

        return torch.tanh(out) # Return normalized updates/deltas
