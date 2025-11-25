import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple

from src.models.phase5.monad.writer_monad import WriterMonad
from src.models.phase5.reflector.meta_network import Reflector

class ConsciousnessMonad(nn.Module):
    """
    Consciousness Monad (State Monad).

    Manages the "State" of the agent's mind, including:
    - Short-term memory (Hidden State)
    - Physical Parameters (Gamma, Bump, Decay)
    - Inner Speech (Writer Monad)

    It integrates the Reflector to update the state based on thoughts.
    """

    def __init__(
        self,
        d_model: int,
        reflector_hidden_dim: int = 256,
    ):
        super().__init__()
        self.writer = WriterMonad()
        self.reflector = Reflector(d_model=d_model, hidden_dim=reflector_hidden_dim)

        # Internal State Parameters (Scalar or Vector)
        # We maintain these as buffers or parameters depending on if we want gradients
        # For now, they are dynamic states passed through the forward loop,
        # but we store base values here.
        self.register_buffer('gamma', torch.tensor(0.0))
        self.register_buffer('bump_scale', torch.tensor(0.02))
        self.register_buffer('decay_rate', torch.tensor(0.01))

    def update_physics(self, hidden_state: torch.Tensor, log_embedding: Optional[torch.Tensor] = None):
        """
        Use the Reflector to update physics parameters based on current state and thoughts.
        """
        # Reflector outputs deltas
        deltas = self.reflector(hidden_state, log_embedding) # (B, 4)

        # Average deltas across batch if needed, or keep per-batch
        # Assuming batch processing, we might average for global parameters
        # or keep them per sequence. For simplicity, let's update global state average.

        avg_delta = deltas.mean(dim=0) # (4,)

        # Update states (soft updates)
        # params: [gamma, bump, decay, gain]
        self.gamma = self.gamma + avg_delta[0] * 0.1
        self.bump_scale = self.bump_scale + avg_delta[1] * 0.01
        self.decay_rate = self.decay_rate + avg_delta[2] * 0.001

        # Clamp values to sane ranges
        self.bump_scale = torch.clamp(self.bump_scale, 0.0, 1.0)
        self.decay_rate = torch.clamp(self.decay_rate, 0.0, 1.0)

        return {
            'gamma': self.gamma,
            'bump_scale': self.bump_scale,
            'decay_rate': self.decay_rate
        }

    def log_thought(self, message: str, embedding: Optional[torch.Tensor] = None):
        """Delegate to Writer Monad."""
        self.writer.tell(message, embedding)

    def get_inner_voice(self):
        """Get accumulated thoughts."""
        return self.writer.listen()

    def clear_mind(self):
        """Flush thoughts."""
        self.writer.flush()
