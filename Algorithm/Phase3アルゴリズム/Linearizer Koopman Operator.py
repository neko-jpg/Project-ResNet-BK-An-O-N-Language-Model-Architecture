"""
Phase 3.5: Koopman Global Linearization

非線形力学系を無限次元の線形空間（Koopman空間）に持ち上げ、推論を高速化する。

Algorithm:
    1. Encoder (Psi): x_t -> g_t (Observables)
    2. Evolution (K): g_{t+1} = K * g_t (Linear Dynamics)
    3. Decoder (Psi_inv): g_{t+1} -> x_{t+1}

Benefit:
    - Inference becomes Matrix Multiplication
    - Spectral Analysis of K reveals "concepts" (eigenvalues)
"""

import torch
import torch.nn as nn

class KoopmanOperator(nn.Module):
    def __init__(self, d_model, d_koopman, n_steps_pred=1):
        super().__init__()
        self.d_model = d_model
        self.d_koopman = d_koopman
        self.n_steps_pred = n_steps_pred
        
        # Observable function (Encoder)
        # Maps physical state x to Koopman invariant subspace g(x)
        self.psi = nn.Sequential(
            nn.Linear(d_model, d_koopman * 2),
            nn.GELU(),
            nn.Linear(d_koopman * 2, d_koopman)
        )
        
        # Koopman Operator K (Complex matrix allowed, but simulated with real block)
        # To ensure stable evolution, K should be close to unitary or strictly bounded
        self.K = nn.Linear(d_koopman, d_koopman, bias=False)
        
        # Inverse Observable (Decoder)
        self.psi_inv = nn.Sequential(
            nn.Linear(d_koopman, d_koopman * 2),
            nn.GELU(),
            nn.Linear(d_koopman * 2, d_model)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize K to be close to Identity for stable start
        nn.init.eye_(self.K.weight)
        # Add small noise
        self.K.weight.data += torch.randn_like(self.K.weight.data) * 0.01

    def forward(self, x):
        """
        Forward pass:
        1. Lift x to g space
        2. Evolve linearly K*g
        3. Project back to x space
        """
        # Lift
        g = self.psi(x)
        
        # Linear Evolution
        g_next = self.K(g)
        
        # Project back
        x_next_pred = self.psi_inv(g_next)
        
        return x_next_pred, g, g_next

    def predict_future(self, x, steps=10):
        """
        Multi-step prediction using only linear operator K
        Extremely fast inference for autoregressive generation
        """
        g = self.psi(x)
        predictions = []
        
        curr_g = g
        for _ in range(steps):
            curr_g = self.K(curr_g)
            pred_x = self.psi_inv(curr_g)
            predictions.append(pred_x)
            
        return torch.stack(predictions, dim=1)

    def linearity_loss(self, g, g_next):
        """
        Auxiliary loss to enforce linearity: ||g(x_{t+1}) - K*g(x_t)||
        However, in forward we computed g_next = K*g. 
        We need true next state.
        """
        # This should be called in training loop with actual x_t and x_{t+1}
        pass