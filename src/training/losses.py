# src/training/losses.py
# This file contains custom loss functions for Project MUSE.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FreeEnergyLoss(nn.Module):
    """
    Implements the Free Energy Principle (FEP) loss, which is composed of
    an Accuracy term and a Complexity term.

    L = Complexity - Accuracy

    This is equivalent to minimizing:
    L = Accuracy_Loss + Complexity_Loss

    Where:
    - Accuracy_Loss is the standard Cross-Entropy loss (Negative Log-Likelihood).
    - Complexity_Loss is the KL divergence between the posterior Q(z|x) and a prior P(z).
    """

    def __init__(self, hidden_dim: int, kl_weight: float = 1.0):
        """
        Args:
            hidden_dim (int): The dimensionality of the hidden state vectors (z).
            kl_weight (float): A weighting factor for the KL divergence term.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kl_weight = kl_weight

        # This small linear layer acts as the "Uncertainty Head", predicting the
        # log variance of the posterior distribution Q(z|x) from the hidden state.
        self.uncertainty_head = nn.Linear(hidden_dim, hidden_dim)

        # The Accuracy term is the standard cross-entropy loss.
        self.accuracy_loss = nn.CrossEntropyLoss()

    def _calculate_kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL divergence between a diagonal Gaussian Q(z|x) = N(mu, exp(log_var))
        and a standard Gaussian prior P(z) = N(0, I).

        The formula is: 0.5 * sum(exp(log_var) + mu^2 - 1 - log_var)

        Args:
            mu (torch.Tensor): The mean of the posterior distribution.
            log_var (torch.Tensor): The log variance of the posterior distribution.

        Returns:
            torch.Tensor: The KL divergence loss.
        """
        # Calculate KL divergence per element
        kl_div_per_element = 0.5 * (torch.exp(log_var) + mu.pow(2) - 1 - log_var)

        # Sum over the feature dimension and average over the batch and sequence length
        # This gives a single scalar value for the complexity loss.
        kl_div_loss = kl_div_per_element.sum(dim=-1).mean()

        return kl_div_loss

    def forward(self, logits: torch.Tensor, hidden_states: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Free Energy Loss.

        Args:
            logits (torch.Tensor): The raw output from the language model head.
                                   Shape: (batch_size, seq_len, vocab_size)
            hidden_states (torch.Tensor): The final hidden states from the model backbone.
                                          Shape: (batch_size, seq_len, hidden_dim)
            targets (torch.Tensor): The target token IDs.
                                    Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: The total scalar loss.
        """
        # Reshape logits and targets for CrossEntropyLoss
        # (batch, seq_len, vocab_size) -> (batch * seq_len, vocab_size)
        # (batch, seq_len) -> (batch * seq_len)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        targets_flat = targets.view(batch_size * seq_len)

        # 1. Accuracy Term
        acc_loss = self.accuracy_loss(logits_flat, targets_flat)

        # 2. Complexity Term
        # The hidden state is treated as the mean (mu) of the posterior distribution Q(z|x)
        mu = hidden_states

        # Predict the log variance from the hidden state using the uncertainty head
        log_var = self.uncertainty_head(mu)

        # Calculate the KL divergence
        comp_loss = self._calculate_kl_divergence(mu, log_var)

        # 3. Total Loss
        # The final loss is a weighted sum of the two terms.
        total_loss = acc_loss + self.kl_weight * comp_loss

        return total_loss


# =============================================================================
# Phase 2: BK-Core Physics Loss (ResNet-BK Specialized)
# =============================================================================

class BKCorePhysicsLoss(nn.Module):
    """
    BK-Core Physics Loss for ResNet-BK architecture.
    
    Enforces physical constraints derived from Green's function theory:
    
    1. Causality: Im(G_ii) > 0 (time-ordering / retarded Green function)
       - Penalty for negative imaginary parts
       - Ensures physically meaningful propagators
    
    2. Unitarity: W†W ≈ I for BK-Core projection layers
       - Scattering matrix must be unitary: S†S = I
       - Energy conservation in scattering processes
    
    3. Energy Conservation: ||x_out|| ≈ ||x_in|| per layer
       - Signal energy should be preserved through layers
       - Prevents gradient vanishing/explosion
    
    4. Spectral Regularization: σ_max(W) ≤ threshold
       - Largest singular value bounded
       - Prevents weight explosion
    
    These loss terms provide auxiliary gradients that flow directly to deep layers,
    bypassing the vanishing gradient problem.
    
    Args:
        causality_weight: Weight for Im(G) > 0 penalty
        unitarity_weight: Weight for W†W = I deviation
        energy_weight: Weight for energy conservation
        spectral_weight: Weight for spectral norm regularization
        spectral_threshold: Maximum allowed singular value
    """
    
    def __init__(
        self,
        causality_weight: float = 0.1,
        unitarity_weight: float = 0.1,
        energy_weight: float = 0.05,
        spectral_weight: float = 0.01,
        spectral_threshold: float = 2.0,
    ):
        super().__init__()
        self.causality_weight = causality_weight
        self.unitarity_weight = unitarity_weight
        self.energy_weight = energy_weight
        self.spectral_weight = spectral_weight
        self.spectral_threshold = spectral_threshold
    
    def forward(
        self,
        model: nn.Module,
        G_ii: torch.Tensor = None,
        intermediate_states: list = None,
    ) -> dict:
        """
        Compute physics-based auxiliary loss terms.
        
        Args:
            model: The ResNet-BK model (for weight access)
            G_ii: Green function diagonal from BK-Core (optional)
                  Can be complex tensor or real tensor with shape (..., 2)
            intermediate_states: List of (input, output) tensor pairs per layer (optional)
        
        Returns:
            Dict with 'total' and individual loss components:
            - 'causality': Penalty for Im(G) < 0
            - 'unitarity': Deviation from W†W = I
            - 'energy': Energy non-conservation penalty
            - 'spectral': Singular value regularization
        """
        losses = {}
        device = next(model.parameters()).device
        
        # 1. Causality Loss: Im(G_ii) must be positive
        if G_ii is not None:
            if G_ii.is_complex():
                imag_G = G_ii.imag
            elif G_ii.dim() >= 1 and G_ii.shape[-1] == 2:
                imag_G = G_ii[..., 1]
            else:
                imag_G = None
            
            if imag_G is not None:
                # Penalty for negative imaginary parts (violates causality)
                causality_violation = F.relu(-imag_G)  # Only positive when Im(G) < 0
                losses['causality'] = self.causality_weight * causality_violation.mean()
            else:
                losses['causality'] = torch.tensor(0.0, device=device)
        else:
            losses['causality'] = torch.tensor(0.0, device=device)
        
        # 2. Unitarity Loss: v_proj, output_proj should be approximately orthogonal
        unitarity_loss = torch.tensor(0.0, device=device)
        unitary_layer_count = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'v_proj') and isinstance(module.v_proj, nn.Linear):
                W = module.v_proj.weight
                if W.dim() == 2:
                    m, n = W.shape
                    if m <= n:
                        # W @ W^T = I_m
                        WWT = W @ W.T
                        I = torch.eye(m, device=device, dtype=W.dtype)
                        unitarity_loss = unitarity_loss + torch.norm(WWT - I, p='fro')
                    else:
                        # W^T @ W = I_n
                        WTW = W.T @ W
                        I = torch.eye(n, device=device, dtype=W.dtype)
                        unitarity_loss = unitarity_loss + torch.norm(WTW - I, p='fro')
                    unitary_layer_count += 1
            
            if hasattr(module, 'output_proj') and isinstance(module.output_proj, nn.Linear):
                W = module.output_proj.weight
                if W.dim() == 2:
                    m, n = W.shape
                    if m <= n:
                        WWT = W @ W.T
                        I = torch.eye(m, device=device, dtype=W.dtype)
                        unitarity_loss = unitarity_loss + torch.norm(WWT - I, p='fro')
                    else:
                        WTW = W.T @ W
                        I = torch.eye(n, device=device, dtype=W.dtype)
                        unitarity_loss = unitarity_loss + torch.norm(WTW - I, p='fro')
                    unitary_layer_count += 1
        
        if unitary_layer_count > 0:
            losses['unitarity'] = self.unitarity_weight * unitarity_loss / unitary_layer_count
        else:
            losses['unitarity'] = torch.tensor(0.0, device=device)
        
        # 3. Energy Conservation: ||x_out|| ≈ ||x_in|| for each layer
        if intermediate_states is not None and len(intermediate_states) > 0:
            energy_loss = torch.tensor(0.0, device=device)
            for x_in, x_out in intermediate_states:
                in_norm = x_in.norm(dim=-1)
                out_norm = x_out.norm(dim=-1)
                # Penalize deviation from unity ratio
                ratio = out_norm / (in_norm + 1e-8)
                energy_loss = energy_loss + (ratio - 1.0).abs().mean()
            losses['energy'] = self.energy_weight * energy_loss / len(intermediate_states)
        else:
            losses['energy'] = torch.tensor(0.0, device=device)
        
        # 4. Spectral Regularization: σ_max(W) ≤ threshold
        spectral_loss = torch.tensor(0.0, device=device)
        linear_count = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() == 2:
                W = module.weight
                # Approximate largest singular value via power iteration
                # Fast approximation: ||W||_F / sqrt(min(m,n))
                # More accurate: power iteration
                try:
                    # Power iteration (3 steps for speed)
                    v = torch.randn(W.shape[1], device=device, dtype=W.dtype)
                    v = v / (v.norm() + 1e-8)
                    for _ in range(3):
                        u = W @ v
                        u = u / (u.norm() + 1e-8)
                        v = W.T @ u
                        v = v / (v.norm() + 1e-8)
                    sigma_max = (W @ v).norm()
                    
                    # Penalty if exceeds threshold
                    spectral_loss = spectral_loss + F.relu(sigma_max - self.spectral_threshold)
                    linear_count += 1
                except:
                    pass
        
        if linear_count > 0:
            losses['spectral'] = self.spectral_weight * spectral_loss / linear_count
        else:
            losses['spectral'] = torch.tensor(0.0, device=device)
        
        # Total loss
        losses['total'] = sum(v for v in losses.values() if v is not None and v != losses)
        
        return losses
    
    def get_loss_str(self, losses: dict) -> str:
        """Format losses for logging."""
        return " | ".join(f"{k}:{v.item():.4f}" for k, v in losses.items())


def create_bk_physics_loss(
    causality_weight: float = 0.1,
    unitarity_weight: float = 0.1,
    energy_weight: float = 0.05,
    spectral_weight: float = 0.01,
) -> BKCorePhysicsLoss:
    """Factory function for BKCorePhysicsLoss."""
    return BKCorePhysicsLoss(
        causality_weight=causality_weight,
        unitarity_weight=unitarity_weight,
        energy_weight=energy_weight,
        spectral_weight=spectral_weight,
    )
