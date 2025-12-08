"""
Hyperbolic Loss Functions

Loss functions that respect hyperbolic geometry for training
neural networks in non-Euclidean spaces.

Key insight: Standard cross-entropy assumes Euclidean distance.
In hyperbolic space, distances grow exponentially from the origin,
so we need geodesic-aware loss functions.

Features:
- Hyperbolic Cross-Entropy using geodesic distances
- Fisher Information-based quantization loss (RSAVQ)
- Manifold-aware label smoothing
- Koopman consistency loss (for dynamics stability)

References:
- RSAVQ: Riemannian Sensitivity-Aware Vector Quantization
- Lorentzian Distance Learning (Law et al., ICML 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# =============================================================================
# Hyperbolic Distance Functions
# =============================================================================

def poincare_distance(u: torch.Tensor, v: torch.Tensor, curvature: float = -1.0) -> torch.Tensor:
    """
    Geodesic distance in Poincaré ball model.
    
    d(u, v) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-u) ⊕ v||)
    
    where ⊕ is Möbius addition.
    """
    c = abs(curvature)
    sqrt_c = math.sqrt(c)
    
    # Möbius addition: (-u) ⊕ v
    # For Poincaré ball with curvature c:
    # (-u) ⊕ v = ((1 + 2c<-u, v> + c||v||²)(-u) + (1 - c||u||²)v) / 
    #            (1 + 2c<-u, v> + c²||u||²||v||²)
    
    u_sq = (u ** 2).sum(dim=-1, keepdim=True)
    v_sq = (v ** 2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    
    # Numerator
    num_u_coeff = 1 - 2 * c * uv + c * v_sq
    num_v_coeff = 1 - c * u_sq
    num = num_u_coeff * (-u) + num_v_coeff * v
    
    # Denominator
    denom = 1 - 2 * c * uv + c * c * u_sq * v_sq
    
    # Möbius addition result
    mobius = num / (denom + 1e-8)
    
    # Distance
    mobius_norm = torch.sqrt((mobius ** 2).sum(dim=-1).clamp(min=1e-10))
    dist = (2 / sqrt_c) * torch.atanh((sqrt_c * mobius_norm).clamp(max=1 - 1e-5))
    
    return dist


def lorentz_distance_batch(x: torch.Tensor, y: torch.Tensor, curvature: float = -1.0) -> torch.Tensor:
    """
    Batch geodesic distance in Lorentz model.
    
    d_L(x, y) = (1/sqrt(-c)) * arcosh(-c * <x, y>_L)
    
    Args:
        x: Shape (batch, dim)
        y: Shape (classes, dim) or (batch, dim)
        
    Returns:
        Distances shape (batch, classes) or (batch,)
    """
    c = abs(curvature)
    
    # Minkowski inner product: -x_0*y_0 + x_1*y_1 + ...
    if y.dim() == 2 and x.shape[0] != y.shape[0]:
        # Broadcasting: x @ y^T with Minkowski metric
        # <x, y>_L = -x_0*y_0 + x_space @ y_space^T
        inner = -x[:, 0:1] @ y[:, 0:1].T + x[:, 1:] @ y[:, 1:].T
    else:
        inner = -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(dim=-1)
    
    # Clamp for numerical stability (arcosh domain is [1, inf))
    clamped = torch.clamp(-c * inner, min=1.0 + 1e-7)
    return torch.acosh(clamped) / math.sqrt(c)


# =============================================================================
# Hyperbolic Cross-Entropy Loss
# =============================================================================

class HyperbolicCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss using hyperbolic (geodesic) distances.
    
    Instead of logits, we use negative geodesic distances to class prototypes
    as the basis for softmax.
    
    P(class=k | x) = exp(-d(x, prototype_k) / τ) / Σ_j exp(-d(x, prototype_j) / τ)
    
    Args:
        num_classes: Number of classes
        embed_dim: Embedding dimension
        curvature: Hyperbolic curvature (negative)
        temperature: Softmax temperature
        model: "poincare" or "lorentz"
        margin: Optional margin for contrastive-style loss
    """
    
    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        curvature: float = -1.0,
        temperature: float = 0.1,
        model: str = "lorentz",
        margin: float = 0.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.curvature = curvature
        self.temperature = temperature
        self.model = model
        self.margin = margin
        self.label_smoothing = label_smoothing
        
        # Initialize class prototypes on the manifold
        if model == "lorentz":
            # Lorentz: (time, space), need dim+1 dimensions
            self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
            self._init_lorentz_prototypes()
        else:
            # Poincaré: points in the ball
            self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.1)
    
    def _init_lorentz_prototypes(self):
        """Initialize prototypes on the hyperboloid."""
        with torch.no_grad():
            # Random spatial directions, project to hyperboloid
            spatial = self.prototypes.data[:, 1:]
            spatial = F.normalize(spatial, dim=-1) * 0.5  # Scale for reasonable distance
            
            # Compute time component: t = sqrt(1/c + ||space||²)
            c = abs(self.curvature)
            space_sq = (spatial ** 2).sum(dim=-1, keepdim=True)
            time = torch.sqrt(1.0 / c + space_sq)
            
            self.prototypes.data = torch.cat([time, spatial], dim=-1)
    
    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """Project points onto the hyperbolic manifold."""
        if self.model == "lorentz":
            c = abs(self.curvature)
            space = x[..., 1:]
            space_sq = (space ** 2).sum(dim=-1, keepdim=True)
            time = torch.sqrt(1.0 / c + space_sq)
            return torch.cat([time, space], dim=-1)
        else:
            # Poincaré: project inside ball
            norm = x.norm(dim=-1, keepdim=True)
            max_norm = 1.0 - 1e-5
            return x * (max_norm / norm.clamp(min=max_norm))
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute hyperbolic cross-entropy loss.
        
        Args:
            embeddings: Hyperbolic embeddings, shape (batch, embed_dim)
            targets: Class indices, shape (batch,)
            reduction: "mean", "sum", or "none"
            
        Returns:
            Loss value
        """
        # Project embeddings to manifold
        embeddings = self.project_to_manifold(embeddings)
        prototypes = self.project_to_manifold(self.prototypes)
        
        # Compute geodesic distances to all prototypes
        if self.model == "lorentz":
            distances = lorentz_distance_batch(embeddings, prototypes, self.curvature)
        else:
            # Poincaré: compute distance to each prototype
            distances = torch.stack([
                poincare_distance(embeddings, prototypes[i:i+1].expand_as(embeddings), self.curvature)
                for i in range(self.num_classes)
            ], dim=-1)
        
        # Add margin to non-target classes (optional contrastive component)
        if self.margin > 0:
            target_mask = F.one_hot(targets, self.num_classes).float()
            distances = distances + self.margin * (1 - target_mask)
        
        # Convert distances to logits (negative distance = similarity)
        logits = -distances / self.temperature
        
        # Cross-entropy with optional label smoothing
        if self.label_smoothing > 0:
            log_probs = F.log_softmax(logits, dim=-1)
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / self.num_classes)
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.label_smoothing + self.label_smoothing / self.num_classes)
            loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            loss = F.cross_entropy(logits, targets, reduction="none")
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss


class HyperbolicLanguageModelLoss(nn.Module):
    """
    Language Model Loss for hyperbolic embeddings.
    
    Adapts standard LM loss to work with hyperbolic output embeddings.
    Uses geodesic distances for vocabulary matching.
    
    For efficiency, supports:
    - Standard CE for output layer (most common)
    - Hyperbolic distance for embedding matching
    - Mixed mode with geometric regularization
    
    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        curvature: Hyperbolic curvature
        use_hyperbolic_output: Use full hyperbolic output
        hyperbolic_weight: Weight for hyperbolic regularization
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        curvature: float = -1.0,
        use_hyperbolic_output: bool = False,
        hyperbolic_weight: float = 0.1,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        self.use_hyperbolic_output = use_hyperbolic_output
        self.hyperbolic_weight = hyperbolic_weight
        self.label_smoothing = label_smoothing
        
        # Standard output projection for efficiency
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Initialize with small values for stability
        nn.init.normal_(self.output_proj.weight, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean"
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute language model loss.
        
        Args:
            hidden_states: Shape (batch, seq_len, hidden_dim)
            targets: Shape (batch, seq_len)
            reduction: "mean" or "sum"
            
        Returns:
            loss: Scalar loss
            metrics: Dictionary of additional metrics
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Flatten for loss computation
        hidden_flat = hidden_states.view(-1, self.hidden_dim)
        targets_flat = targets.view(-1)
        
        # Standard cross-entropy through projection
        logits = self.output_proj(hidden_flat)
        
        # Main CE loss
        if self.label_smoothing > 0:
            ce_loss = F.cross_entropy(
                logits, targets_flat, 
                reduction=reduction,
                label_smoothing=self.label_smoothing,
                ignore_index=-100
            )
        else:
            ce_loss = F.cross_entropy(
                logits, targets_flat,
                reduction=reduction,
                ignore_index=-100
            )
        
        total_loss = ce_loss
        metrics = {"ce_loss": ce_loss.item()}
        
        # Optional hyperbolic regularization
        if self.use_hyperbolic_output and self.hyperbolic_weight > 0:
            # Regularize hidden states to stay near manifold surface
            # For Poincaré: ||x|| should be < 1 but not too small
            norms = hidden_flat.norm(dim=-1)
            
            # Encourage norms to be in a good range [0.3, 0.9]
            too_small = F.relu(0.3 - norms).mean()
            too_large = F.relu(norms - 0.9).mean()
            hyperbolic_reg = too_small + too_large
            
            total_loss = ce_loss + self.hyperbolic_weight * hyperbolic_reg
            metrics["hyperbolic_reg"] = hyperbolic_reg.item()
        
        return total_loss, metrics


# =============================================================================
# RSAVQ Loss (Riemannian Sensitivity-Aware Quantization)
# =============================================================================

class RSAVQLoss(nn.Module):
    """
    Riemannian Sensitivity-Aware Vector Quantization Loss.
    
    Instead of minimizing ||w_quant - w||₂, minimize ||w_quant - w||_F
    where F is the Fisher Information Matrix.
    
    This directs quantization error away from directions that affect loss.
    
    Args:
        alpha: Weight for RSAVQ term
        use_diagonal_fisher: Use diagonal approximation (efficient)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        use_diagonal_fisher: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.use_diagonal_fisher = use_diagonal_fisher
        
        # EMA for Fisher estimation
        self.register_buffer("fisher_diag", None)
        self.ema_decay = 0.99
    
    def update_fisher(self, model: nn.Module, gradients: dict):
        """
        Update Fisher diagonal estimate from gradients.
        
        Fisher diagonal ≈ E[g²] where g is the gradient.
        """
        for name, param in model.named_parameters():
            if name in gradients and param.requires_grad:
                grad_sq = gradients[name] ** 2
                
                if self.fisher_diag is None:
                    self.fisher_diag = {}
                
                if name not in self.fisher_diag:
                    self.fisher_diag[name] = grad_sq.detach()
                else:
                    self.fisher_diag[name] = (
                        self.ema_decay * self.fisher_diag[name] + 
                        (1 - self.ema_decay) * grad_sq.detach()
                    )
    
    def forward(
        self,
        w_continuous: torch.Tensor,
        w_quantized: torch.Tensor,
        fisher_diag: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute RSAVQ loss.
        
        Args:
            w_continuous: Original continuous weights
            w_quantized: Quantized weights
            fisher_diag: Optional Fisher diagonal for this parameter
            
        Returns:
            RSAVQ loss
        """
        # Quantization error
        error = w_quantized - w_continuous
        
        if fisher_diag is not None and self.use_diagonal_fisher:
            # Weight error by Fisher (sensitivity)
            weighted_error = error * torch.sqrt(fisher_diag + 1e-8)
            return self.alpha * (weighted_error ** 2).mean()
        else:
            # Standard L2 quantization error
            return self.alpha * (error ** 2).mean()


# =============================================================================
# Koopman Consistency Loss
# =============================================================================

class KoopmanConsistencyLoss(nn.Module):
    """
    Koopman Consistency Loss for dynamics stability.
    
    Ensures that the learned Koopman operator K satisfies:
    1. Spectral stability: |λ_max(K)| ≤ 1
    2. Consistency: φ(x_{t+1}) ≈ K φ(x_t)
    
    Args:
        spectral_weight: Weight for spectral penalty
        consistency_weight: Weight for consistency term
        target_spectral_radius: Target max eigenvalue magnitude (default: 0.95)
    """
    
    def __init__(
        self,
        spectral_weight: float = 0.01,
        consistency_weight: float = 0.1,
        target_spectral_radius: float = 0.95
    ):
        super().__init__()
        self.spectral_weight = spectral_weight
        self.consistency_weight = consistency_weight
        self.target_spectral_radius = target_spectral_radius
    
    def spectral_penalty(self, K: torch.Tensor) -> torch.Tensor:
        """
        Compute eigenloss: penalize eigenvalues outside unit circle.
        
        L_stable = Σ max(0, |λ_i| - target_radius)
        """
        try:
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(K)
            magnitudes = eigenvalues.abs()
            
            # Penalize eigenvalues exceeding target
            excess = F.relu(magnitudes - self.target_spectral_radius)
            return excess.sum()
        except Exception:
            # Fallback: use spectral norm as proxy
            spectral_norm = torch.linalg.matrix_norm(K, ord=2)
            return F.relu(spectral_norm - self.target_spectral_radius)
    
    def consistency_loss(
        self,
        phi_t: torch.Tensor,
        phi_t1: torch.Tensor,
        K: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency loss: φ(x_{t+1}) ≈ K φ(x_t)
        
        Args:
            phi_t: Encoded state at time t, shape (batch, dim)
            phi_t1: Encoded state at time t+1, shape (batch, dim)
            K: Koopman operator, shape (dim, dim)
            
        Returns:
            Consistency loss
        """
        # Predicted next state
        phi_t1_pred = phi_t @ K.T
        
        # MSE between prediction and actual
        return F.mse_loss(phi_t1_pred, phi_t1)
    
    def forward(
        self,
        K: torch.Tensor,
        phi_t: Optional[torch.Tensor] = None,
        phi_t1: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute total Koopman consistency loss.
        
        Args:
            K: Koopman operator matrix
            phi_t: Optional encoded states at time t
            phi_t1: Optional encoded states at time t+1
            
        Returns:
            loss: Total loss
            metrics: Dictionary of component losses
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=K.device)
        
        # Spectral penalty
        if self.spectral_weight > 0:
            spectral = self.spectral_penalty(K)
            total_loss = total_loss + self.spectral_weight * spectral
            metrics["koopman_spectral"] = spectral.item()
        
        # Consistency loss
        if self.consistency_weight > 0 and phi_t is not None and phi_t1 is not None:
            consistency = self.consistency_loss(phi_t, phi_t1, K)
            total_loss = total_loss + self.consistency_weight * consistency
            metrics["koopman_consistency"] = consistency.item()
        
        # Compute max eigenvalue for monitoring
        try:
            eigenvalues = torch.linalg.eigvals(K)
            metrics["koopman_max_eigenvalue"] = eigenvalues.abs().max().item()
        except Exception:
            metrics["koopman_max_eigenvalue"] = float("nan")
        
        return total_loss, metrics


# =============================================================================
# Combined Hyperbolic Training Loss
# =============================================================================

class HyperbolicTrainingLoss(nn.Module):
    """
    Combined loss for hyperbolic training.
    
    Combines:
    - Standard language model loss
    - Hyperbolic geometry regularization
    - RSAVQ quantization loss
    - Koopman consistency (optional)
    
    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        curvature: Hyperbolic curvature
        ce_weight: Cross-entropy weight
        hyperbolic_weight: Hyperbolic regularization weight
        rsavq_weight: RSAVQ quantization weight
        koopman_weight: Koopman consistency weight
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        curvature: float = -1.0,
        ce_weight: float = 1.0,
        hyperbolic_weight: float = 0.1,
        rsavq_weight: float = 0.01,
        koopman_weight: float = 0.01,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.ce_weight = ce_weight
        self.hyperbolic_weight = hyperbolic_weight
        self.rsavq_weight = rsavq_weight
        self.koopman_weight = koopman_weight
        
        # Sub-losses
        self.lm_loss = HyperbolicLanguageModelLoss(
            vocab_size, hidden_dim, curvature,
            use_hyperbolic_output=hyperbolic_weight > 0,
            hyperbolic_weight=hyperbolic_weight,
            label_smoothing=label_smoothing
        )
        
        self.rsavq_loss = RSAVQLoss(alpha=rsavq_weight) if rsavq_weight > 0 else None
        
        self.koopman_loss = KoopmanConsistencyLoss(
            spectral_weight=koopman_weight
        ) if koopman_weight > 0 else None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        koopman_matrix: Optional[torch.Tensor] = None,
        w_continuous: Optional[torch.Tensor] = None,
        w_quantized: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined hyperbolic training loss.
        
        Returns:
            total_loss: Combined loss
            metrics: Dictionary of all component losses
        """
        metrics = {}
        
        # Main LM loss
        lm_loss, lm_metrics = self.lm_loss(hidden_states, targets, reduction)
        total_loss = self.ce_weight * lm_loss
        metrics.update(lm_metrics)
        
        # RSAVQ loss
        if self.rsavq_loss is not None and w_continuous is not None and w_quantized is not None:
            rsavq = self.rsavq_loss(w_continuous, w_quantized)
            total_loss = total_loss + rsavq
            metrics["rsavq_loss"] = rsavq.item()
        
        # Koopman loss
        if self.koopman_loss is not None and koopman_matrix is not None:
            koopman, koopman_metrics = self.koopman_loss(koopman_matrix)
            total_loss = total_loss + koopman
            metrics.update(koopman_metrics)
        
        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics
