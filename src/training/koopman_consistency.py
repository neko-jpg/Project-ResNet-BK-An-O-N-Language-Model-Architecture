"""
Koopman Consistency

Training regularization for dynamics stability using Koopman operator theory.

Key insight: For stable long-term dynamics, the Koopman operator's eigenvalues
must lie within or on the unit circle. If any eigenvalue has |λ| > 1, the
system will diverge over time.

Features:
- Eigenloss: Penalize eigenvalues outside the unit circle
- Spectral radius monitoring
- Multi-step consistency regularization
- Lyapunov stability constraints

References:
- Miller et al., "Eigenvalue Initialization and Regularization for Koopman Autoencoders" (2023)
- Koopman Operator Theory for dynamical systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class EigenlossRegularizer(nn.Module):
    """
    Eigenvalue-based loss for Koopman operator stability.
    
    Penalizes eigenvalues that exceed the target spectral radius.
    
    L_eigen = Σ max(0, |λ_i| - target_radius)^p
    
    Args:
        target_spectral_radius: Maximum allowed eigenvalue magnitude (default: 0.95)
        penalty_power: Exponent for penalty (1 = linear, 2 = quadratic)
        penalty_weight: Weight for eigenloss term
    """
    
    def __init__(
        self,
        target_spectral_radius: float = 0.95,
        penalty_power: float = 2.0,
        penalty_weight: float = 0.01
    ):
        super().__init__()
        self.target_spectral_radius = target_spectral_radius
        self.penalty_power = penalty_power
        self.penalty_weight = penalty_weight
    
    def forward(self, K: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute eigenloss for Koopman matrix K.
        
        Args:
            K: Koopman operator matrix of shape (dim, dim)
            
        Returns:
            loss: Eigenloss value
            metrics: Dictionary with eigenvalue statistics
        """
        metrics = {}
        
        try:
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvals(K)
            magnitudes = eigenvalues.abs()
            
            # Maximum eigenvalue magnitude
            max_eigen = magnitudes.max()
            mean_eigen = magnitudes.mean()
            
            metrics['max_eigenvalue'] = max_eigen.item()
            metrics['mean_eigenvalue'] = mean_eigen.item()
            
            # Penalize eigenvalues exceeding target
            excess = F.relu(magnitudes - self.target_spectral_radius)
            loss = (excess ** self.penalty_power).sum() * self.penalty_weight
            
            # Count how many eigenvalues are unstable
            num_unstable = (magnitudes > 1.0).sum().item()
            metrics['num_unstable_eigenvalues'] = num_unstable
            
        except Exception as e:
            # Fallback: use spectral norm as proxy
            try:
                spectral_norm = torch.linalg.matrix_norm(K, ord=2)
                loss = F.relu(spectral_norm - self.target_spectral_radius) ** self.penalty_power
                loss = loss * self.penalty_weight
                metrics['spectral_norm'] = spectral_norm.item()
                metrics['max_eigenvalue'] = spectral_norm.item()  # Upper bound
            except Exception:
                loss = torch.tensor(0.0, device=K.device, dtype=K.dtype)
                metrics['error'] = str(e)
        
        return loss, metrics


class SpectralNormConstraint(nn.Module):
    """
    Spectral norm constraint for implicit eigenvalue control.
    
    Faster than computing eigenvalues, provides an upper bound on spectral radius.
    
    Args:
        target_norm: Target spectral norm (default: 0.99)
        n_power_iterations: Power iterations for spectral norm (default: 1)
    """
    
    def __init__(
        self,
        target_norm: float = 0.99,
        n_power_iterations: int = 1
    ):
        super().__init__()
        self.target_norm = target_norm
        self.n_power_iterations = n_power_iterations
    
    def forward(self, K: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute spectral norm penalty."""
        try:
            # Power iteration approximation of spectral norm
            u = torch.randn(K.shape[0], device=K.device, dtype=K.dtype)
            u = u / u.norm()
            
            for _ in range(self.n_power_iterations):
                v = K.T @ u
                v = v / (v.norm() + 1e-10)
                u = K @ v
                u = u / (u.norm() + 1e-10)
            
            spectral_norm = (u @ K @ v).abs()
            
            loss = F.relu(spectral_norm - self.target_norm) ** 2
            
            return loss, {'spectral_norm': spectral_norm.item()}
            
        except Exception:
            return torch.tensor(0.0, device=K.device), {}


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for Koopman dynamics.
    
    Ensures that the encoded dynamics are consistent:
    φ(x_{t+1}) ≈ K @ φ(x_t)
    
    Args:
        multi_step: Include multi-step consistency (2, 4, 8 steps)
        step_weights: Weights for each step horizon
    """
    
    def __init__(
        self,
        multi_step: bool = True,
        step_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.multi_step = multi_step
        if step_weights is None:
            # Default: decreasing weights for longer horizons
            self.step_weights = [1.0, 0.5, 0.25, 0.125]
        else:
            self.step_weights = step_weights
    
    def forward(
        self,
        K: torch.Tensor,
        phi_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute consistency loss.
        
        Args:
            K: Koopman operator, shape (dim, dim)
            phi_sequence: Encoded state sequence, shape (batch, seq_len, dim)
            
        Returns:
            loss: Consistency loss
            metrics: Dictionary with per-step losses
        """
        batch_size, seq_len, dim = phi_sequence.shape
        metrics = {}
        total_loss = torch.tensor(0.0, device=K.device, dtype=K.dtype)
        
        if seq_len < 2:
            return total_loss, metrics
        
        # 1-step consistency
        for t in range(seq_len - 1):
            phi_t = phi_sequence[:, t, :]  # (batch, dim)
            phi_t1_pred = phi_t @ K.T  # (batch, dim)
            phi_t1_true = phi_sequence[:, t + 1, :]  # (batch, dim)
            
            step_loss = F.mse_loss(phi_t1_pred, phi_t1_true)
            total_loss = total_loss + step_loss * self.step_weights[0]
        
        metrics['1_step_loss'] = (total_loss / max(1, seq_len - 1)).item()
        
        if self.multi_step:
            # Multi-step consistency
            K_power = K
            for step_idx, horizon in enumerate([2, 4, 8]):
                if horizon >= seq_len:
                    break
                
                weight = self.step_weights[min(step_idx + 1, len(self.step_weights) - 1)]
                K_power = K_power @ K  # K^horizon
                
                horizon_loss = torch.tensor(0.0, device=K.device, dtype=K.dtype)
                count = 0
                
                for t in range(seq_len - horizon):
                    phi_t = phi_sequence[:, t, :]
                    phi_th_pred = phi_t @ K_power.T
                    phi_th_true = phi_sequence[:, t + horizon, :]
                    
                    horizon_loss = horizon_loss + F.mse_loss(phi_th_pred, phi_th_true)
                    count += 1
                
                if count > 0:
                    horizon_loss = horizon_loss / count
                    total_loss = total_loss + horizon_loss * weight
                    metrics[f'{horizon}_step_loss'] = horizon_loss.item()
        
        return total_loss, metrics


class LyapunovConstraint(nn.Module):
    """
    Lyapunov stability constraint.
    
    Ensures existence of a Lyapunov function V(x) such that:
    V(Kx) < V(x) for all x ≠ 0
    
    This is equivalent to K^T P K - P < 0 for some positive definite P.
    
    Args:
        dim: State dimension
        penalty_weight: Weight for Lyapunov constraint
    """
    
    def __init__(self, dim: int, penalty_weight: float = 0.01):
        super().__init__()
        
        # Learnable Lyapunov matrix P (parameterized as L @ L^T + εI for PD)
        self.L = nn.Parameter(torch.eye(dim) * 0.1)
        self.penalty_weight = penalty_weight
    
    def get_P(self) -> torch.Tensor:
        """Get the positive definite Lyapunov matrix P = L @ L^T + εI."""
        return self.L @ self.L.T + 1e-4 * torch.eye(self.L.shape[0], device=self.L.device)
    
    def forward(self, K: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute Lyapunov constraint violation.
        
        Args:
            K: Koopman operator, shape (dim, dim)
            
        Returns:
            loss: Lyapunov constraint penalty
            metrics: Constraint statistics
        """
        P = self.get_P()
        
        # Compute K^T P K - P (should be negative definite for stability)
        KtPK = K.T @ P @ K
        diff = KtPK - P
        
        # Check positive definiteness of -diff (eigenvalues should be positive)
        try:
            eigenvalues = torch.linalg.eigvalsh(diff)
            
            # Penalize positive eigenvalues of diff (violations of Lyapunov condition)
            violations = F.relu(eigenvalues)
            loss = violations.sum() * self.penalty_weight
            
            max_violation = eigenvalues.max().item()
            is_stable = max_violation < 0
            
            return loss, {
                'lyapunov_max_violation': max_violation,
                'lyapunov_stable': is_stable
            }
            
        except Exception:
            # Fallback: check trace (necessary condition)
            trace_cond = torch.trace(diff)
            loss = F.relu(trace_cond) * self.penalty_weight
            
            return loss, {'lyapunov_trace': trace_cond.item()}


class KoopmanConsistencyLoss(nn.Module):
    """
    Combined Koopman consistency loss for training.
    
    Combines:
    - Eigenloss for spectral stability
    - Consistency loss for dynamics accuracy
    - Optional Lyapunov constraint
    
    Args:
        dim: State dimension
        target_spectral_radius: Target max eigenvalue magnitude
        eigenloss_weight: Weight for eigenloss
        consistency_weight: Weight for consistency loss
        lyapunov_weight: Weight for Lyapunov constraint (0 to disable)
    """
    
    def __init__(
        self,
        dim: Optional[int] = None,
        target_spectral_radius: float = 0.95,
        eigenloss_weight: float = 0.01,
        consistency_weight: float = 0.1,
        lyapunov_weight: float = 0.0
    ):
        super().__init__()
        
        self.eigenloss = EigenlossRegularizer(
            target_spectral_radius=target_spectral_radius,
            penalty_weight=eigenloss_weight
        )
        
        self.consistency = ConsistencyLoss(multi_step=True)
        self.consistency_weight = consistency_weight
        
        if lyapunov_weight > 0 and dim is not None:
            self.lyapunov = LyapunovConstraint(dim, penalty_weight=lyapunov_weight)
        else:
            self.lyapunov = None
    
    def forward(
        self,
        K: torch.Tensor,
        phi_sequence: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined Koopman consistency loss.
        
        Args:
            K: Koopman operator matrix
            phi_sequence: Optional encoded state sequence for consistency
            
        Returns:
            total_loss: Combined loss
            metrics: All component metrics
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=K.device, dtype=K.dtype)
        
        # Eigenloss
        eigen_loss, eigen_metrics = self.eigenloss(K)
        total_loss = total_loss + eigen_loss
        metrics.update({f'koopman_{k}': v for k, v in eigen_metrics.items()})
        
        # Consistency loss
        if phi_sequence is not None:
            consist_loss, consist_metrics = self.consistency(K, phi_sequence)
            total_loss = total_loss + consist_loss * self.consistency_weight
            metrics.update({f'koopman_{k}': v for k, v in consist_metrics.items()})
        
        # Lyapunov constraint
        if self.lyapunov is not None:
            lyap_loss, lyap_metrics = self.lyapunov(K)
            total_loss = total_loss + lyap_loss
            metrics.update({f'koopman_{k}': v for k, v in lyap_metrics.items()})
        
        metrics['koopman_total_loss'] = total_loss.item()
        
        return total_loss, metrics


# =============================================================================
# Utility Functions
# =============================================================================

def init_stable_koopman(dim: int, target_spectral_radius: float = 0.9) -> torch.Tensor:
    """
    Initialize a Koopman matrix with eigenvalues inside the unit circle.
    
    Args:
        dim: Matrix dimension
        target_spectral_radius: Maximum eigenvalue magnitude
        
    Returns:
        Initialized Koopman matrix
    """
    # Create random orthogonal matrix (eigenvalues on unit circle)
    Q, _ = torch.linalg.qr(torch.randn(dim, dim))
    
    # Scale to bring eigenvalues inside circle
    K = Q * target_spectral_radius
    
    return K


def get_koopman_stability_report(K: torch.Tensor) -> Dict:
    """
    Generate a comprehensive stability report for a Koopman matrix.
    
    Args:
        K: Koopman operator matrix
        
    Returns:
        Report dictionary with stability metrics
    """
    report = {}
    
    try:
        eigenvalues = torch.linalg.eigvals(K)
        magnitudes = eigenvalues.abs()
        
        report['max_eigenvalue'] = magnitudes.max().item()
        report['min_eigenvalue'] = magnitudes.min().item()
        report['mean_eigenvalue'] = magnitudes.mean().item()
        report['spectral_radius'] = magnitudes.max().item()
        report['num_unstable'] = (magnitudes > 1.0).sum().item()
        report['num_stable'] = (magnitudes <= 1.0).sum().item()
        report['is_stable'] = magnitudes.max().item() < 1.0
        
        # Condition number
        report['condition_number'] = torch.linalg.cond(K).item()
        
        # Frobenius norm
        report['frobenius_norm'] = torch.linalg.matrix_norm(K, 'fro').item()
        
    except Exception as e:
        report['error'] = str(e)
    
    return report
