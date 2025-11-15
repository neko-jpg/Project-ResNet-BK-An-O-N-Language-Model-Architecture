"""
Sparse BK-Core: Learned Sparsity for BK-Core Computation

Implements learned sparsity where an importance predictor determines which
G_ii diagonal elements to compute, with interpolation for masked positions.

Includes adaptive sparsity scheduling for balancing sparsity vs accuracy during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bk_core import BKCoreFunction, vmapped_get_diag


class AdaptiveSparsityScheduler:
    """
    Adaptive scheduler for sparsity target and loss weight.
    
    Dynamically adjusts sparsity target and loss weight during training to:
    1. Gradually increase sparsity (curriculum learning for sparsity)
    2. Adjust loss weight based on accuracy performance
    3. Balance exploration (trying different sparsity levels) vs exploitation (optimizing current level)
    
    Args:
        initial_sparsity: starting sparsity target (e.g., 0.2 = 20% sparse)
        final_sparsity: final sparsity target (e.g., 0.5 = 50% sparse)
        initial_weight: starting sparsity loss weight
        final_weight: final sparsity loss weight
        warmup_steps: number of steps to reach final sparsity
        schedule_type: 'linear', 'cosine', or 'step'
        accuracy_threshold: if accuracy drops below this, reduce sparsity
    """
    
    def __init__(
        self,
        initial_sparsity=0.2,
        final_sparsity=0.5,
        initial_weight=0.001,
        final_weight=0.01,
        warmup_steps=1000,
        schedule_type='cosine',
        accuracy_threshold=None
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.accuracy_threshold = accuracy_threshold
        
        self.current_step = 0
        self.best_accuracy = float('inf')  # Lower is better for loss
    
    def step(self, current_accuracy=None):
        """
        Update scheduler state.
        
        Args:
            current_accuracy: current accuracy loss (optional, for adaptive adjustment)
        
        Returns:
            dict with current_sparsity_target and current_weight
        """
        self.current_step += 1
        
        # Compute progress
        progress = min(1.0, self.current_step / self.warmup_steps)
        
        # Compute sparsity target based on schedule
        if self.schedule_type == 'linear':
            sparsity_target = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing: smooth transition
            sparsity_target = self.initial_sparsity + 0.5 * (self.final_sparsity - self.initial_sparsity) * (
                1 - torch.cos(torch.tensor(progress * 3.14159)).item()
            )
        
        elif self.schedule_type == 'step':
            # Step schedule: increase sparsity in discrete steps
            if progress < 0.25:
                sparsity_target = self.initial_sparsity
            elif progress < 0.5:
                sparsity_target = self.initial_sparsity + 0.33 * (self.final_sparsity - self.initial_sparsity)
            elif progress < 0.75:
                sparsity_target = self.initial_sparsity + 0.67 * (self.final_sparsity - self.initial_sparsity)
            else:
                sparsity_target = self.final_sparsity
        
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")
        
        # Compute loss weight
        weight = self.initial_weight + progress * (self.final_weight - self.initial_weight)
        
        # Adaptive adjustment based on accuracy
        if current_accuracy is not None and self.accuracy_threshold is not None:
            if current_accuracy > self.accuracy_threshold:
                # Accuracy is poor, reduce sparsity target
                sparsity_target *= 0.9
                weight *= 0.5
            
            # Track best accuracy
            if current_accuracy < self.best_accuracy:
                self.best_accuracy = current_accuracy
        
        return {
            'sparsity_target': sparsity_target,
            'loss_weight': weight,
            'progress': progress,
            'step': self.current_step
        }
    
    def reset(self):
        """Reset scheduler to initial state."""
        self.current_step = 0
        self.best_accuracy = float('inf')


def sparse_theta_recursion(he_diag, h0_super, h0_sub, z, mask):
    """
    Sparse theta recursion: skip computation for masked positions.
    
    Theta recursion: theta[i] = (a[i]-z)*theta[i-1] - b[i-1]*c[i-1]*theta[i-2]
    
    For masked positions, we still need to compute theta values because
    they're needed for subsequent positions. However, we can use a simplified
    computation or interpolation.
    
    Args:
        he_diag: (B, N) effective Hamiltonian diagonal
        h0_super: (B, N-1) super-diagonal
        h0_sub: (B, N-1) sub-diagonal
        z: complex spectral shift
        mask: (B, N) binary mask (1 = compute, 0 = skip)
    
    Returns:
        theta: (B, N+1) complex theta values
    """
    B, N = he_diag.shape
    device = he_diag.device
    dtype = torch.complex128
    
    # Convert to complex
    a = he_diag.to(dtype)
    b = h0_super.to(dtype)
    c = h0_sub.to(dtype)
    z_complex = z.to(dtype)
    
    # Initialize theta
    theta = torch.zeros(B, N + 1, dtype=dtype, device=device)
    theta[:, 0] = 1.0 + 0.0j
    
    # First element (always computed)
    a_shifted = a[:, 0] - z_complex
    theta[:, 1] = a_shifted
    
    # Recursion with sparsity
    for i in range(1, N):
        # Check if this position is masked (for any batch element)
        is_masked = (mask[:, i] < 0.5).any()
        
        if is_masked:
            # For masked positions, use simplified computation
            # Option 1: Linear interpolation from neighbors
            # Option 2: Use only the diagonal term (ignore coupling)
            # We use Option 2 for efficiency
            a_shifted = a[:, i] - z_complex
            theta[:, i + 1] = a_shifted * theta[:, i]
        else:
            # Full computation for important positions
            a_shifted = a[:, i] - z_complex
            term1 = a_shifted * theta[:, i]
            term2 = c[:, i - 1] * b[:, i - 1] * theta[:, i - 1]
            theta[:, i + 1] = term1 - term2
    
    return theta


def sparse_phi_recursion(he_diag, h0_super, h0_sub, z, mask):
    """
    Sparse phi recursion: skip computation for masked positions.
    
    Phi recursion: phi[i] = (a[i+1]-z)*phi[i+1] - b[i]*c[i]*phi[i+2]
    
    Args:
        he_diag: (B, N) effective Hamiltonian diagonal
        h0_super: (B, N-1) super-diagonal
        h0_sub: (B, N-1) sub-diagonal
        z: complex spectral shift
        mask: (B, N) binary mask (1 = compute, 0 = skip)
    
    Returns:
        phi: (B, N) complex phi values
    """
    B, N = he_diag.shape
    device = he_diag.device
    dtype = torch.complex128
    
    # Convert to complex
    a = he_diag.to(dtype)
    b = h0_super.to(dtype)
    c = h0_sub.to(dtype)
    z_complex = z.to(dtype)
    
    # Initialize phi
    phi = torch.zeros(B, N + 2, dtype=dtype, device=device)
    phi[:, N + 1] = 0.0 + 0.0j
    phi[:, N] = 1.0 + 0.0j
    
    # Backward recursion with sparsity
    for i in range(N - 1, -1, -1):
        # Check if this position is masked
        is_masked = (mask[:, i] < 0.5).any()
        
        if is_masked:
            # Simplified computation for masked positions
            a_shifted = a[:, i] - z_complex
            phi[:, i] = a_shifted * phi[:, i + 1]
        else:
            # Full computation for important positions
            a_shifted = a[:, i] - z_complex
            term1 = a_shifted * phi[:, i + 1]
            if i < N - 1:
                term2 = b[:, i] * c[:, i] * phi[:, i + 2]
            else:
                term2 = 0.0
            phi[:, i] = term1 - term2
    
    return phi[:, :N]


def optimized_sparse_bk_core(he_diag, h0_super, h0_sub, z, mask):
    """
    Optimized sparse BK-Core computation.
    
    This function implements a sparse-aware algorithm that:
    1. Skips full recursion for masked positions
    2. Uses simplified computation (diagonal-only) for masked positions
    3. Maintains numerical stability
    
    Args:
        he_diag: (B, N) effective Hamiltonian diagonal
        h0_super: (B, N-1) super-diagonal
        h0_sub: (B, N-1) sub-diagonal
        z: complex spectral shift
        mask: (B, N) binary mask (1 = compute, 0 = skip)
    
    Returns:
        features: (B, N, 2) [real(G_ii), imag(G_ii)]
    """
    B, N = he_diag.shape
    
    # Sparse theta recursion
    theta = sparse_theta_recursion(he_diag, h0_super, h0_sub, z, mask)
    
    # Sparse phi recursion
    phi = sparse_phi_recursion(he_diag, h0_super, h0_sub, z, mask)
    
    # Compute G_ii = theta[:-1] * phi / det_T
    det_T = theta[:, -1]
    
    # Numerical stability: avoid division by very small numbers
    det_T_mag = det_T.abs()
    det_T_stable = torch.where(
        det_T_mag < 1e-9,
        det_T / det_T_mag * 1e-9,
        det_T
    )
    
    # G_ii computation
    numerator = theta[:, :-1] * phi
    G_ii = numerator / (det_T_stable.unsqueeze(1) + 1e-18)
    
    # Convert to features
    features = torch.stack([G_ii.real, G_ii.imag], dim=-1).to(torch.float32)
    
    return features


class SparseBKCore(nn.Module):
    """
    BK-Core with learned sparsity: predict which G_ii elements to compute.
    
    Architecture:
        1. Importance Predictor: x -> importance_scores
        2. Gumbel-Sigmoid: scores -> binary mask (differentiable)
        3. Sparse BK-Core: compute G_ii only for masked positions
        4. Interpolation Network: fill in masked positions
    
    Args:
        d_model: hidden dimension
        n_seq: sequence length
        target_sparsity: target sparsity ratio (0.5 = 50% sparse)
        tau: temperature for Gumbel-Sigmoid (lower = more discrete)
    """
    
    def __init__(self, d_model, n_seq, target_sparsity=0.5, tau=1.0):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.target_sparsity = target_sparsity
        self.tau = tau
        
        # Importance predictor: which positions are important
        self.importance_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Standard BK-Core function
        self.bk_core = BKCoreFunction.apply
        
        # Interpolation network: fill in masked positions
        # Uses 1D convolution to interpolate based on neighboring values
        self.interpolator = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1)
        )
        
        # H0 (discrete Laplacian) as buffers
        self.register_buffer("h0_diag_base", torch.full((1, n_seq), -2.0, dtype=torch.float32))
        self.register_buffer("h0_sub_base", torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))
        self.register_buffer("h0_super_base", torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))
        
        # Spectral shift z as buffer
        self.register_buffer("z", torch.tensor(1.0j, dtype=torch.complex64))
    
    def gumbel_sigmoid(self, logits, tau=1.0, hard=True):
        """
        Gumbel-Sigmoid: differentiable approximation to binary sampling.
        
        Args:
            logits: (B, N) importance scores
            tau: temperature (lower = more discrete)
            hard: if True, return hard binary mask (straight-through estimator)
        
        Returns:
            mask: (B, N) binary mask (0 or 1)
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        
        # Add noise to logits
        noisy_logits = (logits + gumbel_noise) / tau
        
        # Sigmoid
        soft_mask = torch.sigmoid(noisy_logits)
        
        if hard:
            # Straight-through estimator: hard mask in forward, soft mask in backward
            hard_mask = (soft_mask > 0.5).float()
            mask = hard_mask - soft_mask.detach() + soft_mask
        else:
            mask = soft_mask
        
        return mask
    
    def forward(self, x, v, use_sparse_computation=True):
        """
        Forward pass with learned sparsity.
        
        Args:
            x: (B, N, D) - input features (for importance prediction)
            v: (B, N) - potential
            use_sparse_computation: if True, use optimized sparse algorithm
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)]
            mask: (B, N) - binary mask of computed positions
            sparsity_ratio: scalar - actual sparsity ratio
        """
        B, N, D = x.shape
        device = x.device
        
        # 1. Predict importance scores
        importance_scores = self.importance_predictor(x).squeeze(-1)  # (B, N)
        
        # 2. Gumbel-Sigmoid for differentiable binary mask
        mask = self.gumbel_sigmoid(importance_scores, tau=self.tau, hard=True)  # (B, N)
        
        # 3. Expand H0 for batch
        h0_diag = self.h0_diag_base.expand(B, -1)   # (B, N)
        h0_sub = self.h0_sub_base.expand(B, -1)     # (B, N-1)
        h0_super = self.h0_super_base.expand(B, -1) # (B, N-1)
        
        he_diag = h0_diag + v  # (B, N)
        
        if use_sparse_computation:
            # 4a. Optimized sparse BK-Core computation
            # Skip full recursion for masked positions
            features_sparse = optimized_sparse_bk_core(
                he_diag, h0_super, h0_sub, self.z, mask
            )  # (B, N, 2)
            
            # 5a. Interpolate missing positions
            # Zero out masked positions before interpolation
            features_masked = features_sparse * mask.unsqueeze(-1)
            
            # Interpolator expects (B, C, N) format
            features_interpolated = self.interpolator(
                features_masked.permute(0, 2, 1)
            ).permute(0, 2, 1)  # (B, N, 2)
            
            # 6a. Combine: use computed values where available, interpolated otherwise
            features_final = torch.where(
                mask.unsqueeze(-1) > 0.5,
                features_sparse,
                features_interpolated
            )
        else:
            # 4b. Standard full BK-Core computation (for comparison)
            features_full = self.bk_core(he_diag, h0_super, h0_sub, self.z)  # (B, N, 2)
            
            # 5b. Apply mask: zero out non-important positions
            features_sparse = features_full * mask.unsqueeze(-1)  # (B, N, 2)
            
            # 6b. Interpolate missing positions
            features_interpolated = self.interpolator(
                features_sparse.permute(0, 2, 1)
            ).permute(0, 2, 1)  # (B, N, 2)
            
            # 7b. Combine: use computed values where available, interpolated otherwise
            features_final = torch.where(
                mask.unsqueeze(-1) > 0.5,
                features_sparse,
                features_interpolated
            )
        
        # Compute actual sparsity ratio
        sparsity_ratio = 1.0 - mask.mean()
        
        return features_final, mask, sparsity_ratio
    
    def sparsity_loss(self, mask, loss_type='l2'):
        """
        Encourage target sparsity level with multiple loss formulations.
        
        Supports different loss types:
        - 'l2': Squared error (current_sparsity - target_sparsity)^2
        - 'l1': Absolute error |current_sparsity - target_sparsity|
        - 'kl': KL divergence between current and target sparsity distributions
        - 'adaptive': Adaptive loss that penalizes more when far from target
        
        Args:
            mask: (B, N) binary mask (1 = compute, 0 = skip)
            loss_type: type of sparsity loss ('l2', 'l1', 'kl', 'adaptive')
        
        Returns:
            loss: scalar sparsity loss
        """
        current_sparsity = 1.0 - mask.mean()  # Fraction of masked (not computed) positions
        target_sparsity = self.target_sparsity
        
        if loss_type == 'l2':
            # Squared error: penalizes large deviations quadratically
            loss = (current_sparsity - target_sparsity) ** 2
        
        elif loss_type == 'l1':
            # Absolute error: linear penalty
            loss = torch.abs(current_sparsity - target_sparsity)
        
        elif loss_type == 'kl':
            # KL divergence: treats sparsity as probability distribution
            # KL(target || current) = target * log(target/current) + (1-target) * log((1-target)/(1-current))
            eps = 1e-8
            current_sparsity_clamped = torch.clamp(current_sparsity, eps, 1.0 - eps)
            
            if target_sparsity > eps and target_sparsity < 1.0 - eps:
                kl_term1 = target_sparsity * torch.log(target_sparsity / current_sparsity_clamped)
                kl_term2 = (1 - target_sparsity) * torch.log((1 - target_sparsity) / (1 - current_sparsity_clamped))
                loss = kl_term1 + kl_term2
            else:
                # Fallback to L2 for edge cases
                loss = (current_sparsity - target_sparsity) ** 2
        
        elif loss_type == 'adaptive':
            # Adaptive loss: stronger penalty when far from target
            # Uses smooth L1 loss (Huber loss) for robustness
            diff = current_sparsity - target_sparsity
            abs_diff = torch.abs(diff)
            
            # Huber loss: quadratic for small errors, linear for large errors
            delta = 0.1  # Threshold for switching from quadratic to linear
            if abs_diff < delta:
                loss = 0.5 * diff ** 2
            else:
                loss = delta * (abs_diff - 0.5 * delta)
        
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'l2', 'l1', 'kl', 'adaptive'")
        
        return loss
    
    def balanced_sparsity_loss(self, mask, accuracy_loss, sparsity_weight=1.0, accuracy_weight=1.0):
        """
        Balanced loss that trades off sparsity and accuracy.
        
        Total Loss = accuracy_weight * accuracy_loss + sparsity_weight * sparsity_loss
        
        The weights can be adjusted to control the trade-off:
        - Higher sparsity_weight: prioritize sparsity (more aggressive pruning)
        - Higher accuracy_weight: prioritize accuracy (less pruning)
        
        Args:
            mask: (B, N) binary mask
            accuracy_loss: scalar accuracy loss (e.g., cross-entropy)
            sparsity_weight: weight for sparsity loss
            accuracy_weight: weight for accuracy loss
        
        Returns:
            total_loss: scalar balanced loss
            loss_dict: dictionary with individual loss components
        """
        # Compute sparsity loss
        sparsity_loss_val = self.sparsity_loss(mask, loss_type='adaptive')
        
        # Compute total loss
        total_loss = accuracy_weight * accuracy_loss + sparsity_weight * sparsity_loss_val
        
        # Return loss components for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'accuracy_loss': accuracy_loss,
            'sparsity_loss': sparsity_loss_val,
            'sparsity_weight': sparsity_weight,
            'accuracy_weight': accuracy_weight,
            'current_sparsity': 1.0 - mask.mean(),
            'target_sparsity': self.target_sparsity
        }
        
        return total_loss, loss_dict


class SparseMoEResNetBKLayer(nn.Module):
    """
    MoE-ResNet-BK Layer with Sparse BK-Core.
    
    Replaces standard BK-Core with SparseBKCore for learned sparsity.
    """
    
    def __init__(
        self,
        d_model,
        n_seq,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        target_sparsity=0.5,
        sparsity_loss_weight=0.01,
        sparsity_loss_type='adaptive',
        use_sparse_computation=True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.sparsity_loss_weight = sparsity_loss_weight
        self.sparsity_loss_type = sparsity_loss_type
        self.use_sparse_computation = use_sparse_computation
        
        # Import here to avoid circular dependency
        from .moe import SparseMoELayer
        
        self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k, dropout_p)
        self.v_proj = nn.Linear(d_model, 1)
        
        # Sparse BK-Core (replaces standard BK-Core)
        self.sparse_bk_core = SparseBKCore(d_model, n_seq, target_sparsity)
        
        # BK-Core output (real, imag) -> d_model
        self.output_proj = nn.Linear(2, d_model)
        
        # Learnable scale for BK branch contribution
        self.bk_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        # Numerical stability parameters
        self.v_max = 3.0
        self.feature_clamp = 10.0
        
        # Store last mask and sparsity for monitoring
        self.last_mask = None
        self.last_sparsity = None
    
    def forward(self, x):
        """
        Forward pass with sparse BK-Core.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) combined FFN + sparse BK features
        """
        B, N, D = x.shape
        assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"
        
        # MoE-FFN
        ffn_out = self.moe_ffn(x)  # (B, N, D)
        
        # Potential v_i
        v = self.v_proj(ffn_out).squeeze(-1)  # (B, N)
        v = torch.clamp(v, -self.v_max, self.v_max)
        
        # Sparse BK-Core
        features, mask, sparsity_ratio = self.sparse_bk_core(
            x, v, use_sparse_computation=self.use_sparse_computation
        )  # (B, N, 2), (B, N), scalar
        
        # Store for monitoring
        self.last_mask = mask.detach()
        self.last_sparsity = sparsity_ratio.detach()
        
        # Clip BK features
        if self.feature_clamp is not None:
            features = torch.clamp(features, -self.feature_clamp, self.feature_clamp)
        
        spec_out = self.output_proj(features)  # (B, N, D)
        
        # Mix BK branch with learnable scale
        return ffn_out + self.bk_scale * spec_out
    
    def get_sparsity_loss(self, loss_type=None):
        """
        Get sparsity regularization loss.
        
        Args:
            loss_type: type of sparsity loss (None = use default from init)
        
        Returns:
            loss: scalar sparsity loss (or 0 if no mask available)
        """
        if self.last_mask is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if loss_type is None:
            loss_type = self.sparsity_loss_type
        
        loss = self.sparse_bk_core.sparsity_loss(self.last_mask, loss_type=loss_type)
        return self.sparsity_loss_weight * loss
    
    def get_balanced_loss(self, accuracy_loss, sparsity_weight=None, accuracy_weight=1.0):
        """
        Get balanced loss that trades off sparsity and accuracy.
        
        Args:
            accuracy_loss: scalar accuracy loss (e.g., cross-entropy)
            sparsity_weight: weight for sparsity loss (None = use default from init)
            accuracy_weight: weight for accuracy loss
        
        Returns:
            total_loss: scalar balanced loss
            loss_dict: dictionary with individual loss components
        """
        if self.last_mask is None:
            return accuracy_loss, {
                'total_loss': accuracy_loss,
                'accuracy_loss': accuracy_loss,
                'sparsity_loss': torch.tensor(0.0, device=accuracy_loss.device),
                'sparsity_weight': 0.0,
                'accuracy_weight': accuracy_weight,
                'current_sparsity': 0.0,
                'target_sparsity': self.sparse_bk_core.target_sparsity
            }
        
        if sparsity_weight is None:
            sparsity_weight = self.sparsity_loss_weight
        
        return self.sparse_bk_core.balanced_sparsity_loss(
            self.last_mask, accuracy_loss, sparsity_weight, accuracy_weight
        )
    
    def get_sparsity_stats(self):
        """
        Get sparsity statistics for monitoring.
        
        Returns:
            dict with sparsity_ratio and num_computed
        """
        if self.last_mask is None or self.last_sparsity is None:
            return {'sparsity_ratio': 0.0, 'num_computed': self.n_seq}
        
        num_computed = self.last_mask.sum(dim=-1).mean().item()
        
        return {
            'sparsity_ratio': self.last_sparsity.item(),
            'num_computed': num_computed,
            'target_sparsity': self.sparse_bk_core.target_sparsity
        }
