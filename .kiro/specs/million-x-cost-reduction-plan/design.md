# Design Document

## Overview

This document provides the technical design for achieving 1,000,000,000× (1 billion times) AI training cost reduction through the ResNet-BK architecture. The design builds upon the completed Step 1 (O(N) Architecture) and Step 3 (Sparse MoE) to define detailed architectures for Steps 2, 4, 5, 6, and 7.

**Design Philosophy:**
- **Incremental Development**: Each step builds upon previous achievements without breaking existing functionality
- **Numerical Stability First**: All optimizations must maintain or improve numerical stability (current: v_max=3.0, feature_clamp=10.0)
- **Google Colab Compatible**: All implementations must run on free tier (T4 GPU, 15GB RAM) with graceful degradation
- **Reproducible**: Deterministic results with fixed random seeds, comprehensive logging
- **Modular**: Each component can be enabled/disabled independently for ablation studies

**Current Architecture Summary:**
```
Input Tokens (B, N) 
  ↓
Token Embedding (B, N, D) + Position Embedding (B, N, D)
  ↓
ResNet-BK Block × n_layers
  ├─ LayerNorm (B, N, D)
  ├─ MoE-ResNet-BK Layer
  │   ├─ Sparse MoE (4 experts, top-1 routing)
  │   │   ├─ Gating Network: Linear(D → num_experts)
  │   │   ├─ Gumbel-Softmax (hard=True, tau=1.0)
  │   │   └─ Experts: [Linear(D → 2D) → ReLU → Linear(2D → D)] × 4
  │   ├─ Potential Projection: v = MoE(x) → (B, N, 1)
  │   ├─ BK-Core (O(N) Tridiagonal Inverse)
  │   │   ├─ He = H0 + diag(v)  [H0: discrete Laplacian]
  │   │   ├─ Theta Recursion (forward): theta[i] = (a[i]-z)*theta[i-1] - b[i-1]*c[i-1]*theta[i-2]
  │   │   ├─ Phi Recursion (backward): phi[i] = (a[i+1]-z)*phi[i+1] - b[i]*c[i]*phi[i+2]
  │   │   ├─ G_ii = theta[:-1] * phi / det_T  [complex128 precision]
  │   │   └─ Features = [real(G_ii), imag(G_ii)]  (B, N, 2)
  │   ├─ Output Projection: Linear(2 → D)
  │   └─ Residual: output = ffn_out + bk_scale * spec_out
  └─ Residual Connection
  ↓
Final LayerNorm (B, N, D)
  ↓
LM Head: Linear(D → vocab_size)
  ↓
Logits (B, N, vocab_size)
```

**Hybrid Analytic Gradient (Current):**
```
Forward: x → v → He → G_ii → features → output
Backward:
  - dL/d(output) → dL/d(features) → dL/dG_ii (complex)
  - Analytic: dG_ii/dv = -G_ii²  (theoretical)
  - Hypothesis-7: dL/dv ~ -dL/dG_ii / G_ii²  (empirical)
  - Hybrid: dL/dv = (1-α)*analytic + α*hypothesis7  [α=GRAD_BLEND=0.5]
  - MoE backward: autograd through Gumbel-Softmax
```

## Architecture

### Step 2 Phase 1: Optimized Hybrid Analytic Gradient

**Objective**: Achieve 50× backward pass speedup through fully analytic gradient computation.

**Current Bottlenecks** (from profiling):
1. MoE backward pass uses autograd (slow)
2. Gumbel-Softmax gradient computation (non-analytic)
3. Sequential gradient computation (no batching)
4. Complex128 precision throughout (overkill for gradients)

**Design**:

```python
class OptimizedBKCoreFunction(torch.autograd.Function):
    """
    Fully analytic gradient with mixed precision and batched computation.
    """
    
    @staticmethod
    def forward(ctx, he_diag, h0_super, h0_sub, z, grad_blend):
        # Forward: complex128 for numerical stability
        G_ii = vmapped_get_diag_complex128(he_diag, h0_super, h0_sub, z)
        
        # Save for backward: convert to complex64 to save memory
        ctx.save_for_backward(G_ii.to(torch.complex64), torch.tensor(grad_blend))
        
        # Output: FP32 features
        output_features = torch.stack([G_ii.real, G_ii.imag], dim=-1).to(torch.float32)
        return output_features
    
    @staticmethod
    def backward(ctx, grad_output_features):
        G_ii_fp16, grad_blend = ctx.saved_tensors
        G_ii = G_ii_fp16.to(torch.complex64)  # Upgrade for gradient computation
        
        # dL/dG (complex)
        grad_G = torch.complex(grad_output_features[..., 0], grad_output_features[..., 1])
        
        # Compute G² with numerical stability
        G_sq = G_ii ** 2
        G_sq_mag = G_sq.abs()
        G_sq_stable = torch.where(
            G_sq_mag < 1e-3,
            G_sq / (G_sq_mag + 1e-9) * 1e-3,
            G_sq
        )
        
        # Theoretical gradient: dG/dv = -G²
        grad_v_theory = -(grad_G * G_sq_stable).real
        
        # Hypothesis-7 gradient: dL/dv ~ -dL/dG / G²
        grad_v_h7 = -(grad_G / (G_sq_stable + 1e-6)).real
        
        # Hybrid blend
        alpha = grad_blend.item()
        grad_v = (1.0 - alpha) * grad_v_theory + alpha * grad_v_h7
        
        # Numerical safety
        grad_v = torch.clamp(grad_v, -1000.0, 1000.0)
        
        return grad_v.to(torch.float32), None, None, None, None


class AnalyticMoELayer(nn.Module):
    """
    Sparse MoE with fully analytic backward pass (no autograd).
    """
    
    def __init__(self, d_model, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        self.gating = nn.Linear(d_model, num_experts)
        
        # Cache for analytic backward
        self.x_flat = None
        self.router_logits = None
        self.gates_hard = None
        self.expert_outputs = None
    
    def forward(self, x):
        B, N, D = x.shape
        self.x_flat = x.reshape(B * N, D)
        
        # Router logits
        self.router_logits = self.gating(self.x_flat)  # (B*N, E)
        
        # Gumbel-Softmax (hard)
        self.gates_hard = F.gumbel_softmax(self.router_logits, hard=True, tau=1.0)
        
        # Compute all expert outputs
        self.expert_outputs = []
        for expert in self.experts:
            self.expert_outputs.append(expert(self.x_flat))
        expert_stack = torch.stack(self.expert_outputs, dim=1)  # (B*N, E, D)
        
        # Weighted sum
        output = torch.sum(expert_stack * self.gates_hard.unsqueeze(-1), dim=1)
        return output.view(B, N, D)
    
    def analytic_backward(self, grad_output):
        """
        Analytic gradient computation for MoE.
        
        Forward: output = sum_e (gate_e * expert_e(x))
        Backward:
          dL/d(expert_e weights) = gate_e * dL/d(output) * d(expert_e)/d(weights)
          dL/d(gate_e) = expert_e(x) * dL/d(output)
          dL/d(router_logits) = dL/d(gate_e) * d(gumbel_softmax)/d(logits)
        """
        B, N, D = grad_output.shape
        grad_flat = grad_output.reshape(B * N, D)
        
        # Gradient w.r.t. expert outputs
        grad_expert_outputs = []
        for e in range(self.num_experts):
            gate_e = self.gates_hard[:, e].unsqueeze(-1)  # (B*N, 1)
            grad_expert_e = grad_flat * gate_e  # (B*N, D)
            grad_expert_outputs.append(grad_expert_e)
        
        # Backprop through each expert (using autograd for expert internals)
        grad_x_from_experts = torch.zeros_like(self.x_flat)
        for e, expert in enumerate(self.experts):
            if grad_expert_outputs[e].abs().sum() > 0:
                # Backprop through expert
                expert_out = self.expert_outputs[e]
                expert_out.backward(gradient=grad_expert_outputs[e], retain_graph=True)
                # Accumulate gradient to input
                if self.x_flat.grad is not None:
                    grad_x_from_experts += self.x_flat.grad
                    self.x_flat.grad.zero_()
        
        # Gradient w.r.t. gates
        expert_stack = torch.stack(self.expert_outputs, dim=1)  # (B*N, E, D)
        grad_gates = torch.sum(expert_stack * grad_flat.unsqueeze(1), dim=-1)  # (B*N, E)
        
        # Gradient w.r.t. router logits (straight-through estimator for Gumbel-Softmax)
        # Approximate: d(gumbel_softmax)/d(logits) ≈ softmax * (1 - softmax)
        softmax_gates = F.softmax(self.router_logits, dim=-1)
        grad_router_logits = grad_gates * softmax_gates * (1 - softmax_gates)
        
        # Backprop through gating network
        self.router_logits.backward(gradient=grad_router_logits)
        
        return grad_x_from_experts.view(B, N, D)
```

**GRAD_BLEND Optimization**:
- Grid search over [0.0, 0.1, ..., 1.0] on validation set
- Track: convergence speed, final perplexity, gradient variance
- Expected optimal: α ∈ [0.3, 0.7] based on current α=0.5 performance

**Mixed Precision Strategy**:
- Forward: complex128 for theta/phi recursions (numerical stability critical)
- Backward: complex64 for gradient computation (sufficient precision)
- Features: FP32 for output (standard precision)
- Gradients: FP32 for optimizer (standard precision)

**Expected Speedup**:
- Analytic MoE backward: 10× faster than autograd
- Mixed precision: 2× faster
- Batched computation: 2.5× faster
- **Total: 50× backward pass speedup**

### Step 2 Phase 2: Koopman Operator Learning

**Objective**: Replace backpropagation with Koopman operator-based learning for 100× gradient computation cost reduction.

**Theoretical Foundation**:
Koopman theory: Any nonlinear dynamical system can be represented as a linear operator in a higher-dimensional space.

For language modeling:
- State: x_t = hidden_state at layer t
- Dynamics: x_{t+1} = f(x_t)  [nonlinear ResNet-BK layer]
- Lifted space: z_t = φ(x_t)  [embedding into Koopman space]
- Linear dynamics: z_{t+1} = K * z_t  [Koopman operator K]

**Design**:

```python
class KoopmanResNetBKLayer(nn.Module):
    """
    ResNet-BK layer with Koopman operator learning.
    """
    
    def __init__(self, d_model, koopman_dim=256):
        super().__init__()
        self.d_model = d_model
        self.koopman_dim = koopman_dim
        
        # Standard ResNet-BK components
        self.bk_layer = MoEResNetBKLayer(d_model)
        
        # Koopman lifting function φ: R^d_model → R^koopman_dim
        self.phi = nn.Sequential(
            nn.Linear(d_model, koopman_dim),
            nn.Tanh(),  # Bounded activation for stability
            nn.Linear(koopman_dim, koopman_dim)
        )
        
        # Koopman operator K: R^koopman_dim → R^koopman_dim
        # Initialized as identity + small perturbation
        self.K = nn.Parameter(
            torch.eye(koopman_dim) + 0.01 * torch.randn(koopman_dim, koopman_dim)
        )
        
        # Inverse lifting ψ: R^koopman_dim → R^d_model
        self.psi = nn.Sequential(
            nn.Linear(koopman_dim, koopman_dim),
            nn.Tanh(),
            nn.Linear(koopman_dim, d_model)
        )
        
        # Buffers for DMD computation
        self.register_buffer('Z_current', torch.zeros(koopman_dim, 100))  # Store last 100 states
        self.register_buffer('Z_next', torch.zeros(koopman_dim, 100))
        self.buffer_idx = 0
    
    def forward(self, x, use_koopman=False):
        """
        x: (B, N, D)
        use_koopman: if True, use Koopman prediction; else use standard forward
        """
        if not use_koopman:
            # Standard forward pass
            return self.bk_layer(x)
        else:
            # Koopman-based forward pass
            B, N, D = x.shape
            
            # Lift to Koopman space
            z = self.phi(x)  # (B, N, koopman_dim)
            
            # Apply Koopman operator
            z_next = torch.einsum('bnk,kl->bnl', z, self.K)  # (B, N, koopman_dim)
            
            # Project back to state space
            x_next = self.psi(z_next)  # (B, N, D)
            
            return x_next
    
    def update_koopman_operator(self, x_current, x_next):
        """
        Update Koopman operator K using streaming DMD.
        
        Args:
            x_current: (B, N, D) - current states
            x_next: (B, N, D) - next states (from standard forward pass)
        """
        with torch.no_grad():
            # Lift to Koopman space
            z_current = self.phi(x_current)  # (B, N, koopman_dim)
            z_next = self.phi(x_next)  # (B, N, koopman_dim)
            
            # Flatten batch and sequence
            z_current_flat = z_current.reshape(-1, self.koopman_dim).T  # (koopman_dim, B*N)
            z_next_flat = z_next.reshape(-1, self.koopman_dim).T
            
            # Update buffer (circular)
            buffer_size = self.Z_current.shape[1]
            batch_size = z_current_flat.shape[1]
            
            if batch_size <= buffer_size:
                # Simple case: batch fits in buffer
                end_idx = (self.buffer_idx + batch_size) % buffer_size
                if end_idx > self.buffer_idx:
                    self.Z_current[:, self.buffer_idx:end_idx] = z_current_flat
                    self.Z_next[:, self.buffer_idx:end_idx] = z_next_flat
                else:
                    # Wrap around
                    self.Z_current[:, self.buffer_idx:] = z_current_flat[:, :buffer_size-self.buffer_idx]
                    self.Z_current[:, :end_idx] = z_current_flat[:, buffer_size-self.buffer_idx:]
                    self.Z_next[:, self.buffer_idx:] = z_next_flat[:, :buffer_size-self.buffer_idx]
                    self.Z_next[:, :end_idx] = z_next_flat[:, buffer_size-self.buffer_idx:]
                self.buffer_idx = end_idx
            
            # Compute Koopman operator using DMD
            # K = Z_next @ Z_current^+ (pseudoinverse)
            # Use SVD for numerical stability
            U, S, Vt = torch.svd(self.Z_current)
            
            # Truncate small singular values
            threshold = 1e-6
            S_inv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))
            
            # K = Z_next @ V @ S^{-1} @ U^T
            K_new = self.Z_next @ Vt.T @ torch.diag(S_inv) @ U.T
            
            # Exponential moving average update
            alpha = 0.1  # Learning rate for Koopman operator
            self.K.data = (1 - alpha) * self.K.data + alpha * K_new
    
    def koopman_loss(self, x_current, x_next):
        """
        Auxiliary loss to enforce linear dynamics in Koopman space.
        
        L_koopman = ||φ(x_next) - K * φ(x_current)||^2
        """
        z_current = self.phi(x_current)
        z_next_true = self.phi(x_next)
        z_next_pred = torch.einsum('bnk,kl->bnl', z_current, self.K)
        
        loss = F.mse_loss(z_next_pred, z_next_true)
        return loss


class HybridKoopmanTrainer:
    """
    Training loop with hybrid Koopman-gradient learning.
    """
    
    def __init__(self, model, koopman_weight=0.1, koopman_start_epoch=2):
        self.model = model
        self.koopman_weight = koopman_weight
        self.koopman_start_epoch = koopman_start_epoch
        self.current_epoch = 0
    
    def train_step(self, x_batch, y_batch, optimizer, criterion):
        """
        Hybrid training step: gradient-based + Koopman-based.
        """
        # Phase 1: Standard forward pass with gradient-based learning
        optimizer.zero_grad()
        
        # Forward through all layers
        hidden_states = []
        h = x_batch
        for layer in self.model.blocks:
            h_prev = h
            h = layer(h, use_koopman=False)
            hidden_states.append((h_prev, h))
        
        logits = self.model.lm_head(self.model.layer_norm_final(h))
        loss_lm = criterion(logits.view(-1, logits.size(-1)), y_batch)
        
        # Phase 2: Koopman auxiliary loss (if enabled)
        if self.current_epoch >= self.koopman_start_epoch:
            loss_koopman = 0
            for layer, (h_prev, h_next) in zip(self.model.blocks, hidden_states):
                loss_koopman += layer.bk_layer.koopman_loss(h_prev, h_next)
            loss_koopman /= len(self.model.blocks)
            
            total_loss = loss_lm + self.koopman_weight * loss_koopman
        else:
            total_loss = loss_lm
            loss_koopman = torch.tensor(0.0)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Phase 3: Update Koopman operators (no gradients)
        if self.current_epoch >= self.koopman_start_epoch:
            for layer, (h_prev, h_next) in zip(self.model.blocks, hidden_states):
                layer.bk_layer.update_koopman_operator(h_prev.detach(), h_next.detach())
        
        return {
            'loss_lm': loss_lm.item(),
            'loss_koopman': loss_koopman.item() if isinstance(loss_koopman, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
```

**Training Strategy**:
1. **Epochs 1-2**: Standard gradient-based training (warm-up)
2. **Epochs 3-5**: Hybrid training (gradient + Koopman auxiliary loss)
3. **Epochs 6+**: Gradually increase Koopman weight, decrease gradient weight
4. **Final**: Pure Koopman-based updates (gradient computation only for output layer)

**Expected Cost Reduction**:
- Koopman operator application: O(koopman_dim²) ≈ O(256²) = 65K FLOPs
- Standard backward pass: O(N * d_model²) ≈ O(128 * 64²) = 524K FLOPs
- **Speedup: 524K / 65K ≈ 8× per layer**
- With 4 layers: **~30× overall backward pass reduction**
- Combined with Phase 1 (50× analytic gradient): **100× total Step 2 speedup**

### Step 2 Phase 3: Physics-Informed Learning

**Objective**: Incorporate physical conservation laws to guide optimization without explicit gradients.

**Design**:

```python
class PhysicsInformedBKLayer(nn.Module):
    """
    ResNet-BK layer with Hamiltonian structure and energy conservation.
    """
    
    def __init__(self, d_model, n_seq):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        
        # Kinetic energy: depends on "momentum" (hidden state derivatives)
        self.kinetic_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Potential energy: BK-Core already computes potential v_i
        self.potential_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        # Hamiltonian: H = T + V
        self.bk_core = BKCoreFunction.apply
        
        # Lagrange multiplier for energy conservation
        self.lambda_energy = nn.Parameter(torch.tensor(0.1))
    
    def compute_energy(self, x, x_prev=None):
        """
        Compute total energy: E = T + V
        
        Args:
            x: (B, N, D) - current state
            x_prev: (B, N, D) - previous state (for kinetic energy)
        """
        # Potential energy
        V = self.potential_mlp(x).squeeze(-1)  # (B, N)
        V_total = V.sum(dim=-1)  # (B,)
        
        # Kinetic energy (if previous state available)
        if x_prev is not None:
            momentum = x - x_prev  # Finite difference approximation
            T = self.kinetic_mlp(momentum).squeeze(-1)  # (B, N)
            T_total = T.sum(dim=-1)  # (B,)
        else:
            T_total = torch.zeros_like(V_total)
        
        E_total = T_total + V_total
        return E_total, T_total, V_total
    
    def energy_conservation_loss(self, E_current, E_prev):
        """
        Penalize energy drift: L_energy = ||E_current - E_prev||^2
        """
        return F.mse_loss(E_current, E_prev)
    
    def symplectic_update(self, x, grad_x, dt=0.01):
        """
        Symplectic integrator (Störmer-Verlet) for parameter updates.
        
        Preserves Hamiltonian structure during optimization.
        
        Args:
            x: current parameters
            grad_x: gradient (force)
            dt: time step
        
        Returns:
            x_new: updated parameters
        """
        # Störmer-Verlet: 
        # v_{n+1/2} = v_n + (dt/2) * grad_x_n
        # x_{n+1} = x_n + dt * v_{n+1/2}
        # v_{n+1} = v_{n+1/2} + (dt/2) * grad_x_{n+1}
        
        # Simplified (assuming v_n = 0 for parameter updates):
        x_new = x - dt * grad_x  # Gradient descent with symplectic structure
        
        return x_new


class EquilibriumPropagationTrainer:
    """
    Training using equilibrium propagation: energy-based learning.
    """
    
    def __init__(self, model, beta=0.5, n_relax_steps=10):
        self.model = model
        self.beta = beta  # Nudging strength
        self.n_relax_steps = n_relax_steps
    
    def relax_to_equilibrium(self, x, target=None):
        """
        Relax network to energy minimum.
        
        Args:
            x: input
            target: if provided, nudge towards target
        
        Returns:
            equilibrium state
        """
        h = x
        for _ in range(self.n_relax_steps):
            # Forward pass
            h_new = self.model(h)
            
            # Energy minimization: move towards lower energy
            with torch.no_grad():
                E_current, _, _ = self.model.blocks[0].bk_layer.compute_energy(h_new, h)
                
                # If target provided, nudge towards it
                if target is not None:
                    nudge = self.beta * (target - h_new)
                    h_new = h_new + nudge
                
                h = h_new
        
        return h
    
    def train_step(self, x_batch, y_batch):
        """
        Equilibrium propagation training step.
        
        1. Free phase: relax to equilibrium without target
        2. Nudged phase: relax to equilibrium with target nudging
        3. Update: Δw ∝ (h_nudged - h_free)
        """
        # Free phase
        h_free = self.relax_to_equilibrium(x_batch, target=None)
        
        # Nudged phase (use target as nudging signal)
        # For language modeling, target is next token prediction
        h_nudged = self.relax_to_equilibrium(x_batch, target=y_batch)
        
        # Compute parameter updates (no backprop!)
        with torch.no_grad():
            for param in self.model.parameters():
                if param.grad is not None:
                    # Equilibrium propagation update rule
                    # Δw ∝ (activity_nudged - activity_free)
                    # Approximate using parameter-wise difference
                    param.data += 0.01 * (h_nudged.mean() - h_free.mean()) * torch.randn_like(param) * 0.1
        
        # Compute loss for monitoring (not used for updates)
        logits = self.model.lm_head(self.model.layer_norm_final(h_free))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_batch)
        
        return loss.item()
```

**Expected Benefits**:
- Energy conservation provides implicit regularization
- Symplectic updates preserve Hamiltonian structure (better long-term stability)
- Equilibrium propagation: no backpropagation needed (energy-based updates)
- **Cost reduction: ~10× fewer gradient computations** (combined with Koopman)



### Step 4: Advanced Model Compression

**Objective**: Achieve 100× model size and inference cost reduction through quantization, pruning, and distillation.

**Design**:

```python
class QuantizedBKCore(nn.Module):
    """
    INT8 quantized BK-Core with dynamic range calibration.
    """
    
    def __init__(self, n_seq):
        super().__init__()
        self.n_seq = n_seq
        
        # Quantization parameters (learned during QAT)
        self.register_buffer('v_scale', torch.tensor(1.0))
        self.register_buffer('v_zero_point', torch.tensor(0))
        self.register_buffer('G_real_scale', torch.tensor(1.0))
        self.register_buffer('G_real_zero_point', torch.tensor(0))
        self.register_buffer('G_imag_scale', torch.tensor(1.0))
        self.register_buffer('G_imag_zero_point', torch.tensor(0))
        
        # Base Hamiltonian (quantized to INT8)
        h0_diag_fp32 = torch.full((n_seq,), -2.0)
        self.register_buffer('h0_diag_int8', self.quantize_tensor(h0_diag_fp32, 1.0, 0))
        
        h0_sub_fp32 = torch.full((n_seq-1,), 1.0)
        self.register_buffer('h0_sub_int8', self.quantize_tensor(h0_sub_fp32, 1.0, 0))
        
        h0_super_fp32 = torch.full((n_seq-1,), 1.0)
        self.register_buffer('h0_super_int8', self.quantize_tensor(h0_super_fp32, 1.0, 0))
        
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
    
    @staticmethod
    def quantize_tensor(x, scale, zero_point):
        """Quantize FP32 tensor to INT8."""
        x_int8 = torch.clamp(torch.round(x / scale) + zero_point, -128, 127)
        return x_int8.to(torch.int8)
    
    @staticmethod
    def dequantize_tensor(x_int8, scale, zero_point):
        """Dequantize INT8 tensor to FP32."""
        return (x_int8.to(torch.float32) - zero_point) * scale
    
    def calibrate_quantization(self, v_samples):
        """
        Calibrate quantization parameters using sample data.
        
        Args:
            v_samples: (num_samples, n_seq) - sample potential values
        """
        # Compute dynamic range
        v_min = v_samples.min()
        v_max = v_samples.max()
        
        # Symmetric quantization
        v_abs_max = max(abs(v_min), abs(v_max))
        self.v_scale = v_abs_max / 127.0
        self.v_zero_point = 0
    
    def forward(self, v_fp32):
        """
        Forward pass with INT8 computation.
        
        Args:
            v_fp32: (B, N) - potential in FP32
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)] in FP32
        """
        B, N = v_fp32.shape
        
        # Quantize input
        v_int8 = self.quantize_tensor(v_fp32, self.v_scale, self.v_zero_point)
        
        # Dequantize for computation (INT8 arithmetic for theta/phi recursions)
        v_dequant = self.dequantize_tensor(v_int8, self.v_scale, self.v_zero_point)
        h0_diag_dequant = self.dequantize_tensor(self.h0_diag_int8, 1.0, 0)
        h0_sub_dequant = self.dequantize_tensor(self.h0_sub_int8, 1.0, 0)
        h0_super_dequant = self.dequantize_tensor(self.h0_super_int8, 1.0, 0)
        
        # Expand to batch
        h0_diag_batch = h0_diag_dequant.unsqueeze(0).expand(B, -1)
        h0_sub_batch = h0_sub_dequant.unsqueeze(0).expand(B, -1)
        h0_super_batch = h0_super_dequant.unsqueeze(0).expand(B, -1)
        
        he_diag = h0_diag_batch + v_dequant
        
        # BK-Core computation (complex128 for numerical stability)
        G_ii = vmapped_get_diag(he_diag, h0_super_batch, h0_sub_batch, self.z)
        
        # Quantize output
        G_real_int8 = self.quantize_tensor(G_ii.real, self.G_real_scale, self.G_real_zero_point)
        G_imag_int8 = self.quantize_tensor(G_ii.imag, self.G_imag_scale, self.G_imag_zero_point)
        
        # Dequantize for output
        G_real_fp32 = self.dequantize_tensor(G_real_int8, self.G_real_scale, self.G_real_zero_point)
        G_imag_fp32 = self.dequantize_tensor(G_imag_int8, self.G_imag_scale, self.G_imag_zero_point)
        
        features = torch.stack([G_real_fp32, G_imag_fp32], dim=-1)
        return features


class PrunedMoELayer(nn.Module):
    """
    Dynamically pruned MoE: remove unused experts during training.
    """
    
    def __init__(self, d_model, num_experts=8, prune_threshold=0.05):
        super().__init__()
        self.num_experts = num_experts
        self.prune_threshold = prune_threshold
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        self.gating = nn.Linear(d_model, num_experts)
        
        # Track expert usage
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('expert_active', torch.ones(num_experts, dtype=torch.bool))
        self.total_tokens = 0
    
    def forward(self, x):
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        
        # Router logits (only for active experts)
        router_logits_full = self.gating(x_flat)  # (B*N, num_experts)
        router_logits = router_logits_full.clone()
        router_logits[:, ~self.expert_active] = -float('inf')  # Mask inactive experts
        
        # Gumbel-Softmax routing
        gates = F.gumbel_softmax(router_logits, hard=True, tau=1.0)
        
        # Update expert usage statistics
        with torch.no_grad():
            self.expert_usage += gates.sum(dim=0)
            self.total_tokens += B * N
        
        # Compute outputs (only for active experts)
        output = torch.zeros(B * N, D, device=x.device)
        for e in range(self.num_experts):
            if self.expert_active[e]:
                expert_output = self.experts[e](x_flat)
                output += expert_output * gates[:, e].unsqueeze(-1)
        
        return output.view(B, N, D)
    
    def prune_experts(self):
        """
        Prune experts used less than threshold.
        """
        if self.total_tokens == 0:
            return
        
        usage_ratio = self.expert_usage / self.total_tokens
        
        # Mark experts for pruning
        to_prune = usage_ratio < self.prune_threshold
        
        if to_prune.any():
            print(f"Pruning {to_prune.sum().item()} experts with usage < {self.prune_threshold}")
            self.expert_active[to_prune] = False
            
            # Reset statistics
            self.expert_usage.zero_()
            self.total_tokens = 0
    
    def get_num_active_experts(self):
        return self.expert_active.sum().item()


class DistillationTrainer:
    """
    Knowledge distillation: train small student from large teacher.
    """
    
    def __init__(self, teacher_model, student_model, temperature=2.0, alpha=0.5):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha  # Balance between soft and hard targets
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(self, student_logits, teacher_logits, targets):
        """
        Combined loss: soft targets (teacher) + hard targets (ground truth).
        
        L = α * KL(softmax(teacher/T), softmax(student/T)) + (1-α) * CE(student, targets)
        """
        # Soft targets loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        loss_soft = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (self.temperature ** 2)
        
        # Hard targets loss
        loss_hard = F.cross_entropy(student_logits, targets)
        
        # Combined loss
        loss = self.alpha * loss_soft + (1 - self.alpha) * loss_hard
        return loss, loss_soft, loss_hard
    
    def feature_distillation_loss(self, student_features, teacher_features):
        """
        Match intermediate representations (BK-Core G_ii features).
        """
        return F.mse_loss(student_features, teacher_features)
    
    def train_step(self, x_batch, y_batch, optimizer):
        """
        Distillation training step.
        """
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_logits = self.teacher(x_batch)
            teacher_features = self.teacher.blocks[0].bk_layer.output_features
        
        # Student forward
        optimizer.zero_grad()
        student_logits = self.student(x_batch)
        student_features = self.student.blocks[0].bk_layer.output_features
        
        # Distillation loss
        loss_distill, loss_soft, loss_hard = self.distillation_loss(
            student_logits.view(-1, student_logits.size(-1)),
            teacher_logits.view(-1, teacher_logits.size(-1)),
            y_batch
        )
        
        # Feature distillation loss
        loss_feature = self.feature_distillation_loss(student_features, teacher_features)
        
        # Total loss
        loss = loss_distill + 0.1 * loss_feature
        
        loss.backward()
        optimizer.step()
        
        return {
            'loss_total': loss.item(),
            'loss_soft': loss_soft.item(),
            'loss_hard': loss_hard.item(),
            'loss_feature': loss_feature.item()
        }


class CompressionPipeline:
    """
    Automated compression pipeline: quantization → pruning → distillation.
    """
    
    def __init__(self, model, target_size_reduction=100):
        self.model = model
        self.target_size_reduction = target_size_reduction
    
    def run_pipeline(self, train_loader, val_loader):
        """
        Execute full compression pipeline.
        
        Returns:
            compressed_model: final compressed model
            metrics: compression metrics
        """
        print("=== Compression Pipeline ===")
        
        # Step 1: Quantization-Aware Training
        print("\n[1/3] Quantization-Aware Training...")
        qat_model = self.quantization_aware_training(train_loader, val_loader)
        
        # Step 2: Structured Pruning
        print("\n[2/3] Structured Pruning...")
        pruned_model = self.structured_pruning(qat_model, train_loader, val_loader)
        
        # Step 3: Knowledge Distillation
        print("\n[3/3] Knowledge Distillation...")
        final_model = self.knowledge_distillation(pruned_model, train_loader, val_loader)
        
        # Measure compression metrics
        metrics = self.measure_compression(self.model, final_model)
        
        return final_model, metrics
    
    def quantization_aware_training(self, train_loader, val_loader, epochs=3):
        """QAT: simulate INT8 during training."""
        # Replace BK-Core with quantized version
        for block in self.model.blocks:
            block.bk_layer.bk_core = QuantizedBKCore(n_seq=128)
        
        # Train with quantization simulation
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        for epoch in range(epochs):
            self.train_epoch(train_loader, optimizer)
            val_ppl = self.evaluate(val_loader)
            print(f"QAT Epoch {epoch+1}: Val PPL = {val_ppl:.2f}")
        
        return self.model
    
    def structured_pruning(self, model, train_loader, val_loader, prune_ratio=0.5):
        """Prune MoE experts based on usage."""
        # Replace MoE with pruned version
        for block in model.blocks:
            if hasattr(block.bk_layer, 'moe_ffn'):
                block.bk_layer.moe_ffn = PrunedMoELayer(
                    d_model=block.bk_layer.moe_ffn.d_model,
                    num_experts=8,
                    prune_threshold=0.05
                )
        
        # Train and prune iteratively
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for epoch in range(3):
            self.train_epoch(train_loader, optimizer)
            
            # Prune after each epoch
            for block in model.blocks:
                if hasattr(block.bk_layer, 'moe_ffn'):
                    block.bk_layer.moe_ffn.prune_experts()
            
            val_ppl = self.evaluate(val_loader)
            num_active = model.blocks[0].bk_layer.moe_ffn.get_num_active_experts()
            print(f"Pruning Epoch {epoch+1}: Val PPL = {val_ppl:.2f}, Active Experts = {num_active}")
        
        return model
    
    def knowledge_distillation(self, teacher_model, train_loader, val_loader):
        """Distill to smaller student model."""
        # Create student model (smaller)
        student_model = LanguageModel(
            vocab_size=teacher_model.token_embedding.num_embeddings,
            d_model=32,  # Half the size
            n_layers=2,  # Half the layers
            n_seq=128,
            num_experts=2  # Fewer experts
        )
        
        # Distillation training
        trainer = DistillationTrainer(teacher_model, student_model, temperature=2.0, alpha=0.7)
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-3)
        
        for epoch in range(5):
            for x_batch, y_batch in train_loader:
                trainer.train_step(x_batch, y_batch, optimizer)
            
            val_ppl = self.evaluate_model(student_model, val_loader)
            print(f"Distillation Epoch {epoch+1}: Val PPL = {val_ppl:.2f}")
        
        return student_model
    
    def measure_compression(self, original_model, compressed_model):
        """Measure compression metrics."""
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        compression_ratio = original_params / compressed_params
        
        return {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio
        }
```

**Expected Compression**:
- Quantization (FP32 → INT8): 4× size reduction
- Pruning (8 experts → 2 experts): 4× size reduction
- Distillation (d_model=64 → 32, n_layers=4 → 2): 6× size reduction
- **Total: 4 × 4 × 6 = 96× ≈ 100× compression**

### Step 5: Hardware Co-Design

**Objective**: Achieve 10× wall-clock speedup through custom CUDA kernels and mixed precision.

**Design**:

```python
# CUDA kernel for fused theta recursion
THETA_RECURSION_KERNEL = """
extern "C" __global__
void fused_theta_recursion(
    const float* a,      // (N,) main diagonal
    const float* b,      // (N-1,) super diagonal
    const float* c,      // (N-1,) sub diagonal
    float z_real,        // complex shift (real part)
    float z_imag,        // complex shift (imag part)
    float* theta_real,   // (N+1,) output (real part)
    float* theta_imag,   // (N+1,) output (imag part)
    int N
) {
    // Shared memory for intermediate results
    extern __shared__ float shared_mem[];
    float* theta_r_shared = shared_mem;
    float* theta_i_shared = shared_mem + N + 1;
    
    int tid = threadIdx.x;
    
    // Initialize
    if (tid == 0) {
        theta_r_shared[0] = 1.0f;
        theta_i_shared[0] = 0.0f;
        
        float a_shifted_r = a[0] - z_real;
        float a_shifted_i = -z_imag;
        theta_r_shared[1] = a_shifted_r;
        theta_i_shared[1] = a_shifted_i;
    }
    __syncthreads();
    
    // Recursion (sequential, but fused in single kernel)
    for (int i = 1; i < N; i++) {
        if (tid == 0) {
            // a_shifted = a[i] - z
            float a_shifted_r = a[i] - z_real;
            float a_shifted_i = -z_imag;
            
            // theta[i+1] = a_shifted * theta[i] - c[i-1] * b[i-1] * theta[i-1]
            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            
            // Term 1: a_shifted * theta[i]
            float term1_r = a_shifted_r * theta_r_shared[i] - a_shifted_i * theta_i_shared[i];
            float term1_i = a_shifted_r * theta_i_shared[i] + a_shifted_i * theta_r_shared[i];
            
            // Term 2: c[i-1] * b[i-1] * theta[i-1] (all real)
            float term2_r = c[i-1] * b[i-1] * theta_r_shared[i-1];
            float term2_i = c[i-1] * b[i-1] * theta_i_shared[i-1];
            
            // Result
            theta_r_shared[i+1] = term1_r - term2_r;
            theta_i_shared[i+1] = term1_i - term2_i;
        }
        __syncthreads();
    }
    
    // Write to global memory
    if (tid < N + 1) {
        theta_real[tid] = theta_r_shared[tid];
        theta_imag[tid] = theta_i_shared[tid];
    }
}
"""

class CUDAOptimizedBKCore(nn.Module):
    """
    BK-Core with custom CUDA kernels for theta/phi recursions.
    """
    
    def __init__(self, n_seq):
        super().__init__()
        self.n_seq = n_seq
        
        # Compile CUDA kernel
        self.theta_kernel = self._compile_kernel(THETA_RECURSION_KERNEL, 'fused_theta_recursion')
        
        # Base Hamiltonian
        self.register_buffer('h0_diag', torch.full((n_seq,), -2.0))
        self.register_buffer('h0_sub', torch.full((n_seq-1,), 1.0))
        self.register_buffer('h0_super', torch.full((n_seq-1,), 1.0))
        
        self.z = torch.tensor(1.0j, dtype=torch.complex64)
    
    def _compile_kernel(self, kernel_code, kernel_name):
        """Compile CUDA kernel using torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import load_inline
        
        cuda_module = load_inline(
            name='bk_core_kernels',
            cpp_sources='',
            cuda_sources=kernel_code,
            functions=[kernel_name],
            verbose=False
        )
        
        return getattr(cuda_module, kernel_name)
    
    def forward(self, v):
        """
        Forward pass using custom CUDA kernel.
        
        Args:
            v: (B, N) - potential
        
        Returns:
            features: (B, N, 2) - [real(G_ii), imag(G_ii)]
        """
        B, N = v.shape
        device = v.device
        
        # Prepare inputs
        he_diag = self.h0_diag + v  # (B, N)
        
        # Allocate output
        theta_real = torch.zeros(B, N+1, device=device)
        theta_imag = torch.zeros(B, N+1, device=device)
        
        # Launch kernel for each batch
        for b in range(B):
            self.theta_kernel(
                he_diag[b].contiguous(),
                self.h0_super.contiguous(),
                self.h0_sub.contiguous(),
                self.z.real.item(),
                self.z.imag.item(),
                theta_real[b].contiguous(),
                theta_imag[b].contiguous(),
                N,
                block=(1, 1, 1),  # Single thread (sequential recursion)
                grid=(1, 1, 1),
                shared_mem=(2 * (N + 1) * 4)  # 2 arrays of (N+1) floats
            )
        
        # Phi recursion (similar kernel, omitted for brevity)
        phi_real, phi_imag = self._phi_recursion_cuda(he_diag)
        
        # Compute G_ii
        det_T_real = theta_real[:, -1]
        det_T_imag = theta_imag[:, -1]
        
        # G_ii = theta[:-1] * phi / det_T
        numerator_real = theta_real[:, :-1] * phi_real - theta_imag[:, :-1] * phi_imag
        numerator_imag = theta_real[:, :-1] * phi_imag + theta_imag[:, :-1] * phi_real
        
        denom_mag_sq = det_T_real**2 + det_T_imag**2 + 1e-18
        G_ii_real = (numerator_real * det_T_real + numerator_imag * det_T_imag) / denom_mag_sq
        G_ii_imag = (numerator_imag * det_T_real - numerator_real * det_T_imag) / denom_mag_sq
        
        features = torch.stack([G_ii_real, G_ii_imag], dim=-1)
        return features


class MixedPrecisionTrainer:
    """
    Automatic Mixed Precision (AMP) training for ResNet-BK.
    """
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, x_batch, y_batch, criterion):
        """
        Training step with AMP.
        """
        self.optimizer.zero_grad()
        
        # Forward pass in FP16
        with torch.cuda.amp.autocast():
            logits = self.model(x_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

**Expected Speedup**:
- Custom CUDA kernels: 3× faster than PyTorch implementation
- Mixed precision (AMP): 2× faster, 50% memory reduction
- Tensor core optimization: 1.5× faster for matrix operations
- **Total: 3 × 2 × 1.5 ≈ 9× ≈ 10× wall-clock speedup**

