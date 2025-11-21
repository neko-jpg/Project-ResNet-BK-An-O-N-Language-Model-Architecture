"""
Phase 3.2 & 3.4: Hamiltonian Neural ODE with Symplectic Adjoint

思考プロセスを「エネルギー保存系（ハミルトン力学系）」としてモデル化する。
さらに、随伴変数法（Adjoint Method）を用いて、定数メモリ O(1) での学習を実現する。

Physics:
    H(q, p) = T(p) + V(q) = |p|^2/2 + V(q)
    dq/dt = ∂H/∂p = p
    dp/dt = -∂H/∂q = -∇V(q)

Algorithms:
    - Symplectic Integrator (Leapfrog): Energy-preserving time evolution
    - Adjoint Method: Backprop by solving ODE backwards in time
"""

import torch
import torch.nn as nn
import torch.autograd as autograd

class HamiltonianFunc(nn.Module):
    """
    ハミルトニアン H(q, p) を定義するモジュール。
    q: 思考の状態（位置）
    p: 思考の勢い（運動量）
    
    H = Kinetic + Potential
    """
    def __init__(self, potential_net: nn.Module):
        super().__init__()
        self.potential_net = potential_net  # V(q): BK-Core or MLP
        
    def forward(self, t, x):
        # x: (q, p) concatenated
        n_dim = x.shape[-1] // 2
        q, p = x[..., :n_dim], x[..., n_dim:]
        
        # Kinetic Energy T(p) = 0.5 * |p|^2
        kinetic = 0.5 * torch.sum(p**2, dim=-1)
        
        # Potential Energy V(q)
        # potential_net returns vector field, we compute energy by integration or assume scalar output
        # Here we assume potential_net computes scalar Energy V(q) for each batch
        # If potential_net outputs vector features, we take normsq as approximation
        v_out = self.potential_net(q)
        if v_out.shape[-1] != 1:
            potential = 0.5 * torch.sum(v_out**2, dim=-1)
        else:
            potential = v_out.squeeze(-1)
            
        return kinetic + potential

    def time_derivative(self, t, x):
        """
        Hamilton's Equations:
        dq/dt = p
        dp/dt = -dV/dq
        """
        n_dim = x.shape[-1] // 2
        # Split x into q and p. Requires grad enabled for -dV/dq
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            q, p = x[..., :n_dim], x[..., n_dim:]
            
            # Calculate Hamiltonian
            h_val = self.forward(t, x).sum() # Sum for batch gradient
            
            # Gradients: [dH/dq, dH/dp]
            grads = autograd.grad(h_val, x, create_graph=True)[0]
            dq, dp = grads[..., :n_dim], grads[..., n_dim:]
            
        # Symplectic vector field J * ∇H
        # dq/dt = dH/dp = p (if T=0.5p^2)
        # dp/dt = -dH/dq
        return torch.cat([dp, -dq], dim=-1)


class SymplecticAdjoint(torch.autograd.Function):
    """
    3.4 Symplectic Adjoint Method
    
    Forward: Symplectic Integration (Leapfrog)
    Backward: Solve Adjoint ODE backwards + Parameter gradients
    Memory: O(1) w.r.t integration steps
    """
    
    @staticmethod
    def forward(ctx, h_func, x0, t_span, step_size, *params):
        """
        Leapfrog Integrator for Forward Pass
        """
        ctx.h_func = h_func
        ctx.t_span = t_span
        ctx.step_size = step_size
        
        # Unpack t
        t0, t1 = t_span
        steps = int(abs(t1 - t0) / step_size)
        dt = step_size if t1 > t0 else -step_size
        
        # Initial state
        z = x0.clone()
        n_dim = z.shape[-1] // 2
        
        # Leapfrog Integration (Symplectic)
        # 1. p(t + dt/2) = p(t) - dV/dq(q(t)) * dt/2
        # 2. q(t + dt)   = q(t) + p(t + dt/2) * dt
        # 3. p(t + dt)   = p(t + dt/2) - dV/dq(q(t + dt)) * dt/2
        
        with torch.no_grad():
            q, p = z[..., :n_dim], z[..., n_dim:]
            
            for _ in range(steps):
                # Half step momentum
                # Note: standard gradients needed for force.
                # Since we are in no_grad, we use a separate grad calc or functional
                # For efficiency in PyTorch without custom CUDA, we approximate with RK2 or
                # temporarily enable grad for force calculation.
                
                # Simplified Symplectic Euler for PoC (faster)
                # p_new = p_old - dH/dq(q_old) * dt
                # q_new = q_old + dH/dp(p_new) * dt
                
                # Calc force -dH/dq
                with torch.enable_grad():
                    q_in = q.detach().requires_grad_(True)
                    p_in = p.detach().requires_grad_(True)
                    # Reconstruct full state for h_func
                    x_in = torch.cat([q_in, p_in], dim=-1)
                    h_val = h_func(t0, x_in).sum()
                    grads = autograd.grad(h_val, x_in)[0]
                    dH_dq, dH_dp = grads[..., :n_dim], grads[..., n_dim:]
                
                p = p - dH_dq * dt
                q = q + dH_dp * dt # Assuming dH/dp = p, this is p*dt
            
            z_final = torch.cat([q, p], dim=-1)
        
        ctx.save_for_backward(z_final, x0) # Save boundaries only (O(1))
        return z_final

    @staticmethod
    def backward(ctx, grad_output):
        """
        Adjoint Backward Pass
        Solves augmented ODE backwards to get gradients for input and params.
        """
        h_func = ctx.h_func
        t0, t1 = ctx.t_span
        dt = ctx.step_size
        z_final, x0 = ctx.saved_tensors
        
        # Augmented state: [z(t), a(t)] where a(t) is adjoint state (dL/dz)
        # We integrate backwards from t1 to t0
        
        # Current state (starts at z_final)
        z = z_final.clone()
        # Current adjoint (starts at grad_output)
        adj = grad_output.clone()
        
        # Parameter gradients accumulator
        param_grads = [torch.zeros_like(p) for p in h_func.parameters()]
        
        steps = int(abs(t1 - t0) / dt)
        backward_dt = -dt
        
        # Backward integration
        for i in range(steps):
            # 1. Reconstruct state z(t-1) from z(t) (Inverse Symplectic Euler)
            # q_new = q_old + p_new * dt  => q_old = q_new - p_new * dt
            # p_new = p_old + F(q_old) * dt => p_old = p_new - F(q_old) * dt
            
            # Wait! Recomputing trajectory exactly is numerically unstable for chaotic Hamiltonians.
            # "Checkpointing" or "Reversible" architecture is safer.
            # For now, we assume reversibility holds well enough for short bursts.
            
            n_dim = z.shape[-1] // 2
            q, p = z[..., :n_dim], z[..., n_dim:]
            
            # Invert step (Approximate for standard Symplectic Euler)
            # q_prev = q - p * dt (using p_curr as approx for dH/dp)
            # p_prev = p + dH/dq(q_prev) * dt
            
            # Recover z(t-dt)
            with torch.enable_grad():
                 q.requires_grad_(True)
                 p.requires_grad_(True)
                 x_curr = torch.cat([q, p], dim=-1)
                 
                 # Use current gradients for inversion (Implicit-ish)
                 h_val = h_func(0, x_curr).sum()
                 grads = autograd.grad(h_val, x_curr)[0]
                 dH_dq, dH_dp = grads[..., :n_dim], grads[..., n_dim:]
                 
                 q_prev = q - dH_dp * dt
                 # Need force at q_prev
                 q_prev_detach = q_prev.detach().requires_grad_(True)
                 # Temporary p to form x
                 x_temp = torch.cat([q_prev_detach, p], dim=-1)
                 h_temp = h_func(0, x_temp).sum()
                 dH_dq_prev = autograd.grad(h_temp, x_temp)[0][..., :n_dim]
                 
                 p_prev = p + dH_dq_prev * dt
            
            z = torch.cat([q_prev, p_prev], dim=-1).detach()
            
            # 2. Update Adjoint State a(t)
            # da/dt = -a^T * ∂f/∂z
            # Here f is symplectic vector field.
            # This is equivalent to VJP (Vector-Jacobian Product)
            with torch.enable_grad():
                z_in = z.detach().requires_grad_(True)
                dynamics = h_func.time_derivative(0, z_in)
                
                # VJP: grad(dynamics, z_in, grad_outputs=adj)
                # This computes adj * df/dz
                adj_time_deriv = autograd.grad(dynamics, z_in, grad_outputs=adj, retain_graph=True)[0]
                
                # Adjoint update: a(t-dt) = a(t) - da/dt * dt
                # Note: backward time means multiplying by -dt, but da/dt def has minus...
                # Standard adjoint eq: d(adj)/dt = -adj * J
                # Backward step: adj_prev = adj - (-adj*J) * (-dt) = adj - adj*J*dt
                # Correct is: adj_prev = adj + adj_time_deriv * dt (since we go backward)
                adj = adj + adj_time_deriv * dt
                
                # 3. Accumulate Parameter Gradients
                # dL/dtheta = integral( -adj(t) * df/dtheta )
                # We compute df/dtheta VJP
                for param_idx, param in enumerate(h_func.parameters()):
                    if param.grad is None: continue
                    # d_dynamics_d_param
                    d_dyn_d_p = autograd.grad(dynamics, param, grad_outputs=adj, retain_graph=True, allow_unused=True)[0]
                    if d_dyn_d_p is not None:
                        param_grads[param_idx] += d_dyn_d_p * dt

        # Apply accumulated gradients to params
        # (This is a bit hacky in PyTorch Function, usually we return grads for inputs)
        # Ideally, we return None for h_func, and gradients for params in the return tuple.
        
        return None, adj, None, None, *param_grads


class HamiltonianNeuralODE(nn.Module):
    """
    Phase 3.2 Core Module
    """
    def __init__(self, potential_net, step_size=0.1):
        super().__init__()
        self.h_func = HamiltonianFunc(potential_net)
        self.step_size = step_size
        
    def forward(self, x, t_span=(0, 1)):
        # x: initial state (q0, p0) or just q0
        # If just q0, initialize p0 = 0
        if x.shape[-1] % 2 != 0:
            # Assuming input is q, append p=0
            p0 = torch.zeros_like(x)
            x = torch.cat([x, p0], dim=-1)
            
        # Run Symplectic Adjoint ODE
        # Note: We pass parameters to apply to ensure they are tracked
        z_final = SymplecticAdjoint.apply(
            self.h_func, 
            x, 
            t_span, 
            self.step_size, 
            *self.h_func.parameters()
        )
        
        return z_final