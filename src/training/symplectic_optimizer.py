"""
Symplectic Optimizer
Implements Störmer-Verlet integrator for parameter updates that preserve Hamiltonian structure.
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional


class SymplecticSGD(Optimizer):
    """
    Symplectic Stochastic Gradient Descent using Störmer-Verlet integrator.
    
    Preserves Hamiltonian structure during optimization by using symplectic integration.
    
    Störmer-Verlet method:
        v_{n+1/2} = v_n + (dt/2) * F_n
        x_{n+1} = x_n + dt * v_{n+1/2}
        v_{n+1} = v_{n+1/2} + (dt/2) * F_{n+1}
    
    For parameter optimization:
        - x: parameters (position)
        - v: velocity (momentum)
        - F: -gradient (force)
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (time step dt)
        momentum: momentum coefficient (default: 0.9)
        dampening: dampening for momentum (default: 0)
        weight_decay: weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0,
        weight_decay: float = 0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step using Störmer-Verlet integrator.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            loss: loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Gradient (force F = -grad)
                grad = p.grad
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # Initialize velocity if not present
                if 'velocity' not in param_state:
                    param_state['velocity'] = torch.zeros_like(p)
                
                v = param_state['velocity']
                
                # Störmer-Verlet integration
                # Step 1: v_{n+1/2} = v_n + (dt/2) * F_n
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        param_state['momentum_buffer'] = torch.zeros_like(p)
                    
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    
                    # Half-step velocity update
                    v.add_(buf, alpha=lr / 2)
                else:
                    # No momentum: simple half-step
                    v.add_(grad, alpha=-lr / 2)
                
                # Step 2: x_{n+1} = x_n + dt * v_{n+1/2}
                p.add_(v, alpha=lr)
                
                # Step 3: v_{n+1} = v_{n+1/2} + (dt/2) * F_{n+1}
                # Note: F_{n+1} requires recomputing gradient, which is expensive
                # For efficiency, we approximate F_{n+1} ≈ F_n (semi-implicit)
                v.add_(grad, alpha=-lr / 2)
        
        return loss


class SymplecticAdam(Optimizer):
    """
    Symplectic Adam optimizer combining adaptive learning rates with symplectic integration.
    
    Combines the benefits of:
    - Adam: adaptive learning rates, momentum, RMSprop
    - Symplectic integration: Hamiltonian structure preservation
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate (default: 1e-3)
        betas: coefficients for computing running averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        amsgrad: whether to use AMSGrad variant (default: False)
        symplectic: whether to use symplectic integration (default: True)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        symplectic: bool = True
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            symplectic=symplectic
        )
        super().__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('symplectic', True)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        
        Returns:
            loss: loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            symplectic = group['symplectic']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                param_state = self.state[p]
                
                # State initialization
                if len(param_state) == 0:
                    param_state['step'] = 0
                    # Exponential moving average of gradient values
                    param_state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    param_state['exp_avg_sq'] = torch.zeros_like(p)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        param_state['max_exp_avg_sq'] = torch.zeros_like(p)
                    if symplectic:
                        # Velocity for symplectic integration
                        param_state['velocity'] = torch.zeros_like(p)
                
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = param_state['max_exp_avg_sq']
                
                param_state['step'] += 1
                step = param_state['step']
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                
                # Compute update direction
                update = exp_avg / denom
                
                if symplectic:
                    # Symplectic integration (Störmer-Verlet)
                    v = param_state['velocity']
                    
                    # Half-step velocity update: v_{n+1/2} = v_n - (dt/2) * update
                    v.add_(update, alpha=-step_size / 2)
                    
                    # Position update: x_{n+1} = x_n + dt * v_{n+1/2}
                    p.add_(v, alpha=step_size)
                    
                    # Half-step velocity update: v_{n+1} = v_{n+1/2} - (dt/2) * update
                    # (approximating update_{n+1} ≈ update_n)
                    v.add_(update, alpha=-step_size / 2)
                else:
                    # Standard Adam update
                    p.add_(update, alpha=-step_size)
        
        return loss


def create_symplectic_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'adam',
    lr: float = 1e-3,
    **kwargs
) -> Optimizer:
    """
    Factory function to create symplectic optimizer.
    
    Args:
        model: PyTorch model
        optimizer_type: 'sgd' or 'adam'
        lr: learning rate
        **kwargs: additional optimizer arguments
    
    Returns:
        optimizer: symplectic optimizer instance
    """
    if optimizer_type.lower() == 'sgd':
        return SymplecticSGD(
            model.parameters(),
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    elif optimizer_type.lower() == 'adam':
        return SymplecticAdam(
            model.parameters(),
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0),
            symplectic=kwargs.get('symplectic', True)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
