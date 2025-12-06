"""
Gradient Teleportation via Green Function - Moonshot #9

Non-local gradient propagation using Green function to solve vanishing gradient problem.
Instead of relying solely on local chain rule, teleports gradients directly between
strongly correlated layers using the Green function propagator.

Theory (from docs/research):
    δ_l^total = δ_l^chain + λ Σ_{k>l} G(l,k) δ_k

    Where:
    - δ_l^chain: Standard backprop gradient
    - G(l,k): Green function propagator between layers l and k
    - λ: Teleportation strength

    Dyson Equation for gradient flow:
    G = G_0 + G_0 Σ G
    
    Where Σ is the self-energy (interaction term from weight matrices)

Reference: docs/research/物理概念による深層学習革新リサーチ.md, Section 3
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
import math

# Try to import Green function LUT for fast propagator computation
try:
    from .green_function_lut import GreenFunctionLUT, create_green_function_lut
    _GREEN_LUT_AVAILABLE = True
except ImportError:
    _GREEN_LUT_AVAILABLE = False
    GreenFunctionLUT = None


class DysonPropagator(nn.Module):
    """
    Dyson equation-based propagator for gradient flow analysis.
    
    Computes the full Green function G from free propagator G_0 and 
    self-energy Σ using iterative Born approximation:
    
    G = G_0 + G_0 Σ G
    G ≈ G_0 + G_0 Σ G_0 + G_0 Σ G_0 Σ G_0 + ...
    
    This allows gradient flow prediction and regularization for 
    dynamical isometry (preventing gradient vanishing/explosion).
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        max_iterations: int = 3,
        regularization: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.regularization = regularization
        
        # Free propagator G_0 (identity-based, ResNet-like)
        # Represents skip connections
        self.register_buffer(
            'G_0',
            torch.eye(n_layers) * math.exp(-1.0)  # Exponential decay
        )
        
        # Learnable self-energy Σ (interaction strength between layers)
        self.self_energy = nn.Parameter(
            torch.randn(n_layers, n_layers) * 0.01
        )
        
        # Layer-wise teleportation weights
        self.teleport_weights = nn.Parameter(
            torch.ones(n_layers) * 0.1
        )
    
    def compute_full_propagator(self) -> torch.Tensor:
        """
        Compute full Green function using Born series.
        
        Returns:
            G: Full propagator matrix [n_layers, n_layers]
        """
        G = self.G_0.clone()
        G_0_Sigma = self.G_0 @ self.self_energy
        
        G_term = self.G_0.clone()
        for _ in range(self.max_iterations):
            G_term = G_0_Sigma @ G_term
            G = G + G_term
        
        # Add regularization for stability
        G = G + self.regularization * torch.eye(
            self.n_layers, device=G.device
        )
        
        return G
    
    def forward(
        self, 
        layer_gradients: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply non-local gradient correction using Dyson propagator.
        
        Args:
            layer_gradients: List of gradients for each layer
            
        Returns:
            Corrected gradients with teleportation terms
        """
        G = self.compute_full_propagator()
        n_layers = len(layer_gradients)
        
        corrected_grads = []
        for l in range(n_layers):
            # Start with original gradient
            grad_l = layer_gradients[l].clone()
            
            # Add teleported gradients from deeper layers
            teleport_weight = torch.sigmoid(self.teleport_weights[l])
            for k in range(l + 1, n_layers):
                if k < len(layer_gradients):
                    # G(l,k) * δ_k
                    propagator_strength = G[l, k]
                    grad_l = grad_l + teleport_weight * propagator_strength * layer_gradients[k]
            
            corrected_grads.append(grad_l)
        
        return corrected_grads


class GradientTeleporter:
    """
    Gradient Teleportation Manager.
    
    Registers hooks on model layers to capture and modify gradients
    using non-local propagation via Green function.
    
    Usage:
        teleporter = GradientTeleporter(model)
        teleporter.register_hooks()
        
        # Normal training loop...
        loss.backward()
        teleporter.apply_teleportation()
        optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        teleport_strength: float = 0.1,
        use_dyson: bool = True,
        green_fn_lut: Optional['GreenFunctionLUT'] = None,
    ):
        self.model = model
        self.teleport_strength = teleport_strength
        self.use_dyson = use_dyson
        self.green_fn_lut = green_fn_lut
        
        # Find target layers (typically attention and FFN layers)
        self.target_layers = self._find_teleportable_layers()
        self.n_layers = len(self.target_layers)
        
        # Initialize Dyson propagator
        if use_dyson and self.n_layers > 0:
            d_model = self._get_d_model()
            self.dyson = DysonPropagator(
                d_model=d_model,
                n_layers=self.n_layers,
            )
        else:
            self.dyson = None
        
        # Storage for captured gradients
        self.layer_gradients: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
        # Statistics
        self.teleport_count = 0
        self.total_teleport_magnitude = 0.0
    
    def _find_teleportable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Find layers suitable for gradient teleportation."""
        target_layers = []
        for name, module in self.model.named_modules():
            # Target main computation layers (attention/FFN output projections)
            if any(x in name.lower() for x in ['attn', 'attention', 'ffn', 'mlp']):
                if isinstance(module, nn.Linear):
                    target_layers.append((name, module))
        return target_layers
    
    def _get_d_model(self) -> int:
        """Infer d_model from model architecture."""
        for name, param in self.model.named_parameters():
            if 'embed' in name.lower() and param.dim() >= 2:
                return param.shape[-1]
        return 256  # Default fallback
    
    def register_hooks(self):
        """Register backward hooks on target layers."""
        self.hooks = []
        
        for name, layer in self.target_layers:
            def make_hook(layer_name):
                def hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        self.layer_gradients[layer_name] = grad_output[0].detach()
                return hook
            
            hook = layer.register_full_backward_hook(make_hook(name))
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def apply_teleportation(self) -> Dict[str, float]:
        """
        Apply gradient teleportation after backward pass.
        
        Modifies gradients in-place using non-local propagation.
        
        Returns:
            Statistics about teleportation applied
        """
        if not self.layer_gradients or self.dyson is None:
            return {'teleported': False, 'reason': 'no gradients or dyson'}
        
        # Collect gradients in layer order
        ordered_grads = []
        layer_names = []
        for name, _ in self.target_layers:
            if name in self.layer_gradients:
                ordered_grads.append(self.layer_gradients[name])
                layer_names.append(name)
        
        if len(ordered_grads) < 2:
            return {'teleported': False, 'reason': 'not enough layers'}
        
        # Apply Dyson propagator correction
        try:
            corrected_grads = self.dyson(ordered_grads)
            
            # Calculate teleportation magnitude
            teleport_magnitude = 0.0
            for orig, corrected in zip(ordered_grads, corrected_grads):
                diff = (corrected - orig).norm().item()
                teleport_magnitude += diff
            
            self.total_teleport_magnitude += teleport_magnitude
            self.teleport_count += 1
            
            # Apply corrections to model parameters
            # Note: This modifies the gradient that will be used by optimizer
            self._apply_corrections_to_params(layer_names, ordered_grads, corrected_grads)
            
            return {
                'teleported': True,
                'magnitude': teleport_magnitude,
                'layers': len(corrected_grads),
            }
            
        except Exception as e:
            return {'teleported': False, 'error': str(e)}
    
    def _apply_corrections_to_params(
        self,
        layer_names: List[str],
        original_grads: List[torch.Tensor],
        corrected_grads: List[torch.Tensor],
    ):
        """Apply corrected gradients to parameter gradients."""
        for i, name in enumerate(layer_names):
            # Find the layer and modify its weight gradients
            for param_name, param in self.model.named_parameters():
                if name in param_name and param.grad is not None:
                    # Scale correction based on teleport_strength
                    correction = (corrected_grads[i].mean() - original_grads[i].mean())
                    param.grad.add_(correction * self.teleport_strength)
                    break
    
    def reset_stats(self):
        """Reset teleportation statistics."""
        self.teleport_count = 0
        self.total_teleport_magnitude = 0.0
        self.layer_gradients = {}
    
    def get_stats(self) -> Dict[str, float]:
        """Get teleportation statistics."""
        avg_magnitude = (
            self.total_teleport_magnitude / self.teleport_count 
            if self.teleport_count > 0 else 0.0
        )
        return {
            'teleport_count': self.teleport_count,
            'total_magnitude': self.total_teleport_magnitude,
            'avg_magnitude': avg_magnitude,
        }


class TeleportedBackward(torch.autograd.Function):
    """
    Custom autograd function with gradient teleportation.
    
    Can be used to wrap forward passes for automatic teleportation
    during backward.
    """
    
    @staticmethod
    def forward(ctx, x, teleporter):
        ctx.teleporter = teleporter
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # Teleportation is applied externally via hooks
        return grad_output, None


class NesterovTeleportOptimizer:
    """
    Nesterov-like look-ahead optimizer using teleported gradients.
    
    Uses teleported gradients as a "preview" of where the optimization
    should go, similar to Nesterov momentum but using physical propagation.
    
    θ_new = θ - η * (∇_local + λ * ∇_teleport)
    """
    
    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        teleporter: GradientTeleporter,
        lookahead_strength: float = 0.1,
    ):
        self.base_optimizer = base_optimizer
        self.teleporter = teleporter
        self.lookahead_strength = lookahead_strength
    
    def step(self):
        """Optimization step with teleportation."""
        # Apply teleportation corrections
        teleport_stats = self.teleporter.apply_teleportation()
        
        # Normal optimizer step (with modified gradients)
        self.base_optimizer.step()
        
        return teleport_stats
    
    def zero_grad(self):
        """Zero gradients and reset teleporter state."""
        self.base_optimizer.zero_grad()
        self.teleporter.layer_gradients = {}


def create_gradient_teleporter(
    model: nn.Module,
    teleport_strength: float = 0.1,
    use_dyson: bool = True,
) -> GradientTeleporter:
    """
    Factory function for GradientTeleporter.
    
    Args:
        model: The model to apply teleportation to
        teleport_strength: Strength of teleportation (0.0-1.0)
        use_dyson: Whether to use Dyson propagator for corrections
        
    Returns:
        Configured GradientTeleporter instance
    """
    return GradientTeleporter(
        model=model,
        teleport_strength=teleport_strength,
        use_dyson=use_dyson,
    )
