"""
Stochastic Resonance for Quantized Training

Harnesses the counter-intuitive phenomenon where adding noise to a sub-threshold
signal can actually improve its detectability.

Key insight: In 1.58-bit (ternary) networks, gradients smaller than the
quantization threshold produce zero updates. By adding carefully tuned noise,
we enable these small gradients to "tunnel through" the quantization barrier.

Features:
- Adaptive noise injection with learnable scale
- Stochastic rounding with unbiased expectation
- Multiple noise distributions (Gaussian, Uniform, Levy)
- Integration with BitNet quantization

References:
- Duan et al., "Adaptive Stochastic Resonance based CNN" (2022)
- Chen et al., "Q-RGT: Quantized Riemannian Gradient Tracking" (2024)
- Stochastic Resonance in physics and neuroscience
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math


class StochasticResonanceQuantizer(nn.Module):
    """
    Stochastic Resonance-enhanced quantizer for BitNet.
    
    Adds carefully tuned noise before quantization to enable gradient flow
    for sub-threshold updates.
    
    Args:
        noise_type: "gaussian", "uniform", or "levy"
        initial_noise_scale: Initial noise standard deviation
        learnable_scale: Make noise scale learnable
        temperature: Temperature for stochastic rounding
    """
    
    def __init__(
        self,
        noise_type: Literal["gaussian", "uniform", "levy"] = "gaussian",
        initial_noise_scale: float = 0.1,
        learnable_scale: bool = True,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.noise_type = noise_type
        self.temperature = temperature
        
        if learnable_scale:
            self.noise_scale = nn.Parameter(torch.tensor(initial_noise_scale))
        else:
            self.register_buffer('noise_scale', torch.tensor(initial_noise_scale))
    
    def generate_noise(self, shape: torch.Size, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate noise of the configured type.
        
        Args:
            shape: Output shape
            device: Target device
            dtype: Target dtype
            
        Returns:
            Noise tensor
        """
        if self.noise_type == "gaussian":
            return torch.randn(shape, device=device, dtype=dtype)
        
        elif self.noise_type == "uniform":
            return torch.rand(shape, device=device, dtype=dtype) * 2 - 1  # [-1, 1]
        
        elif self.noise_type == "levy":
            # Lévy flight: heavier tails than Gaussian
            # Approximated using ratio of Gaussians (Cauchy has infinite variance)
            # Use a mixture: mostly Gaussian with occasional larger jumps
            base = torch.randn(shape, device=device, dtype=dtype)
            heavy_tail = torch.randn(shape, device=device, dtype=dtype) / (
                torch.randn(shape, device=device, dtype=dtype).abs() + 0.1
            )
            # Mix: 90% Gaussian, 10% heavy-tail
            mask = torch.rand(shape, device=device, dtype=dtype) < 0.1
            return torch.where(mask, heavy_tail.clamp(-10, 10), base)
        
        else:
            return torch.randn(shape, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply stochastic resonance-enhanced quantization.
        
        Args:
            x: Input weights to quantize
            scale: Optional pre-computed scale (if None, uses mean(|x|))
            
        Returns:
            Quantized weights (ternary: {-1, 0, 1} * scale)
        """
        if x.numel() == 0:
            return x
        
        # Compute scale
        if scale is None:
            scale = x.abs().mean().clamp(min=1e-4)
        
        # Normalize by scale
        x_scaled = x / scale
        
        # Add stochastic resonance noise
        if self.training:
            noise = self.generate_noise(x.shape, x.device, x.dtype)
            noise = noise * self.noise_scale.abs()  # Use abs to ensure positive scale
            x_noisy = x_scaled + noise
        else:
            x_noisy = x_scaled
        
        # Stochastic rounding
        x_floor = torch.floor(x_noisy)
        prob = (x_noisy - x_floor) / self.temperature
        prob = prob.clamp(0, 1)  # Ensure valid probability
        
        # Sample
        samples = torch.rand_like(prob)
        x_rounded = x_floor + (samples < prob).float()
        
        # Clamp to ternary {-1, 0, 1}
        x_ternary = x_rounded.clamp(-1, 1)
        
        # Rescale
        x_quant = x_ternary * scale
        
        # Straight-Through Estimator (STE)
        return (x_quant - x).detach() + x


class AdaptiveStochasticResonance(nn.Module):
    """
    Adaptive Stochastic Resonance layer.
    
    Automatically adjusts noise level based on gradient statistics.
    When gradients are small (stuck at local minimum), increase noise.
    When gradients are healthy, decrease noise.
    
    Args:
        dim: Feature dimension (for per-channel adaptation)
        initial_noise: Initial noise scale
        adaptation_rate: How fast to adapt noise level
        min_noise: Minimum noise level
        max_noise: Maximum noise level
    """
    
    def __init__(
        self,
        dim: Optional[int] = None,
        initial_noise: float = 0.1,
        adaptation_rate: float = 0.01,
        min_noise: float = 0.01,
        max_noise: float = 0.5
    ):
        super().__init__()
        
        self.adaptation_rate = adaptation_rate
        self.min_noise = min_noise
        self.max_noise = max_noise
        
        if dim is not None:
            # Per-channel noise
            self.noise_scale = nn.Parameter(torch.full((dim,), initial_noise))
        else:
            # Global noise
            self.noise_scale = nn.Parameter(torch.tensor(initial_noise))
        
        # EMA of gradient magnitude for adaptation
        self.register_buffer('grad_ema', torch.tensor(1.0))
        self.register_buffer('target_grad', torch.tensor(0.1))  # Target gradient magnitude
    
    def adapt_noise(self, grad_norm: float):
        """
        Adapt noise level based on gradient magnitude.
        
        Args:
            grad_norm: Current gradient norm
        """
        with torch.no_grad():
            # Update EMA
            self.grad_ema = 0.99 * self.grad_ema + 0.01 * grad_norm
            
            # If gradients are smaller than target, increase noise
            # If gradients are larger, decrease noise
            ratio = self.target_grad / (self.grad_ema + 1e-10)
            
            # Apply adaptation
            self.noise_scale.data = self.noise_scale.data * (1 + self.adaptation_rate * (ratio - 1))
            
            # Clamp to valid range
            self.noise_scale.data = self.noise_scale.data.clamp(self.min_noise, self.max_noise)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive stochastic resonance noise.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with noise added (during training)
        """
        if not self.training:
            return x
        
        # Generate noise
        noise = torch.randn_like(x)
        
        # Scale noise
        if self.noise_scale.dim() == 0:
            scaled_noise = noise * self.noise_scale.abs()
        else:
            # Per-channel scaling
            scaled_noise = noise * self.noise_scale.abs().view(-1, *([1] * (x.dim() - 1)))
        
        return x + scaled_noise


class GradientStochasticResonance(nn.Module):
    """
    Apply stochastic resonance directly to gradients.
    
    Instead of adding noise to weights, add noise to gradients before
    applying the update. This helps small gradients "break through"
    the quantization threshold.
    
    This is implemented as a module that wraps another module and
    modifies its gradients.
    """
    
    def __init__(
        self,
        wrapped_module: nn.Module,
        noise_scale: float = 0.1,
        gradient_threshold: float = 0.01
    ):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.noise_scale = noise_scale
        self.gradient_threshold = gradient_threshold
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register gradient hooks on all parameters."""
        for param in self.wrapped_module.parameters():
            param.register_hook(self._gradient_hook)
    
    def _gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic resonance to gradient.
        
        Add noise proportional to how "stuck" the gradient is
        (smaller gradients get more noise boost).
        """
        grad_norm = grad.abs()
        
        # Compute boost factor: more noise for smaller gradients
        boost = torch.exp(-grad_norm / (self.gradient_threshold + 1e-10))
        boost = boost.clamp(max=10.0)  # Prevent explosion
        
        # Generate and scale noise
        noise = torch.randn_like(grad) * self.noise_scale * boost
        
        return grad + noise
    
    def forward(self, *args, **kwargs):
        return self.wrapped_module(*args, **kwargs)


# =============================================================================
# Stochastic Rounding Functions
# =============================================================================

def stochastic_round(x: torch.Tensor) -> torch.Tensor:
    """
    Basic stochastic rounding.
    
    P(round to ceil) = x - floor(x)
    P(round to floor) = ceil(x) - x
    
    This ensures E[round(x)] = x (unbiased).
    """
    floor = torch.floor(x)
    prob = x - floor
    return floor + (torch.rand_like(prob) < prob).float()


def stochastic_round_ternary(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    noise_scale: float = 0.0
) -> torch.Tensor:
    """
    Stochastic rounding to ternary values {-1, 0, 1}.
    
    Args:
        x: Input tensor
        scale: Quantization scale (if None, uses mean absolute value)
        noise_scale: Optional noise to add before rounding
        
    Returns:
        Ternary quantized tensor
    """
    if x.numel() == 0:
        return x
    
    # Compute scale
    if scale is None:
        scale = x.abs().mean().clamp(min=1e-4)
    
    # Scale input
    x_scaled = x / scale
    
    # Add noise if specified
    if noise_scale > 0:
        x_scaled = x_scaled + torch.randn_like(x_scaled) * noise_scale
    
    # Stochastic rounding
    x_rounded = stochastic_round(x_scaled.clamp(-1.5, 1.5))
    
    # Clamp to ternary
    x_ternary = x_rounded.clamp(-1, 1)
    
    # Rescale
    x_out = x_ternary * scale
    
    # STE
    return (x_out - x).detach() + x


# =============================================================================
# Integration with Training
# =============================================================================

class StochasticResonanceTrainingCallback:
    """
    Callback for integrating stochastic resonance into training loops.
    
    Usage:
        callback = StochasticResonanceTrainingCallback(model)
        for step in range(num_steps):
            loss = train_step()
            callback.on_step_end(step, loss, optimizer)
    """
    
    def __init__(
        self,
        model: nn.Module,
        initial_noise: float = 0.1,
        noise_decay: float = 0.999,
        min_noise: float = 0.01,
        quantize_weights: bool = True
    ):
        self.model = model
        self.noise_scale = initial_noise
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.quantize_weights = quantize_weights
        
        self.quantizer = StochasticResonanceQuantizer(
            initial_noise_scale=initial_noise
        )
    
    def on_step_end(
        self,
        step: int,
        loss: float,
        optimizer: torch.optim.Optimizer
    ):
        """
        Called at the end of each training step.
        
        Args:
            step: Current step number
            loss: Current loss value
            optimizer: The optimizer being used
        """
        # Decay noise over time
        self.noise_scale = max(
            self.min_noise,
            self.noise_scale * self.noise_decay
        )
        
        # Update quantizer
        with torch.no_grad():
            self.quantizer.noise_scale.fill_(self.noise_scale)
        
        # Optionally quantize weights
        if self.quantize_weights:
            with torch.no_grad():
                for param in self.model.parameters():
                    if param.dim() >= 2 and param.numel() >= 1024:
                        param.copy_(stochastic_round_ternary(param, noise_scale=self.noise_scale))
    
    def get_metrics(self) -> dict:
        """Return current SR metrics."""
        return {
            'sr_noise_scale': self.noise_scale
        }


def apply_stochastic_resonance(
    model: nn.Module,
    noise_scale: float = 0.1,
    apply_to_weights: bool = True,
    apply_to_gradients: bool = False
) -> nn.Module:
    """
    Apply stochastic resonance to a model.
    
    Args:
        model: PyTorch model
        noise_scale: Noise scale for SR
        apply_to_weights: Add SR noise to weights during forward
        apply_to_gradients: Add SR noise to gradients during backward
        
    Returns:
        Modified model with SR applied
    """
    if apply_to_gradients:
        model = GradientStochasticResonance(model, noise_scale=noise_scale)
    
    if apply_to_weights:
        sr = AdaptiveStochasticResonance(initial_noise=noise_scale)
        
        # Register forward hook to add noise
        def add_noise_hook(module, input, output):
            if module.training:
                return sr(output)
            return output
        
        # Add to final layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_hook(add_noise_hook)
    
    return model
