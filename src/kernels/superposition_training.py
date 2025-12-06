"""
Quantum-Inspired Superposition Training - Moonshot #12

Treats model weights as a wave function Ψ(W) instead of a single point.
Multiple weight configurations are explored in parallel, with
interference patterns guiding optimization.

Theory (from research docs):
- Standard SGD updates a single weight point
- Quantum-inspired approach maintains K "particles" (weight perturbations)
- Each particle's loss determines its "probability amplitude"
- Final update is weighted average based on loss (interference)
- Path integral formulation for escaping local minima

Expected: Better optimization direction, escaping local minima

Reference: docs/research/物理概念による深層学習革新リサーチ.md, Section 6
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
import math
import copy


class SuperpositionOptimizer:
    """
    Quantum-Inspired Superposition Optimizer.
    
    Maintains multiple "particle" states around the current weights
    and uses interference (loss-weighted averaging) to find better
    optimization directions.
    
    Algorithm (PI2 - Path Integral Policy Improvement):
    1. Generate K perturbed weight configurations
    2. Evaluate loss for each configuration
    3. Compute probability weights: P_k = exp(-λ * L_k) / Z
    4. Update: W_new = Σ P_k * W_k
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        num_particles: int = 5,
        noise_scale: float = 0.01,
        temperature: float = 1.0,
        update_frequency: int = 10,
    ):
        """
        Args:
            model: The model to optimize
            base_optimizer: Base optimizer (e.g., AdamW) for fallback
            num_particles: Number of parallel weight configurations
            noise_scale: Scale of Gaussian perturbations
            temperature: λ in exp(-λL) for probability computation
            update_frequency: How often to use superposition (every N steps)
        """
        self.model = model
        self.base_optimizer = base_optimizer
        self.num_particles = num_particles
        self.noise_scale = noise_scale
        self.temperature = temperature
        self.update_frequency = update_frequency
        
        self.step_count = 0
        self.superposition_steps = 0
        self.fallback_steps = 0
        
        # Store parameter shapes for efficient particle generation
        self.param_shapes = {
            name: p.shape for name, p in model.named_parameters() 
            if p.requires_grad
        }
    
    def _generate_particles(self) -> List[Dict[str, torch.Tensor]]:
        """Generate K perturbed weight configurations."""
        particles = []
        
        # Get current weights as reference
        base_state = {
            name: p.data.clone() 
            for name, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        for k in range(self.num_particles):
            particle = {}
            for name, base_weight in base_state.items():
                # Gaussian perturbation
                noise = torch.randn_like(base_weight) * self.noise_scale
                particle[name] = base_weight + noise
            particles.append(particle)
        
        return particles
    
    def _apply_particle(self, particle: Dict[str, torch.Tensor]):
        """Apply a particle's weights to the model."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in particle:
                    p.data.copy_(particle[name])
    
    def _compute_loss(
        self,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute loss with current model weights."""
        with torch.no_grad():
            outputs = self.model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, targets)
            return loss.item()
    
    def superposition_step(
        self,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one superposition optimization step.
        
        1. Generate K perturbed configurations
        2. Evaluate loss for each
        3. Compute interference weights
        4. Update to weighted average
        """
        # Save current state
        original_state = {
            name: p.data.clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        # Generate particles
        particles = self._generate_particles()
        
        # Evaluate each particle
        losses = []
        for particle in particles:
            self._apply_particle(particle)
            loss = self._compute_loss(loss_fn, inputs, targets)
            losses.append(loss)
        
        losses = torch.tensor(losses)
        
        # Compute probability weights (Boltzmann distribution)
        # P_k = exp(-λ * L_k) / Z
        log_probs = -self.temperature * losses
        log_probs = log_probs - log_probs.max()  # Numerical stability
        probs = torch.softmax(log_probs, dim=0)
        
        # Compute weighted average (interference)
        new_state = {}
        for name in original_state:
            weighted_sum = torch.zeros_like(original_state[name])
            for k, particle in enumerate(particles):
                weighted_sum += probs[k].item() * particle[name]
            new_state[name] = weighted_sum
        
        # Apply new state
        self._apply_particle(new_state)
        
        self.superposition_steps += 1
        
        return {
            'min_loss': losses.min().item(),
            'max_loss': losses.max().item(),
            'mean_loss': losses.mean().item(),
            'entropy': -(probs * probs.log().clamp(min=-100)).sum().item(),
            'best_particle': losses.argmin().item(),
        }
    
    def step(
        self,
        loss: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Optimization step with optional superposition.
        
        If loss_fn, inputs, and targets are provided, uses superposition
        every update_frequency steps. Otherwise falls back to base optimizer.
        """
        self.step_count += 1
        
        use_superposition = (
            self.step_count % self.update_frequency == 0 and
            loss_fn is not None and
            inputs is not None and
            targets is not None
        )
        
        if use_superposition:
            return self.superposition_step(loss_fn, inputs, targets)
        else:
            # Standard optimizer step
            self.base_optimizer.step()
            self.fallback_steps += 1
            return {'mode': 'standard'}
    
    def zero_grad(self):
        """Zero gradients."""
        self.base_optimizer.zero_grad()
    
    def state_dict(self) -> Dict:
        """Return optimizer state for checkpointing."""
        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'superposition_steps': self.superposition_steps,
            'fallback_steps': self.fallback_steps,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state from checkpoint."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        self.superposition_steps = state_dict['superposition_steps']
        self.fallback_steps = state_dict['fallback_steps']


class ImaginaryTimeEvolution:
    """
    Imaginary Time Evolution for optimization.
    
    Uses exp(-τH) instead of exp(-iHt) to naturally decay
    high-energy (high-loss) states toward ground state (minimum).
    
    This is equivalent to Riemannian gradient descent with
    the quantum Fisher information matrix as the metric.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tau: float = 0.01,  # Imaginary time step
        use_fisher: bool = True,
    ):
        self.model = model
        self.tau = tau
        self.use_fisher = use_fisher
        
        # Store running estimate of Fisher information
        self.fisher_diag = {}
        self.fisher_momentum = 0.99
    
    def _update_fisher(self, gradients: Dict[str, torch.Tensor]):
        """Update running estimate of diagonal Fisher information."""
        for name, grad in gradients.items():
            if name not in self.fisher_diag:
                self.fisher_diag[name] = grad ** 2
            else:
                self.fisher_diag[name] = (
                    self.fisher_momentum * self.fisher_diag[name] +
                    (1 - self.fisher_momentum) * grad ** 2
                )
    
    def step(self, loss: torch.Tensor):
        """
        Perform imaginary time evolution step.
        
        W_new = W - τ * G^{-1} * ∇L
        
        where G is the Fisher information (or identity if not using Fisher).
        """
        loss.backward()
        
        gradients = {
            name: p.grad.clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        
        if self.use_fisher:
            self._update_fisher(gradients)
        
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    if self.use_fisher and name in self.fisher_diag:
                        # Natural gradient: G^{-1} * grad
                        fisher = self.fisher_diag[name].clamp(min=1e-8)
                        natural_grad = p.grad / fisher.sqrt()
                        p.data -= self.tau * natural_grad
                    else:
                        p.data -= self.tau * p.grad
        
        self.model.zero_grad()


class QuantumEnsembleDistillation:
    """
    Distill quantum ensemble (superposition) into single network.
    
    After training with SuperpositionOptimizer, distill the
    interference pattern (output distribution) into a student network.
    """
    
    def __init__(
        self,
        teacher_ensemble: List[nn.Module],
        student: nn.Module,
        temperature: float = 2.0,
    ):
        self.teachers = teacher_ensemble
        self.student = student
        self.temperature = temperature
    
    def distill_step(
        self,
        inputs: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One distillation step."""
        # Get teacher ensemble output (interference pattern)
        with torch.no_grad():
            teacher_outputs = []
            for teacher in self.teachers:
                out = teacher(inputs)
                if isinstance(out, tuple):
                    out = out[0]
                teacher_outputs.append(out)
            
            # Average teacher outputs (quantum average)
            teacher_avg = torch.stack(teacher_outputs).mean(dim=0)
        
        # Student forward
        student_out = self.student(inputs)
        if isinstance(student_out, tuple):
            student_out = student_out[0]
        
        # KL divergence loss
        loss = nn.functional.kl_div(
            nn.functional.log_softmax(student_out / self.temperature, dim=-1),
            nn.functional.softmax(teacher_avg / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


def create_superposition_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    num_particles: int = 5,
    noise_scale: float = 0.01,
) -> SuperpositionOptimizer:
    """Factory function for Superposition Optimizer."""
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return SuperpositionOptimizer(
        model=model,
        base_optimizer=base_optimizer,
        num_particles=num_particles,
        noise_scale=noise_scale,
    )
