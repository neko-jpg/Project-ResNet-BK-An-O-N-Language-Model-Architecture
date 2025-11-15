"""
Structured pruning for MoE with automatic expert removal.

This module implements dynamic expert pruning based on usage statistics,
removing experts that are rarely selected by the routing network.

Integrates with existing MoE implementation from src/models/moe.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
from .moe import SparseMoELayer


class PrunedMoELayer(nn.Module):
    """
    MoE layer with usage tracking and automatic expert pruning.
    
    Tracks which experts are used and removes those with usage < threshold.
    """
    
    def __init__(self, d_model: int, num_experts: int = 8, prune_threshold: float = 0.05,
                 top_k: int = 1):
        """
        Args:
            d_model: Model dimension
            num_experts: Initial number of experts
            prune_threshold: Prune experts with usage < this fraction (e.g., 0.05 = 5%)
            top_k: Number of experts to route to per token
        """
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.prune_threshold = prune_threshold
        self.top_k = top_k
        
        # Create experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating = nn.Linear(d_model, num_experts)
        
        # Track expert usage
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('expert_active', torch.ones(num_experts, dtype=torch.bool))
        self.total_tokens = 0
        
        # Pruning history
        self.pruning_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with usage tracking.
        
        Args:
            x: (B, N, D) - input
        
        Returns:
            output: (B, N, D)
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)
        
        # Router logits (only for active experts)
        router_logits_full = self.gating(x_flat)  # (B*N, num_experts)
        router_logits = router_logits_full.clone()
        
        # Mask inactive experts
        router_logits[:, ~self.expert_active] = -float('inf')
        
        # Gumbel-Softmax routing (hard)
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
    
    def get_expert_usage_stats(self) -> Dict[int, float]:
        """
        Get usage statistics for each expert.
        
        Returns:
            Dictionary mapping expert_id -> usage_ratio
        """
        if self.total_tokens == 0:
            return {e: 0.0 for e in range(self.num_experts)}
        
        usage_ratio = (self.expert_usage / self.total_tokens).cpu().numpy()
        return {e: float(usage_ratio[e]) for e in range(self.num_experts)}
    
    def prune_experts(self, verbose: bool = True) -> int:
        """
        Prune experts with usage below threshold.
        
        Args:
            verbose: If True, print pruning information
        
        Returns:
            Number of experts pruned
        """
        if self.total_tokens == 0:
            if verbose:
                print("No tokens processed yet, skipping pruning")
            return 0
        
        # Compute usage ratio
        usage_ratio = self.expert_usage / self.total_tokens
        
        # Identify experts to prune
        to_prune = (usage_ratio < self.prune_threshold) & self.expert_active
        num_pruned = to_prune.sum().item()
        
        if num_pruned > 0:
            # Don't prune all experts
            num_active = self.expert_active.sum().item()
            if num_active - num_pruned < 1:
                if verbose:
                    print("Cannot prune all experts, keeping at least 1")
                # Keep the most used expert
                usage_ratio_masked = usage_ratio.clone()
                usage_ratio_masked[~self.expert_active] = -1
                best_expert = usage_ratio_masked.argmax()
                to_prune[best_expert] = False
                num_pruned = to_prune.sum().item()
            
            if num_pruned > 0:
                # Mark experts as inactive
                self.expert_active[to_prune] = False
                
                # Record pruning event
                pruned_experts = torch.where(to_prune)[0].tolist()
                self.pruning_history.append({
                    'step': self.total_tokens,
                    'pruned_experts': pruned_experts,
                    'usage_ratios': {e: usage_ratio[e].item() for e in pruned_experts}
                })
                
                if verbose:
                    print(f"\n=== Expert Pruning ===")
                    print(f"Pruned {num_pruned} experts with usage < {self.prune_threshold:.1%}")
                    print(f"Pruned expert IDs: {pruned_experts}")
                    for e in pruned_experts:
                        print(f"  Expert {e}: usage = {usage_ratio[e].item():.4%}")
                    print(f"Active experts: {self.get_num_active_experts()}/{self.num_experts}")
        
        # Reset statistics
        self.expert_usage.zero_()
        self.total_tokens = 0
        
        return num_pruned
    
    def get_num_active_experts(self) -> int:
        """Get number of currently active experts."""
        return self.expert_active.sum().item()
    
    def get_pruning_history(self) -> List[Dict]:
        """Get history of pruning events."""
        return self.pruning_history
    
    def reset_usage_stats(self):
        """Reset usage statistics without pruning."""
        self.expert_usage.zero_()
        self.total_tokens = 0


class ProgressivePruningScheduler:
    """
    Scheduler for progressive expert pruning during training.
    
    Gradually reduces number of experts over training epochs.
    """
    
    def __init__(self, moe_layer: PrunedMoELayer, 
                 target_experts: int = 2,
                 prune_epochs: List[int] = None,
                 prune_steps: List[int] = None):
        """
        Args:
            moe_layer: MoE layer to prune
            target_experts: Target number of experts after pruning
            prune_epochs: List of epochs at which to prune (if using epoch-based)
            prune_steps: List of steps at which to prune (if using step-based)
        """
        self.moe_layer = moe_layer
        self.target_experts = target_experts
        self.prune_epochs = prune_epochs or []
        self.prune_steps = prune_steps or []
        
        self.current_epoch = 0
        self.current_step = 0
        
        # Calculate pruning schedule
        initial_experts = moe_layer.num_experts
        self.pruning_schedule = self._calculate_schedule(initial_experts, target_experts)
    
    def _calculate_schedule(self, initial: int, target: int) -> List[int]:
        """
        Calculate progressive pruning schedule.
        
        Returns:
            List of target expert counts at each pruning step
        """
        if len(self.prune_epochs) == 0 and len(self.prune_steps) == 0:
            # Default: prune every 2 epochs
            num_prune_steps = 3
            self.prune_epochs = [2, 4, 6]
        else:
            num_prune_steps = max(len(self.prune_epochs), len(self.prune_steps))
        
        # Exponential decay schedule
        schedule = []
        for i in range(num_prune_steps):
            progress = (i + 1) / num_prune_steps
            num_experts = int(initial * (1 - progress) + target * progress)
            schedule.append(max(num_experts, target))
        
        return schedule
    
    def step_epoch(self, epoch: int, verbose: bool = True):
        """
        Step the scheduler at epoch boundary.
        
        Args:
            epoch: Current epoch number
            verbose: If True, print pruning information
        """
        self.current_epoch = epoch
        
        if epoch in self.prune_epochs:
            idx = self.prune_epochs.index(epoch)
            target = self.pruning_schedule[idx] if idx < len(self.pruning_schedule) else self.target_experts
            
            if verbose:
                print(f"\n=== Progressive Pruning at Epoch {epoch} ===")
                print(f"Target experts: {target}")
            
            # Prune until we reach target
            current_active = self.moe_layer.get_num_active_experts()
            while current_active > target:
                num_pruned = self.moe_layer.prune_experts(verbose=verbose)
                if num_pruned == 0:
                    break
                current_active = self.moe_layer.get_num_active_experts()
            
            if verbose:
                print(f"After pruning: {current_active} active experts")
    
    def step_iteration(self, step: int, verbose: bool = True):
        """
        Step the scheduler at iteration boundary.
        
        Args:
            step: Current training step
            verbose: If True, print pruning information
        """
        self.current_step = step
        
        if step in self.prune_steps:
            idx = self.prune_steps.index(step)
            target = self.pruning_schedule[idx] if idx < len(self.pruning_schedule) else self.target_experts
            
            if verbose:
                print(f"\n=== Progressive Pruning at Step {step} ===")
                print(f"Target experts: {target}")
            
            # Prune until we reach target
            current_active = self.moe_layer.get_num_active_experts()
            while current_active > target:
                num_pruned = self.moe_layer.prune_experts(verbose=verbose)
                if num_pruned == 0:
                    break
                current_active = self.moe_layer.get_num_active_experts()
            
            if verbose:
                print(f"After pruning: {current_active} active experts")


class MagnitudePruner:
    """
    Magnitude-based weight pruning for linear layers.
    
    Prunes weights with |w| < threshold, with support for:
    - Iterative pruning with retraining
    - Gradual sparsity increase
    - Mask persistence across training
    """
    
    def __init__(self, threshold: float = 0.01, target_sparsity: float = 0.5):
        """
        Args:
            threshold: Prune weights with |w| < threshold
            target_sparsity: Target sparsity ratio (0.5 = 50% weights pruned)
        """
        self.threshold = threshold
        self.target_sparsity = target_sparsity
        
        # Store masks for each layer
        self.masks = {}
        
        # Pruning history
        self.pruning_history = []
    
    def prune_layer(self, layer: nn.Linear, layer_name: str = "", 
                   sparsity: Optional[float] = None, verbose: bool = True) -> int:
        """
        Prune weights in a linear layer.
        
        Args:
            layer: Linear layer to prune
            layer_name: Name of the layer (for tracking)
            sparsity: If provided, prune to this sparsity level; else use threshold
            verbose: If True, print pruning information
        
        Returns:
            Number of weights pruned
        """
        with torch.no_grad():
            weight = layer.weight.data
            
            if sparsity is not None:
                # Prune to target sparsity by magnitude
                num_total = weight.numel()
                num_to_prune = int(num_total * sparsity)
                
                # Get absolute values and find threshold
                weight_abs = weight.abs().flatten()
                if num_to_prune > 0:
                    threshold_value = torch.kthvalue(weight_abs, num_to_prune).values.item()
                    mask = weight.abs() >= threshold_value
                else:
                    mask = torch.ones_like(weight, dtype=torch.bool)
            else:
                # Prune by fixed threshold
                mask = weight.abs() >= self.threshold
            
            # Count pruned weights
            num_total = weight.numel()
            num_pruned = (~mask).sum().item()
            prune_ratio = num_pruned / num_total
            
            # Apply mask
            weight.mul_(mask.float())
            
            # Store mask for this layer
            if layer_name:
                self.masks[layer_name] = mask
            
            if verbose:
                print(f"Pruned {num_pruned}/{num_total} weights ({prune_ratio:.2%})")
            
            return num_pruned
    
    def apply_masks(self, model: nn.Module):
        """
        Apply stored masks to model weights.
        
        This should be called after each optimizer step to maintain sparsity.
        
        Args:
            model: Model to apply masks to
        """
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and name in self.masks:
                    module.weight.data.mul_(self.masks[name].float())
    
    def prune_model(self, model: nn.Module, layer_names: Optional[List[str]] = None,
                   sparsity: Optional[float] = None, verbose: bool = True) -> Dict[str, int]:
        """
        Prune all linear layers in a model.
        
        Args:
            model: Model to prune
            layer_names: If provided, only prune these layers
            sparsity: If provided, prune to this sparsity level; else use threshold
            verbose: If True, print pruning information
        
        Returns:
            Dictionary mapping layer_name -> num_pruned
        """
        pruning_stats = {}
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if layer_names is None or name in layer_names:
                    if verbose:
                        print(f"\nPruning layer: {name}")
                    num_pruned = self.prune_layer(module, layer_name=name, 
                                                  sparsity=sparsity, verbose=verbose)
                    pruning_stats[name] = num_pruned
        
        return pruning_stats
    
    def get_model_sparsity(self, model: nn.Module, 
                          layer_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate sparsity for each layer in the model.
        
        Args:
            model: Model to analyze
            layer_names: If provided, only analyze these layers
        
        Returns:
            Dictionary mapping layer_name -> sparsity_ratio
        """
        sparsity_stats = {}
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if layer_names is None or name in layer_names:
                        weight = module.weight.data
                        num_total = weight.numel()
                        num_zero = (weight == 0).sum().item()
                        sparsity = num_zero / num_total
                        sparsity_stats[name] = sparsity
        
        return sparsity_stats


class IterativeMagnitudePruner:
    """
    Iterative magnitude-based pruning with retraining.
    
    Gradually increases sparsity over multiple pruning-retraining cycles.
    """
    
    def __init__(self, initial_sparsity: float = 0.2, 
                 final_sparsity: float = 0.8,
                 num_iterations: int = 5,
                 prune_layers: Optional[List[str]] = None):
        """
        Args:
            initial_sparsity: Starting sparsity level (e.g., 0.2 = 20%)
            final_sparsity: Target final sparsity (e.g., 0.8 = 80%)
            num_iterations: Number of prune-retrain cycles
            prune_layers: Layer name patterns to prune (e.g., ['output_proj', 'fc'])
        """
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.num_iterations = num_iterations
        self.prune_layers = prune_layers
        
        # Calculate sparsity schedule (exponential)
        self.sparsity_schedule = self._calculate_sparsity_schedule()
        
        # Create magnitude pruner
        self.pruner = MagnitudePruner()
        
        # Track iteration
        self.current_iteration = 0
        
        # History
        self.history = []
    
    def _calculate_sparsity_schedule(self) -> List[float]:
        """
        Calculate exponential sparsity schedule.
        
        Returns:
            List of sparsity values for each iteration
        """
        schedule = []
        for i in range(self.num_iterations):
            # Exponential schedule: s(t) = s_0 * (s_f / s_0)^(t / T)
            progress = (i + 1) / self.num_iterations
            sparsity = self.initial_sparsity * (
                (self.final_sparsity / self.initial_sparsity) ** progress
            )
            schedule.append(sparsity)
        
        return schedule
    
    def _filter_layer_names(self, model: nn.Module) -> List[str]:
        """
        Filter layer names based on prune_layers patterns.
        
        Args:
            model: Model to filter layers from
        
        Returns:
            List of layer names to prune
        """
        if self.prune_layers is None:
            # Prune all linear layers
            return None
        
        filtered_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if any pattern matches
                for pattern in self.prune_layers:
                    if pattern in name:
                        filtered_names.append(name)
                        break
        
        return filtered_names if filtered_names else None
    
    def prune_step(self, model: nn.Module, verbose: bool = True) -> Dict:
        """
        Execute one pruning step.
        
        Args:
            model: Model to prune
            verbose: If True, print pruning information
        
        Returns:
            Dictionary with pruning statistics
        """
        if self.current_iteration >= self.num_iterations:
            if verbose:
                print("All pruning iterations complete")
            return {}
        
        target_sparsity = self.sparsity_schedule[self.current_iteration]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"ITERATIVE PRUNING - Iteration {self.current_iteration + 1}/{self.num_iterations}")
            print(f"Target sparsity: {target_sparsity:.1%}")
            print(f"{'='*60}")
        
        # Filter layers to prune
        layer_names = self._filter_layer_names(model)
        
        if verbose and layer_names:
            print(f"Pruning {len(layer_names)} layers matching patterns: {self.prune_layers}")
        
        # Prune model
        pruning_stats = self.pruner.prune_model(
            model, 
            layer_names=layer_names,
            sparsity=target_sparsity,
            verbose=verbose
        )
        
        # Get current sparsity
        sparsity_stats = self.pruner.get_model_sparsity(model, layer_names=layer_names)
        
        # Record history
        iteration_stats = {
            'iteration': self.current_iteration,
            'target_sparsity': target_sparsity,
            'pruning_stats': pruning_stats,
            'sparsity_stats': sparsity_stats,
            'avg_sparsity': sum(sparsity_stats.values()) / len(sparsity_stats) if sparsity_stats else 0.0
        }
        self.history.append(iteration_stats)
        
        if verbose:
            print(f"\nAverage sparsity: {iteration_stats['avg_sparsity']:.2%}")
            print(f"Total weights pruned: {sum(pruning_stats.values()):,}")
        
        self.current_iteration += 1
        
        return iteration_stats
    
    def train_step_with_mask(self, model: nn.Module):
        """
        Apply masks after optimizer step to maintain sparsity.
        
        Call this after each optimizer.step() during retraining.
        
        Args:
            model: Model being trained
        """
        self.pruner.apply_masks(model)
    
    def get_pruning_summary(self) -> Dict:
        """
        Get summary of all pruning iterations.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.history:
            return {}
        
        summary = {
            'num_iterations': len(self.history),
            'initial_sparsity': self.history[0]['avg_sparsity'],
            'final_sparsity': self.history[-1]['avg_sparsity'],
            'target_final_sparsity': self.final_sparsity,
            'history': self.history
        }
        
        return summary
