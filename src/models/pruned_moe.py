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
    
    Prunes weights with |w| < threshold.
    """
    
    def __init__(self, threshold: float = 0.01):
        """
        Args:
            threshold: Prune weights with |w| < threshold
        """
        self.threshold = threshold
    
    def prune_layer(self, layer: nn.Linear, verbose: bool = True) -> int:
        """
        Prune weights in a linear layer.
        
        Args:
            layer: Linear layer to prune
            verbose: If True, print pruning information
        
        Returns:
            Number of weights pruned
        """
        with torch.no_grad():
            weight = layer.weight.data
            
            # Create mask
            mask = weight.abs() >= self.threshold
            
            # Count pruned weights
            num_total = weight.numel()
            num_pruned = (~mask).sum().item()
            prune_ratio = num_pruned / num_total
            
            # Apply mask
            weight.mul_(mask.float())
            
            if verbose:
                print(f"Pruned {num_pruned}/{num_total} weights ({prune_ratio:.2%})")
            
            return num_pruned
    
    def prune_model(self, model: nn.Module, layer_names: Optional[List[str]] = None,
                   verbose: bool = True) -> Dict[str, int]:
        """
        Prune all linear layers in a model.
        
        Args:
            model: Model to prune
            layer_names: If provided, only prune these layers
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
                    num_pruned = self.prune_layer(module, verbose=verbose)
                    pruning_stats[name] = num_pruned
        
        return pruning_stats
