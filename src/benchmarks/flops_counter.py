"""
FLOPs Counter Infrastructure
Comprehensive FLOPs counting for ResNet-BK model components.

Tracks forward and backward FLOPs separately for:
- BK-Core (theta/phi recursions)
- MoE (expert computation and routing)
- Linear layers
- Optimizer steps
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class FLOPsCount:
    """Container for FLOPs measurements."""
    forward: int = 0
    backward: int = 0
    optimizer: int = 0
    
    @property
    def total(self) -> int:
        """Total FLOPs (forward + backward + optimizer)."""
        return self.forward + self.backward + self.optimizer
    
    def __add__(self, other):
        """Add two FLOPsCount objects."""
        return FLOPsCount(
            forward=self.forward + other.forward,
            backward=self.backward + other.backward,
            optimizer=self.optimizer + other.optimizer
        )
    
    def __mul__(self, scalar):
        """Multiply FLOPs by scalar."""
        return FLOPsCount(
            forward=int(self.forward * scalar),
            backward=int(self.backward * scalar),
            optimizer=int(self.optimizer * scalar)
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'forward': self.forward,
            'backward': self.backward,
            'optimizer': self.optimizer,
            'total': self.total
        }


class FLOPsCounter:
    """
    Comprehensive FLOPs counter for ResNet-BK model.
    
    Counts FLOPs for:
    - BK-Core: theta/phi recursions, complex arithmetic
    - MoE: expert computation, routing, gating
    - Linear layers: matrix multiplications
    - Embeddings: lookup and addition
    - Optimizer: parameter updates (AdamW)
    
    Usage:
        counter = FLOPsCounter(model, batch_size=32, seq_len=128)
        flops = counter.count_forward_flops()
        flops_backward = counter.count_backward_flops()
        flops_total = counter.count_total_flops()
    """
    
    def __init__(self, model: nn.Module, batch_size: int, seq_len: int):
        """
        Initialize FLOPs counter.
        
        Args:
            model: ResNet-BK model
            batch_size: batch size for FLOPs calculation
            seq_len: sequence length
        """
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
        
        # Extract model dimensions
        # Handle both ConfigurableResNetBK (has .model) and LanguageModel (direct)
        if hasattr(model, 'model'):
            # ConfigurableResNetBK wrapper
            inner_model = model.model
            self.d_model = inner_model.d_model
            self.n_layers = len(inner_model.blocks)
            self.vocab_size = inner_model.lm_head.out_features
            first_block = inner_model.blocks[0]
        else:
            # Direct LanguageModel
            self.d_model = model.d_model
            self.n_layers = len(model.blocks)
            self.vocab_size = model.lm_head.out_features
            first_block = model.blocks[0]
        
        # Extract MoE configuration
        self.num_experts = first_block.bk_layer.moe_ffn.num_experts
        self.top_k = first_block.bk_layer.moe_ffn.top_k
        
        # Component-wise FLOPs breakdown
        self.component_flops: Dict[str, FLOPsCount] = {}
    
    def count_bk_core_flops(self) -> FLOPsCount:
        """
        Count FLOPs for BK-Core computation.
        
        Forward pass:
        - Theta recursion: N iterations, each with 2 complex multiplies + 1 complex add
        - Phi recursion: N iterations, each with 2 complex multiplies + 1 complex add
        - Final division: N complex divisions
        - Complex multiply: 6 real ops (4 muls, 2 adds)
        - Complex add: 2 real ops
        - Complex divide: 6 real ops (approximation)
        
        Backward pass:
        - Gradient computation: O(N) operations
        - G² computation: N complex multiplies
        - Gradient blending: N operations
        """
        N = self.seq_len
        B = self.batch_size
        
        # Forward FLOPs per sequence
        # Theta recursion: N iterations × (2 complex_mul + 1 complex_add)
        theta_flops = N * (2 * 6 + 2)  # 14 ops per iteration
        
        # Phi recursion: N iterations × (2 complex_mul + 1 complex_add)
        phi_flops = N * (2 * 6 + 2)  # 14 ops per iteration
        
        # Final division: N complex divisions
        division_flops = N * 6
        
        # Real/imag extraction: negligible
        
        forward_flops_per_seq = theta_flops + phi_flops + division_flops
        forward_flops = B * forward_flops_per_seq
        
        # Backward FLOPs per sequence
        # G² computation: N complex multiplies
        g_square_flops = N * 6
        
        # Gradient computation (theoretical + hypothesis-7): 2 × N operations
        grad_computation_flops = 2 * N * 10  # Approximate: complex ops + blending
        
        backward_flops_per_seq = g_square_flops + grad_computation_flops
        backward_flops = B * backward_flops_per_seq
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_moe_flops(self) -> FLOPsCount:
        """
        Count FLOPs for MoE layer.
        
        Forward pass:
        - Gating network: Linear(D → E)
        - Expert computation: top_k experts × [Linear(D → 2D) + ReLU + Linear(2D → D)]
        - Routing: softmax over E experts
        
        Backward pass:
        - Gradient through experts: same as forward
        - Gradient through gating: same as forward
        """
        N = self.seq_len
        B = self.batch_size
        D = self.d_model
        E = self.num_experts
        K = self.top_k
        
        # Forward FLOPs
        # Gating network: (B*N, D) @ (D, E) = B*N*D*E multiplies
        gating_flops = B * N * D * E * 2  # 2 ops per multiply-add
        
        # Softmax: B*N*E operations (exp + sum + divide)
        softmax_flops = B * N * E * 3
        
        # Expert computation (per token, K experts selected)
        # Linear(D → 2D): D * 2D = 2D² operations
        # ReLU: 2D operations
        # Linear(2D → D): 2D * D = 2D² operations
        expert_flops_per_token = K * (2 * D * D * 2 + 2 * D + 2 * D * D * 2)
        expert_flops_per_token = K * (8 * D * D + 2 * D)
        expert_flops = B * N * expert_flops_per_token
        
        forward_flops = gating_flops + softmax_flops + expert_flops
        
        # Backward FLOPs (approximately 2× forward for standard backprop)
        backward_flops = 2 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_linear_flops(self, in_features: int, out_features: int) -> FLOPsCount:
        """
        Count FLOPs for a single linear layer.
        
        Args:
            in_features: input dimension
            out_features: output dimension
        
        Returns:
            FLOPsCount for this linear layer
        """
        B = self.batch_size
        N = self.seq_len
        
        # Forward: (B*N, in_features) @ (in_features, out_features)
        forward_flops = B * N * in_features * out_features * 2  # multiply-add
        
        # Backward: gradient w.r.t. input + gradient w.r.t. weights
        # grad_input: (B*N, out_features) @ (out_features, in_features)
        # grad_weight: (in_features, B*N) @ (B*N, out_features)
        backward_flops = 2 * forward_flops
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_embedding_flops(self) -> FLOPsCount:
        """
        Count FLOPs for embedding layers.
        
        Token embedding + position embedding + addition.
        """
        B = self.batch_size
        N = self.seq_len
        D = self.d_model
        
        # Embedding lookup: negligible (just indexing)
        # Addition: B*N*D operations
        forward_flops = B * N * D
        
        # Backward: gradient propagation through embeddings
        backward_flops = B * N * D
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_layernorm_flops(self) -> FLOPsCount:
        """
        Count FLOPs for LayerNorm.
        
        Operations: mean, variance, normalize, scale, shift
        """
        B = self.batch_size
        N = self.seq_len
        D = self.d_model
        
        # Forward: mean (D ops) + variance (D ops) + normalize (D ops) + scale (D ops) + shift (D ops)
        forward_flops = B * N * D * 5
        
        # Backward: similar complexity
        backward_flops = B * N * D * 5
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_output_projection_flops(self) -> FLOPsCount:
        """
        Count FLOPs for BK-Core output projection.
        
        Linear(2 → D) projection from [real(G_ii), imag(G_ii)] to d_model.
        """
        return self.count_linear_flops(2, self.d_model)
    
    def count_lm_head_flops(self) -> FLOPsCount:
        """
        Count FLOPs for language modeling head.
        
        Linear(D → vocab_size) projection.
        """
        return self.count_linear_flops(self.d_model, self.vocab_size)
    
    def count_forward_flops(self) -> FLOPsCount:
        """
        Count total forward pass FLOPs.
        
        Returns:
            FLOPsCount with forward FLOPs only
        """
        total = FLOPsCount()
        
        # Embeddings
        embedding_flops = self.count_embedding_flops()
        total.forward += embedding_flops.forward
        self.component_flops['embedding'] = FLOPsCount(forward=embedding_flops.forward)
        
        # Per-layer computation
        for layer_idx in range(self.n_layers):
            layer_total = FLOPsCount()
            
            # LayerNorm
            ln_flops = self.count_layernorm_flops()
            layer_total.forward += ln_flops.forward
            
            # MoE
            moe_flops = self.count_moe_flops()
            layer_total.forward += moe_flops.forward
            
            # BK-Core
            bk_flops = self.count_bk_core_flops()
            layer_total.forward += bk_flops.forward
            
            # Output projection
            proj_flops = self.count_output_projection_flops()
            layer_total.forward += proj_flops.forward
            
            self.component_flops[f'layer_{layer_idx}'] = FLOPsCount(forward=layer_total.forward)
            total.forward += layer_total.forward
        
        # Final LayerNorm
        final_ln_flops = self.count_layernorm_flops()
        total.forward += final_ln_flops.forward
        self.component_flops['final_layernorm'] = FLOPsCount(forward=final_ln_flops.forward)
        
        # LM Head
        lm_head_flops = self.count_lm_head_flops()
        total.forward += lm_head_flops.forward
        self.component_flops['lm_head'] = FLOPsCount(forward=lm_head_flops.forward)
        
        return total
    
    def count_backward_flops(self) -> FLOPsCount:
        """
        Count total backward pass FLOPs.
        
        Returns:
            FLOPsCount with backward FLOPs only
        """
        total = FLOPsCount()
        
        # LM Head backward
        lm_head_flops = self.count_lm_head_flops()
        total.backward += lm_head_flops.backward
        
        # Final LayerNorm backward
        final_ln_flops = self.count_layernorm_flops()
        total.backward += final_ln_flops.backward
        
        # Per-layer backward
        for layer_idx in range(self.n_layers):
            # Output projection backward
            proj_flops = self.count_output_projection_flops()
            total.backward += proj_flops.backward
            
            # BK-Core backward (analytic gradient)
            bk_flops = self.count_bk_core_flops()
            total.backward += bk_flops.backward
            
            # MoE backward
            moe_flops = self.count_moe_flops()
            total.backward += moe_flops.backward
            
            # LayerNorm backward
            ln_flops = self.count_layernorm_flops()
            total.backward += ln_flops.backward
        
        # Embeddings backward
        embedding_flops = self.count_embedding_flops()
        total.backward += embedding_flops.backward
        
        return total
    
    def count_optimizer_flops(self, optimizer_name: str = 'adamw') -> FLOPsCount:
        """
        Count FLOPs for optimizer step.
        
        Args:
            optimizer_name: 'sgd', 'adam', or 'adamw'
        
        Returns:
            FLOPsCount with optimizer FLOPs only
        """
        # Count total parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if optimizer_name.lower() == 'sgd':
            # SGD: param = param - lr * grad
            optimizer_flops = num_params * 2  # 1 multiply, 1 add
        elif optimizer_name.lower() in ['adam', 'adamw']:
            # Adam/AdamW: 
            # - momentum update: m = beta1 * m + (1-beta1) * grad (3 ops)
            # - variance update: v = beta2 * v + (1-beta2) * grad² (4 ops)
            # - bias correction: m_hat, v_hat (4 ops)
            # - parameter update: param = param - lr * m_hat / (sqrt(v_hat) + eps) (4 ops)
            optimizer_flops = num_params * 15  # Approximate
        else:
            optimizer_flops = num_params * 2  # Default to SGD
        
        return FLOPsCount(optimizer=optimizer_flops)
    
    def count_total_flops(self, optimizer_name: str = 'adamw') -> FLOPsCount:
        """
        Count total FLOPs for one training step.
        
        Args:
            optimizer_name: optimizer type
        
        Returns:
            FLOPsCount with forward, backward, and optimizer FLOPs
        """
        forward = self.count_forward_flops()
        backward = self.count_backward_flops()
        optimizer = self.count_optimizer_flops(optimizer_name)
        
        return FLOPsCount(
            forward=forward.forward,
            backward=backward.backward,
            optimizer=optimizer.optimizer
        )
    
    def get_breakdown(self) -> Dict[str, Dict]:
        """
        Get detailed FLOPs breakdown by component.
        
        Returns:
            Dictionary with component-wise FLOPs
        """
        # Ensure we've counted FLOPs
        if not self.component_flops:
            self.count_forward_flops()
        
        breakdown = {}
        for component, flops in self.component_flops.items():
            breakdown[component] = flops.to_dict()
        
        return breakdown
    
    def print_summary(self, optimizer_name: str = 'adamw'):
        """
        Print human-readable FLOPs summary.
        
        Args:
            optimizer_name: optimizer type
        """
        total = self.count_total_flops(optimizer_name)
        
        print("=" * 70)
        print(f"FLOPs Counter Summary")
        print("=" * 70)
        print(f"Model: ResNet-BK (d={self.d_model}, L={self.n_layers}, N={self.seq_len})")
        print(f"Batch Size: {self.batch_size}")
        print(f"MoE: {self.num_experts} experts, top-{self.top_k}")
        print("-" * 70)
        print(f"Forward Pass:   {total.forward:>15,} FLOPs ({total.forward/1e9:.3f} GFLOPs)")
        print(f"Backward Pass:  {total.backward:>15,} FLOPs ({total.backward/1e9:.3f} GFLOPs)")
        print(f"Optimizer Step: {total.optimizer:>15,} FLOPs ({total.optimizer/1e9:.3f} GFLOPs)")
        print("-" * 70)
        print(f"Total per Step: {total.total:>15,} FLOPs ({total.total/1e9:.3f} GFLOPs)")
        print("=" * 70)
        
        # Component breakdown
        print("\nComponent Breakdown (Forward Pass):")
        print("-" * 70)
        breakdown = self.get_breakdown()
        for component, flops in sorted(breakdown.items()):
            if flops['forward'] > 0:
                pct = 100 * flops['forward'] / total.forward
                print(f"  {component:20s}: {flops['forward']:>12,} FLOPs ({pct:5.1f}%)")
        print("=" * 70)
    
    def save_to_json(self, filepath: str, optimizer_name: str = 'adamw'):
        """
        Save FLOPs count to JSON file.
        
        Args:
            filepath: output JSON file path
            optimizer_name: optimizer type
        """
        total = self.count_total_flops(optimizer_name)
        breakdown = self.get_breakdown()
        
        data = {
            'model_config': {
                'd_model': self.d_model,
                'n_layers': self.n_layers,
                'seq_len': self.seq_len,
                'batch_size': self.batch_size,
                'vocab_size': self.vocab_size,
                'num_experts': self.num_experts,
                'top_k': self.top_k,
            },
            'total_flops': total.to_dict(),
            'component_breakdown': breakdown,
            'optimizer': optimizer_name,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"FLOPs count saved to {filepath}")


def compare_models(model1: nn.Module, model2: nn.Module, 
                   batch_size: int, seq_len: int,
                   model1_name: str = "Model 1",
                   model2_name: str = "Model 2") -> Dict:
    """
    Compare FLOPs between two models.
    
    Args:
        model1: first model
        model2: second model (baseline)
        batch_size: batch size
        seq_len: sequence length
        model1_name: name for first model
        model2_name: name for second model (baseline)
    
    Returns:
        Dictionary with comparison results
    """
    counter1 = FLOPsCounter(model1, batch_size, seq_len)
    counter2 = FLOPsCounter(model2, batch_size, seq_len)
    
    flops1 = counter1.count_total_flops()
    flops2 = counter2.count_total_flops()
    
    # Calculate speedup
    speedup_forward = flops2.forward / flops1.forward if flops1.forward > 0 else 0
    speedup_backward = flops2.backward / flops1.backward if flops1.backward > 0 else 0
    speedup_total = flops2.total / flops1.total if flops1.total > 0 else 0
    
    comparison = {
        model1_name: flops1.to_dict(),
        model2_name: flops2.to_dict(),
        'speedup': {
            'forward': speedup_forward,
            'backward': speedup_backward,
            'total': speedup_total,
        }
    }
    
    print("=" * 70)
    print(f"Model Comparison: {model1_name} vs {model2_name}")
    print("=" * 70)
    print(f"{model1_name}:")
    print(f"  Forward:  {flops1.forward:>15,} FLOPs ({flops1.forward/1e9:.3f} GFLOPs)")
    print(f"  Backward: {flops1.backward:>15,} FLOPs ({flops1.backward/1e9:.3f} GFLOPs)")
    print(f"  Total:    {flops1.total:>15,} FLOPs ({flops1.total/1e9:.3f} GFLOPs)")
    print()
    print(f"{model2_name} (Baseline):")
    print(f"  Forward:  {flops2.forward:>15,} FLOPs ({flops2.forward/1e9:.3f} GFLOPs)")
    print(f"  Backward: {flops2.backward:>15,} FLOPs ({flops2.backward/1e9:.3f} GFLOPs)")
    print(f"  Total:    {flops2.total:>15,} FLOPs ({flops2.total/1e9:.3f} GFLOPs)")
    print()
    print(f"Speedup:")
    print(f"  Forward:  {speedup_forward:.2f}×")
    print(f"  Backward: {speedup_backward:.2f}×")
    print(f"  Total:    {speedup_total:.2f}×")
    print("=" * 70)
    
    return comparison


class ACTFLOPsCounter:
    """
    FLOPs counter for ACT-enabled models.
    
    Tracks actual FLOPs based on dynamic layer execution. Accounts for:
    - Variable number of layers executed per token
    - Early exit savings
    - ACT overhead (halting computation)
    
    Usage:
        counter = ACTFLOPsCounter(model, batch_size=32, seq_len=128)
        flops = counter.count_actual_flops(avg_layers_executed=5.2)
    """
    
    def __init__(self, model: nn.Module, batch_size: int, seq_len: int):
        """
        Initialize ACT FLOPs counter.
        
        Args:
            model: ACT-enabled ResNet-BK model
            batch_size: batch size
            seq_len: sequence length
        """
        self.base_counter = FLOPsCounter(model, batch_size, seq_len)
        self.model = model
        self.batch_size = batch_size
        self.seq_len = seq_len
    
    def count_act_overhead_flops(self) -> FLOPsCount:
        """
        Count FLOPs for ACT halting computation.
        
        Per layer:
        - Scattering phase extraction: O(D) operations
        - Halting probability computation: O(1) operations
        - Weight computation: O(1) operations
        
        Returns:
            FLOPsCount for ACT overhead
        """
        B = self.batch_size
        N = self.seq_len
        D = self.base_counter.d_model
        n_layers = self.base_counter.n_layers
        
        # Phase extraction: mean over D dimensions
        phase_extraction_flops = B * N * D * n_layers
        
        # Halting probability: simple arithmetic operations
        halting_computation_flops = B * N * 10 * n_layers  # ~10 ops per token
        
        # Weight computation: conditional operations
        weight_computation_flops = B * N * 5 * n_layers  # ~5 ops per token
        
        forward_flops = phase_extraction_flops + halting_computation_flops + weight_computation_flops
        
        # Backward: gradient through ACT logic (minimal)
        backward_flops = forward_flops * 0.5  # Approximate
        
        return FLOPsCount(forward=forward_flops, backward=backward_flops)
    
    def count_actual_flops(
        self,
        avg_layers_executed: float,
        include_act_overhead: bool = True
    ) -> FLOPsCount:
        """
        Count actual FLOPs based on average layers executed.
        
        Args:
            avg_layers_executed: average number of layers executed per token
            include_act_overhead: include ACT computation overhead
        
        Returns:
            FLOPsCount with actual FLOPs
        """
        # Get per-layer FLOPs
        full_forward = self.base_counter.count_forward_flops()
        full_backward = self.base_counter.count_backward_flops()
        
        # Compute layer FLOPs (excluding embeddings and LM head)
        embedding_flops = self.base_counter.count_embedding_flops()
        lm_head_flops = self.base_counter.count_lm_head_flops()
        final_ln_flops = self.base_counter.count_layernorm_flops()
        
        layer_forward_flops = (
            full_forward.forward 
            - embedding_flops.forward 
            - lm_head_flops.forward 
            - final_ln_flops.forward
        )
        layer_backward_flops = (
            full_backward.backward 
            - embedding_flops.backward 
            - lm_head_flops.backward 
            - final_ln_flops.backward
        )
        
        # Scale by actual layers executed
        n_layers = self.base_counter.n_layers
        layer_scale = avg_layers_executed / n_layers
        
        actual_forward = (
            embedding_flops.forward
            + layer_forward_flops * layer_scale
            + final_ln_flops.forward
            + lm_head_flops.forward
        )
        
        actual_backward = (
            embedding_flops.backward
            + layer_backward_flops * layer_scale
            + final_ln_flops.backward
            + lm_head_flops.backward
        )
        
        # Add ACT overhead
        if include_act_overhead:
            act_overhead = self.count_act_overhead_flops()
            actual_forward += act_overhead.forward
            actual_backward += act_overhead.backward
        
        # Optimizer FLOPs unchanged
        optimizer_flops = self.base_counter.count_optimizer_flops()
        
        return FLOPsCount(
            forward=int(actual_forward),
            backward=int(actual_backward),
            optimizer=optimizer_flops.optimizer
        )
    
    def compute_flops_reduction(self, avg_layers_executed: float) -> float:
        """
        Compute FLOPs reduction percentage.
        
        Args:
            avg_layers_executed: average layers executed
        
        Returns:
            reduction: FLOPs reduction as fraction (e.g., 0.4 = 40% reduction)
        """
        full_flops = self.base_counter.count_total_flops()
        actual_flops = self.count_actual_flops(avg_layers_executed)
        
        reduction = 1.0 - (actual_flops.total / full_flops.total)
        return reduction
    
    def print_act_summary(self, avg_layers_executed: float):
        """
        Print ACT FLOPs summary.
        
        Args:
            avg_layers_executed: average layers executed per token
        """
        full_flops = self.base_counter.count_total_flops()
        actual_flops = self.count_actual_flops(avg_layers_executed)
        reduction = self.compute_flops_reduction(avg_layers_executed)
        
        print("=" * 70)
        print(f"ACT FLOPs Counter Summary")
        print("=" * 70)
        print(f"Model: ResNet-BK with ACT")
        print(f"Max Layers: {self.base_counter.n_layers}")
        print(f"Avg Layers Executed: {avg_layers_executed:.2f}")
        print(f"Batch Size: {self.batch_size}, Seq Length: {self.seq_len}")
        print("-" * 70)
        print(f"Full Model (no ACT):")
        print(f"  Forward:  {full_flops.forward:>15,} FLOPs ({full_flops.forward/1e9:.3f} GFLOPs)")
        print(f"  Backward: {full_flops.backward:>15,} FLOPs ({full_flops.backward/1e9:.3f} GFLOPs)")
        print(f"  Total:    {full_flops.total:>15,} FLOPs ({full_flops.total/1e9:.3f} GFLOPs)")
        print()
        print(f"With ACT (avg {avg_layers_executed:.2f} layers):")
        print(f"  Forward:  {actual_flops.forward:>15,} FLOPs ({actual_flops.forward/1e9:.3f} GFLOPs)")
        print(f"  Backward: {actual_flops.backward:>15,} FLOPs ({actual_flops.backward/1e9:.3f} GFLOPs)")
        print(f"  Total:    {actual_flops.total:>15,} FLOPs ({actual_flops.total/1e9:.3f} GFLOPs)")
        print("-" * 70)
        print(f"FLOPs Reduction: {reduction:.1%}")
        print(f"Speedup: {1.0/(1.0-reduction):.2f}×")
        print("=" * 70)
    
    def save_act_results(
        self,
        filepath: str,
        avg_layers_executed: float,
        early_exit_rate: float,
        full_depth_rate: float
    ):
        """
        Save ACT results to JSON.
        
        Args:
            filepath: output JSON file path
            avg_layers_executed: average layers executed
            early_exit_rate: fraction of tokens that exited early
            full_depth_rate: fraction of tokens that used full depth
        """
        full_flops = self.base_counter.count_total_flops()
        actual_flops = self.count_actual_flops(avg_layers_executed)
        reduction = self.compute_flops_reduction(avg_layers_executed)
        
        data = {
            'model_config': {
                'd_model': self.base_counter.d_model,
                'n_layers': self.base_counter.n_layers,
                'seq_len': self.seq_len,
                'batch_size': self.batch_size,
            },
            'act_config': {
                'avg_layers_executed': avg_layers_executed,
                'early_exit_rate': early_exit_rate,
                'full_depth_rate': full_depth_rate,
            },
            'flops': {
                'full_model': full_flops.to_dict(),
                'with_act': actual_flops.to_dict(),
                'reduction': reduction,
                'speedup': 1.0 / (1.0 - reduction) if reduction < 1.0 else float('inf'),
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"ACT results saved to {filepath}")


def measure_act_flops(
    model: nn.Module,
    dataloader,
    device: str = 'cuda',
    max_batches: int = 100
) -> Dict:
    """
    Measure actual FLOPs for ACT model on real data.
    
    Args:
        model: ACT-enabled model
        dataloader: data loader
        device: device to run on
        max_batches: maximum batches to process
    
    Returns:
        Dictionary with FLOPs measurements and statistics
    """
    model.eval()
    model.to(device)
    
    # Check if model has ACT
    if not hasattr(model, 'act_module'):
        raise ValueError("Model does not have ACT enabled")
    
    # Reset ACT statistics
    model.act_module.reset_statistics()
    
    # Process batches
    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            x_batch = x_batch.to(device)
            _ = model(x_batch)
    
    # Get ACT statistics
    stats = model.act_module.get_statistics()
    
    # Create FLOPs counter
    batch_size = next(iter(dataloader))[0].shape[0]
    seq_len = next(iter(dataloader))[0].shape[1]
    counter = ACTFLOPsCounter(model, batch_size, seq_len)
    
    # Compute FLOPs
    actual_flops = counter.count_actual_flops(stats['avg_layers_executed'])
    full_flops = counter.base_counter.count_total_flops()
    
    results = {
        'act_statistics': stats,
        'flops': {
            'full_model': full_flops.to_dict(),
            'with_act': actual_flops.to_dict(),
            'reduction': stats['flops_reduction'],
        },
        'avg_flops_per_token': actual_flops.total / (batch_size * seq_len),
    }
    
    return results


if __name__ == '__main__':
    # Example usage
    from src.models.configurable_resnet_bk import ConfigurableResNetBK, BASELINE_CONFIG
    
    # Create model
    config = BASELINE_CONFIG
    model = ConfigurableResNetBK(config)
    
    # Count FLOPs
    counter = FLOPsCounter(model, batch_size=32, seq_len=128)
    counter.print_summary()
    
    # Save to JSON
    counter.save_to_json('flops_count.json')
    
    print("\n" + "=" * 70)
    print("ACT FLOPs Counter Example")
    print("=" * 70)
    
    # Simulate ACT with average 5.2 layers executed (out of 8)
    act_counter = ACTFLOPsCounter(model, batch_size=32, seq_len=128)
    act_counter.print_act_summary(avg_layers_executed=5.2)
    
    # Save ACT results
    act_counter.save_act_results(
        'act_flops_count.json',
        avg_layers_executed=5.2,
        early_exit_rate=0.35,
        full_depth_rate=0.15
    )
