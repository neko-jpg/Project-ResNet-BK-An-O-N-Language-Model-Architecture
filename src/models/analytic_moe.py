"""
Analytic MoE Layer with Manual Gradient Computation
Implements fully analytic backward pass without autograd dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AnalyticMoELayer(nn.Module):
    """
    Sparse MoE with fully analytic backward pass.
    
    Removes autograd dependency for routing gradients using:
    - Straight-through estimator for Gumbel-Softmax
    - Manual gradient computation for expert routing
    - Explicit gradient propagation through expert networks
    
    Args:
        d_model: hidden dimension
        num_experts: number of expert networks
        top_k: number of experts to route to (1 for sparse)
        dropout_p: dropout probability
        tau: Gumbel-Softmax temperature
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 1,
        dropout_p: float = 0.1,
        tau: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout_p = dropout_p
        self.tau = tau
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating = nn.Linear(d_model, num_experts)
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with caching for analytic backward.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) routed through experts
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # (T, D), T = B*N
        
        # Router logits
        router_logits = self.gating(x_flat)  # (T, E)
        
        # Gumbel-Softmax routing (hard)
        if self.training:
            gates_hard = F.gumbel_softmax(router_logits, hard=True, tau=self.tau)
        else:
            # Inference: use argmax
            indices = router_logits.argmax(dim=-1)
            gates_hard = F.one_hot(indices, num_classes=self.num_experts).float()
        
        # Compute expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x_flat)  # (T, D)
            expert_outputs.append(expert_out)
        expert_stack = torch.stack(expert_outputs, dim=1)  # (T, E, D)
        
        # Weighted sum
        output = torch.sum(expert_stack * gates_hard.unsqueeze(-1), dim=1)  # (T, D)
        
        # Cache for backward
        if self.training:
            self.cache = {
                'x_flat': x_flat,
                'router_logits': router_logits,
                'gates_hard': gates_hard,
                'expert_outputs': expert_outputs,
                'expert_stack': expert_stack,
                'shape': (B, N, D)
            }
        
        return output.view(B, N, D)
    
    def analytic_backward(
        self,
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Analytic backward pass without autograd.
        
        Computes gradients manually for:
        1. Expert network parameters
        2. Gating network parameters
        3. Input x
        
        Args:
            grad_output: (B, N, D) gradient w.r.t. output
        
        Returns:
            grad_input: (B, N, D) gradient w.r.t. input
            grad_dict: dictionary of parameter gradients
        """
        if not self.cache:
            raise RuntimeError("Must call forward() before analytic_backward()")
        
        B, N, D = self.cache['shape']
        grad_flat = grad_output.reshape(B * N, D)  # (T, D)
        
        x_flat = self.cache['x_flat']
        router_logits = self.cache['router_logits']
        gates_hard = self.cache['gates_hard']
        expert_outputs = self.cache['expert_outputs']
        expert_stack = self.cache['expert_stack']
        
        T, E = gates_hard.shape
        
        # Initialize gradient accumulators
        grad_dict = {}
        grad_x_from_experts = torch.zeros_like(x_flat)
        
        # ===================================================================
        # 1. Gradient w.r.t. expert outputs
        # ===================================================================
        # output = sum_e (gate_e * expert_e(x))
        # dL/d(expert_e_output) = gate_e * dL/d(output)
        
        grad_expert_outputs = []
        for e in range(self.num_experts):
            gate_e = gates_hard[:, e].unsqueeze(-1)  # (T, 1)
            grad_expert_e = grad_flat * gate_e  # (T, D)
            grad_expert_outputs.append(grad_expert_e)
        
        # ===================================================================
        # 2. Backprop through each expert network
        # ===================================================================
        # For each expert: compute gradients w.r.t. parameters and input
        
        for e, expert in enumerate(self.experts):
            if grad_expert_outputs[e].abs().sum() > 0:
                # Get expert layers
                linear1 = expert[0]  # Linear(D, 2D)
                relu = expert[1]     # ReLU
                dropout = expert[2]  # Dropout
                linear2 = expert[3]  # Linear(2D, D)
                
                # Forward pass through expert (to get activations)
                h1 = linear1(x_flat)  # (T, 2D)
                h1_relu = relu(h1)    # (T, 2D)
                h1_drop = dropout(h1_relu)  # (T, 2D)
                # expert_outputs[e] = linear2(h1_drop)
                
                # Backward through linear2
                grad_h1_drop = grad_expert_outputs[e] @ linear2.weight  # (T, 2D)
                grad_linear2_weight = h1_drop.T @ grad_expert_outputs[e]  # (2D, D)
                grad_linear2_bias = grad_expert_outputs[e].sum(dim=0)  # (D,)
                
                # Store gradients
                grad_dict[f'expert_{e}_linear2_weight'] = grad_linear2_weight
                grad_dict[f'expert_{e}_linear2_bias'] = grad_linear2_bias
                
                # Backward through dropout (straight-through in eval mode)
                grad_h1_relu = grad_h1_drop
                
                # Backward through ReLU
                grad_h1 = grad_h1_relu * (h1 > 0).float()
                
                # Backward through linear1
                grad_x_expert = grad_h1 @ linear1.weight  # (T, D)
                grad_linear1_weight = x_flat.T @ grad_h1  # (D, 2D)
                grad_linear1_bias = grad_h1.sum(dim=0)  # (2D,)
                
                # Store gradients
                grad_dict[f'expert_{e}_linear1_weight'] = grad_linear1_weight
                grad_dict[f'expert_{e}_linear1_bias'] = grad_linear1_bias
                
                # Accumulate gradient to input
                grad_x_from_experts += grad_x_expert
        
        # ===================================================================
        # 3. Gradient w.r.t. gates (routing)
        # ===================================================================
        # output = sum_e (gate_e * expert_e(x))
        # dL/d(gate_e) = expert_e(x) * dL/d(output)
        
        grad_gates = torch.sum(expert_stack * grad_flat.unsqueeze(1), dim=-1)  # (T, E)
        
        # ===================================================================
        # 4. Gradient w.r.t. router logits (straight-through estimator)
        # ===================================================================
        # Gumbel-Softmax gradient approximation:
        # For hard=True, use straight-through estimator
        # dL/d(logits) ≈ dL/d(gates) * softmax(logits) * (1 - softmax(logits))
        
        softmax_gates = F.softmax(router_logits / self.tau, dim=-1)  # (T, E)
        
        # Straight-through estimator: treat hard gates as soft gates for gradient
        grad_router_logits = grad_gates * softmax_gates * (1 - softmax_gates)  # (T, E)
        
        # ===================================================================
        # 5. Backprop through gating network
        # ===================================================================
        # router_logits = gating(x_flat) = x_flat @ W_gate + b_gate
        
        grad_x_from_gating = grad_router_logits @ self.gating.weight  # (T, D)
        grad_gating_weight = x_flat.T @ grad_router_logits  # (D, E)
        grad_gating_bias = grad_router_logits.sum(dim=0)  # (E,)
        
        grad_dict['gating_weight'] = grad_gating_weight
        grad_dict['gating_bias'] = grad_gating_bias
        
        # ===================================================================
        # 6. Total gradient w.r.t. input
        # ===================================================================
        grad_x_total = grad_x_from_experts + grad_x_from_gating
        grad_input = grad_x_total.view(B, N, D)
        
        return grad_input, grad_dict
    
    def apply_gradients(self, grad_dict: dict, learning_rate: float = 1e-3):
        """
        Apply computed gradients to parameters.
        
        Args:
            grad_dict: dictionary of parameter gradients
            learning_rate: learning rate for gradient descent
        """
        with torch.no_grad():
            # Apply gating network gradients
            if 'gating_weight' in grad_dict:
                self.gating.weight -= learning_rate * grad_dict['gating_weight'].T
            if 'gating_bias' in grad_dict:
                self.gating.bias -= learning_rate * grad_dict['gating_bias']
            
            # Apply expert network gradients
            for e in range(self.num_experts):
                expert = self.experts[e]
                linear1 = expert[0]
                linear2 = expert[3]
                
                if f'expert_{e}_linear1_weight' in grad_dict:
                    linear1.weight -= learning_rate * grad_dict[f'expert_{e}_linear1_weight'].T
                if f'expert_{e}_linear1_bias' in grad_dict:
                    linear1.bias -= learning_rate * grad_dict[f'expert_{e}_linear1_bias']
                if f'expert_{e}_linear2_weight' in grad_dict:
                    linear2.weight -= learning_rate * grad_dict[f'expert_{e}_linear2_weight'].T
                if f'expert_{e}_linear2_bias' in grad_dict:
                    linear2.bias -= learning_rate * grad_dict[f'expert_{e}_linear2_bias']


class AnalyticMoEFunction(torch.autograd.Function):
    """
    Autograd-compatible wrapper for AnalyticMoELayer.
    
    Allows integration with PyTorch's autograd while using custom backward pass.
    """
    
    @staticmethod
    def forward(ctx, x, moe_layer):
        """
        Forward pass through MoE layer.
        
        Args:
            x: (B, N, D) input tensor
            moe_layer: AnalyticMoELayer instance
        
        Returns:
            output: (B, N, D) routed through experts
        """
        ctx.moe_layer = moe_layer
        output = moe_layer(x)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using analytic gradients.
        
        Args:
            grad_output: (B, N, D) gradient w.r.t. output
        
        Returns:
            grad_input: (B, N, D) gradient w.r.t. input
            None: no gradient for moe_layer argument
        """
        moe_layer = ctx.moe_layer
        grad_input, grad_dict = moe_layer.analytic_backward(grad_output)
        
        # Store gradients in parameter .grad attributes for optimizer
        with torch.no_grad():
            if 'gating_weight' in grad_dict:
                if moe_layer.gating.weight.grad is None:
                    moe_layer.gating.weight.grad = grad_dict['gating_weight'].T
                else:
                    moe_layer.gating.weight.grad += grad_dict['gating_weight'].T
            
            if 'gating_bias' in grad_dict:
                if moe_layer.gating.bias.grad is None:
                    moe_layer.gating.bias.grad = grad_dict['gating_bias']
                else:
                    moe_layer.gating.bias.grad += grad_dict['gating_bias']
            
            for e in range(moe_layer.num_experts):
                expert = moe_layer.experts[e]
                linear1 = expert[0]
                linear2 = expert[3]
                
                if f'expert_{e}_linear1_weight' in grad_dict:
                    if linear1.weight.grad is None:
                        linear1.weight.grad = grad_dict[f'expert_{e}_linear1_weight'].T
                    else:
                        linear1.weight.grad += grad_dict[f'expert_{e}_linear1_weight'].T
                
                if f'expert_{e}_linear1_bias' in grad_dict:
                    if linear1.bias.grad is None:
                        linear1.bias.grad = grad_dict[f'expert_{e}_linear1_bias']
                    else:
                        linear1.bias.grad += grad_dict[f'expert_{e}_linear1_bias']
                
                if f'expert_{e}_linear2_weight' in grad_dict:
                    if linear2.weight.grad is None:
                        linear2.weight.grad = grad_dict[f'expert_{e}_linear2_weight'].T
                    else:
                        linear2.weight.grad += grad_dict[f'expert_{e}_linear2_weight'].T
                
                if f'expert_{e}_linear2_bias' in grad_dict:
                    if linear2.bias.grad is None:
                        linear2.bias.grad = grad_dict[f'expert_{e}_linear2_bias']
                    else:
                        linear2.bias.grad += grad_dict[f'expert_{e}_linear2_bias']
        
        return grad_input, None


def validate_analytic_gradients(
    moe_layer: AnalyticMoELayer,
    x: torch.Tensor,
    epsilon: float = 1e-4,
    tolerance: float = 1e-3
) -> dict:
    """
    Validate analytic gradients using finite differences.
    
    Args:
        moe_layer: AnalyticMoELayer to validate
        x: (B, N, D) input tensor
        epsilon: finite difference step size
        tolerance: maximum allowed relative error
    
    Returns:
        validation_results: dictionary with validation metrics
    """
    moe_layer.eval()
    
    # Forward pass
    output = moe_layer(x)
    
    # Dummy loss: sum of outputs
    loss = output.sum()
    
    # Analytic backward
    grad_output = torch.ones_like(output)
    grad_input_analytic, grad_dict_analytic = moe_layer.analytic_backward(grad_output)
    
    # Finite difference gradients
    results = {
        'input_gradient_error': 0.0,
        'parameter_gradient_errors': {},
        'max_error': 0.0,
        'passed': True
    }
    
    # Validate input gradient
    grad_input_fd = torch.zeros_like(x)
    for b in range(x.shape[0]):
        for n in range(x.shape[1]):
            for d in range(x.shape[2]):
                x_plus = x.clone()
                x_plus[b, n, d] += epsilon
                output_plus = moe_layer(x_plus)
                loss_plus = output_plus.sum()
                
                x_minus = x.clone()
                x_minus[b, n, d] -= epsilon
                output_minus = moe_layer(x_minus)
                loss_minus = output_minus.sum()
                
                grad_input_fd[b, n, d] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compute error
    error = (grad_input_analytic - grad_input_fd).abs().max().item()
    relative_error = error / (grad_input_fd.abs().max().item() + 1e-9)
    results['input_gradient_error'] = relative_error
    results['max_error'] = max(results['max_error'], relative_error)
    
    if relative_error > tolerance:
        results['passed'] = False
        print(f"WARNING: Input gradient error {relative_error:.6f} exceeds tolerance {tolerance}")
    
    # Validate parameter gradients (sample a few parameters)
    # For brevity, we'll just check gating network weights
    param = moe_layer.gating.weight
    grad_analytic = grad_dict_analytic['gating_weight'].T
    
    grad_fd = torch.zeros_like(param)
    for i in range(min(5, param.shape[0])):  # Sample first 5 rows
        for j in range(min(5, param.shape[1])):  # Sample first 5 cols
            param_original = param[i, j].item()
            
            param.data[i, j] = param_original + epsilon
            output_plus = moe_layer(x)
            loss_plus = output_plus.sum()
            
            param.data[i, j] = param_original - epsilon
            output_minus = moe_layer(x)
            loss_minus = output_minus.sum()
            
            grad_fd[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore original value
            param.data[i, j] = param_original
    
    error = (grad_analytic[:5, :5] - grad_fd[:5, :5]).abs().max().item()
    relative_error = error / (grad_fd[:5, :5].abs().max().item() + 1e-9)
    results['parameter_gradient_errors']['gating_weight'] = relative_error
    results['max_error'] = max(results['max_error'], relative_error)
    
    if relative_error > tolerance:
        results['passed'] = False
        print(f"WARNING: Gating weight gradient error {relative_error:.6f} exceeds tolerance {tolerance}")
    
    return results
