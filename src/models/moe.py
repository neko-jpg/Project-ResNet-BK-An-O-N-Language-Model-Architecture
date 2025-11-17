"""
Sparse Mixture of Experts (MoE) Layer
Implements top-k expert routing with Gumbel-Softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts with top-k routing.
    
    Args:
        d_model: hidden dimension
        num_experts: number of expert networks
        top_k: number of experts to route to (1 for sparse, num_experts for dense)
        dropout_p: dropout probability
    """
    
    def __init__(self, d_model, num_experts=4, top_k=1, dropout_p=0.1, use_scattering_router: bool = False, scattering_scale: float = 0.1, scattering_scale_warmup_steps: int = 0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_scattering_router = use_scattering_router
        self.scattering_scale = scattering_scale
        self.scattering_scale_warmup_steps = scattering_scale_warmup_steps
        self.register_buffer("scattering_step_counter", torch.tensor(0, dtype=torch.long))

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(d_model * 2, d_model),
            )
            for _ in range(num_experts)
        ])
        self.gating_network = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        Forward pass with expert routing.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) routed through experts
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # (T, D), T = B*N
        router_logits = self.gating_network(x_flat)  # (T, E)
        if self.use_scattering_router:
            # Centered, normalized token norm as scattering proxy (reduce blow-up, modulate entropy)
            token_norm = x_flat.norm(dim=-1, keepdim=True)  # (T, 1)
            norm_centered = token_norm - token_norm.mean()
            norm_scaled = norm_centered / (token_norm.std() + 1e-6)
            scale = self.scattering_scale
            # simple warmup: double scale for initial steps
            if self.scattering_scale_warmup_steps > 0:
                if self.scattering_step_counter < self.scattering_scale_warmup_steps:
                    scale = scale * 2.0
            router_logits = router_logits + scale * norm_scaled
            # advance counter once per forward call
            self.scattering_step_counter += 1

        routing_entropy = None
        if self.top_k >= self.num_experts:
            # Dense Mixture (softmax composition) mode
            gates = F.softmax(router_logits, dim=-1)  # (T, E)
            # Entropy per token averaged
            routing_entropy = - (gates * (gates.clamp_min(1e-8).log())).sum(dim=-1).mean()
            expert_outputs = []
            for expert in self.experts:
                expert_outputs.append(expert(x_flat))  # (T, D)
            stacked = torch.stack(expert_outputs, dim=1)  # (T, E, D)
            out_flat = torch.sum(stacked * gates.unsqueeze(-1), dim=1)  # (T, D)
        else:
            # Sparse top-k routing (defaulting to top-1 if k==1)
            topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
            # Normalize logits of the selected experts to obtain mixing weights
            topk_gates = F.softmax(topk_logits, dim=-1)  # (T, K)
            out_flat = torch.zeros_like(x_flat)
            # Entropy on selected experts (approximate)
            routing_entropy = - (topk_gates * (topk_gates.clamp_min(1e-8).log())).sum(dim=-1).mean()
            for expert_idx, expert in enumerate(self.experts):
                mask = (topk_indices == expert_idx)  # (T, K)
                if not mask.any():
                    continue

                token_idx, slot_idx = mask.nonzero(as_tuple=True)
                sub_x = x_flat[token_idx]                # (T_e, D)
                sub_y = expert(sub_x)                    # (T_e, D)
                weights = topk_gates[token_idx, slot_idx].unsqueeze(-1)

                out_flat[token_idx] = out_flat[token_idx] + (sub_y * weights).to(out_flat.dtype)

        out = out_flat.view(B, N, D)
        # store entropy for logging upstream
        self.last_routing_entropy = routing_entropy.item() if routing_entropy is not None else None
        return out, routing_entropy
