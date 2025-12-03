"""
Sparse Mixture of Experts (MoE) Layer
Implements top-k expert routing with Gumbel-Softmax or Scattering-Based Router.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts with top-k routing.
    
    Supports two routing modes:
    1. MLP-based routing (default): Learned gating network
    2. Scattering-based routing: Parameter-free physics-based routing
    
    Args:
        d_model: hidden dimension
        num_experts: number of expert networks
        top_k: number of experts to route to (1 for sparse, num_experts for dense)
        dropout_p: dropout probability
        use_scattering_router: enable scattering-based routing (zero parameters)
        scattering_scale: scaling factor for scattering modulation (legacy, for backward compat)
        scattering_scale_warmup_steps: warmup steps (legacy, for backward compat)
        scattering_router_config: configuration dict for ScatteringRouter
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 1,
        dropout_p: float = 0.1,
        use_scattering_router: bool = False,
        scattering_scale: float = 0.1,
        scattering_scale_warmup_steps: int = 0,
        scattering_router_config: Optional[Dict] = None,
        use_hyperbolic_router: bool = False,
        hyperbolic_curvature: float = 1.0,
        hyperbolic_boundary: float = 0.85,
        hyperbolic_router_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_scattering_router = use_scattering_router
        self.scattering_scale = scattering_scale
        self.scattering_scale_warmup_steps = scattering_scale_warmup_steps
        self.register_buffer("scattering_step_counter", torch.tensor(0, dtype=torch.long))
        self.use_hyperbolic_router = use_hyperbolic_router
        self.hyperbolic_curvature = hyperbolic_curvature
        self.hyperbolic_boundary = hyperbolic_boundary
        self.hyperbolic_update_prototypes = False
        self.hyperbolic_proto_decay = 0.9

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
        
        # Routing: MLP gating or Scattering-based
        if use_scattering_router:
            # Import here to avoid circular dependency
            from src.models.scattering_router import ScatteringRouter
            
            # Default config if not provided
            if scattering_router_config is None:
                scattering_router_config = {
                    'use_clark_measure': False,
                    'resonance_threshold': 0.1,
                    'top_k_resonance': min(2, num_experts),
                    'top_k_normal': min(top_k, num_experts),
                }
            
            self.scattering_router = ScatteringRouter(
                num_experts=num_experts,
                **scattering_router_config
            )
            self.gating_network = None  # No learnable parameters
            self.hyperbolic_prototypes = None
        else:
            self.gating_network = nn.Linear(d_model, num_experts)
            self.scattering_router = None
            self.hyperbolic_prototypes = None

        if self.use_hyperbolic_router:
            # Prototypes are placed near the boundary to encode specialization.
            self.hyperbolic_prototypes = nn.Parameter(
                torch.randn(num_experts, d_model) * 0.05
            )
            self.expert_radii = nn.Parameter(
                torch.full((num_experts,), hyperbolic_boundary)
            )
            if hyperbolic_router_config is not None:
                self.hyperbolic_update_prototypes = hyperbolic_router_config.get("update_prototypes", False)
                self.hyperbolic_proto_decay = hyperbolic_router_config.get("proto_decay", 0.9)

    def forward(
        self,
        x: torch.Tensor,
        G_ii: Optional[torch.Tensor] = None,
        epsilon: float = 1.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass with expert routing.
        
        Args:
            x: (B, N, D) input tensor
            G_ii: (B, N) or (B, N, 2) complex resolvent diagonal from BirmanSchwingerCore
                  Required if use_scattering_router=True
            epsilon: regularization parameter for scattering router
        
        Returns:
            output: (B, N, D) routed through experts
            routing_entropy: scalar entropy value (for logging)
            routing_diagnostics: dictionary with routing statistics (only for scattering router)
        """
        B, N, D = x.shape
        x_flat = x.reshape(B * N, D)  # (T, D), T = B*N
        
        routing_diagnostics = None
        
        if self.use_hyperbolic_router:
            return self._hyperbolic_route(x_flat, B, N, G_ii)
        
        if self.use_scattering_router:
            # Scattering-based routing (zero parameters)
            if G_ii is None:
                raise ValueError("G_ii (resolvent diagonal) required for scattering-based routing")
            
            # Get routing from scattering router
            expert_indices, routing_weights, routing_diagnostics = self.scattering_router(
                G_ii, epsilon
            )
            
            # expert_indices: (B, N, top_k)
            # routing_weights: (B, N, top_k)
            
            # Flatten for processing
            expert_indices_flat = expert_indices.reshape(B * N, -1)  # (T, top_k)
            routing_weights_flat = routing_weights.reshape(B * N, -1)  # (T, top_k)
            
            # Route through experts
            out_flat = torch.zeros_like(x_flat)
            
            for expert_idx, expert in enumerate(self.experts):
                # Find tokens routed to this expert
                mask = (expert_indices_flat == expert_idx)  # (T, top_k)
                if not mask.any():
                    continue
                
                token_idx, slot_idx = mask.nonzero(as_tuple=True)
                sub_x = x_flat[token_idx]  # (T_e, D)
                sub_y = expert(sub_x)  # (T_e, D)
                weights = routing_weights_flat[token_idx, slot_idx].unsqueeze(-1)
                
                out_flat[token_idx] = out_flat[token_idx] + (sub_y * weights).to(out_flat.dtype)
            
            # Compute entropy from routing weights
            routing_entropy = -(routing_weights_flat * (routing_weights_flat.clamp_min(1e-8).log())).sum(dim=-1).mean()
            
        else:
            # MLP-based routing (learned parameters)
            router_logits = self.gating_network(x_flat)  # (T, E)
            
            # Legacy scattering modulation (for backward compatibility)
            if self.scattering_scale > 0:
                token_norm = x_flat.norm(dim=-1, keepdim=True)  # (T, 1)
                norm_centered = token_norm - token_norm.mean()
                norm_scaled = norm_centered / (token_norm.std() + 1e-6)
                scale = self.scattering_scale
                if self.scattering_scale_warmup_steps > 0:
                    if self.scattering_step_counter < self.scattering_scale_warmup_steps:
                        scale = scale * 2.0
                router_logits = router_logits + scale * norm_scaled
                self.scattering_step_counter += 1
            
            routing_entropy = None
            if self.top_k >= self.num_experts:
                # Dense Mixture (softmax composition) mode
                gates = F.softmax(router_logits, dim=-1)  # (T, E)
                routing_entropy = -(gates * (gates.clamp_min(1e-8).log())).sum(dim=-1).mean()
                expert_outputs = []
                for expert in self.experts:
                    expert_outputs.append(expert(x_flat))  # (T, D)
                stacked = torch.stack(expert_outputs, dim=1)  # (T, E, D)
                out_flat = torch.sum(stacked * gates.unsqueeze(-1), dim=1)  # (T, D)
            else:
                # Sparse top-k routing
                topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
                topk_gates = F.softmax(topk_logits, dim=-1)  # (T, K)
                out_flat = torch.zeros_like(x_flat)
                routing_entropy = -(topk_gates * (topk_gates.clamp_min(1e-8).log())).sum(dim=-1).mean()
                
                for expert_idx, expert in enumerate(self.experts):
                    mask = (topk_indices == expert_idx)  # (T, K)
                    if not mask.any():
                        continue
                    
                    token_idx, slot_idx = mask.nonzero(as_tuple=True)
                    sub_x = x_flat[token_idx]  # (T_e, D)
                    sub_y = expert(sub_x)  # (T_e, D)
                    weights = topk_gates[token_idx, slot_idx].unsqueeze(-1)
                    
                    out_flat[token_idx] = out_flat[token_idx] + (sub_y * weights).to(out_flat.dtype)
        
        out = out_flat.view(B, N, D)
        
        # Store entropy for logging upstream
        self.last_routing_entropy = routing_entropy.item() if routing_entropy is not None else None
        
        return out, routing_entropy, routing_diagnostics

    def _project_to_ball(self, x: torch.Tensor) -> torch.Tensor:
        """Project vectors to the Poincaré ball with soft clipping."""
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.tanh(norm) / norm
        return x * scale

    def _hyperbolic_route(
        self,
        x_flat: torch.Tensor,
        B: int,
        N: int,
        G_ii: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """Hyperbolic distance based routing."""
        if self.hyperbolic_prototypes is None:
            raise RuntimeError("Hyperbolic router requested but prototypes are uninitialized.")

        c = torch.tensor(self.hyperbolic_curvature, device=x_flat.device, dtype=x_flat.dtype)
        sqrt_c = torch.sqrt(c.clamp(min=1e-6))

        x_proj = self._project_to_ball(x_flat)
        proto = self._project_to_ball(self.hyperbolic_prototypes)
        proto = proto * torch.tanh(self.expert_radii).unsqueeze(-1)

        x_norm_sq = (x_proj ** 2).sum(dim=-1, keepdim=True)
        proto_norm_sq = (proto ** 2).sum(dim=-1)

        diff = x_proj.unsqueeze(1) - proto.unsqueeze(0)  # (T, E, D)
        diff_sq = (diff ** 2).sum(dim=-1)

        denom = (1 - c * x_norm_sq) * (1 - c * proto_norm_sq.unsqueeze(0))
        denom = denom.clamp(min=1e-6)
        cosh_arg = 1 + 2 * c * diff_sq / denom
        dist = torch.acosh(cosh_arg.clamp_min(1 + 1e-6)) / sqrt_c

        router_logits = -dist
        if G_ii is not None:
            energy = torch.abs(G_ii.reshape(-1, 1))
            router_logits = router_logits + 0.1 * energy

        routing_entropy = None
        out_flat = torch.zeros_like(x_flat)
        routing_diagnostics = {
            'hyperbolic_distance_mean': dist.mean().item(),
        }

        if self.top_k >= self.num_experts:
            gates = F.softmax(router_logits, dim=-1)
            routing_entropy = -(gates * (gates.clamp_min(1e-8).log())).sum(dim=-1).mean()
            expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            out_flat = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=1)

            # EMA update prototypes (dense only)
            if self.hyperbolic_update_prototypes:
                proto_targets = torch.einsum("te,td->ed", gates, x_proj) / (gates.sum(dim=0).unsqueeze(-1) + 1e-6)
                with torch.no_grad():
                    decay = self.hyperbolic_proto_decay
                    self.hyperbolic_prototypes.mul_(decay).add_((1 - decay) * proto_targets)
        else:
            topk_logits, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
            topk_gates = F.softmax(topk_logits, dim=-1)
            routing_entropy = -(topk_gates * (topk_gates.clamp_min(1e-8).log())).sum(dim=-1).mean()

            for expert_idx, expert in enumerate(self.experts):
                mask = (topk_indices == expert_idx)
                if not mask.any():
                    continue

                token_idx, slot_idx = mask.nonzero(as_tuple=True)
                sub_x = x_flat[token_idx]
                sub_y = expert(sub_x)
                weights = topk_gates[token_idx, slot_idx].unsqueeze(-1)
                out_flat[token_idx] = out_flat[token_idx] + (sub_y * weights).to(out_flat.dtype)

        out = out_flat.view(B, N, self.d_model)
        self.last_routing_entropy = routing_entropy.item() if routing_entropy is not None else None
        return out, routing_entropy, routing_diagnostics
