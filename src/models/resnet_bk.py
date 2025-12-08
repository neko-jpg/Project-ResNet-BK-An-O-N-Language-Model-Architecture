"""
ResNet-BK Architecture
Combines BK-Core with MoE for O(N) language modeling.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import sys

from .bk_core import BKCoreFunction
from .moe import SparseMoELayer
from .birman_schwinger_core import BirmanSchwingerCore
from .prime_bump_potential import PrimeBumpPotential
from src.models.phase4.homeostasis import HomeostasisController
from src.kernels.fused_moe_kernel import fused_moe_forward
from src.models.phase7.hybrid_attention import HybridHyperbolicAttention


from .config import ResNetBKConfig
from src.models.bitnet import LowRankLinear

class LowRankFFN(nn.Module):
    """
    Feed-Forward Network with Low-Rank Linear Layers.
    """
    def __init__(self, config: ResNetBKConfig):
        super().__init__()
        self.d_model = config.d_model
        self.rank = config.low_rank_rank
        self.expansion_factor = 4
        d_ff = self.d_model * self.expansion_factor
        
        self.up_proj = LowRankLinear(self.d_model, d_ff, self.rank, use_bitnet=config.use_bitnet)
        self.down_proj = LowRankLinear(d_ff, self.d_model, self.rank, use_bitnet=config.use_bitnet)
        self.act = nn.GELU()

    def forward(self, x):
        # Apply with soft-capping for numerical stability
        up = self.act(self.up_proj(x))
        up = torch.tanh(up / 10.0) * 10.0  # Soft clamp intermediate
        out = self.down_proj(up)
        # Final safety: clamp output to reasonable range
        out = torch.tanh(out / 30.0) * 30.0
        out = torch.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)
        return out

class MoEResNetBKLayer(nn.Module):
    def __init__(self, config: ResNetBKConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_seq = config.n_seq
        self.use_hybrid_attention = config.use_hybrid_attention
        self.use_fused_moe_kernel = config.use_fused_moe_kernel

        if self.use_hybrid_attention:
            self.hybrid_attn = HybridHyperbolicAttention(config)
        else:
            self.hybrid_attn = None

        # FFN Selection
        self.ffn = None
        if config.low_rank_ffn:
            self.ffn = LowRankFFN(config)
        elif not self.use_hybrid_attention:
            # Only use MoE if not hybrid and not low_rank (legacy behavior)
            self.ffn = SparseMoELayer(
                config.d_model,
                config.num_experts,
                config.top_k,
                config.dropout_p,
                use_scattering_router=config.use_scattering_router,
                scattering_scale=config.scattering_scale,
                scattering_scale_warmup_steps=config.scattering_scale_warmup_steps,
                use_hyperbolic_router=config.use_hyperbolic_router,
                hyperbolic_curvature=config.hyperbolic_router_curvature,
                hyperbolic_boundary=config.hyperbolic_router_boundary,
                hyperbolic_router_config={
                    "update_prototypes": config.hyperbolic_router_update_proto,
                    "proto_decay": config.hyperbolic_router_proto_decay,
                },
            )

        self.v_proj = nn.Linear(config.d_model, 1)
        self.output_proj = nn.Linear(2, config.d_model)
        self.bk_scale = nn.Parameter(torch.ones(config.d_model, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.homeostasis = HomeostasisController()
        
        # Initialize v_proj with ultra-small weights (Muon + 327M model)
        # Critical: v_proj feeds into BK Core - must be extremely conservative
        with torch.no_grad():
            nn.init.normal_(self.v_proj.weight, mean=0.0, std=0.0001)  # 0.02 → 0.0001 (200x smaller!)
            if self.v_proj.bias is not None:
                nn.init.zeros_(self.v_proj.bias)
        
        # Add gradient clamp hook to v_proj (防止勾配爆発)
        def clamp_v_proj_grad(grad):
            if grad is None:
                return grad
            # Muon用: 非常に厳しいクランプ
            return torch.clamp(grad, -0.1, 0.1)
        self.v_proj.weight.register_hook(clamp_v_proj_grad)

        self.use_birman_schwinger = config.use_birman_schwinger
        if config.use_birman_schwinger:
            self.birman_schwinger_core = BirmanSchwingerCore(
                n_seq=config.n_seq,
                epsilon=config.epsilon,
                use_mourre=config.use_mourre,
                use_lap=config.use_lap,
                schatten_threshold=config.schatten_threshold,
                precision_upgrade_threshold=config.precision_upgrade_threshold,
                enable_gradient_checkpointing=config.use_gradient_checkpointing,
                use_bitnet=config.use_bitnet,
            )
            self.last_bs_diagnostics = {}
            self.bk_core = None
        else:
            self.birman_schwinger_core = None
            # --- Identity Initialization (Transparent Pipe) ---
            # Initialize strictly to Identity Matrix (Diag=1, Off=0) for Cold Start
            self.register_buffer("h0_diag_base", torch.full((1, config.n_seq), 1.0, dtype=torch.float32))
            self.register_buffer("h0_sub_base",  torch.zeros((1, config.n_seq - 1), dtype=torch.float32))
            self.register_buffer("h0_super_base",torch.zeros((1, config.n_seq - 1), dtype=torch.float32))
            self.epsilon_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
            self.bk_core = BKCoreFunction.apply

        self.v_max = 5.0  # Increased from 1.0 back to 5.0 for gradient flow
        self.feature_clamp = 5.0  # Reduced from 10.0 for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"
        
        v_prelim = self.v_proj(x).squeeze(-1)
        # Use tanh for smoother saturation instead of hard clamp
        v_prelim = self.v_max * torch.tanh(v_prelim / self.v_max)
        gamma_val = self.gamma

        features, G_ii = None, None
        if self.use_birman_schwinger:
            features, diagnostics = self.birman_schwinger_core(v_prelim, z=1.0j, gamma=gamma_val)
            self.last_bs_diagnostics = diagnostics
            G_ii = torch.complex(features[..., 0], features[..., 1]).unsqueeze(-1)
            if torch.isnan(G_ii).any():
                print(f"🚨 NaN in G_ii! Max: {G_ii.abs().max().item()}", flush=True)
        else:
            h0_diag  = self.h0_diag_base.expand(B, -1)
            h0_sub   = self.h0_sub_base.expand(B, -1)
            h0_super = self.h0_super_base.expand(B, -1)
            he_diag = h0_diag + v_prelim
            epsilon = torch.nn.functional.softplus(self.epsilon_param) + 1e-4  # 1e-6 → 1e-4 (Muon安定化)
            z = 1.0j * epsilon + 1j * gamma_val
            z = z.to(dtype=torch.complex64, device=he_diag.device)
            
            # Use bfloat16 for BK-Core (wider exponent range than fp16)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                he_diag_f = he_diag.to(torch.bfloat16)
                h0_super_f = h0_super.to(torch.bfloat16)
                h0_sub_f = h0_sub.to(torch.bfloat16)
                z_f = z  # z is complex64, compatible with bfloat16
                
                features, g_ii_scalar = self.bk_core(he_diag_f, h0_super_f, h0_sub_f, z_f)
                
            G_ii = g_ii_scalar.unsqueeze(-1)
            if torch.isnan(G_ii).any():
                print(f"🚨 NaN in G_ii! Max: {G_ii.abs().max().item()}", flush=True)

        # 物理的に重要な Green 関数の対角成分を保持してハイブリッド注意に渡す
        self.last_g_ii = G_ii
        
        if self.use_hybrid_attention:
            out_tuple = self.hybrid_attn(x, g_ii=G_ii, return_diagnostics=True)
            if isinstance(out_tuple, tuple):
                output, diagnostics = out_tuple
                self.last_hybrid_diagnostics = diagnostics
            else:
                output = out_tuple
            
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"🚨 NaN/Inf detected after HybridAttention! Max: {output.abs().max().item()}", flush=True)
            
            # Apply FFN if present (Sequential: Attn -> FFN)
            if self.ffn is not None:
                ffn_out = self.ffn(output)
                if torch.isnan(ffn_out).any() or torch.isinf(ffn_out).any():
                    print(f"🚨 NaN/Inf detected after FFN! Max: {ffn_out.abs().max().item()}", flush=True)
                output = output + ffn_out 
                
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"🚨 NaN/Inf detected after FFN Addition! Max: {output.abs().max().item()}", flush=True)
                # RECOVERY: Replace NaN/Inf with safe values
                output = torch.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)

        else:
            epsilon = self.birman_schwinger_core.epsilon if self.use_birman_schwinger else 1.0
            if self.ffn is not None:
                # MoE or LowRank FFN
                if isinstance(self.ffn, SparseMoELayer):
                    ffn_out, routing_entropy, routing_diagnostics = self.ffn(x, G_ii=G_ii, epsilon=epsilon)
                    self.last_routing_diagnostics = routing_diagnostics
                else:
                    ffn_out = self.ffn(x)
            else:
                ffn_out = torch.zeros_like(x)

            if self.feature_clamp is not None:
                features = torch.clamp(features, -self.feature_clamp, self.feature_clamp)
            spec_out = self.output_proj(features)
            output = ffn_out + self.bk_scale * spec_out
        
        # Unified FFN application for Hybrid mode if not handled above
        if self.use_hybrid_attention and self.ffn is not None:
             pass
             
        return output

class ResNetBKBlock(nn.Module):
    """
    ResNet-BK Block with LayerNorm and residual connection.
    """
    def __init__(self, config: ResNetBKConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.bk_layer = MoEResNetBKLayer(config)

    def forward(self, x):
        """Pre-Norm residual structure."""
        if torch.isnan(x).any():
             print(f"🚨 NaN detected in ResNetBKBlock Input!")
             # RECOVERY: Replace NaN with zeros
             x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
             
        norm_x = self.layer_norm(x)
        out = self.bk_layer(norm_x)
        
        if torch.isnan(out).any():
             print(f"🚨 NaN detected in ResNetBKBlock Output (before residual)!")
             # RECOVERY: Replace NaN with zeros
             out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
             
        return x + out


class SymplecticBKBlock(nn.Module):
    """
    Symplectic Integrator Block (The "Time Machine").

    Replaces standard residual connection with a physical Hamiltonian time evolution
    using Velocity Verlet integration (Order 2) or Symplectic Euler (Order 1).
    """

    def __init__(
        self,
        d_model,
        n_seq,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        use_scattering_router: bool = False,
        scattering_scale: float = 0.1,
        scattering_scale_warmup_steps: int = 0,
        use_birman_schwinger: bool = False,
        epsilon: float = 1.0,
        use_mourre: bool = True,
        use_lap: bool = True,
        schatten_threshold: float = 100.0,
        precision_upgrade_threshold: float = 1e6,
        use_bitnet: bool = False,
        dt: float = 0.1, # Time step size
        enable_gradient_checkpointing: bool = False,
        integration_mode: str = 'verlet', # 'verlet' or 'euler'
        use_fused_moe_kernel: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dt = dt
        self.integration_mode = integration_mode

        self.d_q = d_model // 2
        assert d_model % 2 == 0, "d_model must be even for Symplectic Block (q, p split)"

        self.layer_norm_q = nn.LayerNorm(self.d_q)

        self.force_field = MoEResNetBKLayer(
            d_model=self.d_q,
            n_seq=n_seq,
            num_experts=num_experts,
            top_k=top_k,
            dropout_p=dropout_p,
            use_scattering_router=use_scattering_router,
            scattering_scale=scattering_scale,
            scattering_scale_warmup_steps=scattering_scale_warmup_steps,
            use_birman_schwinger=use_birman_schwinger,
            epsilon=epsilon,
            use_mourre=use_mourre,
            use_lap=use_lap,
            schatten_threshold=schatten_threshold,
            precision_upgrade_threshold=precision_upgrade_threshold,
            use_bitnet=use_bitnet,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            use_fused_moe_kernel=use_fused_moe_kernel,
        )

    def forward(self, x):
        B, N, D = x.shape
        q = x[..., :self.d_q]
        p = x[..., self.d_q:]
        dt = self.dt

        if self.integration_mode == 'euler':
            q_norm = self.layer_norm_q(q)
            force_t = self.force_field(q_norm)
            p_next = p + dt * force_t
            q_next = q + dt * p_next
        else:
            q_norm = self.layer_norm_q(q)
            force_t = self.force_field(q_norm)
            assert force_t.shape == p.shape, f"Force shape {force_t.shape} mismatch with p shape {p.shape}"
            p_half = p + 0.5 * dt * force_t
            q_next = q + dt * p_half
            q_next_norm = self.layer_norm_q(q_next)
            force_next = self.force_field(q_next_norm)
            p_next = p_half + 0.5 * dt * force_next

        next_state = torch.cat([q_next, p_next], dim=-1)

        with torch.no_grad():
            ke_prev = 0.5 * p.norm(p=2, dim=-1).mean()
            ke_next = 0.5 * p_next.norm(p=2, dim=-1).mean()
            self.last_ke_diff = (ke_next - ke_prev).abs()

        return next_state

    def get_energy_diff(self):
        return getattr(self, 'last_ke_diff', torch.tensor(0.0))


from .config import ResNetBKConfig

class LanguageModel(nn.Module):
    def __init__(self, config: ResNetBKConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_seq = config.n_seq
        self.use_birman_schwinger = config.use_birman_schwinger
        self.prime_bump_init = config.prime_bump_init
        self.use_symplectic = config.use_symplectic
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.use_hybrid_attention = config.use_hybrid_attention
        self.num_heads = config.num_heads

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.n_seq, config.d_model)
        
        # Register gradient clamp hooks for embeddings (Muon用: より厳しく)
        def clamp_position_grad(grad):
            if grad is not None:
                # CRITICAL: Sanitize NaN/Inf BEFORE clamping
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                return torch.clamp(grad, -1.0, 1.0)
            return grad
        self.position_embedding.weight.register_hook(clamp_position_grad)
        
        def clamp_token_grad(grad):
            if grad is not None:
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
                return torch.clamp(grad, -1.0, 1.0)
            return grad
        self.token_embedding.weight.register_hook(clamp_token_grad)

        if config.use_birman_schwinger and config.prime_bump_init:
            self.prime_bump_potential = PrimeBumpPotential(
                n_seq=config.n_seq,
                epsilon=config.epsilon,
                k_max=config.k_max,
                scale=config.prime_bump_scale,
            )
        else:
            self.prime_bump_potential = None

        block_class = SymplecticBKBlock if config.use_symplectic else ResNetBKBlock

        if config.use_symplectic and config.d_model % 2 != 0:
            raise ValueError("d_model must be even when use_symplectic=True")

        # Create a dictionary of arguments for the block class, excluding those not in its __init__
        block_args = config.__dict__.copy()
        if config.use_symplectic:
            # SymplecticBKBlock does not take all ResNetBKConfig args
            valid_args = SymplecticBKBlock.__init__.__code__.co_varnames
            block_args = {k: v for k, v in block_args.items() if k in valid_args}
            block_args['dt'] = config.symplectic_dt
            block_args['integration_mode'] = config.symplectic_mode
            self.blocks = nn.ModuleList([
                SymplecticBKBlock(**block_args) for _ in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                ResNetBKBlock(config=config) for _ in range(config.n_layers)
            ])

        self.layer_norm_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self._reset_parameters(prime_bump_init=config.prime_bump_init, prime_bump_scale=config.prime_bump_scale)

    @staticmethod
    def _prime_indices(n: int):
        if n < 2: return []
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for p in range(2, int(n ** 0.5) + 1):
            if sieve[p]:
                step = p
                start = p * p
                sieve[start:n:step] = [False] * len(range(start, n, step))
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def _reset_parameters(self, prime_bump_init: bool, prime_bump_scale: float):
        # Poincaré Centering: Initialize embeddings deep within the hyperbolic space (norm < 1e-3)
        # to prevent boundary collapse during Cold Start.
        embed_std = 0.0001  # 1e-4
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=embed_std)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=embed_std)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use much smaller initialization for stability
                # Especially important for v_proj which feeds into BK Core
                fan_in = module.weight.size(1)
                std = min(0.0001, (1.0 / fan_in) ** 0.5)  # Cap at 1e-4 for ultra-stability
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        if prime_bump_init:
            if self.use_birman_schwinger and self.prime_bump_potential is not None:
                with torch.no_grad():
                    V = self.prime_bump_potential.compute_potential()
                    pos_weight = self.position_embedding.weight.data
                    pos_weight.add_(V.unsqueeze(-1).expand_as(pos_weight))
            else:
                primes = self._prime_indices(self.n_seq)
                if primes:
                    pos_weight = self.position_embedding.weight.data
                    bump = torch.zeros_like(pos_weight)
                    bump[primes] = prime_bump_scale
                    pos_weight.add_(bump)

    def forward(self, x):
        batch_size, n_seq = x.shape
        assert n_seq == self.n_seq, f"n_seq mismatch: expected {self.n_seq}, got {n_seq}"

        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        h = tok_emb + pos_emb
        
        # === Embedding → Block Safety Zone ===
        # Clamp magnitude to prevent explosion before entering first LayerNorm
        h = torch.nan_to_num(h, nan=0.0, posinf=10.0, neginf=-10.0)
        h = torch.clamp(h, -10.0, 10.0)

        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                # use_reentrant=True for vmap compatibility
                h = checkpoint(lambda inp, blk=block: blk(inp), h, use_reentrant=True)
            else:
                h = block(h)

        h = self.layer_norm_final(h)
        logits = self.lm_head(h)
        
        return logits
