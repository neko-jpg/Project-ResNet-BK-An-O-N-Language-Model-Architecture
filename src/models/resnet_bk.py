"""
ResNet-BK Architecture
Combines BK-Core with MoE for O(N) language modeling.

This file defines the main building blocks of the ResNet-BK model.
The architecture is designed to implement the operator-theoretic concepts
described in the accompanying paper "Weil's Explicit Formula as a Birman-Krein
Phase Identity" (see `paper/theory/riemann_hypothesis_main.tex`).

The core idea is to treat the input sequence as a potential `V_ε` in a
quantum scattering problem, and use the spectral properties of the corresponding
Hamiltonian `H_ε = H_0 + V_ε` as features for the language model.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .bk_core import BKCoreFunction
from .moe import SparseMoELayer
from .birman_schwinger_core import BirmanSchwingerCore
from .prime_bump_potential import PrimeBumpPotential
from src.models.phase4.homeostasis import HomeostasisController
from src.kernels.fused_moe_kernel import fused_moe_forward

# === 追加 ===
from src.models.phase7 import HybridHyperbolicAttention
# ===========


class MoEResNetBKLayer(nn.Module):
    """
    MoE-ResNet-BK Layer: combines MoE FFN with BK-Core spectral features.

    This layer forms the core of the ResNet-BK model. It treats the input
    features `x` as a source to define a potential `V_ε`, which perturbs a
    base Hamiltonian `H_0`. The spectral response of this perturbed system,
    `H_ε = H_0 + V_ε`, is then computed and used as learned features.

    Now supports Non-Hermitian Physics via learnable gamma decay term:
    H_eff = H - i*gamma

    Architectural Flow (Forward Pass):
    1.  Input `x` (B, N, D) is projected to a scalar potential `v_prelim` (B, N).
        This corresponds to defining the perturbation `V_ε` in the theory.
    2.  The `BirmanSchwingerCore` or `BKCoreFunction` computes the spectral
        features (diagonal of the Green's function) of the system. This step
        is a numerical implementation of solving the scattering problem
        described in `riemann_hypothesis_main.tex`.
    3.  These spectral features are projected back to `d_model` and added to
        the output of a standard MoE-FFN layer, mixing sequence-level spectral
        information with token-level features.
    
    The output is a residual combination: `Output = FFN_out + bk_scale * BK_out`.
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
        enable_gradient_checkpointing: bool = False,
        use_hybrid_attention: bool = False,
        hyperbolic_window_size: int = 64,
        num_heads: int = 4,
        use_fused_moe_kernel: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.use_hybrid_attention = use_hybrid_attention
        self.use_fused_moe_kernel = use_fused_moe_kernel

        if self.use_hybrid_attention:
            self.hybrid_attn = HybridHyperbolicAttention(
                d_model=d_model,
                num_heads=num_heads,
                local_window_size=hyperbolic_window_size
            )
            self.moe_ffn = None
            self.birman_schwinger_core = None
            self.v_proj = None
            self.output_proj = None
            self.bk_scale = None
            self.gamma = None
            self.homeostasis = None
            self.bk_core = None
        else:
            self.use_birman_schwinger = use_birman_schwinger
            self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k, dropout_p, use_scattering_router=use_scattering_router, scattering_scale=scattering_scale, scattering_scale_warmup_steps=scattering_scale_warmup_steps)
            self.v_proj = nn.Linear(d_model, 1)
            self.output_proj = nn.Linear(2, d_model)
            self.bk_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
            self.homeostasis = HomeostasisController()
            if use_birman_schwinger:
                self.birman_schwinger_core = BirmanSchwingerCore(
                    n_seq=n_seq,
                    epsilon=epsilon,
                    use_mourre=use_mourre,
                    use_lap=use_lap,
                    schatten_threshold=schatten_threshold,
                    precision_upgrade_threshold=precision_upgrade_threshold,
                    enable_gradient_checkpointing=enable_gradient_checkpointing,
                    use_bitnet=use_bitnet,
                )
                self.last_bs_diagnostics = {}
            else:
                self.register_buffer("h0_diag_base", torch.full((1, n_seq), -2.0, dtype=torch.float32))
                self.register_buffer("h0_sub_base",  torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))
                self.register_buffer("h0_super_base",torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))
                self.epsilon_param = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
                self.bk_core = BKCoreFunction.apply

        self.v_max = 3.0
        self.feature_clamp = 10.0

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) combined FFN + BK features
        """
        if self.use_hybrid_attention:
            out_tuple = self.hybrid_attn(x, return_diagnostics=True)
            if isinstance(out_tuple, tuple):
                output, diagnostics = out_tuple
                self.last_hybrid_diagnostics = diagnostics
                return output
            else:
                return out_tuple
        else:
            B, N, D = x.shape
            assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"
            v_prelim = self.v_proj(x).squeeze(-1)
            v_prelim = torch.clamp(v_prelim, -self.v_max, self.v_max)
            gamma_val = self.gamma
            G_ii = None
            if self.use_birman_schwinger:
                features, diagnostics = self.birman_schwinger_core(v_prelim, z=1.0j, gamma=gamma_val)
                self.last_bs_diagnostics = diagnostics
                G_ii = torch.complex(features[..., 0], features[..., 1])
            else:
                h0_diag  = self.h0_diag_base.expand(B, -1)
                h0_sub   = self.h0_sub_base.expand(B, -1)
                h0_super = self.h0_super_base.expand(B, -1)
                he_diag = h0_diag + v_prelim
                epsilon = torch.nn.functional.softplus(self.epsilon_param) + 1e-6
                z = 1.0j * epsilon
                z = z + 1j * gamma_val
                z = z.to(dtype=torch.complex64, device=he_diag.device)
                features = self.bk_core(he_diag, h0_super, h0_sub, z)
                if self.moe_ffn.use_scattering_router:
                    G_ii = torch.complex(features[..., 0], features[..., 1])
            epsilon = self.birman_schwinger_core.epsilon if self.use_birman_schwinger else 1.0

            routing_entropy = None # Initialize to a default

            if self.use_fused_moe_kernel and not self.moe_ffn.use_scattering_router:
                # --- Fused MoE Kernel Path ---
                gate_w_t = self.moe_ffn.gating_network.weight.t()
                experts_w1_t = torch.stack([expert[0].weight.t() for expert in self.moe_ffn.experts])
                experts_w2_t = torch.stack([expert[3].weight.t() for expert in self.moe_ffn.experts])

                ffn_out = fused_moe_forward(
                    x,
                    gate_w_t,
                    experts_w1_t,
                    experts_w2_t,
                    self.moe_ffn.top_k
                )
                routing_diagnostics = None
                self.moe_ffn.last_routing_entropy = None
            else:
                # --- Original SparseMoELayer Path ---
                ffn_out, routing_entropy, routing_diagnostics = self.moe_ffn(x, G_ii=G_ii, epsilon=epsilon)
                # The original moe_ffn.forward call sets the .last_routing_entropy attribute itself

            self.last_routing_diagnostics = routing_diagnostics
            if self.feature_clamp is not None:
                features = torch.clamp(features, -self.feature_clamp, self.feature_clamp)
            spec_out = self.output_proj(features)
            output = ffn_out + self.bk_scale * spec_out

            # Restore the entropy assignment here to fix the regression
            if routing_entropy is not None:
                self.last_routing_entropy = routing_entropy.item()
            else:
                 # Ensure it's cleared if the path doesn't produce it (like the fused kernel)
                self.last_routing_entropy = None

            with torch.no_grad():
                norm_in = x.norm(p=2, dim=-1).mean()
                norm_out = output.norm(p=2, dim=-1).mean()
                self.last_unitarity_violation = (norm_out / (norm_in + 1e-9) - 1.0).abs()
                if self.training:
                    diagnostics = {
                        'unitarity_violation': self.last_unitarity_violation.item(),
                        'growth_ratio': (norm_out / (norm_in + 1e-9)).item()
                    }
                    new_gamma = self.homeostasis(self.gamma, diagnostics)
                    self.gamma.data.copy_(new_gamma)
            return output

    def check_unitarity_violation(self):
        """Return the last recorded unitarity violation metric."""
        return getattr(self, 'last_unitarity_violation', torch.tensor(0.0))


class ResNetBKBlock(nn.Module):
    """
    ResNet-BK Block with LayerNorm and residual connection.
    
    Architecture:
        Input -> LayerNorm -> MoEResNetBKLayer -> Add(Input) -> Output
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
        enable_gradient_checkpointing: bool = False,
        use_hybrid_attention: bool = False,
        hyperbolic_window_size: int = 64,
        num_heads: int = 4,
        use_fused_moe_kernel: bool = False,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(
            d_model,
            n_seq,
            num_experts,
            top_k,
            dropout_p,
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
            use_hybrid_attention=use_hybrid_attention,
            hyperbolic_window_size=hyperbolic_window_size,
            num_heads=num_heads,
            use_fused_moe_kernel=use_fused_moe_kernel,
        )

    def forward(self, x):
        """Pre-Norm residual structure."""
        out = self.bk_layer(self.layer_norm(x))
        return x + out


class SymplecticBKBlock(nn.Module):
    """
    Symplectic Integrator Block (The "Time Machine").

    Replaces standard residual connection with a physical Hamiltonian time evolution
    using Velocity Verlet integration (Order 2) or Symplectic Euler (Order 1).

    State:
        x -> separated into (q, p)
        q: Position (canonical coordinate)
        p: Momentum (conjugate momentum)

    Dynamics (Velocity Verlet, Order 2):
        1. p_{t+0.5} = p_t - 0.5 * eps * dV/dq(q_t)
        2. q_{t+1}   = q_t + eps * p_{t+0.5}
        3. p_{t+1}   = p_{t+0.5} - 0.5 * eps * dV/dq(q_{t+1})

    Dynamics (Symplectic Euler, Order 1):
        1. p_{t+1} = p_t - dt * dV/dq(q_t)
        2. q_{t+1} = q_t + dt * p_{t+1}

    Force term:
        F(q) = -dV/dq ≈ MoEResNetBKLayer(q)
        We interpret the BK layer output as the "force" acting on the system.

    Energy Conservation:
        H(q, p) = |p|^2 / 2 + V(q)
        The symplectic integrator approximately conserves this energy.
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

        # The BK Layer acts as the Force Field F(q)
        # Note: It acts on q (d_model/2) and must return Force (d_model/2)
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
        """
        Symplectic Step.

        Args:
            x: Tensor of shape (B, N, D).
               We split this into q = x[..., :D/2] and p = x[..., D/2:]

        Returns:
            next_state: Tensor of shape (B, N, D) containing [next_q, next_p]
        """
        B, N, D = x.shape
        q = x[..., :self.d_q]
        p = x[..., self.d_q:]

        # Velocity Verlet Integration
        dt = self.dt

        if self.integration_mode == 'euler':
            # Symplectic Euler (1st Order, 1 Force Eval)
            # p_{t+1} = p_t + dt * F(q_t)
            q_norm = self.layer_norm_q(q)
            force_t = self.force_field(q_norm)

            p_next = p + dt * force_t

            # q_{t+1} = q_t + dt * p_{t+1}
            q_next = q + dt * p_next

        else: # Default: 'verlet'
            # Velocity Verlet (2nd Order, 2 Force Evals)

            # 1. First half-step for momentum
            # p_{t+0.5} = p_t + 0.5 * dt * F(q_t)
            q_norm = self.layer_norm_q(q)
            force_t = self.force_field(q_norm)

            assert force_t.shape == p.shape, f"Force shape {force_t.shape} mismatch with p shape {p.shape}"

            p_half = p + 0.5 * dt * force_t

            # 2. Full step for position
            # q_{t+1} = q_t + dt * p_{t+0.5}
            q_next = q + dt * p_half

            # 3. Second half-step for momentum
            # p_{t+1} = p_{t+0.5} + 0.5 * dt * F(q_{t+1})
            q_next_norm = self.layer_norm_q(q_next)
            force_next = self.force_field(q_next_norm)

            p_next = p_half + 0.5 * dt * force_next

        # Re-assemble state
        next_state = torch.cat([q_next, p_next], dim=-1)

        # Diagnostic: Energy Conservation Check
        with torch.no_grad():
            ke_prev = 0.5 * p.norm(p=2, dim=-1).mean()
            ke_next = 0.5 * p_next.norm(p=2, dim=-1).mean()
            self.last_ke_diff = (ke_next - ke_prev).abs()

        return next_state

    def get_energy_diff(self):
        return getattr(self, 'last_ke_diff', torch.tensor(0.0))


class LanguageModel(nn.Module):
    """
    ResNet-BK Language Model.
    
    This class combines the ResNet-BK blocks into a full language model.
    It includes token and position embeddings, a stack of ResNetBKBlocks,
    and a final layer norm and language model head.

    A key feature is the optional "Prime-Bump Initialization", which is
    inspired by the theoretical construction of the potential `V_ε` from
    the prime numbers. See `riemann_hypothesis_main.tex` for the mathematical
    background. When enabled, the position embeddings are initialized with a
    pattern derived from the prime numbers, providing a structured inductive
    bias related to the spectral properties of the Riemann zeta function.

    Architecture:
        Token Embedding + Position Embedding (with optional Prime-Bump)
        -> ResNetBKBlock × n_layers
        -> LayerNorm
        -> LM Head
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=64,
        n_layers=4,
        n_seq=128,
        num_experts=4,
        top_k=1,
        dropout_p=0.1,
        use_scattering_router: bool = False,
        scattering_scale: float = 0.1,
        scattering_scale_warmup_steps: int = 0,
        prime_bump_init: bool = False,
        prime_bump_scale: float = 0.02,
        use_birman_schwinger: bool = False,
        epsilon: float = 1.0,
        use_mourre: bool = True,
        use_lap: bool = True,
        schatten_threshold: float = 100.0,
        precision_upgrade_threshold: float = 1e6,
        k_max: int = 3,
        use_bitnet: bool = False,
        use_symplectic: bool = False, # New Flag
        symplectic_dt: float = 0.1,
        symplectic_mode: str = 'verlet', # 'verlet' or 'euler'
        use_gradient_checkpointing: bool = False,
        use_hybrid_attention: bool = False,
        hyperbolic_window_size: int = 64,
        num_heads: int = 4,
        use_fused_moe_kernel: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.use_birman_schwinger = use_birman_schwinger
        self.prime_bump_init = prime_bump_init
        self.use_symplectic = use_symplectic
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)

        # Initialize Prime-Bump potential if using Birman-Schwinger
        if use_birman_schwinger and prime_bump_init:
            self.prime_bump_potential = PrimeBumpPotential(
                n_seq=n_seq,
                epsilon=epsilon,
                k_max=k_max,
                scale=prime_bump_scale,
            )
        else:
            self.prime_bump_potential = None

        block_class = SymplecticBKBlock if use_symplectic else ResNetBKBlock

        # If symplectic, d_model must be even and acts as combined [q,p]
        if use_symplectic and d_model % 2 != 0:
            raise ValueError("d_model must be even when use_symplectic=True")

        self.blocks = nn.ModuleList([
            block_class(
                d_model=d_model,
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
                enable_gradient_checkpointing=use_gradient_checkpointing,
                use_hybrid_attention=use_hybrid_attention,
                hyperbolic_window_size=hyperbolic_window_size,
                num_heads=num_heads,
                use_fused_moe_kernel=use_fused_moe_kernel,
                **({'dt': symplectic_dt, 'integration_mode': symplectic_mode} if use_symplectic else {})
            )
            for _ in range(n_layers)
        ])

        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self._reset_parameters(prime_bump_init=prime_bump_init, prime_bump_scale=prime_bump_scale)

    @staticmethod
    def _prime_indices(n: int):
        """Return list of primes < n using simple sieve."""
        if n < 2:
            return []
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for p in range(2, int(n ** 0.5) + 1):
            if sieve[p]:
                step = p
                start = p * p
                sieve[start:n:step] = [False] * len(range(start, n, step))
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def _reset_parameters(self, prime_bump_init: bool, prime_bump_scale: float):
        """Initialize weights. Optionally add prime-bump pattern to position embeddings."""
        # Base initializations
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=prime_bump_scale)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=prime_bump_scale)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Prime-bump: add structured offsets to position embeddings following prime indices
        if prime_bump_init:
            if self.use_birman_schwinger and self.prime_bump_potential is not None:
                # Use Prime-Bump potential for initialization
                # Add potential values to position embeddings
                with torch.no_grad():
                    V = self.prime_bump_potential.compute_potential()  # (N,)
                    pos_weight = self.position_embedding.weight.data  # (N, D)
                    # Add potential to each dimension (broadcast)
                    pos_weight.add_(V.unsqueeze(-1).expand_as(pos_weight))
            else:
                # Simple prime-bump: add offsets at prime positions
                primes = self._prime_indices(self.n_seq)
                if primes:
                    pos_weight = self.position_embedding.weight.data
                    bump = torch.zeros_like(pos_weight)
                    bump[primes] = prime_bump_scale
                    pos_weight.add_(bump)

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, n_seq) token indices
        
        Returns:
            logits: (batch_size, n_seq, vocab_size)
        """
        batch_size, n_seq = x.shape
        assert n_seq == self.n_seq, f"n_seq mismatch: expected {self.n_seq}, got {n_seq}"

        tok_emb = self.token_embedding(x)  # (B, N, D)

        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)  # (1, N, D)

        h = tok_emb + pos_emb

        routing_entropies = []
        routing_diagnostics_list = []
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                # use_reentrant=True avoids saved tensor hooks incompatibility with vmap
                h = checkpoint(lambda inp, blk=block: blk(inp), h)
            else:
                h = block(h)

            # Use appropriate attribute access depending on block type
            if self.use_symplectic:
                 # Symplectic block has .force_field instead of .bk_layer
                 if hasattr(block.force_field.moe_ffn, "last_routing_entropy"):
                    ent = block.force_field.moe_ffn.last_routing_entropy
                    if ent is not None:
                        routing_entropies.append(ent)
                 if hasattr(block.force_field, "last_routing_diagnostics"):
                    diag = block.force_field.last_routing_diagnostics
                    if diag is not None:
                        routing_diagnostics_list.append(diag)
            else:
                if hasattr(block.bk_layer.moe_ffn, "last_routing_entropy"):
                    ent = block.bk_layer.moe_ffn.last_routing_entropy
                    if ent is not None:
                        routing_entropies.append(ent)
                if hasattr(block.bk_layer, "last_routing_diagnostics"):
                    diag = block.bk_layer.last_routing_diagnostics
                    if diag is not None:
                        routing_diagnostics_list.append(diag)
        
        h = self.layer_norm_final(h)
        logits = self.lm_head(h)           # (B, N, vocab_size)
        
        # store average routing entropy for logging
        if routing_entropies:
            self.last_routing_entropy = float(sum(routing_entropies) / len(routing_entropies))
        else:
            self.last_routing_entropy = None
        
        # store aggregated routing diagnostics
        self.last_routing_diagnostics_list = routing_diagnostics_list

        return logits
    
    def get_stability_diagnostics(self):
        """
        Get stability diagnostics from Birman-Schwinger cores.
        
        Returns:
            Dictionary with aggregated diagnostics from all layers
        """
        if not self.use_birman_schwinger:
            return {}
        
        diagnostics = {
            'schatten_s1': [],
            'schatten_s2': [],
            'condition_number': [],
            'mourre_verified': [],
            's1_bound_satisfied': [],
            's2_bound_satisfied': [],
            'all_finite': [],
            'precision_upgrades': 0,
        }
        
        for block in self.blocks:
            if self.use_symplectic:
                 layer = block.force_field
            else:
                 layer = block.bk_layer

            if hasattr(layer, 'last_bs_diagnostics'):
                diag = layer.last_bs_diagnostics
                if diag:
                    diagnostics['schatten_s1'].append(diag.get('schatten_s1', 0.0))
                    diagnostics['schatten_s2'].append(diag.get('schatten_s2', 0.0))
                    diagnostics['condition_number'].append(diag.get('condition_number', 0.0))
                    diagnostics['mourre_verified'].append(diag.get('mourre_verified', False))
                    diagnostics['s1_bound_satisfied'].append(diag.get('s1_bound_satisfied', False))
                    diagnostics['s2_bound_satisfied'].append(diag.get('s2_bound_satisfied', False))
                    diagnostics['all_finite'].append(diag.get('all_finite', True))
                    diagnostics['precision_upgrades'] += diag.get('precision_upgrades', 0)
        
        # Compute aggregates
        if diagnostics['schatten_s1']:
            diagnostics['mean_schatten_s1'] = sum(diagnostics['schatten_s1']) / len(diagnostics['schatten_s1'])
            diagnostics['max_schatten_s1'] = max(diagnostics['schatten_s1'])
        if diagnostics['schatten_s2']:
            diagnostics['mean_schatten_s2'] = sum(diagnostics['schatten_s2']) / len(diagnostics['schatten_s2'])
            diagnostics['max_schatten_s2'] = max(diagnostics['schatten_s2'])
        if diagnostics['condition_number']:
            diagnostics['mean_condition_number'] = sum(diagnostics['condition_number']) / len(diagnostics['condition_number'])
            diagnostics['max_condition_number'] = max(diagnostics['condition_number'])
        
        diagnostics['mourre_verified_rate'] = sum(diagnostics['mourre_verified']) / len(diagnostics['mourre_verified']) if diagnostics['mourre_verified'] else 0.0
        diagnostics['s1_bound_satisfied_rate'] = sum(diagnostics['s1_bound_satisfied']) / len(diagnostics['s1_bound_satisfied']) if diagnostics['s1_bound_satisfied'] else 0.0
        diagnostics['s2_bound_satisfied_rate'] = sum(diagnostics['s2_bound_satisfied']) / len(diagnostics['s2_bound_satisfied']) if diagnostics['s2_bound_satisfied'] else 0.0
        diagnostics['all_finite_rate'] = sum(diagnostics['all_finite']) / len(diagnostics['all_finite']) if diagnostics['all_finite'] else 1.0
        
        return diagnostics
    
    def get_routing_diagnostics(self):
        """
        Get routing diagnostics from scattering router.
        
        Returns:
            Dictionary with aggregated routing diagnostics from all layers
        """
        if not hasattr(self, 'last_routing_diagnostics_list') or not self.last_routing_diagnostics_list:
            return {}
        
        diagnostics = {
            'mean_phase': [],
            'std_phase': [],
            'resonance_fraction': [],
            'mean_spectral_shift': [],
        }
        
        for diag in self.last_routing_diagnostics_list:
            if diag:
                diagnostics['mean_phase'].append(diag.get('mean_phase', 0.0))
                diagnostics['std_phase'].append(diag.get('std_phase', 0.0))
                diagnostics['resonance_fraction'].append(diag.get('resonance_fraction', 0.0))
                diagnostics['mean_spectral_shift'].append(diag.get('mean_spectral_shift', 0.0))
        
        # Compute aggregates
        aggregated = {}
        if diagnostics['mean_phase']:
            aggregated['avg_mean_phase'] = sum(diagnostics['mean_phase']) / len(diagnostics['mean_phase'])
            aggregated['avg_std_phase'] = sum(diagnostics['std_phase']) / len(diagnostics['std_phase'])
            aggregated['avg_resonance_fraction'] = sum(diagnostics['resonance_fraction']) / len(diagnostics['resonance_fraction'])
            aggregated['avg_spectral_shift'] = sum(diagnostics['mean_spectral_shift']) / len(diagnostics['mean_spectral_shift'])
        
        return aggregated
