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

from .bk_core import BKCoreFunction
from .moe import SparseMoELayer
from .birman_schwinger_core import BirmanSchwingerCore
from .prime_bump_potential import PrimeBumpPotential


class MoEResNetBKLayer(nn.Module):
    """
    MoE-ResNet-BK Layer: combines MoE FFN with BK-Core spectral features.

    This layer forms the core of the ResNet-BK model. It treats the input
    features `x` as a source to define a potential `V_ε`, which perturbs a
    base Hamiltonian `H_0`. The spectral response of this perturbed system,
    `H_ε = H_0 + V_ε`, is then computed and used as learned features.

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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.use_birman_schwinger = use_birman_schwinger

        self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k, dropout_p, use_scattering_router=use_scattering_router, scattering_scale=scattering_scale, scattering_scale_warmup_steps=scattering_scale_warmup_steps)
        self.v_proj = nn.Linear(d_model, 1)

        # BK-Core output (real, imag) -> d_model
        self.output_proj = nn.Linear(2, d_model)

        # Learnable scale for BK branch contribution (per-channel scaling)
        self.bk_scale = nn.Parameter(torch.ones(d_model, dtype=torch.float32))

        # Initialize BK-Core (either Birman-Schwinger or original)
        if use_birman_schwinger:
            self.birman_schwinger_core = BirmanSchwingerCore(
                n_seq=n_seq,
                epsilon=epsilon,
                use_mourre=use_mourre,
                use_lap=use_lap,
                schatten_threshold=schatten_threshold,
                precision_upgrade_threshold=precision_upgrade_threshold,
            )
            # Store diagnostics
            self.last_bs_diagnostics = {}
        else:
            # Original BK-Core setup
            # H0 (discrete Laplacian) as buffers
            self.register_buffer("h0_diag_base", torch.full((1, n_seq), -2.0, dtype=torch.float32))
            self.register_buffer("h0_sub_base",  torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))
            self.register_buffer("h0_super_base",torch.full((1, n_seq - 1), 1.0, dtype=torch.float32))

            # Spectral shift z as buffer
            self.register_buffer("z", torch.tensor(1.0j, dtype=torch.complex64))

            self.bk_core = BKCoreFunction.apply

        # --- Numerical stability parameters ---
        self.v_max = 3.0          # Potential v_i clipping range
        self.feature_clamp = 10.0 # BK features (ReG, ImG) clipping range

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, N, D) input tensor
        
        Returns:
            output: (B, N, D) combined FFN + BK features
        """
        B, N, D = x.shape
        assert N == self.n_seq, f"Sequence length mismatch: expected {self.n_seq}, got {N}"

        # First, compute potential v_i from input (needed for BK-Core)
        # Use a preliminary projection to get potential
        v_prelim = self.v_proj(x).squeeze(-1)  # (B, N)
        v_prelim = torch.clamp(v_prelim, -self.v_max, self.v_max)

        # Compute BK features using either Birman-Schwinger or original core
        # This gives us G_ii for scattering router
        G_ii = None
        if self.use_birman_schwinger:
            # Use Birman-Schwinger core with LAP stability
            features, diagnostics = self.birman_schwinger_core(v_prelim, z=1.0j)  # (B, N, 2)
            self.last_bs_diagnostics = diagnostics
            
            # Extract G_ii for scattering router
            # features is (B, N, 2) with [real, imag]
            G_ii = torch.complex(features[..., 0], features[..., 1])  # (B, N) complex
        else:
            # Original BK-Core
            # Expand H0 for batch
            h0_diag  = self.h0_diag_base.expand(B, -1)   # (B, N)
            h0_sub   = self.h0_sub_base.expand(B, -1)    # (B, N-1)
            h0_super = self.h0_super_base.expand(B, -1)  # (B, N-1)

            he_diag = h0_diag + v_prelim                # (B, N)

            # BK-Core + hybrid analytic gradient
            features = self.bk_core(he_diag, h0_super, h0_sub, self.z)  # (B, N, 2)
            
            # For original BK-Core, also extract G_ii if using scattering router
            if self.moe_ffn.use_scattering_router:
                G_ii = torch.complex(features[..., 0], features[..., 1])  # (B, N) complex

        # MoE-FFN with optional scattering-based routing
        epsilon = self.birman_schwinger_core.epsilon if self.use_birman_schwinger else 1.0
        ffn_out, routing_entropy, routing_diagnostics = self.moe_ffn(
            x, G_ii=G_ii, epsilon=epsilon
        )  # (B, N, D), scalar, dict

        # Store routing diagnostics
        self.last_routing_diagnostics = routing_diagnostics

        # Clip BK features (prevent explosion with MoE + residual)
        if self.feature_clamp is not None:
            features = torch.clamp(features, -self.feature_clamp, self.feature_clamp)

        spec_out = self.output_proj(features)       # (B, N, D)

        # Mix BK branch with learnable scale
        output = ffn_out + self.bk_scale * spec_out
        # stash routing entropy for logging
        self.last_routing_entropy = routing_entropy

        # Task 5: Check Unitarity Violation (Information Loss)
        # Physics: S = I + K. Unitarity implies S†S = I.
        # In non-Hermitian system, this is violated.
        # We approximate violation by norm change ratio: | ||y||/||x|| - 1 |
        with torch.no_grad():
            norm_in = x.norm(p=2, dim=-1).mean()
            norm_out = output.norm(p=2, dim=-1).mean()
            self.last_unitarity_violation = (norm_out / (norm_in + 1e-9) - 1.0).abs()

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
        )

    def forward(self, x):
        """Pre-Norm residual structure."""
        out = self.bk_layer(self.layer_norm(x))
        return x + out


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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq
        self.use_birman_schwinger = use_birman_schwinger
        self.prime_bump_init = prime_bump_init

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

        self.blocks = nn.ModuleList([
            ResNetBKBlock(
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
            h = block(h)
            # collect routing entropy if available
            if hasattr(block.bk_layer.moe_ffn, "last_routing_entropy"):
                ent = block.bk_layer.moe_ffn.last_routing_entropy
                if ent is not None:
                    routing_entropies.append(ent)
            # collect routing diagnostics if available
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
            if hasattr(block.bk_layer, 'last_bs_diagnostics'):
                diag = block.bk_layer.last_bs_diagnostics
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
