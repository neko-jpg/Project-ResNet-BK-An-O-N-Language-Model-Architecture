"""
ResNet-BK Architecture
Combines BK-Core with MoE for O(N) language modeling.
"""

import torch
import torch.nn as nn

from .bk_core import BKCoreFunction
from .moe import SparseMoELayer


class MoEResNetBKLayer(nn.Module):
    """
    MoE-ResNet-BK Layer: combines MoE FFN with BK-Core spectral features.
    
    Architecture:
        Input -> MoE-FFN -> Potential v_i -> BK-Core -> Features -> Output
        Output = FFN_out + bk_scale * BK_out
    """
    
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1, use_scattering_router: bool = False, scattering_scale: float = 0.1, scattering_scale_warmup_steps: int = 0):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k, dropout_p, use_scattering_router=use_scattering_router, scattering_scale=scattering_scale, scattering_scale_warmup_steps=scattering_scale_warmup_steps)
        self.v_proj = nn.Linear(d_model, 1)

        # BK-Core output (real, imag) -> d_model
        self.output_proj = nn.Linear(2, d_model)

        # Learnable scale for BK branch contribution
        self.bk_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

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

        # MoE-FFN
        ffn_out, routing_entropy = self.moe_ffn(x)             # (B, N, D), scalar

        # Potential v_i (B, N)
        v = self.v_proj(ffn_out).squeeze(-1)    # (B, N)
        # Numerical stability: clip potential
        v = torch.clamp(v, -self.v_max, self.v_max)

        # Expand H0 for batch
        h0_diag  = self.h0_diag_base.expand(B, -1)   # (B, N)
        h0_sub   = self.h0_sub_base.expand(B, -1)    # (B, N-1)
        h0_super = self.h0_super_base.expand(B, -1)  # (B, N-1)

        he_diag = h0_diag + v                       # (B, N)

        # BK-Core + hybrid analytic gradient
        features = self.bk_core(he_diag, h0_super, h0_sub, self.z)  # (B, N, 2)

        # Clip BK features (prevent explosion with MoE + residual)
        if self.feature_clamp is not None:
            features = torch.clamp(features, -self.feature_clamp, self.feature_clamp)

        spec_out = self.output_proj(features)       # (B, N, D)

        # Mix BK branch with learnable scale
        output = ffn_out + self.bk_scale * spec_out
        # stash routing entropy for logging
        self.last_routing_entropy = routing_entropy
        return output


class ResNetBKBlock(nn.Module):
    """
    ResNet-BK Block with LayerNorm and residual connection.
    
    Architecture:
        Input -> LayerNorm -> MoEResNetBKLayer -> Add(Input) -> Output
    """
    
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1, use_scattering_router: bool = False, scattering_scale: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq, num_experts, top_k, dropout_p, use_scattering_router=use_scattering_router, scattering_scale=scattering_scale)

    def forward(self, x):
        """Pre-Norm residual structure."""
        out = self.bk_layer(self.layer_norm(x))
        return x + out


class LanguageModel(nn.Module):
    """
    ResNet-BK Language Model.
    
    Architecture:
        Token Embedding + Position Embedding
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
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_seq, d_model)

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

        for block in self.blocks:
            h = block(h)
            # collect routing entropy if available
            if hasattr(block.bk_layer.moe_ffn, "last_routing_entropy"):
                ent = block.bk_layer.moe_ffn.last_routing_entropy
                if ent is not None:
                    routing_entropies.append(ent)
        h = self.layer_norm_final(h)
        logits = self.lm_head(h)           # (B, N, vocab_size)
        # store average routing entropy for logging
        if routing_entropies:
            self.last_routing_entropy = float(sum(routing_entropies) / len(routing_entropies))
        else:
            self.last_routing_entropy = None

        return logits
