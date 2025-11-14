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
    
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.moe_ffn = SparseMoELayer(d_model, num_experts, top_k, dropout_p)
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
        ffn_out = self.moe_ffn(x)               # (B, N, D)

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
        return ffn_out + self.bk_scale * spec_out


class ResNetBKBlock(nn.Module):
    """
    ResNet-BK Block with LayerNorm and residual connection.
    
    Architecture:
        Input -> LayerNorm -> MoEResNetBKLayer -> Add(Input) -> Output
    """
    
    def __init__(self, d_model, n_seq, num_experts=4, top_k=1, dropout_p=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.bk_layer = MoEResNetBKLayer(d_model, n_seq, num_experts, top_k, dropout_p)

    def forward(self, x):
        """Pre-Norm residual structure."""
        return x + self.bk_layer(self.layer_norm(x))


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
            )
            for _ in range(n_layers)
        ])

        self.layer_norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

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

        h = self.layer_norm_final(h)
        logits = self.lm_head(h)           # (B, N, vocab_size)
        return logits
