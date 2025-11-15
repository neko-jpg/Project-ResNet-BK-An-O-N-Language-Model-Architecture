"""
Standard Transformer language model baseline used for ResNet-BK comparisons.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the Transformer baseline."""

    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    ffn_dim: int = 1024
    max_seq_len: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    tie_weights: bool = True


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block (MHA + FFN)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
        )
        self.ffn_dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention + FFN sublayers."""
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + self.attn_dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.ffn_dropout(ffn_out)
        return x


class TransformerLM(nn.Module):
    """Autoregressive Transformer language model."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def generate_attn_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to preserve autoregressive property."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len) token ids
            targets: (batch, seq_len) token ids for loss computation
        Returns:
            logits: (batch, seq_len, vocab)
            loss (optional)
        """
        bsz, seq_len = x.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}. "
                "Increase TransformerConfig.max_seq_len."
            )

        device = x.device
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        h = self.dropout(h)

        attn_mask = self.generate_attn_mask(seq_len, device)
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask)

        h = self.norm_f(h)
        logits = self.lm_head(h)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
