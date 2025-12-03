"""
Flash Attention 2 Wrapper for Phase 8

Provides integration with Flash Attention 2 if available, with fallback to existing
Triton kernels.
"""

import torch
import torch.nn as nn
from typing import Optional

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None


def is_flash_attention_available() -> bool:
    """Check if Flash Attention 2 is available."""
    return FLASH_ATTN_AVAILABLE


class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for Flash Attention 2 with fallback support.
    
    Provides a unified interface for attention computation that automatically
    uses Flash Attention 2 if available, otherwise falls back to standard
    scaled dot-product attention or custom Triton kernels.
    """
    
    def __init__(
        self,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        use_flash_if_available: bool = True,
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.use_flash_if_available = use_flash_if_available
        
        self.using_flash = FLASH_ATTN_AVAILABLE and use_flash_if_available
        
        if self.using_flash:
            print("✅ Using Flash Attention 2 for optimized attention")
        else:
            print("⚠️  Flash Attention 2 not available, using fallback")
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention 2 or fallback.
        
        Args:
            q: Query tensor [batch, seq_len, num_heads, head_dim]
            k: Key tensor [batch, seq_len, num_heads, head_dim]
            v: Value tensor [batch, seq_len, num_heads, head_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]
        """
        if self.using_flash and attn_mask is None:  # Flash Attention doesn't support arbitrary masks easily
            # Flash Attention 2 expects [batch, seq_len, num_heads, head_dim]
            # which is what we have
            try:
                output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=self.causal,
                )
                return output
            except Exception as e:
                print(f"⚠️  Flash Attention failed ({e}), falling back to standard attention")
                # Fall through to fallback
        
        # Fallback: Use PyTorch scaled dot-product attention
        # Transpose to [batch, num_heads, seq_len, head_dim] for torch.nn.functional
        batch, seq_len, num_heads, head_dim = q.shape
        
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        if self.softmax_scale is None:
            scale = head_dim ** -0.5
        else:
            scale = self.softmax_scale
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        if self.causal:
            # Create causal mask
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=q.device),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        if self.dropout_p > 0.0 and self.training:
            attn_weights = torch.dropout(attn_weights, self.dropout_p, train=True)
        
        output = torch.matmul(attn_weights, v)
        
        # Transpose back to [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2)
        
        return output


def create_flash_attention_layer(
    dropout_p: float = 0.0,
    causal: bool = False,
) -> FlashAttentionWrapper:
    """
    Factory function to create Flash Attention layer.
    
    Args:
        dropout_p: Dropout probability
        causal: Whether to use causal attention
        
    Returns:
        FlashAttentionWrapper instance
    """
    return FlashAttentionWrapper(
        dropout_p=dropout_p,
        causal=causal,
        use_flash_if_available=True,
    )
