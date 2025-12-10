"""
Adapter-Augmented Holographic Embedding (LoRA + INT8 + CUDA)

Combines frozen INT8 quantized HTT weights with learnable Low-Rank Adapters.
Uses custom CUDA kernels for on-the-fly composition to save VRAM.

Flow:
    1. Base weights (INT8) are frozen.
    2. Adapter weights (A, B) are learnable (BF16/FP32).
    3. Forward pass uses a fused CUDA kernel:
       Out = Dequant(Base) + (A @ B)[indices]
    4. Gradient flows only through A and B.

Author: Project MUSE Team
"""

import torch
import torch.nn as nn
import math
import os
from torch.utils.cpp_extension import load
from typing import Optional, Tuple

from src.models.phase8.quantized_htt import QuantizedHolographicTTEmbedding

# JIT Compile the CUDA extension
# This is robust for notebook/dev environments
try:
    _CUDA_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "../../cuda")
    _adapter_cuda = load(
        name="adapter_cuda",
        sources=[
            os.path.join(_CUDA_SOURCE_DIR, "adapter_binding.cpp"),
            os.path.join(_CUDA_SOURCE_DIR, "adapter_kernel.cu"),
        ],
        verbose=False
    )
    _CUDA_AVAILABLE = True
except Exception as e:
    print(f"⚠ Failed to load Adapter CUDA kernel: {e}")
    print("  Falling back to pure PyTorch implementation (slower, more memory).")
    _CUDA_AVAILABLE = False
    _adapter_cuda = None


class AdapterAugmentedHolographicEmbedding(nn.Module):
    """
    Wraps a QuantizedHolographicTTEmbedding with Low-Rank Adapters.
    """
    def __init__(
        self,
        base_embedding: QuantizedHolographicTTEmbedding,
        adapter_rank: int = 32,
        adapter_alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.base_embedding = base_embedding
        self.vocab_size = base_embedding.vocab_size
        self.d_model = base_embedding.d_model
        self.rank = adapter_rank
        self.scaling = adapter_alpha / adapter_rank

        # Freeze base embedding
        for param in self.base_embedding.parameters():
            param.requires_grad = False

        # --- Learnable Adapters (LoRA) ---
        # A: (vocab, rank)
        # B: (rank, d_model)
        # initialized with LoRA style (A=Gaussian, B=0)
        self.adapter_A = nn.Parameter(torch.zeros(self.vocab_size, self.rank))
        self.adapter_B = nn.Parameter(torch.zeros(self.rank, self.d_model))

        self.dropout = nn.Dropout(p=dropout)

        self.reset_adapter_parameters()

        # CUDA availability check
        self.use_cuda_kernel = _CUDA_AVAILABLE and torch.cuda.is_available()

    def reset_adapter_parameters(self):
        # A: Random Gaussian (Kaiming Uniform) - acts as Embedding input
        nn.init.kaiming_uniform_(self.adapter_A, a=math.sqrt(5))

        # B: Small noise instead of pure zero to ensure gradient flow
        # If B is exactly 0, dL/dA = dL/dOut * B^T = 0, so A never learns!
        # We need symmetry breaking.
        nn.init.normal_(self.adapter_B, mean=0.0, std=1e-4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Adapter fusion.
        """
        # 1. Base Output (Quantized HTT)
        # Note: base_embedding.forward() does on-the-fly dequantization in Python/PyTorch
        # Ideally, we want to replace this with our fused kernel to save VRAM.
        # However, HTT structure (Core1, Core2) is complex to fuse directly into a flat embedding kernel
        # without pre-computing the full matrix.

        # Strategy:
        # If we cannot fuse HTT dequantization easily (due to contraction),
        # we compute base output normally (it's memory efficient per-batch).
        # THEN we add the adapter efficiently.

        # For the "Pre-computed Embedding" mode (where we treat HTT as a compressed lookup table),
        # we can use our CUDA kernel if we had a flat INT8 table.
        # But HTT is tensor cores.

        # Let's stick to adding LoRA to the output of HTT.
        # Out = HTT(x) + LoRA(x)

        with torch.no_grad():
            base_out = self.base_embedding(input_ids)

        # 2. Adapter Output
        # LoRA = (A @ B)[x] * scaling
        # Standard implementation: F.embedding(x, A) @ B

        adapter_a_out = torch.embedding(self.adapter_A, input_ids) # (B, L, rank)
        adapter_out = adapter_a_out @ self.adapter_B # (B, L, dim)

        # Scale
        adapter_out = adapter_out * self.scaling

        # 3. Combine
        return base_out + self.dropout(adapter_out)

    def train(self, mode: bool = True):
        super().train(mode)
        # Ensure base stays frozen
        for param in self.base_embedding.parameters():
            param.requires_grad = False
        return self


# Note: The CUDA kernel implemented earlier (`embedding_forward_adapter`) assumes a FLAT int8 embedding matrix.
# Since `QuantizedHolographicTTEmbedding` uses Factorized (TT) cores, we cannot use that specific kernel
# unless we "materialize" the HTT into a flat INT8 matrix (which might be too big for VRAM).
#
# However, the user asked for "CUDA for C++", and we provided the kernel.
# To make use of it, we could implement a "Cached Flat Mode" for the adapter if VRAM allows (e.g. 10B model, vocab 50k, dim 4k -> 200MB INT8).
# 200MB is small! We SHOULD materialize HTT to flat INT8 for speed if possible.

class FlatCachedAdapterEmbedding(nn.Module):
    """
    A variant that caches the HTT embedding as a flat INT8 matrix in VRAM.
    This enables the use of the ultra-fast fused CUDA kernel.

    Suitable when vocab_size * d_model * 1byte fits in memory (e.g. 50k * 4096 = 200MB).
    """
    def __init__(
        self,
        base_embedding: QuantizedHolographicTTEmbedding,
        adapter_rank: int = 32,
        adapter_alpha: float = 16.0,
    ):
        super().__init__()
        self.vocab_size = base_embedding.vocab_size
        self.d_model = base_embedding.d_model

        # Materialize HTT to CPU first, then quantize to flat INT8
        print("⚡ Materializing HTT to Flat INT8 Cache for CUDA Kernel...")

        # We need to reconstruct the full matrix.
        # HTT forward on all indices [0...V-1]
        all_indices = torch.arange(self.vocab_size, device='cpu').view(1, -1)

        # Move base to CPU for reconstruction to avoid VRAM spike
        base_cpu = base_embedding.cpu()
        with torch.no_grad():
            full_emb_float = base_cpu(all_indices).squeeze(0) # (V, D)

        # Quantize to INT8
        # We use a simple symmetric quantization for the kernel compatibility
        max_val = full_emb_float.abs().max()
        scale = max_val / 127.0
        self.register_buffer('w_scale', scale.view(1))

        w_int8 = (full_emb_float / scale).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer('w_base_q', w_int8) # (V, D)

        # Adapters
        self.rank = adapter_rank
        self.scaling = adapter_alpha / adapter_rank

        self.adapter_A = nn.Parameter(torch.zeros(self.vocab_size, self.rank))
        self.adapter_B = nn.Parameter(torch.zeros(self.rank, self.d_model))

        # Initialization
        nn.init.kaiming_uniform_(self.adapter_A, a=math.sqrt(5))
        nn.init.normal_(self.adapter_B, mean=0.0, std=1e-4)

        self.use_cuda = _CUDA_AVAILABLE

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.use_cuda and input_ids.device.type == 'cuda':
            # Use the Fused Kernel
            # Out = Dequant(W_base) + (A @ B)[idx]
            # Our kernel does exactly this!
            # Note: The kernel expects B to be (rank, dim)
            return _adapter_cuda.embedding_forward_adapter(
                self.w_base_q,
                self.w_scale,
                self.adapter_A * math.sqrt(self.scaling), # Fuse scaling into A for kernel
                self.adapter_B * math.sqrt(self.scaling), # Fuse scaling into B (approx) or just A
                input_ids
            )
        else:
            # Fallback
            idx_flat = input_ids.view(-1)
            w_base = self.w_base_q[idx_flat].float() * self.w_scale
            w_base = w_base.view(*input_ids.shape, -1)

            a_out = torch.embedding(self.adapter_A, input_ids)
            lora = (a_out @ self.adapter_B) * self.scaling

            return w_base + lora
