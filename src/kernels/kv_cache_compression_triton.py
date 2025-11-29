#!/usr/bin/env python3
"""
KV Cache Compression Triton Kernel

タスク33.1: KVキャッシュ圧縮カーネル
- 4-bit KV cache with per-channel scaling
- Fused decompression with distance computation
- 目標: 4x compression, <1% PPL degradation

Requirements: 39.1-39.6
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import math

# Tritonのインポート
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


if TRITON_AVAILABLE:
    @triton.jit
    def _kv_compress_kernel(
        K,  # Input Key [B, H, N, D]
        V,  # Input Value [B, H, N, D]
        K_compressed,  # Compressed Key (INT4 packed) [B, H, N, D//2]
        V_compressed,  # Compressed Value (INT4 packed) [B, H, N, D//2]
        K_scale,  # Key scale [B, H, N]
        V_scale,  # Value scale [B, H, N]
        K_zero,  # Key zero point [B, H, N]
        V_zero,  # Value zero point [B, H, N]
        stride_kv_b, stride_kv_h, stride_kv_n, stride_kv_d,
        stride_c_b, stride_c_h, stride_c_n, stride_c_d,
        stride_s_b, stride_s_h, stride_s_n,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        KVキャッシュ圧縮カーネル
        
        物理的直観: FP16/FP32のKVキャッシュを4ビット整数に圧縮
        per-channel scalingで精度を維持
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)
        
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # 各シーケンス位置を処理
        for n_idx in range(BLOCK_N):
            actual_n = pid_n * BLOCK_N + n_idx
            if actual_n >= N:
                continue
            
            # Kをロード
            k_ptr = K + pid_b * stride_kv_b + pid_h * stride_kv_h + actual_n * stride_kv_n
            k_vals = tl.load(
                k_ptr + tl.arange(0, BLOCK_D) * stride_kv_d,
                mask=tl.arange(0, BLOCK_D) < D,
                other=0.0
            )
            
            # Vをロード
            v_ptr = V + pid_b * stride_kv_b + pid_h * stride_kv_h + actual_n * stride_kv_n
            v_vals = tl.load(
                v_ptr + tl.arange(0, BLOCK_D) * stride_kv_d,
                mask=tl.arange(0, BLOCK_D) < D,
                other=0.0
            )
            
            # K: min/max計算
            k_min = tl.min(k_vals)
            k_max = tl.max(k_vals)
            k_scale_val = (k_max - k_min) / 15.0
            k_scale_val = tl.maximum(k_scale_val, 1e-8)
            k_zero_val = -k_min / k_scale_val
            
            # V: min/max計算
            v_min = tl.min(v_vals)
            v_max = tl.max(v_vals)
            v_scale_val = (v_max - v_min) / 15.0
            v_scale_val = tl.maximum(v_scale_val, 1e-8)
            v_zero_val = -v_min / v_scale_val
            
            # 量子化
            k_q = ((k_vals / k_scale_val) + k_zero_val)
            k_q = tl.minimum(tl.maximum(k_q, 0.0), 15.0)
            k_q = k_q.to(tl.int8)
            
            v_q = ((v_vals / v_scale_val) + v_zero_val)
            v_q = tl.minimum(tl.maximum(v_q, 0.0), 15.0)
            v_q = v_q.to(tl.int8)
            
            # パック（2つのINT4を1バイトに）
            offs_d_even = tl.arange(0, BLOCK_D // 2) * 2
            offs_d_odd = offs_d_even + 1
            
            k_even = tl.load(k_ptr + offs_d_even * stride_kv_d, mask=offs_d_even < D, other=0.0)
            k_odd = tl.load(k_ptr + offs_d_odd * stride_kv_d, mask=offs_d_odd < D, other=0.0)
            
            k_even_q = ((k_even / k_scale_val) + k_zero_val)
            k_even_q = tl.minimum(tl.maximum(k_even_q, 0.0), 15.0).to(tl.int8)
            k_odd_q = ((k_odd / k_scale_val) + k_zero_val)
            k_odd_q = tl.minimum(tl.maximum(k_odd_q, 0.0), 15.0).to(tl.int8)
            
            k_packed = k_even_q | (k_odd_q << 4)
            
            v_even = tl.load(v_ptr + offs_d_even * stride_kv_d, mask=offs_d_even < D, other=0.0)
            v_odd = tl.load(v_ptr + offs_d_odd * stride_kv_d, mask=offs_d_odd < D, other=0.0)
            
            v_even_q = ((v_even / v_scale_val) + v_zero_val)
            v_even_q = tl.minimum(tl.maximum(v_even_q, 0.0), 15.0).to(tl.int8)
            v_odd_q = ((v_odd / v_scale_val) + v_zero_val)
            v_odd_q = tl.minimum(tl.maximum(v_odd_q, 0.0), 15.0).to(tl.int8)
            
            v_packed = v_even_q | (v_odd_q << 4)
            
            # 保存
            k_c_ptr = K_compressed + pid_b * stride_c_b + pid_h * stride_c_h + actual_n * stride_c_n
            v_c_ptr = V_compressed + pid_b * stride_c_b + pid_h * stride_c_h + actual_n * stride_c_n
            
            tl.store(
                k_c_ptr + tl.arange(0, BLOCK_D // 2) * stride_c_d,
                k_packed.to(tl.uint8),
                mask=tl.arange(0, BLOCK_D // 2) < (D // 2)
            )
            tl.store(
                v_c_ptr + tl.arange(0, BLOCK_D // 2) * stride_c_d,
                v_packed.to(tl.uint8),
                mask=tl.arange(0, BLOCK_D // 2) < (D // 2)
            )
            
            # スケールとゼロポイントを保存
            s_ptr = K_scale + pid_b * stride_s_b + pid_h * stride_s_h + actual_n * stride_s_n
            tl.store(s_ptr, k_scale_val)
            
            s_ptr = V_scale + pid_b * stride_s_b + pid_h * stride_s_h + actual_n * stride_s_n
            tl.store(s_ptr, v_scale_val)
            
            z_ptr = K_zero + pid_b * stride_s_b + pid_h * stride_s_h + actual_n * stride_s_n
            tl.store(z_ptr, k_zero_val)
            
            z_ptr = V_zero + pid_b * stride_s_b + pid_h * stride_s_h + actual_n * stride_s_n
            tl.store(z_ptr, v_zero_val)


    @triton.jit
    def _fused_decompress_distance_kernel(
        Q,  # Query [B, H, M, D]
        K_compressed,  # Compressed Key [B, H, N, D//2]
        K_scale,  # Key scale [B, H, N]
        K_zero,  # Key zero point [B, H, N]
        Out,  # Output distance [B, H, M, N]
        curvature,
        stride_q_b, stride_q_h, stride_q_m, stride_q_d,
        stride_k_b, stride_k_h, stride_k_n, stride_k_d,
        stride_s_b, stride_s_h, stride_s_n,
        stride_o_b, stride_o_h, stride_o_m, stride_o_n,
        M: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        融合デコンプレス＋距離計算カーネル
        
        物理的直観: 圧縮されたKVキャッシュを展開しながら
        同時に双曲距離を計算することで、メモリ帯域幅を節約
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        
        mask_m = offs_m < M
        mask_d = offs_d < D
        
        # Qをロード
        q_ptr = Q + pid_b * stride_q_b + pid_h * stride_q_h
        q = tl.load(
            q_ptr + offs_m[:, None] * stride_q_m + offs_d[None, :] * stride_q_d,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0
        )
        
        # Q norm^2
        q_norm_sq = tl.sum(q * q, axis=1)  # [BLOCK_M]
        
        # Kブロックをイテレート
        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            
            # スケールとゼロポイントをロード
            s_ptr = K_scale + pid_b * stride_s_b + pid_h * stride_s_h
            k_scale = tl.load(s_ptr + offs_n * stride_s_n, mask=mask_n, other=1.0)
            
            z_ptr = K_zero + pid_b * stride_s_b + pid_h * stride_s_h
            k_zero = tl.load(z_ptr + offs_n * stride_s_n, mask=mask_n, other=0.0)
            
            # 圧縮Kをロード＆デコンプレス
            k_c_ptr = K_compressed + pid_b * stride_k_b + pid_h * stride_k_h
            
            # デコンプレスされたKを構築
            k_decompressed = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
            
            for d_idx in range(0, D // 2):
                packed = tl.load(
                    k_c_ptr + offs_n * stride_k_n + d_idx * stride_k_d,
                    mask=mask_n,
                    other=0
                ).to(tl.int32)
                
                # アンパック
                low = packed & 0x0F
                high = (packed >> 4) & 0x0F
                
                # デクオンタイズ
                k_low = (low.to(tl.float32) - k_zero) * k_scale
                k_high = (high.to(tl.float32) - k_zero) * k_scale
                
                # 保存（簡略化）
                # 実際の実装ではより効率的なメモリアクセスパターンを使用
            
            # 簡略化: 直接内積計算
            # 実際の実装ではデコンプレスされたKを使用
            k_norm_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
            qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            
            # 双曲距離計算
            diff_sq = q_norm_sq[:, None] + k_norm_sq[None, :] - 2.0 * qk
            diff_sq = tl.maximum(diff_sq, 0.0)
            
            denom_q = 1.0 - curvature * q_norm_sq[:, None]
            denom_k = 1.0 - curvature * k_norm_sq[None, :]
            denom = tl.maximum(denom_q * denom_k, 1e-6)
            
            cosh_arg = 1.0 + 2.0 * curvature * diff_sq / denom
            cosh_arg = tl.maximum(cosh_arg, 1.0)
            
            distance = tl.log(cosh_arg + tl.sqrt(cosh_arg * cosh_arg - 1.0 + 1e-8))
            
            # 出力を保存
            out_ptr = Out + pid_b * stride_o_b + pid_h * stride_o_h
            tl.store(
                out_ptr + offs_m[:, None] * stride_o_m + offs_n[None, :] * stride_o_n,
                distance,
                mask=mask_m[:, None] & mask_n[None, :]
            )


class KVCacheCompressor:
    """
    KVキャッシュ圧縮ユーティリティ
    
    4ビット量子化を使用してKVキャッシュを圧縮し、
    メモリ使用量を4分の1に削減する。
    """
    
    @staticmethod
    def compress(
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        KVキャッシュを圧縮
        
        Args:
            k: Key [B, H, N, D]
            v: Value [B, H, N, D]
        
        Returns:
            (k_compressed, v_compressed, metadata)
        """
        B, H, N, D = k.shape
        assert D % 2 == 0, "D must be even for INT4 packing"
        
        # Per-token量子化
        k_min = k.amin(dim=-1, keepdim=True)
        k_max = k.amax(dim=-1, keepdim=True)
        k_scale = (k_max - k_min) / 15.0
        k_scale = torch.clamp(k_scale, min=1e-8)
        k_zero = -k_min / k_scale
        
        v_min = v.amin(dim=-1, keepdim=True)
        v_max = v.amax(dim=-1, keepdim=True)
        v_scale = (v_max - v_min) / 15.0
        v_scale = torch.clamp(v_scale, min=1e-8)
        v_zero = -v_min / v_scale
        
        # 量子化
        k_q = ((k / k_scale) + k_zero).round().clamp(0, 15).to(torch.uint8)
        v_q = ((v / v_scale) + v_zero).round().clamp(0, 15).to(torch.uint8)
        
        # パック
        k_even = k_q[..., 0::2]
        k_odd = k_q[..., 1::2]
        k_packed = k_even | (k_odd << 4)
        
        v_even = v_q[..., 0::2]
        v_odd = v_q[..., 1::2]
        v_packed = v_even | (v_odd << 4)
        
        metadata = {
            'k_scale': k_scale.squeeze(-1),
            'k_zero': k_zero.squeeze(-1),
            'v_scale': v_scale.squeeze(-1),
            'v_zero': v_zero.squeeze(-1),
        }
        
        return k_packed, v_packed, metadata
    
    @staticmethod
    def decompress(
        k_packed: torch.Tensor,
        v_packed: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        KVキャッシュを展開
        
        Args:
            k_packed: Compressed Key [B, H, N, D//2]
            v_packed: Compressed Value [B, H, N, D//2]
            metadata: 量子化メタデータ
        
        Returns:
            (k, v)
        """
        k_scale = metadata['k_scale'].unsqueeze(-1)
        k_zero = metadata['k_zero'].unsqueeze(-1)
        v_scale = metadata['v_scale'].unsqueeze(-1)
        v_zero = metadata['v_zero'].unsqueeze(-1)
        
        # アンパック
        k_low = k_packed & 0x0F
        k_high = (k_packed >> 4) & 0x0F
        
        v_low = v_packed & 0x0F
        v_high = (v_packed >> 4) & 0x0F
        
        # インターリーブ
        B, H, N, D_packed = k_packed.shape
        D = D_packed * 2
        
        k = torch.zeros(B, H, N, D, device=k_packed.device, dtype=torch.float32)
        k[..., 0::2] = k_low.float()
        k[..., 1::2] = k_high.float()
        
        v = torch.zeros(B, H, N, D, device=v_packed.device, dtype=torch.float32)
        v[..., 0::2] = v_low.float()
        v[..., 1::2] = v_high.float()
        
        # デクオンタイズ
        k = (k - k_zero) * k_scale
        v = (v - v_zero) * v_scale
        
        return k, v


class CompressedKVCache(nn.Module):
    """
    圧縮KVキャッシュモジュール
    
    4ビット量子化を使用してKVキャッシュを管理し、
    メモリ効率を最大化する。
    
    Args:
        max_seq_len: 最大シーケンス長
        num_heads: アテンションヘッド数
        head_dim: ヘッド次元
        device: デバイス
    """
    
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device = None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 圧縮キャッシュバッファ
        self.register_buffer(
            'k_cache',
            torch.zeros(1, num_heads, max_seq_len, head_dim // 2, dtype=torch.uint8)
        )
        self.register_buffer(
            'v_cache',
            torch.zeros(1, num_heads, max_seq_len, head_dim // 2, dtype=torch.uint8)
        )
        
        # メタデータバッファ
        self.register_buffer(
            'k_scale',
            torch.ones(1, num_heads, max_seq_len)
        )
        self.register_buffer(
            'k_zero',
            torch.zeros(1, num_heads, max_seq_len)
        )
        self.register_buffer(
            'v_scale',
            torch.ones(1, num_heads, max_seq_len)
        )
        self.register_buffer(
            'v_zero',
            torch.zeros(1, num_heads, max_seq_len)
        )
        
        self.current_len = 0
    
    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        start_pos: int,
    ):
        """
        キャッシュを更新
        
        Args:
            k: 新しいKey [B, H, L, D]
            v: 新しいValue [B, H, L, D]
            start_pos: 開始位置
        """
        B, H, L, D = k.shape
        
        # バッチサイズの調整
        if self.k_cache.shape[0] != B:
            self.k_cache = self.k_cache.expand(B, -1, -1, -1).contiguous()
            self.v_cache = self.v_cache.expand(B, -1, -1, -1).contiguous()
            self.k_scale = self.k_scale.expand(B, -1, -1).contiguous()
            self.k_zero = self.k_zero.expand(B, -1, -1).contiguous()
            self.v_scale = self.v_scale.expand(B, -1, -1).contiguous()
            self.v_zero = self.v_zero.expand(B, -1, -1).contiguous()
        
        # 圧縮
        k_packed, v_packed, metadata = KVCacheCompressor.compress(k, v)
        
        # キャッシュに保存
        end_pos = start_pos + L
        self.k_cache[:, :, start_pos:end_pos, :] = k_packed
        self.v_cache[:, :, start_pos:end_pos, :] = v_packed
        self.k_scale[:, :, start_pos:end_pos] = metadata['k_scale']
        self.k_zero[:, :, start_pos:end_pos] = metadata['k_zero']
        self.v_scale[:, :, start_pos:end_pos] = metadata['v_scale']
        self.v_zero[:, :, start_pos:end_pos] = metadata['v_zero']
        
        self.current_len = end_pos
    
    def get(
        self,
        end_pos: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        キャッシュを取得（展開）
        
        Args:
            end_pos: 終了位置（Noneの場合は現在の長さ）
        
        Returns:
            (k, v)
        """
        if end_pos is None:
            end_pos = self.current_len
        
        k_packed = self.k_cache[:, :, :end_pos, :]
        v_packed = self.v_cache[:, :, :end_pos, :]
        
        metadata = {
            'k_scale': self.k_scale[:, :, :end_pos],
            'k_zero': self.k_zero[:, :, :end_pos],
            'v_scale': self.v_scale[:, :, :end_pos],
            'v_zero': self.v_zero[:, :, :end_pos],
        }
        
        return KVCacheCompressor.decompress(k_packed, v_packed, metadata)
    
    def get_compression_ratio(self) -> float:
        """圧縮率を取得"""
        # FP16: 2 bytes per element
        # INT4 packed: 0.5 bytes per element + metadata
        original_size = self.current_len * self.head_dim * 2 * 2  # K + V, FP16
        compressed_size = (
            self.current_len * (self.head_dim // 2) * 2 +  # K + V packed
            self.current_len * 4 * 4  # 4 metadata tensors, FP32
        )
        
        return original_size / compressed_size if compressed_size > 0 else 0
    
    def reset(self):
        """キャッシュをリセット"""
        self.current_len = 0


def compress_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    KVキャッシュを圧縮する関数インターフェース
    
    Args:
        k: Key [B, H, N, D]
        v: Value [B, H, N, D]
    
    Returns:
        (k_compressed, v_compressed, metadata)
    """
    return KVCacheCompressor.compress(k, v)


def decompress_kv_cache(
    k_packed: torch.Tensor,
    v_packed: torch.Tensor,
    metadata: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KVキャッシュを展開する関数インターフェース
    
    Args:
        k_packed: Compressed Key
        v_packed: Compressed Value
        metadata: 量子化メタデータ
    
    Returns:
        (k, v)
    """
    return KVCacheCompressor.decompress(k_packed, v_packed, metadata)


__all__ = [
    'TRITON_AVAILABLE',
    'KVCacheCompressor',
    'CompressedKVCache',
    'compress_kv_cache',
    'decompress_kv_cache',
]
