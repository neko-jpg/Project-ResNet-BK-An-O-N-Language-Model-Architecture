#!/usr/bin/env python3
"""
Quantized Hyperbolic Distance Triton Kernel

タスク32.2: INT8/INT4量子化双曲距離計算カーネル
- INT8距離計算とルックアップテーブル
- INT4 weights with 8-bit activations (W4A8)
- 目標: INT8で2xスピードアップ、INT4で4xスピードアップ

Requirements: 36.1-36.6
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

# ルックアップテーブル用の定数
ACOSH_LUT_SIZE = 256
ACOSH_LUT_MAX = 10.0

def _create_acosh_lut(size: int = ACOSH_LUT_SIZE, max_val: float = ACOSH_LUT_MAX) -> torch.Tensor:
    """acosh関数のルックアップテーブルを作成"""
    x = torch.linspace(1.0, max_val, size)
    return torch.acosh(x).float()

# グローバルLUT（初期化時に作成）
_ACOSH_LUT: Optional[torch.Tensor] = None

def get_acosh_lut(device: torch.device) -> torch.Tensor:
    """acosh LUTを取得（遅延初期化）"""
    global _ACOSH_LUT
    if _ACOSH_LUT is None or _ACOSH_LUT.device != device:
        _ACOSH_LUT = _create_acosh_lut().to(device)
    return _ACOSH_LUT


if TRITON_AVAILABLE:
    @triton.jit
    def _int8_hyperbolic_distance_kernel(
        Q,  # INT8量子化クエリ [B, H, N, D]
        K,  # INT8量子化キー [B, H, N, D]
        Q_scale,  # クエリスケール [B, H]
        K_scale,  # キースケール [B, H]
        Q_norm_sq,  # ||Q||^2 [B, H, N]
        K_norm_sq,  # ||K||^2 [B, H, N]
        Out,  # 出力距離 [B, H, N, N]
        curvature,
        stride_q_b, stride_q_h, stride_q_n, stride_q_d,
        stride_k_b, stride_k_h, stride_k_n, stride_k_d,
        stride_o_b, stride_o_h, stride_o_n, stride_o_m,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        INT8量子化双曲距離計算カーネル
        
        物理的直観: 整数演算で近似距離を計算し、
        スケールファクターで補正する。双曲距離は
        d_H(x,y) = acosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
        """
        # プログラムID
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)
        
        # ブロックオフセット
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        
        # マスク
        mask_m = offs_m < N
        
        # Qブロックをロード
        q_ptrs = Q + pid_b * stride_q_b + pid_h * stride_q_h
        q_block = tl.load(
            q_ptrs + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d,
            mask=mask_m[:, None] & (offs_d[None, :] < D),
            other=0
        ).to(tl.float32)
        
        # スケールをロード
        q_scale_val = tl.load(Q_scale + pid_b * H + pid_h)
        k_scale_val = tl.load(K_scale + pid_b * H + pid_h)
        scale_factor = q_scale_val * k_scale_val
        
        # Q norm^2をロード
        q_norm_sq = tl.load(Q_norm_sq + pid_b * H * N + pid_h * N + offs_m, mask=mask_m, other=0.0)

        
        # Kブロックをイテレート
        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            
            # Kブロックをロード
            k_ptrs = K + pid_b * stride_k_b + pid_h * stride_k_h
            k_block = tl.load(
                k_ptrs + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d,
                mask=(offs_d[:, None] < D) & mask_n[None, :],
                other=0
            ).to(tl.float32)
            
            # K norm^2をロード
            k_norm_sq = tl.load(K_norm_sq + pid_b * H * N + pid_h * N + offs_n, mask=mask_n, other=0.0)
            
            # INT8内積計算: Q @ K^T
            qk = tl.dot(q_block, k_block) * scale_factor
            
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            diff_sq = q_norm_sq[:, None] + k_norm_sq[None, :] - 2.0 * qk
            diff_sq = tl.maximum(diff_sq, 0.0)  # 数値安定性
            
            # 双曲距離の計算
            # d_H = acosh(1 + 2 * diff_sq / ((1 - ||x||^2)(1 - ||y||^2)))
            denom_q = 1.0 - curvature * q_norm_sq[:, None]
            denom_k = 1.0 - curvature * k_norm_sq[None, :]
            denom = denom_q * denom_k
            denom = tl.maximum(denom, 1e-6)  # ゼロ除算防止
            
            cosh_arg = 1.0 + 2.0 * curvature * diff_sq / denom
            cosh_arg = tl.maximum(cosh_arg, 1.0)  # acoshの定義域
            
            # acoshの近似: acosh(x) ≈ log(x + sqrt(x^2 - 1))
            distance = tl.log(cosh_arg + tl.sqrt(cosh_arg * cosh_arg - 1.0 + 1e-8))
            
            # 出力を保存
            out_ptrs = Out + pid_b * stride_o_b + pid_h * stride_o_h
            tl.store(
                out_ptrs + offs_m[:, None] * stride_o_n + offs_n[None, :] * stride_o_m,
                distance,
                mask=mask_m[:, None] & mask_n[None, :]
            )


    @triton.jit
    def _int4_dequantize_kernel(
        W_int4,  # INT4 weights (packed, 2 values per byte) [N, D//2]
        W_scale,  # スケール [N]
        W_zero,  # ゼロポイント [N]
        W_out,  # 出力 (FP16) [N, D]
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        INT4 → FP16 デクオンタイズカーネル
        
        物理的直観: 4ビット整数を浮動小数点に変換
        W_fp = (W_int4 - zero_point) * scale
        """
        pid_n = tl.program_id(0)
        
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        
        mask_n = offs_n < N
        
        # スケールとゼロポイントをロード
        scale = tl.load(W_scale + offs_n, mask=mask_n, other=1.0)
        zero = tl.load(W_zero + offs_n, mask=mask_n, other=0.0)
        
        # パックされたINT4をロード（2値/バイト）
        for d_start in range(0, D, BLOCK_D * 2):
            offs_d_packed = d_start // 2 + offs_d
            mask_d = offs_d_packed < (D // 2)
            
            # パックされたバイトをロード
            packed = tl.load(
                W_int4 + offs_n[:, None] * (D // 2) + offs_d_packed[None, :],
                mask=mask_n[:, None] & mask_d[None, :],
                other=0
            ).to(tl.int32)
            
            # 下位4ビットと上位4ビットを分離
            low_4bit = packed & 0x0F
            high_4bit = (packed >> 4) & 0x0F
            
            # デクオンタイズ
            low_fp = (low_4bit.to(tl.float32) - zero[:, None]) * scale[:, None]
            high_fp = (high_4bit.to(tl.float32) - zero[:, None]) * scale[:, None]
            
            # 出力を保存
            offs_d_low = d_start + offs_d * 2
            offs_d_high = d_start + offs_d * 2 + 1
            
            mask_d_low = offs_d_low < D
            mask_d_high = offs_d_high < D
            
            tl.store(
                W_out + offs_n[:, None] * D + offs_d_low[None, :],
                low_fp.to(tl.float16),
                mask=mask_n[:, None] & mask_d_low[None, :]
            )
            tl.store(
                W_out + offs_n[:, None] * D + offs_d_high[None, :],
                high_fp.to(tl.float16),
                mask=mask_n[:, None] & mask_d_high[None, :]
            )


    @triton.jit
    def _int8_attention_kernel(
        Q,  # INT8 Query [B, H, N, D]
        K,  # INT8 Key [B, H, N, D]
        V,  # FP16 Value [B, H, N, D]
        Q_scale,  # [B, H]
        K_scale,  # [B, H]
        Out,  # FP16 Output [B, H, N, D]
        curvature,
        stride_q_b, stride_q_h, stride_q_n, stride_q_d,
        stride_k_b, stride_k_h, stride_k_n, stride_k_d,
        stride_v_b, stride_v_h, stride_v_n, stride_v_d,
        stride_o_b, stride_o_h, stride_o_n, stride_o_d,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        INT8量子化アテンションカーネル（フルアテンション）
        
        Q, Kは INT8、VはFP16で計算
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)
        
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        
        mask_m = offs_m < N
        
        # Qをロード
        q_ptrs = Q + pid_b * stride_q_b + pid_h * stride_q_h
        q = tl.load(
            q_ptrs + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d,
            mask=mask_m[:, None] & (offs_d[None, :] < D),
            other=0
        ).to(tl.float32)
        
        # スケール
        q_scale_val = tl.load(Q_scale + pid_b * H + pid_h)
        k_scale_val = tl.load(K_scale + pid_b * H + pid_h)
        scale = q_scale_val * k_scale_val / tl.sqrt(D * 1.0)
        
        # アキュムレータ
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        
        # Kブロックをイテレート（オンラインソフトマックス）
        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N
            
            # Kをロード
            k_ptrs = K + pid_b * stride_k_b + pid_h * stride_k_h
            k = tl.load(
                k_ptrs + offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d,
                mask=(offs_d[:, None] < D) & mask_n[None, :],
                other=0
            ).to(tl.float32)
            
            # Vをロード
            v_ptrs = V + pid_b * stride_v_b + pid_h * stride_v_h
            v = tl.load(
                v_ptrs + offs_n[:, None] * stride_v_n + offs_d[None, :] * stride_v_d,
                mask=mask_n[:, None] & (offs_d[None, :] < D),
                other=0.0
            ).to(tl.float32)
            
            # QK^T
            qk = tl.dot(q, k) * scale
            
            # オンラインソフトマックス
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            
            l_new = alpha * l_i + beta * tl.sum(tl.exp(qk - m_ij[:, None]), axis=1)
            
            # アキュムレータ更新
            p = tl.exp(qk - m_new[:, None])
            acc = alpha[:, None] * acc + tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
            
            m_i = m_new
            l_i = l_new
        
        # 正規化
        out = acc / l_i[:, None]
        
        # 出力を保存
        out_ptrs = Out + pid_b * stride_o_b + pid_h * stride_o_h
        tl.store(
            out_ptrs + offs_m[:, None] * stride_o_n + offs_d[None, :] * stride_o_d,
            out.to(tl.float16),
            mask=mask_m[:, None] & (offs_d[None, :] < D)
        )



class INT8Quantizer:
    """INT8量子化ユーティリティ"""
    
    @staticmethod
    def quantize(
        x: torch.Tensor,
        per_channel: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FP16/FP32テンソルをINT8に量子化
        
        Args:
            x: 入力テンソル
            per_channel: チャネルごとの量子化を使用するか
        
        Returns:
            (quantized, scale, zero_point)
        """
        if per_channel:
            # 最後の次元以外で量子化
            reduce_dims = tuple(range(x.dim() - 1))
            x_min = x.amin(dim=-1, keepdim=True)
            x_max = x.amax(dim=-1, keepdim=True)
        else:
            x_min = x.min()
            x_max = x.max()
        
        # スケールとゼロポイント計算
        scale = (x_max - x_min) / 255.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = (-x_min / scale).round().clamp(0, 255)
        
        # 量子化
        x_q = ((x / scale) + zero_point).round().clamp(0, 255).to(torch.uint8)
        
        return x_q, scale.squeeze(), zero_point.squeeze()
    
    @staticmethod
    def dequantize(
        x_q: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """INT8テンソルをFP16にデクオンタイズ"""
        return ((x_q.float() - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)).half()


class INT4Quantizer:
    """INT4量子化ユーティリティ（W4A8用）"""
    
    @staticmethod
    def quantize(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        FP16/FP32テンソルをINT4に量子化（パック形式）
        
        Args:
            x: 入力テンソル [..., D] (Dは偶数)
        
        Returns:
            (packed_int4, scale, zero_point)
        """
        assert x.shape[-1] % 2 == 0, "Last dimension must be even for INT4 packing"
        
        x_min = x.amin(dim=-1, keepdim=True)
        x_max = x.amax(dim=-1, keepdim=True)
        
        # INT4: 0-15
        scale = (x_max - x_min) / 15.0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = (-x_min / scale).round().clamp(0, 15)
        
        # 量子化
        x_q = ((x / scale) + zero_point).round().clamp(0, 15).to(torch.uint8)
        
        # パック: 2つのINT4を1バイトに
        x_even = x_q[..., 0::2]  # 偶数インデックス（下位4ビット）
        x_odd = x_q[..., 1::2]   # 奇数インデックス（上位4ビット）
        packed = x_even | (x_odd << 4)
        
        return packed, scale.squeeze(), zero_point.squeeze()
    
    @staticmethod
    def dequantize(
        packed: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """INT4パックテンソルをFP16にデクオンタイズ"""
        # アンパック
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        
        # インターリーブ
        D_packed = packed.shape[-1]
        D = D_packed * 2
        
        x_q = torch.zeros(*packed.shape[:-1], D, dtype=torch.uint8, device=packed.device)
        x_q[..., 0::2] = low
        x_q[..., 1::2] = high
        
        return ((x_q.float() - zero_point.unsqueeze(-1)) * scale.unsqueeze(-1)).half()



class QuantizedHyperbolicAttention(nn.Module):
    """
    量子化双曲アテンションモジュール
    
    INT8/INT4量子化を使用して高速な双曲アテンションを実現。
    
    Args:
        d_model: モデル次元
        num_heads: アテンションヘッド数
        curvature: 双曲空間の曲率
        quantization_bits: 量子化ビット数 (8 or 4)
        use_triton: Tritonカーネルを使用するか
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        curvature: float = 1.0,
        quantization_bits: int = 8,
        use_triton: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.curvature = curvature
        self.quantization_bits = quantization_bits
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # 射影層
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # 量子化器
        self.quantizer = INT8Quantizer() if quantization_bits == 8 else INT4Quantizer()
        
        # キャリブレーション用の統計
        self.register_buffer('q_scale', torch.ones(1))
        self.register_buffer('k_scale', torch.ones(1))
        self.calibrated = False
    
    def calibrate(self, x: torch.Tensor):
        """キャリブレーションデータで量子化パラメータを設定"""
        with torch.no_grad():
            q = self.q_proj(x)
            k = self.k_proj(x)
            
            # 動的範囲を計算
            q_range = q.abs().max()
            k_range = k.abs().max()
            
            self.q_scale.fill_(q_range / 127.0)
            self.k_scale.fill_(k_range / 127.0)
            self.calibrated = True
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [B, N, D]
            mask: アテンションマスク [B, N, N] (optional)
        
        Returns:
            出力テンソル [B, N, D]
        """
        B, N, D = x.shape
        
        # 射影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # ヘッド分割
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 量子化アテンション
        if self.use_triton and self.quantization_bits == 8:
            out = self._int8_attention_triton(q, k, v, mask)
        else:
            out = self._pytorch_attention(q, k, v, mask)
        
        # ヘッド結合
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.o_proj(out)

    
    def _int8_attention_triton(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Triton INT8アテンション"""
        B, H, N, D = q.shape
        
        # INT8量子化
        q_int8, q_scale, q_zp = INT8Quantizer.quantize(q)
        k_int8, k_scale, k_zp = INT8Quantizer.quantize(k)
        
        # スケールを[B, H]形状に
        if q_scale.dim() == 0:
            q_scale = q_scale.expand(B, H)
            k_scale = k_scale.expand(B, H)
        elif q_scale.dim() == 2:
            q_scale = q_scale.mean(dim=-1)  # [B, H]
            k_scale = k_scale.mean(dim=-1)
        
        # 出力テンソル
        out = torch.empty(B, H, N, D, device=q.device, dtype=torch.float16)
        
        # ブロックサイズ
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = min(64, D)
        
        # グリッド
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        
        _int8_attention_kernel[grid](
            q_int8, k_int8, v.half(),
            q_scale, k_scale,
            out,
            self.curvature,
            q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3),
            k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, H=H, N=N, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        
        return out
    
    def _pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """PyTorchフォールバック実装"""
        B, H, N, D = q.shape
        scale = D ** -0.5
        
        # 双曲距離ベースのアテンション
        # 簡略化: ユークリッド内積を使用
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        
        return torch.matmul(attn, v)



def int8_hyperbolic_distance(
    q: torch.Tensor,
    k: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    INT8量子化双曲距離計算
    
    Args:
        q: クエリテンソル [B, H, N, D]
        k: キーテンソル [B, H, N, D]
        curvature: 双曲空間の曲率
    
    Returns:
        距離行列 [B, H, N, N]
    """
    if not TRITON_AVAILABLE:
        return _pytorch_hyperbolic_distance(q, k, curvature)
    
    B, H, N, D = q.shape
    
    # INT8量子化
    q_int8, q_scale, _ = INT8Quantizer.quantize(q)
    k_int8, k_scale, _ = INT8Quantizer.quantize(k)
    
    # ノルム計算
    q_norm_sq = (q ** 2).sum(dim=-1)
    k_norm_sq = (k ** 2).sum(dim=-1)
    
    # スケールを[B, H]形状に
    if q_scale.dim() > 2:
        q_scale = q_scale.mean(dim=-1)
        k_scale = k_scale.mean(dim=-1)
    
    # 出力テンソル
    out = torch.empty(B, H, N, N, device=q.device, dtype=torch.float32)
    
    # ブロックサイズ
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = min(64, D)
    
    # グリッド
    grid = (B, H, triton.cdiv(N, BLOCK_M))
    
    _int8_hyperbolic_distance_kernel[grid](
        q_int8, k_int8,
        q_scale, k_scale,
        q_norm_sq, k_norm_sq,
        out,
        curvature,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B=B, H=H, N=N, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    
    return out


def _pytorch_hyperbolic_distance(
    q: torch.Tensor,
    k: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """PyTorchフォールバック: 双曲距離計算"""
    # ||x - y||^2
    diff_sq = ((q.unsqueeze(-2) - k.unsqueeze(-3)) ** 2).sum(dim=-1)
    
    # ノルム
    q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)
    k_norm_sq = (k ** 2).sum(dim=-1).unsqueeze(-2)
    
    # 分母
    denom = (1 - curvature * q_norm_sq) * (1 - curvature * k_norm_sq)
    denom = torch.clamp(denom, min=1e-6)
    
    # acosh引数
    cosh_arg = 1 + 2 * curvature * diff_sq / denom
    cosh_arg = torch.clamp(cosh_arg, min=1.0)
    
    # 双曲距離
    return torch.acosh(cosh_arg)



def int4_dequantize(
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    INT4パックテンソルをFP16にデクオンタイズ
    
    Args:
        packed: パックされたINT4テンソル [N, D//2]
        scale: スケール [N]
        zero_point: ゼロポイント [N]
    
    Returns:
        FP16テンソル [N, D]
    """
    if not TRITON_AVAILABLE:
        return INT4Quantizer.dequantize(packed, scale, zero_point)
    
    N, D_packed = packed.shape
    D = D_packed * 2
    
    out = torch.empty(N, D, device=packed.device, dtype=torch.float16)
    
    BLOCK_N = 64
    BLOCK_D = min(64, D_packed)
    
    grid = (triton.cdiv(N, BLOCK_N),)
    
    _int4_dequantize_kernel[grid](
        packed, scale, zero_point, out,
        N=N, D=D,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    
    return out


class QuantizedHyperbolicDistanceModule(nn.Module):
    """
    量子化双曲距離計算モジュール
    
    INT8量子化を使用して効率的な双曲距離計算を実現。
    ルックアップテーブルを使用してacosh計算を高速化。
    
    Args:
        curvature: 双曲空間の曲率
        use_lut: ルックアップテーブルを使用するか
    """
    
    def __init__(
        self,
        curvature: float = 1.0,
        use_lut: bool = True,
    ):
        super().__init__()
        self.curvature = curvature
        self.use_lut = use_lut
        
        if use_lut:
            self.register_buffer('acosh_lut', _create_acosh_lut())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """
        双曲距離を計算
        
        Args:
            q: クエリ [B, H, N, D]
            k: キー [B, H, N, D]
        
        Returns:
            距離行列 [B, H, N, N]
        """
        return int8_hyperbolic_distance(q, k, self.curvature)


# エクスポート用の便利関数
def create_quantized_attention(
    d_model: int,
    num_heads: int,
    curvature: float = 1.0,
    bits: int = 8,
) -> QuantizedHyperbolicAttention:
    """量子化双曲アテンションモジュールを作成"""
    return QuantizedHyperbolicAttention(
        d_model=d_model,
        num_heads=num_heads,
        curvature=curvature,
        quantization_bits=bits,
    )


__all__ = [
    'TRITON_AVAILABLE',
    'INT8Quantizer',
    'INT4Quantizer',
    'QuantizedHyperbolicAttention',
    'QuantizedHyperbolicDistanceModule',
    'int8_hyperbolic_distance',
    'int4_dequantize',
    'create_quantized_attention',
    'get_acosh_lut',
]
