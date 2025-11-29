#!/usr/bin/env python3
"""
Fused LayerNorm + Hyperbolic Projection Triton Kernel

タスク32.1: LayerNormと双曲射影の融合カーネル
- Warp-level reduction for mean/variance
- 目標: 80%+ memory bandwidth utilization

Requirements: 45.1-45.6
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

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
    def _fused_ln_hyperbolic_fwd_kernel(
        X,  # 入力テンソル [B, N, D]
        Y,  # 出力テンソル [B, N, D]
        W,  # LayerNorm weight [D]
        B,  # LayerNorm bias [D]
        Mean,  # 平均 [B, N]
        Rstd,  # 逆標準偏差 [B, N]
        curvature,  # 曲率
        stride_x_b, stride_x_n, stride_x_d,
        stride_y_b, stride_y_n, stride_y_d,
        N_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        eps: tl.constexpr,
    ):
        """
        融合LayerNorm + 双曲射影のフォワードカーネル
        
        物理的直観: LayerNormで正規化した後、双曲空間に射影する
        これにより、メモリアクセスを1回に削減
        """
        # プログラムID
        row_idx = tl.program_id(0)
        batch_idx = row_idx // N_COLS
        seq_idx = row_idx % N_COLS
        
        # 入力ポインタ計算
        x_ptr = X + batch_idx * stride_x_b + seq_idx * stride_x_n
        y_ptr = Y + batch_idx * stride_y_b + seq_idx * stride_y_n
        
        # ブロックオフセット
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        
        # 入力をロード
        x = tl.load(x_ptr + cols * stride_x_d, mask=mask, other=0.0)
        
        # Step 1: 平均計算 (Warp-level reduction)
        mean = tl.sum(x, axis=0) / N_COLS
        
        # Step 2: 分散計算
        x_centered = x - mean
        var = tl.sum(x_centered * x_centered, axis=0) / N_COLS
        
        # Step 3: 正規化
        rstd = 1.0 / tl.sqrt(var + eps)
        x_norm = x_centered * rstd
        
        # Step 4: アフィン変換
        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        x_ln = x_norm * w + b
        
        # Step 5: 双曲射影 (Poincaré ball)
        # ||x|| を計算
        x_norm_sq = tl.sum(x_ln * x_ln, axis=0)
        x_norm_val = tl.sqrt(x_norm_sq + 1e-8)
        
        # 双曲空間の境界制約: ||x|| < 1/sqrt(c)
        max_norm = 1.0 / tl.sqrt(curvature) - 0.01
        
        # クリッピング
        scale = tl.where(
            x_norm_val > max_norm,
            max_norm / x_norm_val,
            1.0
        )
        y = x_ln * scale
        
        # 出力を保存
        tl.store(y_ptr + cols * stride_y_d, y, mask=mask)
        
        # 統計を保存（逆伝播用）
        if seq_idx == 0:
            tl.store(Mean + row_idx, mean)
            tl.store(Rstd + row_idx, rstd)


    @triton.jit
    def _fused_ln_hyperbolic_bwd_kernel(
        DY,  # 出力勾配 [B, N, D]
        X,   # 入力 [B, N, D]
        W,   # LayerNorm weight [D]
        Mean,  # 平均 [B, N]
        Rstd,  # 逆標準偏差 [B, N]
        DX,  # 入力勾配 [B, N, D]
        DW,  # Weight勾配 [D]
        DB,  # Bias勾配 [D]
        curvature,
        stride_dy_b, stride_dy_n, stride_dy_d,
        stride_x_b, stride_x_n, stride_x_d,
        stride_dx_b, stride_dx_n, stride_dx_d,
        N_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """融合LayerNorm + 双曲射影の逆伝播カーネル"""
        row_idx = tl.program_id(0)
        batch_idx = row_idx // N_COLS
        seq_idx = row_idx % N_COLS
        
        # ポインタ計算
        dy_ptr = DY + batch_idx * stride_dy_b + seq_idx * stride_dy_n
        x_ptr = X + batch_idx * stride_x_b + seq_idx * stride_x_n
        dx_ptr = DX + batch_idx * stride_dx_b + seq_idx * stride_dx_n
        
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        
        # データロード
        dy = tl.load(dy_ptr + cols * stride_dy_d, mask=mask, other=0.0)
        x = tl.load(x_ptr + cols * stride_x_d, mask=mask, other=0.0)
        w = tl.load(W + cols, mask=mask, other=1.0)
        mean = tl.load(Mean + row_idx)
        rstd = tl.load(Rstd + row_idx)
        
        # 双曲射影の逆伝播
        # 簡略化: スケーリングの勾配を近似
        dy_scaled = dy
        
        # LayerNormの逆伝播
        x_centered = x - mean
        x_norm = x_centered * rstd
        
        # dw, db の計算
        dw = dy_scaled * x_norm
        db = dy_scaled
        
        # dx の計算
        dx_norm = dy_scaled * w
        dx = rstd * (dx_norm - tl.sum(dx_norm, axis=0) / N_COLS - 
                     x_norm * tl.sum(dx_norm * x_norm, axis=0) / N_COLS)
        
        # 保存
        tl.store(dx_ptr + cols * stride_dx_d, dx, mask=mask)
        
        # アトミック加算でdw, dbを累積
        tl.atomic_add(DW + cols, dw, mask=mask)
        tl.atomic_add(DB + cols, db, mask=mask)


class FusedLNHyperbolicFunction(torch.autograd.Function):
    """融合LayerNorm + 双曲射影のAutograd関数"""
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        curvature: float = 1.0,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """フォワードパス"""
        if not TRITON_AVAILABLE:
            # フォールバック: PyTorch実装
            return FusedLNHyperbolicFunction._pytorch_forward(
                x, weight, bias, curvature, eps
            )
        
        B, N, D = x.shape
        
        # 出力テンソル
        y = torch.empty_like(x)
        mean = torch.empty(B * N, device=x.device, dtype=x.dtype)
        rstd = torch.empty(B * N, device=x.device, dtype=x.dtype)
        
        # ブロックサイズ
        BLOCK_SIZE = triton.next_power_of_2(D)
        
        # カーネル起動
        grid = (B * N,)
        _fused_ln_hyperbolic_fwd_kernel[grid](
            x, y, weight, bias, mean, rstd,
            curvature,
            x.stride(0), x.stride(1), x.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            N_COLS=D,
            BLOCK_SIZE=BLOCK_SIZE,
            eps=eps,
        )
        
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.curvature = curvature
        ctx.BLOCK_SIZE = BLOCK_SIZE
        
        return y
    
    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        """逆伝播"""
        x, weight, mean, rstd = ctx.saved_tensors
        curvature = ctx.curvature
        
        if not TRITON_AVAILABLE:
            return FusedLNHyperbolicFunction._pytorch_backward(
                dy, x, weight, mean, rstd, curvature
            )
        
        B, N, D = x.shape
        
        dx = torch.empty_like(x)
        dw = torch.zeros_like(weight)
        db = torch.zeros(D, device=x.device, dtype=x.dtype)
        
        BLOCK_SIZE = ctx.BLOCK_SIZE
        grid = (B * N,)
        
        _fused_ln_hyperbolic_bwd_kernel[grid](
            dy, x, weight, mean, rstd, dx, dw, db,
            curvature,
            dy.stride(0), dy.stride(1), dy.stride(2),
            x.stride(0), x.stride(1), x.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            N_COLS=D,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return dx, dw, db, None, None
    
    @staticmethod
    def _pytorch_forward(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        curvature: float,
        eps: float,
    ) -> torch.Tensor:
        """PyTorchフォールバック実装"""
        # LayerNorm
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps)
        x_ln = x_norm * weight + bias
        
        # 双曲射影
        norm = torch.norm(x_ln, dim=-1, keepdim=True)
        max_norm = 1.0 / (curvature ** 0.5) - 0.01
        scale = torch.where(
            norm > max_norm,
            max_norm / norm,
            torch.ones_like(norm)
        )
        return x_ln * scale
    
    @staticmethod
    def _pytorch_backward(
        dy: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        curvature: float,
    ):
        """PyTorchフォールバック逆伝播"""
        # 簡略化した勾配計算
        B, N, D = x.shape
        mean = mean.view(B, N, 1)
        rstd = rstd.view(B, N, 1)
        
        x_centered = x - mean
        x_norm = x_centered * rstd
        
        dw = (dy * x_norm).sum(dim=(0, 1))
        db = dy.sum(dim=(0, 1))
        
        dx_norm = dy * weight
        dx = rstd * (dx_norm - dx_norm.mean(dim=-1, keepdim=True) -
                     x_norm * (dx_norm * x_norm).mean(dim=-1, keepdim=True))
        
        return dx, dw, db, None, None


class FusedLNHyperbolic(nn.Module):
    """
    融合LayerNorm + 双曲射影モジュール
    
    LayerNormと双曲空間への射影を1つのカーネルで実行し、
    メモリ帯域幅の利用効率を最大化する。
    
    Args:
        d_model: モデル次元
        curvature: 双曲空間の曲率
        eps: LayerNormのイプシロン
    """
    
    def __init__(
        self,
        d_model: int,
        curvature: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.curvature = curvature
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [B, N, D]
        
        Returns:
            双曲空間に射影された正規化テンソル [B, N, D]
        """
        return FusedLNHyperbolicFunction.apply(
            x, self.weight, self.bias, self.curvature, self.eps
        )
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, curvature={self.curvature}, eps={self.eps}"


def fused_ln_hyperbolic(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    curvature: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    融合LayerNorm + 双曲射影の関数インターフェース
    
    Args:
        x: 入力テンソル [B, N, D]
        weight: LayerNorm weight [D]
        bias: LayerNorm bias [D]
        curvature: 双曲空間の曲率
        eps: LayerNormのイプシロン
    
    Returns:
        双曲空間に射影された正規化テンソル [B, N, D]
    """
    return FusedLNHyperbolicFunction.apply(x, weight, bias, curvature, eps)
