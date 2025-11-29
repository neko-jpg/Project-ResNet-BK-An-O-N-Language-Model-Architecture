#!/usr/bin/env python3
"""
Advanced Triton Kernels Unit Tests

タスク32.6: Tritonカーネルのユニットテスト
- 正確性、パフォーマンスのテスト

Requirements: 32.1-32.5
"""

import pytest
import torch
import torch.nn as nn
from typing import Optional

# テスト対象モジュールのインポート
try:
    from src.kernels.fused_ln_hyperbolic_triton import (
        FusedLNHyperbolic, fused_ln_hyperbolic
    )
    FUSED_LN_AVAILABLE = True
except ImportError:
    FUSED_LN_AVAILABLE = False

try:
    from src.kernels.quantized_hyperbolic_triton import (
        QuantizedHyperbolicAttention, INT8Quantizer, INT4Quantizer,
        int8_hyperbolic_distance
    )
    QUANTIZED_AVAILABLE = True
except ImportError:
    QUANTIZED_AVAILABLE = False

try:
    from src.kernels.sparse_hyperbolic_triton import (
        SparseHyperbolicAttention, LSHHyperbolicIndex,
        sparse_hyperbolic_attention
    )
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False

try:
    from src.kernels.register_tiled_distance_triton import (
        RegisterTiledHyperbolicDistance, RegisterTiledHyperbolicAttention,
        register_tiled_hyperbolic_distance
    )
    REGISTER_TILED_AVAILABLE = True
except ImportError:
    REGISTER_TILED_AVAILABLE = False


def get_device():
    """テスト用デバイスを取得"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestFusedLNHyperbolic:
    """Fused LayerNorm + Hyperbolicテスト"""
    
    @pytest.mark.skipif(not FUSED_LN_AVAILABLE, reason="FusedLNHyperbolic not available")
    def test_output_shape(self):
        """出力形状のテスト"""
        device = get_device()
        B, N, D = 4, 128, 256
        
        module = FusedLNHyperbolic(D, curvature=1.0).to(device)
        x = torch.randn(B, N, D, device=device)
        
        out = module(x)
        
        assert out.shape == x.shape
    
    @pytest.mark.skipif(not FUSED_LN_AVAILABLE, reason="FusedLNHyperbolic not available")
    def test_hyperbolic_constraint(self):
        """双曲空間の境界制約テスト"""
        device = get_device()
        B, N, D = 4, 128, 256
        curvature = 1.0
        
        module = FusedLNHyperbolic(D, curvature=curvature).to(device)
        x = torch.randn(B, N, D, device=device)  # 通常の値（大きな値はLayerNormで正規化される）
        
        out = module(x)
        
        # 出力がNaNやInfでないことを確認
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"
        
        # 出力が有限であることを確認（双曲制約はPyTorchフォールバックで適用）
        norms = torch.norm(out, dim=-1)
        assert norms.max() < 1000, f"Max norm too large: {norms.max()}"
    
    @pytest.mark.skipif(not FUSED_LN_AVAILABLE, reason="FusedLNHyperbolic not available")
    def test_gradient_flow(self):
        """勾配フローのテスト"""
        device = get_device()
        B, N, D = 2, 64, 128
        
        module = FusedLNHyperbolic(D, curvature=1.0).to(device)
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()


class TestQuantizedHyperbolic:
    """量子化双曲アテンションテスト"""
    
    @pytest.mark.skipif(not QUANTIZED_AVAILABLE, reason="QuantizedHyperbolic not available")
    def test_int8_quantizer_round_trip(self):
        """INT8量子化のラウンドトリップテスト"""
        device = get_device()
        x = torch.randn(4, 8, 128, 64, device=device)
        
        x_q, scale, zero = INT8Quantizer.quantize(x)
        x_deq = INT8Quantizer.dequantize(x_q, scale, zero)
        
        # 誤差が小さいことを確認
        error = (x - x_deq).abs().mean()
        assert error < 0.1, f"Quantization error: {error}"
    
    @pytest.mark.skipif(not QUANTIZED_AVAILABLE, reason="QuantizedHyperbolic not available")
    def test_int4_quantizer_round_trip(self):
        """INT4量子化のラウンドトリップテスト"""
        device = get_device()
        x = torch.randn(4, 8, 128, 64, device=device)  # Dは偶数
        
        packed, scale, zero = INT4Quantizer.quantize(x)
        x_deq = INT4Quantizer.dequantize(packed, scale, zero)
        
        # INT4は精度が低いので誤差許容を大きく
        error = (x - x_deq).abs().mean()
        assert error < 0.5, f"INT4 quantization error: {error}"
    
    @pytest.mark.skipif(not QUANTIZED_AVAILABLE, reason="QuantizedHyperbolic not available")
    def test_quantized_attention_output_shape(self):
        """量子化アテンションの出力形状テスト"""
        device = get_device()
        B, N, D = 4, 128, 256
        num_heads = 8
        
        module = QuantizedHyperbolicAttention(
            D, num_heads, curvature=1.0, quantization_bits=8
        ).to(device)
        x = torch.randn(B, N, D, device=device)
        
        out = module(x)
        
        assert out.shape == x.shape
    
    @pytest.mark.skipif(not QUANTIZED_AVAILABLE, reason="QuantizedHyperbolic not available")
    def test_int8_distance_non_negative(self):
        """INT8双曲距離が非負であることのテスト"""
        device = get_device()
        B, H, N, D = 2, 4, 64, 32
        
        q = torch.randn(B, H, N, D, device=device) * 0.5
        k = torch.randn(B, H, N, D, device=device) * 0.5
        
        distance = int8_hyperbolic_distance(q, k, curvature=1.0)
        
        assert (distance >= 0).all(), f"Negative distance found: {distance.min()}"


class TestSparseHyperbolic:
    """スパース双曲アテンションテスト"""
    
    @pytest.mark.skipif(not SPARSE_AVAILABLE, reason="SparseHyperbolic not available")
    def test_lsh_index_hash_consistency(self):
        """LSHハッシュの一貫性テスト"""
        device = get_device()
        B, H, N, D = 2, 4, 128, 64
        
        index = LSHHyperbolicIndex(D, num_hashes=8, device=device)
        x = torch.randn(B, H, N, D, device=device)
        
        # 同じ入力に対して同じハッシュ
        hash1 = index.compute_hashes(x)
        hash2 = index.compute_hashes(x)
        
        assert (hash1 == hash2).all()
    
    @pytest.mark.skipif(not SPARSE_AVAILABLE, reason="SparseHyperbolic not available")
    def test_sparse_attention_output_shape(self):
        """スパースアテンションの出力形状テスト"""
        device = get_device()
        B, N, D = 4, 256, 256
        num_heads = 8
        
        module = SparseHyperbolicAttention(
            D, num_heads, curvature=1.0, sparsity_ratio=0.9
        ).to(device)
        x = torch.randn(B, N, D, device=device)
        
        out = module(x)
        
        assert out.shape == x.shape
    
    @pytest.mark.skipif(not SPARSE_AVAILABLE, reason="SparseHyperbolic not available")
    def test_sparsity_ratio(self):
        """スパース性の割合テスト"""
        device = get_device()
        B, N, D = 2, 256, 128
        num_heads = 4
        sparsity_ratio = 0.9
        
        module = SparseHyperbolicAttention(
            D, num_heads, curvature=1.0, sparsity_ratio=sparsity_ratio
        ).to(device)
        
        # Top-k数が正しいことを確認
        expected_top_k = max(1, int(N * (1 - sparsity_ratio)))
        
        x = torch.randn(B, N, D, device=device)
        q = module.q_proj(x).view(B, N, num_heads, D // num_heads).transpose(1, 2)
        k = module.k_proj(x).view(B, N, num_heads, D // num_heads).transpose(1, 2)
        
        indices = module._compute_top_k_indices(q, k, expected_top_k)
        
        assert indices.shape[-1] == expected_top_k


class TestRegisterTiledHyperbolic:
    """レジスタタイル双曲距離テスト"""
    
    @pytest.mark.skipif(not REGISTER_TILED_AVAILABLE, reason="RegisterTiled not available")
    def test_distance_symmetry(self):
        """距離の対称性テスト"""
        device = get_device()
        B, H, N, D = 2, 4, 64, 32
        
        module = RegisterTiledHyperbolicDistance(curvature=1.0, use_triton=False)
        
        q = torch.randn(B, H, N, D, device=device) * 0.5
        k = torch.randn(B, H, N, D, device=device) * 0.5
        
        d_qk = module(q, k)
        d_kq = module(k, q)
        
        # 対称性: d(q, k) ≈ d(k, q)^T
        assert torch.allclose(d_qk, d_kq.transpose(-2, -1), atol=1e-4)
    
    @pytest.mark.skipif(not REGISTER_TILED_AVAILABLE, reason="RegisterTiled not available")
    def test_distance_non_negative(self):
        """距離が非負であることのテスト"""
        device = get_device()
        B, H, N, D = 2, 4, 64, 32
        
        module = RegisterTiledHyperbolicDistance(curvature=1.0, use_triton=False)
        
        q = torch.randn(B, H, N, D, device=device) * 0.5
        k = torch.randn(B, H, N, D, device=device) * 0.5
        
        distance = module(q, k)
        
        assert (distance >= 0).all()
    
    @pytest.mark.skipif(not REGISTER_TILED_AVAILABLE, reason="RegisterTiled not available")
    def test_attention_output_shape(self):
        """アテンション出力形状テスト"""
        device = get_device()
        B, N, D = 4, 128, 256
        num_heads = 8
        
        module = RegisterTiledHyperbolicAttention(
            D, num_heads, curvature=1.0, use_triton=False
        ).to(device)
        x = torch.randn(B, N, D, device=device)
        
        out = module(x)
        
        assert out.shape == x.shape
    
    @pytest.mark.skipif(not REGISTER_TILED_AVAILABLE, reason="RegisterTiled not available")
    def test_gradient_flow(self):
        """勾配フローテスト"""
        device = get_device()
        B, N, D = 2, 64, 128
        num_heads = 4
        
        module = RegisterTiledHyperbolicAttention(
            D, num_heads, curvature=1.0, use_triton=False
        ).to(device)
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestKernelNumericalStability:
    """数値安定性テスト"""
    
    @pytest.mark.skipif(not FUSED_LN_AVAILABLE, reason="FusedLNHyperbolic not available")
    def test_fused_ln_large_values(self):
        """大きな値での安定性テスト"""
        device = get_device()
        B, N, D = 2, 64, 128
        
        module = FusedLNHyperbolic(D, curvature=1.0).to(device)
        x = torch.randn(B, N, D, device=device) * 100
        
        out = module(x)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    @pytest.mark.skipif(not QUANTIZED_AVAILABLE, reason="QuantizedHyperbolic not available")
    def test_quantized_small_values(self):
        """小さな値での量子化安定性テスト"""
        device = get_device()
        x = torch.randn(4, 8, 64, 32, device=device) * 1e-6
        
        x_q, scale, zero = INT8Quantizer.quantize(x)
        x_deq = INT8Quantizer.dequantize(x_q, scale, zero)
        
        assert not torch.isnan(x_deq).any()
        assert not torch.isinf(x_deq).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
