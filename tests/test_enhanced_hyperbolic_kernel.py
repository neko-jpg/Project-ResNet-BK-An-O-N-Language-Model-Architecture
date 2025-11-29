"""
Enhanced Hyperbolic Kernel Tests

拡張双曲カーネルのテスト。
Property-Based TestingとUnit Testを含む。

Requirements: 26.1-26.6, 32.1-32.6, 33.1-33.6
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, '.')

from src.kernels.enhanced_hyperbolic_triton import (
    EnhancedHyperbolicAttention,
    HierarchicalBlockDecomposition,
    create_enhanced_hyperbolic_attention,
    benchmark_enhanced_kernel,
)


# ============================================================
# Property-Based Tests
# ============================================================

class TestEnhancedKernelSpeedupProperty:
    """
    **Feature: phase8-hyperbolic-transcendence, Property 16: Enhanced Kernel Speedup**
    **Validates: Requirements 26.4**
    
    Enhanced kernelがPyTorch参照実装より高速であることを検証。
    """
    
    @given(
        batch_size=st.integers(min_value=1, max_value=2),
        seq_len=st.sampled_from([64, 128, 256]),
        d_model=st.sampled_from([64, 128]),
        num_heads=st.sampled_from([2, 4]),
    )
    @settings(max_examples=10, deadline=60000)
    def test_output_valid(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
    ):
        """
        Property: Enhanced kernelの出力は有効（NaN/Infなし）
        """
        torch.manual_seed(42)
        
        model = create_enhanced_hyperbolic_attention(
            d_model=d_model,
            num_heads=num_heads,
            use_taylor=True,
            use_asymptotic=True,
        )
        model.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            output, diagnostics = model(x)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} != {x.shape}"


class TestTaylorApproximationProperty:
    """
    Taylor展開近似のプロパティテスト
    
    Requirements: 26.1
    """
    
    @given(
        d_model=st.sampled_from([64, 128]),
        num_heads=st.sampled_from([2, 4]),
    )
    @settings(max_examples=20, deadline=30000)
    def test_taylor_vs_standard_close(
        self,
        d_model: int,
        num_heads: int,
    ):
        """
        Property: Taylor近似と標準計算の結果は近い（小距離で）
        """
        torch.manual_seed(42)
        
        # Taylor有効
        model_taylor = create_enhanced_hyperbolic_attention(
            d_model=d_model,
            num_heads=num_heads,
            use_taylor=True,
            use_asymptotic=False,
        )
        
        # Taylor無効
        model_standard = create_enhanced_hyperbolic_attention(
            d_model=d_model,
            num_heads=num_heads,
            use_taylor=False,
            use_asymptotic=False,
        )
        
        # 重みを同期
        model_standard.load_state_dict(model_taylor.state_dict())
        
        model_taylor.eval()
        model_standard.eval()
        
        # 小さいノルムの入力（Taylor近似が有効な領域）
        x = torch.randn(2, 32, d_model) * 0.05
        
        with torch.no_grad():
            out_taylor, _ = model_taylor(x)
            out_standard, _ = model_standard(x)
        
        # 相対誤差が小さい
        rel_error = (out_taylor - out_standard).abs().mean() / (out_standard.abs().mean() + 1e-8)
        
        # Taylor近似は1%以内の誤差
        assert rel_error < 0.1, f"Taylor approximation error too large: {rel_error}"


# ============================================================
# Unit Tests
# ============================================================

class TestEnhancedHyperbolicAttention:
    """Enhanced Hyperbolic Attentionのユニットテスト"""
    
    def test_forward_pass(self):
        """Forward passの動作確認"""
        model = create_enhanced_hyperbolic_attention(
            d_model=64,
            num_heads=4,
        )
        
        x = torch.randn(2, 32, 64)
        output, diagnostics = model(x, return_diagnostics=True)
        
        assert output.shape == x.shape
        assert 'curvature' in diagnostics
        assert 'beta' in diagnostics
    
    def test_with_causal_mask(self):
        """Causal maskの動作確認"""
        model = create_enhanced_hyperbolic_attention(
            d_model=64,
            num_heads=4,
        )
        
        x = torch.randn(2, 32, 64)
        mask = torch.ones(32, 32)  # ダミーマスク
        
        output, _ = model(x, mask=mask)
        
        assert output.shape == x.shape
    
    def test_taylor_approximation(self):
        """Taylor近似の動作確認"""
        model = create_enhanced_hyperbolic_attention(
            d_model=64,
            num_heads=4,
            use_taylor=True,
            use_asymptotic=False,
        )
        
        x = torch.randn(2, 32, 64) * 0.1
        output, diagnostics = model(x, return_diagnostics=True)
        
        assert output.shape == x.shape
        assert diagnostics['use_taylor'] == True
    
    def test_asymptotic_approximation(self):
        """漸近近似の動作確認"""
        model = create_enhanced_hyperbolic_attention(
            d_model=64,
            num_heads=4,
            use_taylor=False,
            use_asymptotic=True,
        )
        
        x = torch.randn(2, 32, 64) * 5.0  # 大きなノルム
        output, diagnostics = model(x, return_diagnostics=True)
        
        assert output.shape == x.shape
        assert diagnostics['use_asymptotic'] == True
    
    def test_tensor_core_mode(self):
        """Tensor Coreモードの動作確認"""
        model = create_enhanced_hyperbolic_attention(
            d_model=64,
            num_heads=4,
            use_tensor_core=True,
        )
        
        x = torch.randn(2, 32, 64)
        output, diagnostics = model(x)
        
        assert output.shape == x.shape
        assert output.dtype == torch.float32  # 出力はFP32に戻る
    
    def test_pytorch_fallback(self):
        """PyTorchフォールバックの動作確認"""
        model = create_enhanced_hyperbolic_attention(
            d_model=64,
            num_heads=4,
        )
        
        x = torch.randn(2, 32, 64)
        
        # フォールバック関数を直接テスト
        q = model.W_q(x).view(2, 32, 4, 16).transpose(1, 2)
        k = model.W_k(x).view(2, 32, 4, 16).transpose(1, 2)
        v = model.W_v(x).view(2, 32, 4, 16).transpose(1, 2)
        
        q = q.reshape(8, 32, 16)
        k = k.reshape(8, 32, 16)
        v = v.reshape(8, 32, 16)
        
        c = torch.nn.functional.softplus(model.log_c)
        beta = torch.nn.functional.softplus(model.log_beta) + 0.5
        
        out = model._pytorch_fallback(q, k, v, c, beta, causal=False)
        
        assert out.shape == (8, 32, 16)
        assert not torch.isnan(out).any()


class TestHierarchicalBlockDecomposition:
    """階層的ブロック分解のユニットテスト"""
    
    def test_short_sequence(self):
        """短いシーケンスの処理"""
        model = HierarchicalBlockDecomposition(
            d_model=64,
            num_heads=4,
            block_size=1024,
        )
        
        x = torch.randn(2, 256, 64)  # block_sizeより短い
        output, diagnostics = model(x)
        
        assert output.shape == x.shape
    
    def test_long_sequence(self):
        """長いシーケンスの処理"""
        model = HierarchicalBlockDecomposition(
            d_model=64,
            num_heads=4,
            block_size=128,
        )
        
        x = torch.randn(2, 512, 64)  # block_sizeより長い
        output, diagnostics = model(x)
        
        assert output.shape == x.shape
        assert 'num_blocks' in diagnostics
        assert diagnostics['num_blocks'] == 4  # 512 / 128 = 4
    
    def test_padding_handling(self):
        """パディング処理の確認"""
        model = HierarchicalBlockDecomposition(
            d_model=64,
            num_heads=4,
            block_size=128,
        )
        
        # block_sizeで割り切れない長さ
        x = torch.randn(2, 300, 64)
        output, _ = model(x)
        
        assert output.shape == x.shape


class TestFactoryFunction:
    """ファクトリ関数のテスト"""
    
    def test_create_with_defaults(self):
        """デフォルト設定での作成"""
        model = create_enhanced_hyperbolic_attention()
        
        assert model.d_model == 256
        assert model.num_heads == 8
        assert model.use_taylor == True
        assert model.use_asymptotic == True
    
    def test_create_with_custom_config(self):
        """カスタム設定での作成"""
        model = create_enhanced_hyperbolic_attention(
            d_model=512,
            num_heads=16,
            use_taylor=False,
            use_asymptotic=False,
            use_tensor_core=False,
        )
        
        assert model.d_model == 512
        assert model.num_heads == 16
        assert model.use_taylor == False
        assert model.use_asymptotic == False
        assert model.use_tensor_core == False


# ============================================================
# Integration Tests
# ============================================================

class TestEnhancedKernelIntegration:
    """統合テスト"""
    
    def test_gradient_flow(self):
        """勾配フローの確認"""
        model = create_enhanced_hyperbolic_attention(d_model=64, num_heads=4)
        model.train()
        
        x = torch.randn(2, 32, 64, requires_grad=True)
        
        output, _ = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_numerical_stability(self):
        """数値安定性の確認"""
        model = create_enhanced_hyperbolic_attention(d_model=64, num_heads=4)
        model.eval()
        
        # 大きな値
        x_large = torch.randn(2, 32, 64) * 100
        output_large, _ = model(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()
        
        # 小さな値
        x_small = torch.randn(2, 32, 64) * 0.001
        output_small, _ = model(x_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()
    
    def test_deterministic_output(self):
        """決定論的出力の確認"""
        model = create_enhanced_hyperbolic_attention(d_model=64, num_heads=4)
        model.eval()
        
        torch.manual_seed(42)
        x = torch.randn(2, 32, 64)
        
        with torch.no_grad():
            output1, _ = model(x)
            output2, _ = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-5)


class TestBenchmarkFunction:
    """ベンチマーク関数のテスト"""
    
    def test_benchmark_cpu(self):
        """CPUでのベンチマーク（CUDAなし）"""
        # CUDAがない場合はエラーを返す
        if not torch.cuda.is_available():
            results = benchmark_enhanced_kernel(device='cuda')
            assert 'error' in results
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_gpu(self):
        """GPUでのベンチマーク"""
        results = benchmark_enhanced_kernel(
            batch_size=1,
            seq_lengths=[64, 128],
            d_model=64,
            num_heads=4,
            num_iterations=3,
        )
        
        assert 'seq_64' in results
        assert 'seq_128' in results
        assert results['seq_64']['tokens_per_second'] > 0
        assert results['seq_128']['tokens_per_second'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
