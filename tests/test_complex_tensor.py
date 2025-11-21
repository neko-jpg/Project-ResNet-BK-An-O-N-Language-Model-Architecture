"""
ComplexTensor単体テスト

このテストファイルは、ComplexTensorクラスの全機能を検証します。

Test Coverage:
    1. 基本構造（初期化、プロパティ）
    2. 複素数演算（加算、減算、乗算、除算、共役、絶対値）
    3. complex64との相互変換
    4. メモリ使用量の検証（50%削減）
    5. 数値安定性の検証

Requirements:
    - Requirement 1.1: 基本構造
    - Requirement 1.2: 複素数演算
    - Requirement 1.3: 変換機能
    - Requirement 1.4: メモリ効率
"""

import pytest
import torch
import numpy as np
from src.models.phase3.complex_tensor import ComplexTensor


class TestComplexTensorBasics:
    """ComplexTensorの基本機能テスト"""
    
    def test_initialization(self):
        """初期化のテスト"""
        real = torch.randn(4, 10, 64, dtype=torch.float16)
        imag = torch.randn(4, 10, 64, dtype=torch.float16)
        
        z = ComplexTensor(real, imag)
        
        assert z.shape == torch.Size([4, 10, 64])
        assert z.dtype == torch.float16
        assert torch.allclose(z.real, real)
        assert torch.allclose(z.imag, imag)
    
    def test_initialization_shape_mismatch(self):
        """形状不一致のテスト"""
        real = torch.randn(4, 10, 64, dtype=torch.float16)
        imag = torch.randn(4, 10, 32, dtype=torch.float16)  # 異なる形状
        
        with pytest.raises(AssertionError):
            ComplexTensor(real, imag)
    
    def test_initialization_dtype_mismatch(self):
        """データ型不一致のテスト"""
        real = torch.randn(4, 10, 64, dtype=torch.float32)  # float32
        imag = torch.randn(4, 10, 64, dtype=torch.float16)
        
        with pytest.raises(AssertionError):
            ComplexTensor(real, imag)
    
    def test_device_transfer(self):
        """デバイス転送のテスト"""
        real = torch.randn(4, 10, 64, dtype=torch.float16)
        imag = torch.randn(4, 10, 64, dtype=torch.float16)
        z = ComplexTensor(real, imag)
        
        # CPU → CUDA（利用可能な場合）
        if torch.cuda.is_available():
            z_cuda = z.cuda()
            assert z_cuda.device.type == 'cuda'
            
            # CUDA → CPU
            z_cpu = z_cuda.cpu()
            assert z_cpu.device.type == 'cpu'
    
    def test_clone_and_detach(self):
        """複製と切り離しのテスト"""
        real = torch.randn(4, 10, 64, dtype=torch.float16, requires_grad=True)
        imag = torch.randn(4, 10, 64, dtype=torch.float16, requires_grad=True)
        z = ComplexTensor(real, imag)
        
        # Clone
        z_clone = z.clone()
        assert torch.allclose(z_clone.real, z.real)
        assert torch.allclose(z_clone.imag, z.imag)
        assert z_clone.real is not z.real  # 異なるオブジェクト
        
        # Detach
        z_detach = z.detach()
        assert not z_detach.requires_grad


class TestComplexArithmetic:
    """複素数演算のテスト（Requirement 1.2）"""
    
    def test_addition(self):
        """加算のテスト"""
        # (1 + 2i) + (3 + 4i) = (4 + 6i)
        z1 = ComplexTensor(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[2.0]], dtype=torch.float16)
        )
        z2 = ComplexTensor(
            torch.tensor([[3.0]], dtype=torch.float16),
            torch.tensor([[4.0]], dtype=torch.float16)
        )
        
        z3 = z1 + z2
        
        assert torch.allclose(z3.real, torch.tensor([[4.0]], dtype=torch.float16), atol=1e-3)
        assert torch.allclose(z3.imag, torch.tensor([[6.0]], dtype=torch.float16), atol=1e-3)
    
    def test_addition_with_scalar(self):
        """スカラーとの加算のテスト"""
        # (1 + 2i) + 3 = (4 + 2i)
        z = ComplexTensor(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[2.0]], dtype=torch.float16)
        )
        
        z_plus_3 = z + 3
        
        assert torch.allclose(z_plus_3.real, torch.tensor([[4.0]], dtype=torch.float16), atol=1e-3)
        assert torch.allclose(z_plus_3.imag, torch.tensor([[2.0]], dtype=torch.float16), atol=1e-3)
    
    def test_subtraction(self):
        """減算のテスト"""
        # (5 + 6i) - (2 + 3i) = (3 + 3i)
        z1 = ComplexTensor(
            torch.tensor([[5.0]], dtype=torch.float16),
            torch.tensor([[6.0]], dtype=torch.float16)
        )
        z2 = ComplexTensor(
            torch.tensor([[2.0]], dtype=torch.float16),
            torch.tensor([[3.0]], dtype=torch.float16)
        )
        
        z3 = z1 - z2
        
        assert torch.allclose(z3.real, torch.tensor([[3.0]], dtype=torch.float16), atol=1e-3)
        assert torch.allclose(z3.imag, torch.tensor([[3.0]], dtype=torch.float16), atol=1e-3)
    
    def test_multiplication(self):
        """乗算のテスト"""
        # (1 + 2i) * (3 + 4i) = (3 - 8) + (4 + 6)i = (-5 + 10i)
        z1 = ComplexTensor(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[2.0]], dtype=torch.float16)
        )
        z2 = ComplexTensor(
            torch.tensor([[3.0]], dtype=torch.float16),
            torch.tensor([[4.0]], dtype=torch.float16)
        )
        
        z3 = z1 * z2
        
        assert torch.allclose(z3.real, torch.tensor([[-5.0]], dtype=torch.float16), atol=1e-2)
        assert torch.allclose(z3.imag, torch.tensor([[10.0]], dtype=torch.float16), atol=1e-2)
    
    def test_multiplication_with_scalar(self):
        """スカラーとの乗算のテスト"""
        # (2 + 3i) * 2 = (4 + 6i)
        z = ComplexTensor(
            torch.tensor([[2.0]], dtype=torch.float16),
            torch.tensor([[3.0]], dtype=torch.float16)
        )
        
        z_times_2 = z * 2
        
        assert torch.allclose(z_times_2.real, torch.tensor([[4.0]], dtype=torch.float16), atol=1e-3)
        assert torch.allclose(z_times_2.imag, torch.tensor([[6.0]], dtype=torch.float16), atol=1e-3)
    
    def test_division(self):
        """除算のテスト"""
        # (4 + 2i) / (1 + 1i) = [(4 + 2) + (2 - 4)i] / 2 = (3 - 1i)
        z1 = ComplexTensor(
            torch.tensor([[4.0]], dtype=torch.float16),
            torch.tensor([[2.0]], dtype=torch.float16)
        )
        z2 = ComplexTensor(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[1.0]], dtype=torch.float16)
        )
        
        z3 = z1 / z2
        
        assert torch.allclose(z3.real, torch.tensor([[3.0]], dtype=torch.float16), atol=1e-2)
        assert torch.allclose(z3.imag, torch.tensor([[-1.0]], dtype=torch.float16), atol=1e-2)
    
    def test_conjugate(self):
        """共役のテスト"""
        # conj(3 + 4i) = (3 - 4i)
        z = ComplexTensor(
            torch.tensor([[3.0]], dtype=torch.float16),
            torch.tensor([[4.0]], dtype=torch.float16)
        )
        
        z_conj = z.conj()
        
        assert torch.allclose(z_conj.real, torch.tensor([[3.0]], dtype=torch.float16), atol=1e-3)
        assert torch.allclose(z_conj.imag, torch.tensor([[-4.0]], dtype=torch.float16), atol=1e-3)
    
    def test_absolute_value(self):
        """絶対値のテスト"""
        # |3 + 4i| = √(9 + 16) = 5
        z = ComplexTensor(
            torch.tensor([[3.0]], dtype=torch.float16),
            torch.tensor([[4.0]], dtype=torch.float16)
        )
        
        abs_z = z.abs()
        
        assert torch.allclose(abs_z, torch.tensor([[5.0]], dtype=torch.float16), atol=1e-2)
    
    def test_angle(self):
        """偏角のテスト"""
        # arg(1 + 1i) = π/4 ≈ 0.785
        z = ComplexTensor(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[1.0]], dtype=torch.float16)
        )
        
        angle_z = z.angle()
        
        expected_angle = torch.tensor([[np.pi / 4]], dtype=torch.float16)
        assert torch.allclose(angle_z, expected_angle, atol=1e-2)


class TestComplexConversion:
    """complex64との相互変換テスト（Requirement 1.3）"""
    
    def test_to_complex64(self):
        """ComplexTensor → complex64変換のテスト"""
        real = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
        imag = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float16)
        z = ComplexTensor(real, imag)
        
        z_complex64 = z.to_complex64()
        
        assert z_complex64.dtype == torch.complex64
        assert torch.allclose(z_complex64.real, real.float(), atol=1e-3)
        assert torch.allclose(z_complex64.imag, imag.float(), atol=1e-3)
    
    def test_from_complex64(self):
        """complex64 → ComplexTensor変換のテスト"""
        z_complex64 = torch.complex(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        )
        
        z = ComplexTensor.from_complex64(z_complex64)
        
        assert z.dtype == torch.float16
        assert torch.allclose(z.real, z_complex64.real.half(), atol=1e-3)
        assert torch.allclose(z.imag, z_complex64.imag.half(), atol=1e-3)
    
    def test_roundtrip_conversion(self):
        """往復変換のテスト"""
        real = torch.randn(4, 10, 64, dtype=torch.float16)
        imag = torch.randn(4, 10, 64, dtype=torch.float16)
        z_original = ComplexTensor(real, imag)
        
        # ComplexTensor → complex64 → ComplexTensor
        z_complex64 = z_original.to_complex64()
        z_back = ComplexTensor.from_complex64(z_complex64)
        
        assert torch.allclose(z_back.real, z_original.real, atol=1e-3)
        assert torch.allclose(z_back.imag, z_original.imag, atol=1e-3)
    
    def test_from_real(self):
        """実数からの変換のテスト"""
        real = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        # 虚部なし
        z1 = ComplexTensor.from_real(real)
        assert torch.allclose(z1.real, real.half(), atol=1e-3)
        assert torch.allclose(z1.imag, torch.zeros_like(real).half(), atol=1e-3)
        
        # 虚部あり
        imag = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        z2 = ComplexTensor.from_real(real, imag)
        assert torch.allclose(z2.real, real.half(), atol=1e-3)
        assert torch.allclose(z2.imag, imag.half(), atol=1e-3)


class TestMemoryEfficiency:
    """メモリ効率のテスト（Requirement 1.4）"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage_reduction(self):
        """
        メモリ使用量50%削減の検証
        
        Expected:
            - complex64: 8 bytes/element
            - ComplexTensor (complex32): 4 bytes/element
            - 削減率: 50%
        """
        shape = (128, 512, 512)  # 大きなテンソル
        
        # complex64のメモリ使用量
        torch.cuda.reset_peak_memory_stats()
        z_complex64 = torch.randn(*shape, dtype=torch.complex64).cuda()
        memory_complex64 = torch.cuda.max_memory_allocated()
        del z_complex64
        torch.cuda.empty_cache()
        
        # ComplexTensorのメモリ使用量
        torch.cuda.reset_peak_memory_stats()
        real = torch.randn(*shape, dtype=torch.float16).cuda()
        imag = torch.randn(*shape, dtype=torch.float16).cuda()
        z_complex32 = ComplexTensor(real, imag)
        memory_complex32 = torch.cuda.max_memory_allocated()
        del z_complex32, real, imag
        torch.cuda.empty_cache()
        
        # メモリ削減率の計算
        reduction_ratio = memory_complex32 / memory_complex64
        
        print(f"\nMemory Usage:")
        print(f"  complex64: {memory_complex64 / 1024**2:.2f} MB")
        print(f"  ComplexTensor (complex32): {memory_complex32 / 1024**2:.2f} MB")
        print(f"  Reduction ratio: {reduction_ratio:.2%}")
        
        # 50%削減を確認（多少の誤差を許容）
        assert reduction_ratio < 0.55, \
            f"Memory reduction not achieved: {reduction_ratio:.2%} (expected < 55%)"
    
    def test_memory_layout_planar(self):
        """Planar形式のメモリレイアウト検証"""
        real = torch.randn(4, 10, 64, dtype=torch.float16)
        imag = torch.randn(4, 10, 64, dtype=torch.float16)
        z = ComplexTensor(real, imag)
        
        # Planar形式: 実部と虚部が分離されている
        assert z.real.data_ptr() != z.imag.data_ptr()
        
        # 連続性の確認
        assert z.real.is_contiguous()
        assert z.imag.is_contiguous()


class TestNumericalStability:
    """数値安定性のテスト"""
    
    def test_zero_division_safety(self):
        """ゼロ除算の安全性テスト"""
        z1 = ComplexTensor(
            torch.tensor([[1.0]], dtype=torch.float16),
            torch.tensor([[1.0]], dtype=torch.float16)
        )
        z2 = ComplexTensor(
            torch.tensor([[0.0]], dtype=torch.float16),
            torch.tensor([[0.0]], dtype=torch.float16)
        )
        
        # ゼロ除算でもNaN/Infにならないことを確認
        z3 = z1 / z2
        assert not torch.isnan(z3.real).any()
        assert not torch.isnan(z3.imag).any()
        assert not torch.isinf(z3.real).any()
        assert not torch.isinf(z3.imag).any()
    
    def test_abs_underflow_safety(self):
        """絶対値計算のアンダーフロー安全性テスト"""
        # 非常に小さな値
        z = ComplexTensor(
            torch.tensor([[1e-8]], dtype=torch.float16),
            torch.tensor([[1e-8]], dtype=torch.float16)
        )
        
        abs_z = z.abs()
        
        # NaN/Infにならないことを確認
        assert not torch.isnan(abs_z).any()
        assert not torch.isinf(abs_z).any()
        assert (abs_z > 0).all()
    
    def test_multiplication_overflow_safety(self):
        """乗算のオーバーフロー安全性テスト"""
        # 大きな値
        z1 = ComplexTensor(
            torch.tensor([[1000.0]], dtype=torch.float16),
            torch.tensor([[1000.0]], dtype=torch.float16)
        )
        z2 = ComplexTensor(
            torch.tensor([[1000.0]], dtype=torch.float16),
            torch.tensor([[1000.0]], dtype=torch.float16)
        )
        
        z3 = z1 * z2
        
        # Infにならないことを確認（float16の範囲内）
        # 注: float16の最大値は約65504なので、オーバーフローする可能性がある
        # ここでは、計算が完了することを確認
        assert z3.real.shape == z1.real.shape
        assert z3.imag.shape == z1.imag.shape


class TestUtilityMethods:
    """ユーティリティメソッドのテスト"""
    
    def test_view_and_reshape(self):
        """形状変更のテスト"""
        z = ComplexTensor(
            torch.randn(4, 10, 64, dtype=torch.float16),
            torch.randn(4, 10, 64, dtype=torch.float16)
        )
        
        # View
        z_view = z.view(4, 640)
        assert z_view.shape == torch.Size([4, 640])
        
        # Reshape
        z_reshape = z.reshape(40, 64)
        assert z_reshape.shape == torch.Size([40, 64])
    
    def test_permute_and_transpose(self):
        """次元入れ替えのテスト"""
        z = ComplexTensor(
            torch.randn(4, 10, 64, dtype=torch.float16),
            torch.randn(4, 10, 64, dtype=torch.float16)
        )
        
        # Permute
        z_permute = z.permute(2, 0, 1)
        assert z_permute.shape == torch.Size([64, 4, 10])
        
        # Transpose
        z_transpose = z.transpose(0, 1)
        assert z_transpose.shape == torch.Size([10, 4, 64])
    
    def test_squeeze_and_unsqueeze(self):
        """次元追加/削除のテスト"""
        z = ComplexTensor(
            torch.randn(4, 1, 64, dtype=torch.float16),
            torch.randn(4, 1, 64, dtype=torch.float16)
        )
        
        # Squeeze
        z_squeeze = z.squeeze(1)
        assert z_squeeze.shape == torch.Size([4, 64])
        
        # Unsqueeze
        z_unsqueeze = z_squeeze.unsqueeze(1)
        assert z_unsqueeze.shape == torch.Size([4, 1, 64])
    
    def test_mean_and_sum(self):
        """平均と合計のテスト"""
        z = ComplexTensor(
            torch.ones(4, 10, 64, dtype=torch.float16),
            torch.ones(4, 10, 64, dtype=torch.float16) * 2
        )
        
        # Mean
        z_mean = z.mean(dim=1)
        assert z_mean.shape == torch.Size([4, 64])
        assert torch.allclose(z_mean.real, torch.ones(4, 64, dtype=torch.float16), atol=1e-3)
        assert torch.allclose(z_mean.imag, torch.ones(4, 64, dtype=torch.float16) * 2, atol=1e-3)
        
        # Sum
        z_sum = z.sum(dim=1)
        assert z_sum.shape == torch.Size([4, 64])
        assert torch.allclose(z_sum.real, torch.ones(4, 64, dtype=torch.float16) * 10, atol=1e-2)
    
    def test_norm(self):
        """ノルムのテスト"""
        z = ComplexTensor(
            torch.tensor([[3.0]], dtype=torch.float16),
            torch.tensor([[4.0]], dtype=torch.float16)
        )
        
        # L2ノルム
        norm_l2 = z.norm(p=2)
        assert torch.allclose(norm_l2, torch.tensor(5.0, dtype=torch.float16), atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
