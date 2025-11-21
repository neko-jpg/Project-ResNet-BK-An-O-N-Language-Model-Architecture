"""
Unit Tests for Complex Operations (Phase 3)

このテストファイルは、ComplexLinear、ModReLU、ComplexLayerNormの動作を検証します。

Test Coverage:
    - ComplexLinear: 出力形状、複素行列積の正確性、勾配計算
    - ModReLU: 位相保存、振幅フィルタリング
    - ComplexLayerNorm: 正規化の正確性、勾配計算

Requirements:
    - Requirement 1.8: ComplexLinear単体テスト
    - Requirement 1.10: ModReLU単体テスト
    - Requirement 1.12: ComplexLayerNorm単体テスト
"""

import pytest
import torch
import torch.nn as nn
import math

from src.models.phase3.complex_tensor import ComplexTensor
from src.models.phase3.complex_ops import ComplexLinear, ModReLU, ComplexLayerNorm


class TestComplexLinear:
    """ComplexLinear層のテスト"""
    
    def test_output_shape_complex_tensor(self):
        """
        出力形状の正確性を検証（ComplexTensor入力）
        
        Requirements:
            - Requirement 1.8: 出力形状の正確性
        """
        batch_size = 4
        seq_len = 10
        in_features = 64
        out_features = 128
        
        # ComplexLinear層の作成
        layer = ComplexLinear(in_features, out_features, use_complex32=True)
        
        # ComplexTensor入力
        real = torch.randn(batch_size, seq_len, in_features, dtype=torch.float16)
        imag = torch.randn(batch_size, seq_len, in_features, dtype=torch.float16)
        x = ComplexTensor(real, imag)
        
        # Forward pass
        y = layer(x)
        
        # 出力形状の検証
        assert isinstance(y, ComplexTensor), "Output should be ComplexTensor"
        assert y.shape == (batch_size, seq_len, out_features), \
            f"Expected shape {(batch_size, seq_len, out_features)}, got {y.shape}"
        assert y.dtype == torch.float16, "Output dtype should be float16"
    
    def test_output_shape_complex64(self):
        """
        出力形状の正確性を検証（complex64入力）
        
        Requirements:
            - Requirement 1.8: 出力形状の正確性
        """
        batch_size = 4
        seq_len = 10
        in_features = 64
        out_features = 128
        
        # ComplexLinear層の作成
        layer = ComplexLinear(in_features, out_features, use_complex32=False)
        
        # complex64入力
        x = torch.randn(batch_size, seq_len, in_features, dtype=torch.complex64)
        
        # Forward pass
        y = layer(x)
        
        # 出力形状の検証
        assert y.is_complex(), "Output should be complex type"
        assert y.shape == (batch_size, seq_len, out_features), \
            f"Expected shape {(batch_size, seq_len, out_features)}, got {y.shape}"
        assert y.dtype == torch.complex64, "Output dtype should be complex64"
    
    def test_complex_matrix_multiplication(self):
        """
        複素行列積の正確性を検証
        
        Formula:
            (A + iB)(x + iy) = (Ax - By) + i(Bx + Ay)
        
        Requirements:
            - Requirement 1.8: 複素行列積の正確性
        """
        in_features = 4
        out_features = 3
        
        # ComplexLinear層の作成
        layer = ComplexLinear(in_features, out_features, bias=False, use_complex32=False)
        
        # 重みを手動で設定（検証のため）
        layer.weight_real.data = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=torch.float32)
        
        layer.weight_imag.data = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        # 入力: x = [1, 2, 3, 4] + i[0, 0, 0, 0]
        x_real = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        x_imag = torch.zeros_like(x_real)
        x = torch.complex(x_real, x_imag)
        
        # Forward pass
        y = layer(x)
        
        # 期待される出力:
        # y[0] = 1*1 + 0*2 + 0*3 + 0*4 = 1 + 0i
        # y[1] = 0*1 + 1*2 + 0*3 + 0*4 = 2 + 0i
        # y[2] = 0*1 + 0*2 + 1*3 + 0*4 + i(0*1 + 0*2 + 0*3 + 1*4) = 3 + 4i
        expected_real = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        expected_imag = torch.tensor([[0.0, 0.0, 4.0]], dtype=torch.float32)
        
        assert torch.allclose(y.real, expected_real, atol=1e-5), \
            f"Real part mismatch: expected {expected_real}, got {y.real}"
        assert torch.allclose(y.imag, expected_imag, atol=1e-5), \
            f"Imag part mismatch: expected {expected_imag}, got {y.imag}"
    
    def test_gradient_computation(self):
        """
        勾配計算が正常に動作することを確認
        
        Requirements:
            - Requirement 1.8: 勾配計算の正常動作
        """
        batch_size = 2
        in_features = 8
        out_features = 4
        
        # ComplexLinear層の作成
        layer = ComplexLinear(in_features, out_features, use_complex32=False)
        
        # complex64入力（勾配計算を有効化）
        x = torch.randn(batch_size, in_features, dtype=torch.complex64, requires_grad=True)
        
        # Forward pass
        y = layer(x)
        
        # 損失関数（絶対値の合計）
        loss = y.abs().sum()
        
        # Backward pass
        loss.backward()
        
        # 勾配の検証
        assert layer.weight_real.grad is not None, "weight_real gradient should not be None"
        assert layer.weight_imag.grad is not None, "weight_imag gradient should not be None"
        assert layer.bias_real.grad is not None, "bias_real gradient should not be None"
        assert layer.bias_imag.grad is not None, "bias_imag gradient should not be None"
        
        # 勾配がNaN/Infでないことを確認
        assert not torch.isnan(layer.weight_real.grad).any(), "weight_real gradient contains NaN"
        assert not torch.isnan(layer.weight_imag.grad).any(), "weight_imag gradient contains NaN"
        assert not torch.isinf(layer.weight_real.grad).any(), "weight_real gradient contains Inf"
        assert not torch.isinf(layer.weight_imag.grad).any(), "weight_imag gradient contains Inf"
    
    def test_xavier_initialization(self):
        """
        Xavier初期化が正しく動作することを確認
        
        Xavier初期化の期待値:
            std = √(2 / (in_features + out_features))
        
        Requirements:
            - Requirement 1.8: Xavier初期化の検証
        """
        in_features = 64
        out_features = 128
        
        # ComplexLinear層の作成
        layer = ComplexLinear(in_features, out_features, use_complex32=False)
        
        # 期待される標準偏差
        expected_std = math.sqrt(2.0 / (in_features + out_features))
        
        # 実部の標準偏差
        real_std = layer.weight_real.data.std().item()
        imag_std = layer.weight_imag.data.std().item()
        
        # 標準偏差が期待値に近いことを確認（±50%の誤差を許容）
        # Xavier初期化はuniform分布なので、標準偏差にばらつきがある
        assert abs(real_std - expected_std) < expected_std * 0.5, \
            f"Real weight std {real_std} is too far from expected {expected_std}"
        assert abs(imag_std - expected_std) < expected_std * 0.5, \
            f"Imag weight std {imag_std} is too far from expected {expected_std}"
        
        # バイアスがゼロ初期化されていることを確認
        assert torch.allclose(layer.bias_real.data, torch.zeros_like(layer.bias_real.data)), \
            "bias_real should be zero-initialized"
        assert torch.allclose(layer.bias_imag.data, torch.zeros_like(layer.bias_imag.data)), \
            "bias_imag should be zero-initialized"
    
    def test_bias_disabled(self):
        """バイアスなしの動作を確認"""
        in_features = 8
        out_features = 4
        
        # バイアスなしのComplexLinear層
        layer = ComplexLinear(in_features, out_features, bias=False)
        
        assert layer.bias_real is None, "bias_real should be None when bias=False"
        assert layer.bias_imag is None, "bias_imag should be None when bias=False"
        
        # Forward passが正常に動作することを確認
        x = ComplexTensor(
            torch.randn(2, in_features, dtype=torch.float16),
            torch.randn(2, in_features, dtype=torch.float16)
        )
        y = layer(x)
        
        assert y.shape == (2, out_features), f"Expected shape (2, {out_features}), got {y.shape}"


class TestModReLU:
    """ModReLU活性化関数のテスト"""
    
    def test_phase_preservation(self):
        """
        位相が保存されることを確認
        
        Requirements:
            - Requirement 1.10: 位相保存の検証
        """
        features = 64
        batch_size = 4
        seq_len = 10
        
        # ModReLU層の作成
        modrelu = ModReLU(features, use_half=False)
        
        # complex64入力
        x = torch.randn(batch_size, seq_len, features, dtype=torch.complex64)
        
        # 位相を計算（活性化前）
        phase_before = torch.angle(x)
        
        # Forward pass
        y = modrelu(x)
        
        # 位相を計算（活性化後）
        phase_after = torch.angle(y)
        
        # 位相が保存されていることを確認（振幅がゼロの場合を除く）
        # 振幅が十分大きい要素のみをチェック
        mask = torch.abs(x) > 0.1
        assert torch.allclose(phase_before[mask], phase_after[mask], atol=1e-3), \
            "Phase should be preserved after ModReLU"
    
    def test_amplitude_filtering(self):
        """
        振幅がフィルタリングされることを確認
        
        Requirements:
            - Requirement 1.10: 振幅フィルタリングの検証
        """
        features = 64
        batch_size = 4
        
        # ModReLU層の作成（バイアス=0）
        modrelu = ModReLU(features, use_half=False)
        modrelu.bias.data.zero_()
        
        # 負の振幅を持つ入力を作成
        # 実部と虚部が両方負の場合、振幅は正だが、バイアスを調整して負にできる
        x_real = torch.full((batch_size, features), -0.5, dtype=torch.float32)
        x_imag = torch.full((batch_size, features), -0.5, dtype=torch.float32)
        x = torch.complex(x_real, x_imag)
        
        # 振幅を計算（活性化前）
        mag_before = torch.abs(x)  # √(0.5² + 0.5²) ≈ 0.707
        
        # バイアスを-1に設定（振幅を負にする）
        modrelu.bias.data.fill_(-1.0)
        
        # Forward pass
        y = modrelu(x)
        
        # 振幅を計算（活性化後）
        mag_after = torch.abs(y)
        
        # ReLU(0.707 - 1.0) = ReLU(-0.293) = 0
        # 振幅がゼロになることを確認
        assert torch.allclose(mag_after, torch.zeros_like(mag_after), atol=1e-3), \
            "Amplitude should be filtered to zero when ReLU(|z| + b) < 0"
    
    def test_gradient_computation(self):
        """勾配計算が正常に動作することを確認"""
        features = 32
        batch_size = 2
        
        # ModReLU層の作成
        modrelu = ModReLU(features, use_half=False)
        
        # complex64入力（勾配計算を有効化）
        x = torch.randn(batch_size, features, dtype=torch.complex64, requires_grad=True)
        
        # Forward pass
        y = modrelu(x)
        
        # 損失関数
        loss = y.abs().sum()
        
        # Backward pass
        loss.backward()
        
        # 勾配の検証
        assert modrelu.bias.grad is not None, "bias gradient should not be None"
        assert not torch.isnan(modrelu.bias.grad).any(), "bias gradient contains NaN"
        assert not torch.isinf(modrelu.bias.grad).any(), "bias gradient contains Inf"


class TestComplexLayerNorm:
    """ComplexLayerNorm層のテスト"""
    
    def test_normalization_accuracy(self):
        """
        正規化後の平均が0、分散が1に近いことを確認
        
        Requirements:
            - Requirement 1.12: 正規化の正確性
        """
        normalized_shape = 64
        batch_size = 4
        seq_len = 10
        
        # ComplexLayerNorm層の作成（アフィン変換なし）
        norm = ComplexLayerNorm(normalized_shape, elementwise_affine=False)
        
        # complex64入力
        x = torch.randn(batch_size, seq_len, normalized_shape, dtype=torch.complex64)
        
        # Forward pass
        y = norm(x)
        
        # 正規化後の平均を計算
        mean_real = y.real.mean(dim=-1)
        mean_imag = y.imag.mean(dim=-1)
        
        # 平均が0に近いことを確認
        assert torch.allclose(mean_real, torch.zeros_like(mean_real), atol=1e-5), \
            "Real part mean should be close to 0 after normalization"
        assert torch.allclose(mean_imag, torch.zeros_like(mean_imag), atol=1e-5), \
            "Imag part mean should be close to 0 after normalization"
        
        # 正規化後の分散を計算
        var = (y.real ** 2 + y.imag ** 2).mean(dim=-1)
        
        # 分散が1に近いことを確認
        assert torch.allclose(var, torch.ones_like(var), atol=1e-1), \
            "Variance should be close to 1 after normalization"
    
    def test_affine_transformation(self):
        """アフィン変換が正しく動作することを確認"""
        normalized_shape = 64
        batch_size = 2
        
        # ComplexLayerNorm層の作成（アフィン変換あり）
        norm = ComplexLayerNorm(normalized_shape, elementwise_affine=True)
        
        # gammaとbetaを手動で設定
        norm.gamma.data.fill_(2.0)
        norm.beta.data.fill_(1.0)
        
        # complex64入力
        x = torch.randn(batch_size, normalized_shape, dtype=torch.complex64)
        
        # Forward pass
        y = norm(x)
        
        # アフィン変換なしの正規化
        norm_no_affine = ComplexLayerNorm(normalized_shape, elementwise_affine=False)
        y_no_affine = norm_no_affine(x)
        
        # y = gamma * y_no_affine + beta
        expected = y_no_affine * 2.0 + 1.0
        
        assert torch.allclose(y.real, expected.real, atol=1e-3), \
            "Affine transformation (real part) is incorrect"
        assert torch.allclose(y.imag, expected.imag, atol=1e-3), \
            "Affine transformation (imag part) is incorrect"
    
    def test_gradient_computation(self):
        """
        勾配計算が正常に動作することを確認
        
        Requirements:
            - Requirement 1.12: 勾配計算の正常動作
        """
        normalized_shape = 32
        batch_size = 2
        
        # ComplexLayerNorm層の作成
        norm = ComplexLayerNorm(normalized_shape, elementwise_affine=True)
        
        # complex64入力（勾配計算を有効化）
        x = torch.randn(batch_size, normalized_shape, dtype=torch.complex64, requires_grad=True)
        
        # Forward pass
        y = norm(x)
        
        # 損失関数
        loss = y.abs().sum()
        
        # Backward pass
        loss.backward()
        
        # 勾配の検証
        assert norm.gamma.grad is not None, "gamma gradient should not be None"
        assert norm.beta.grad is not None, "beta gradient should not be None"
        assert not torch.isnan(norm.gamma.grad).any(), "gamma gradient contains NaN"
        assert not torch.isnan(norm.beta.grad).any(), "beta gradient contains NaN"
        assert not torch.isinf(norm.gamma.grad).any(), "gamma gradient contains Inf"
        assert not torch.isinf(norm.beta.grad).any(), "beta gradient contains Inf"
    
    def test_complex_tensor_input(self):
        """ComplexTensor入力でも正常に動作することを確認"""
        normalized_shape = 64
        batch_size = 2
        seq_len = 10
        
        # ComplexLayerNorm層の作成
        norm = ComplexLayerNorm(normalized_shape)
        
        # ComplexTensor入力（3次元: batch, seq, features）
        x = ComplexTensor(
            torch.randn(batch_size, seq_len, normalized_shape, dtype=torch.float16),
            torch.randn(batch_size, seq_len, normalized_shape, dtype=torch.float16)
        )
        
        # Forward pass
        y = norm(x)
        
        # 出力がComplexTensorであることを確認
        assert isinstance(y, ComplexTensor), "Output should be ComplexTensor"
        assert y.shape == (batch_size, seq_len, normalized_shape), \
            f"Expected shape {(batch_size, seq_len, normalized_shape)}, got {y.shape}"


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v", "--tb=short"])
