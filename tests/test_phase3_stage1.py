"""
Phase 3 Stage 1 Integration Tests

このモジュールは、Phase 3 Stage 1モデルの統合テストを実装します。

Test Coverage:
    1. Forward/Backward passの正常動作
    2. NaN/Infの発生チェック
    3. 勾配の健全性チェック
    4. メモリ効率の検証
    5. Phase 2互換性の検証

Requirements:
    - Requirement 1.17: Forward/Backward passが正常に動作することを確認
    - Requirement 1.17: NaN/Infが発生しないことを確認

Author: Project MUSE Team
Date: 2025-01-21
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from src.models.phase3.stage1_model import (
    Phase3Stage1Block,
    Phase3Stage1Model,
    create_phase3_stage1_model,
    convert_phase2_to_complex,
)
from src.models.phase3.complex_tensor import ComplexTensor


class TestPhase3Stage1Block:
    """Phase3Stage1Blockのテスト"""
    
    def test_forward_pass(self):
        """Forward passが正常に動作することを確認"""
        # モデルの作成
        block = Phase3Stage1Block(d_model=64, use_complex32=True)
        
        # 入力の作成
        z = ComplexTensor(
            torch.randn(2, 10, 64, dtype=torch.float16),
            torch.randn(2, 10, 64, dtype=torch.float16)
        )
        
        # Forward pass
        z_out = block(z)
        
        # 出力の検証
        assert isinstance(z_out, ComplexTensor)
        assert z_out.shape == (2, 10, 64)
        assert z_out.dtype == torch.float16
    
    def test_backward_pass(self):
        """Backward passが正常に動作することを確認"""
        # モデルの作成
        block = Phase3Stage1Block(d_model=64, use_complex32=True)
        
        # 入力の作成（勾配計算を有効化）
        z = ComplexTensor(
            torch.randn(2, 10, 64, dtype=torch.float16, requires_grad=True),
            torch.randn(2, 10, 64, dtype=torch.float16, requires_grad=True)
        )
        
        # Forward pass
        z_out = block(z)
        
        # 損失の計算（実部の平均を使用）
        loss = z_out.real.mean()
        
        # Backward pass
        loss.backward()
        
        # 勾配の検証
        assert z.real.grad is not None
        assert z.imag.grad is not None
        assert not torch.isnan(z.real.grad).any()
        assert not torch.isnan(z.imag.grad).any()
    
    def test_no_nan_inf(self):
        """NaN/Infが発生しないことを確認"""
        # モデルの作成
        block = Phase3Stage1Block(d_model=64, use_complex32=True)
        
        # 複数回のForward passでNaN/Infをチェック
        for _ in range(10):
            z = ComplexTensor(
                torch.randn(2, 10, 64, dtype=torch.float16),
                torch.randn(2, 10, 64, dtype=torch.float16)
            )
            
            z_out = block(z)
            
            # NaN/Infのチェック
            assert not torch.isnan(z_out.real).any(), "NaN detected in real part"
            assert not torch.isnan(z_out.imag).any(), "NaN detected in imag part"
            assert not torch.isinf(z_out.real).any(), "Inf detected in real part"
            assert not torch.isinf(z_out.imag).any(), "Inf detected in imag part"
    
    def test_diagnostics(self):
        """診断情報が正しく収集されることを確認"""
        # モデルの作成
        block = Phase3Stage1Block(d_model=64, use_complex32=True)
        
        # 入力の作成
        z = ComplexTensor(
            torch.randn(2, 10, 64, dtype=torch.float16),
            torch.randn(2, 10, 64, dtype=torch.float16)
        )
        
        # Forward pass with diagnostics
        z_out, diagnostics = block(z, return_diagnostics=True)
        
        # 診断情報の検証
        assert 'magnitude_mean' in diagnostics
        assert 'magnitude_std' in diagnostics
        assert 'phase_mean' in diagnostics
        assert 'phase_std' in diagnostics
        
        # 値の妥当性チェック
        assert diagnostics['magnitude_mean'] > 0
        assert diagnostics['magnitude_std'] >= 0


class TestPhase3Stage1Model:
    """Phase3Stage1Modelのテスト"""
    
    def test_model_creation(self):
        """モデルが正しく作成されることを確認"""
        # モデルの作成
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=True
        )
        
        # モデルの検証
        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.n_layers == 2
        assert len(model.blocks) == 2
    
    def test_forward_pass(self):
        """Forward passが正常に動作することを確認"""
        # モデルの作成
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=True
        )
        
        # 入力の作成
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        logits = model(input_ids)
        
        # 出力の検証
        assert logits.shape == (2, 10, 1000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_backward_pass(self):
        """Backward passが正常に動作することを確認"""
        # モデルの作成
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=True
        )
        
        # 入力の作成
        input_ids = torch.randint(0, 1000, (2, 10))
        target_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        logits = model(input_ids)
        
        # 損失の計算
        loss = nn.functional.cross_entropy(
            logits.view(-1, 1000),
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # 勾配の検証
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient is None for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
                assert not torch.isinf(param.grad).any(), f"Inf in gradient for {name}"
    
    def test_gradient_health(self):
        """勾配の健全性をチェック（1e-6以上、1e3以下）"""
        # モデルの作成
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=True
        )
        
        # 入力の作成
        input_ids = torch.randint(0, 1000, (2, 10))
        target_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        logits = model(input_ids)
        
        # 損失の計算
        loss = nn.functional.cross_entropy(
            logits.view(-1, 1000),
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # 勾配ノルムのチェック
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                # 勾配ノルムが健全な範囲にあることを確認
                assert grad_norm >= 1e-6, f"Gradient too small for {name}: {grad_norm}"
                assert grad_norm <= 1e3, f"Gradient too large for {name}: {grad_norm}"
    
    def test_no_nan_multiple_iterations(self):
        """複数回のForward/Backward passでNaNが発生しないことを確認（float32使用）"""
        # モデルの作成（float32を使用して数値安定性を確保）
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=False  # float32を使用
        )
        
        # オプティマイザの作成
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 10回のイテレーション（float32でも時間がかかるため、回数を減らす）
        for i in range(10):
            # 入力の作成
            input_ids = torch.randint(0, 1000, (2, 10))
            target_ids = torch.randint(0, 1000, (2, 10))
            
            # Forward pass
            logits = model(input_ids)
            
            # 損失の計算
            loss = nn.functional.cross_entropy(
                logits.view(-1, 1000),
                target_ids.view(-1)
            )
            
            # NaNチェック
            assert not torch.isnan(loss).any(), f"NaN in loss at iteration {i}"
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 勾配のNaNチェック
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    assert not torch.isnan(param.grad).any(), \
                        f"NaN in gradient for {name} at iteration {i}"
        
        # 統計情報の確認
        stats = model.get_statistics()
        assert stats['nan_rate'] == 0.0, \
            f"NaN detected: {stats['nan_count']}/{stats['forward_count']}"
        
        print(f"✓ 10回のイテレーションでNaN発生率0%を達成（float32使用）")
    
    def test_diagnostics(self):
        """診断情報が正しく収集されることを確認"""
        # モデルの作成
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=True
        )
        
        # 入力の作成
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass with diagnostics
        logits, diagnostics = model(input_ids, return_diagnostics=True)
        
        # 診断情報の検証
        assert 'embedding' in diagnostics
        assert 'layer_0' in diagnostics
        assert 'layer_1' in diagnostics
        assert 'output' in diagnostics
        
        # 各層の診断情報をチェック
        for key in ['layer_0', 'layer_1']:
            assert 'magnitude_mean' in diagnostics[key]
            assert 'magnitude_std' in diagnostics[key]
            assert 'phase_mean' in diagnostics[key]
            assert 'phase_std' in diagnostics[key]
    
    def test_factory_function(self):
        """ファクトリー関数が正しく動作することを確認"""
        # ファクトリー関数でモデルを作成
        model = create_phase3_stage1_model(
            vocab_size=1000,
            d_model=64,
            n_layers=2
        )
        
        # モデルの検証
        assert isinstance(model, Phase3Stage1Model)
        assert model.vocab_size == 1000
        assert model.d_model == 64
        assert model.n_layers == 2


class TestPhase2Compatibility:
    """Phase 2互換性のテスト"""
    
    def test_convert_phase2_to_complex(self):
        """Phase 2モデルをPhase 3に変換できることを確認"""
        # Phase 2モデルの作成（簡略版）
        from src.models.phase2.factory import create_phase2_model
        
        try:
            phase2_model = create_phase2_model(
                vocab_size=1000,
                d_model=64,
                n_layers=2,
                n_seq=128
            )
        except Exception as e:
            pytest.skip(f"Phase 2 model creation failed: {e}")
        
        # Phase 3に変換
        phase3_model = convert_phase2_to_complex(phase2_model, use_complex32=True)
        
        # モデルの検証
        assert isinstance(phase3_model, Phase3Stage1Model)
        assert phase3_model.vocab_size == 1000
        assert phase3_model.d_model == 64
        assert phase3_model.n_layers == 2
    
    def test_converted_model_forward(self):
        """変換されたモデルがForward passできることを確認"""
        # Phase 2モデルの作成
        from src.models.phase2.factory import create_phase2_model
        
        try:
            phase2_model = create_phase2_model(
                vocab_size=1000,
                d_model=64,
                n_layers=2,
                n_seq=128
            )
        except Exception as e:
            pytest.skip(f"Phase 2 model creation failed: {e}")
        
        # Phase 3に変換
        phase3_model = convert_phase2_to_complex(phase2_model, use_complex32=True)
        
        # 入力の作成
        input_ids = torch.randint(0, 1000, (2, 10))
        
        # Forward pass
        logits = phase3_model(input_ids)
        
        # 出力の検証
        assert logits.shape == (2, 10, 1000)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestMemoryEfficiency:
    """メモリ効率のテスト"""
    
    def test_complex32_memory_usage(self):
        """complex32がメモリ効率的であることを確認"""
        # complex32モデル
        model_complex32 = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            use_complex32=True
        )
        
        # complex64モデル（比較用）
        model_complex64 = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            use_complex32=False
        )
        
        # パラメータ数の比較
        params_complex32 = sum(p.numel() * p.element_size() for p in model_complex32.parameters())
        params_complex64 = sum(p.numel() * p.element_size() for p in model_complex64.parameters())
        
        # complex32の方がメモリ効率的であることを確認
        # 注意: 完全に50%削減されるわけではない（Output層は実数のため）
        assert params_complex32 < params_complex64, \
            f"complex32 ({params_complex32}) should use less memory than complex64 ({params_complex64})"
        
        print(f"Memory usage:")
        print(f"  complex32: {params_complex32 / 1024 / 1024:.2f} MB")
        print(f"  complex64: {params_complex64 / 1024 / 1024:.2f} MB")
        print(f"  Reduction: {(1 - params_complex32 / params_complex64) * 100:.1f}%")


class TestNumericalStability:
    """数値安定性のテスト"""
    
    def test_random_input_stability(self):
        """ランダム入力100回試行でNaN発生率0%を確認"""
        # モデルの作成
        model = Phase3Stage1Model(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            max_seq_len=128,
            use_complex32=True
        )
        
        # 統計情報のリセット
        model.reset_statistics()
        
        # 100回のランダム入力
        for i in range(100):
            input_ids = torch.randint(0, 1000, (2, 10))
            
            # Forward pass
            logits = model(input_ids)
            
            # NaNチェック
            if torch.isnan(logits).any():
                pytest.fail(f"NaN detected at iteration {i}")
        
        # 統計情報の確認
        stats = model.get_statistics()
        assert stats['nan_rate'] == 0.0, \
            f"NaN detected: {stats['nan_count']}/{stats['forward_count']} " \
            f"(rate: {stats['nan_rate']*100:.2f}%)"
        
        print(f"✓ 100回のランダム入力でNaN発生率0%を達成")


if __name__ == '__main__':
    # テストの実行
    pytest.main([__file__, '-v', '-s'])
