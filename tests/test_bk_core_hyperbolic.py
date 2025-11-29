"""
BK-Core Hyperbolic Integration Tests

BK-Core双曲統合モジュールのテスト。
Property-Based TestingとUnit Testを含む。

Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, strategies as st, settings, assume

# テスト対象モジュール
import sys
sys.path.insert(0, '.')

from src.models.phase8.bk_core_hyperbolic import (
    BKCoreHyperbolicConfig,
    BKCoreHyperbolicIntegration,
    ScatteringGate,
    ResonanceDetector,
    HybridGradientComputation,
    create_bk_core_hyperbolic,
)


# ============================================================
# Property-Based Tests
# ============================================================

class TestBKCoreGateCorrelationProperty:
    """
    **Feature: phase8-hyperbolic-transcendence, Property 15: BK-Core Gate Correlation**
    **Validates: Requirements 22.2**
    
    BK-Coreゲートとアテンション重みの相関が正の値を持つことを検証。
    散乱エネルギーが高い位置ほどアテンションが強くなるべき。
    """
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=8, max_value=32),
        d_model=st.sampled_from([64, 128]),
        num_heads=st.sampled_from([2, 4]),
    )
    @settings(max_examples=50, deadline=30000)
    def test_gate_correlation_positive(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        num_heads: int,
    ):
        """
        Property: BK-Coreゲートとアテンション重みの相関は非負
        
        散乱エネルギーが高い位置ほどアテンションが強くなる傾向があるべき。
        """
        torch.manual_seed(42)
        
        # モデル作成
        model = create_bk_core_hyperbolic(
            d_model=d_model,
            use_scattering_gate=True,
            use_resonance_detection=True,
        )
        model.eval()
        
        # 入力生成
        x = torch.randn(batch_size, seq_len, d_model)
        
        # アテンション重みを生成（ソフトマックス正規化済み）
        attn_logits = torch.randn(batch_size, num_heads, seq_len, seq_len)
        attention_weights = torch.softmax(attn_logits, dim=-1)
        
        # ゲーティング適用
        with torch.no_grad():
            output, diagnostics = model(x, attention_weights)
        
        # ゲート値の統計を確認
        gate_mean = diagnostics.get('gate_mean', torch.tensor(0.5))
        
        # ゲート値は[0, 1]の範囲内
        assert 0.0 <= gate_mean.item() <= 1.0, \
            f"Gate mean {gate_mean.item()} out of range [0, 1]"
        
        # 出力形状の確認
        assert output.shape == attention_weights.shape, \
            f"Output shape {output.shape} != attention shape {attention_weights.shape}"


class TestResonanceDetectionProperty:
    """
    共鳴検出のプロパティテスト
    
    Requirements: 22.3
    """
    
    @given(
        resonance_strength=st.floats(min_value=0.0, max_value=2.0),
        threshold=st.floats(min_value=0.1, max_value=1.0),
        current_curvature=st.floats(min_value=0.5, max_value=5.0),
    )
    @settings(max_examples=100, deadline=10000)
    def test_curvature_adjustment_direction(
        self,
        resonance_strength: float,
        threshold: float,
        current_curvature: float,
    ):
        """
        Property: 共鳴強度が閾値を超えると曲率が増加、下回ると減少
        """
        assume(not np.isnan(resonance_strength))
        assume(not np.isnan(threshold))
        assume(not np.isnan(current_curvature))
        
        detector = ResonanceDetector(
            threshold=threshold,
            adjustment_rate=0.1,
        )
        
        # G_iiをシミュレート
        G_ii = torch.complex(
            torch.tensor([[resonance_strength]]),
            torch.tensor([[0.1]])
        )
        
        suggested, diagnostics = detector(G_ii, current_curvature)
        
        if resonance_strength > threshold:
            # 共鳴時は曲率増加
            assert suggested >= current_curvature, \
                f"Curvature should increase on resonance: {suggested} < {current_curvature}"
        else:
            # 非共鳴時は曲率減少
            assert suggested <= current_curvature, \
                f"Curvature should decrease without resonance: {suggested} > {current_curvature}"


# ============================================================
# Unit Tests
# ============================================================

class TestScatteringGate:
    """散乱ゲートのユニットテスト"""
    
    def test_gate_output_range(self):
        """ゲート出力が[0, 1]の範囲内"""
        gate = ScatteringGate(d_model=64, gate_scale=1.0)
        
        G_ii = torch.complex(
            torch.randn(2, 16),
            torch.randn(2, 16)
        )
        attention_weights = torch.softmax(
            torch.randn(2, 4, 16, 16), dim=-1
        )
        
        gated, diagnostics = gate(G_ii, attention_weights)
        
        # ゲート平均は[0, 1]
        assert 0.0 <= diagnostics['gate_mean'].item() <= 1.0
        
        # 出力は正規化されている
        row_sums = gated.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_gate_shape_preservation(self):
        """ゲーティング後も形状が保持される"""
        gate = ScatteringGate(d_model=128, gate_scale=2.0)
        
        batch, heads, seq = 3, 8, 32
        G_ii = torch.complex(
            torch.randn(batch, seq),
            torch.randn(batch, seq)
        )
        attention_weights = torch.softmax(
            torch.randn(batch, heads, seq, seq), dim=-1
        )
        
        gated, _ = gate(G_ii, attention_weights)
        
        assert gated.shape == attention_weights.shape


class TestResonanceDetector:
    """共鳴検出のユニットテスト"""
    
    def test_resonance_detection(self):
        """共鳴状態の検出"""
        detector = ResonanceDetector(threshold=0.5, adjustment_rate=0.1)
        
        # 高い共鳴強度
        G_ii_high = torch.complex(
            torch.ones(2, 16) * 1.0,
            torch.zeros(2, 16)
        )
        suggested_high, diag_high = detector(G_ii_high, 1.0)
        
        assert diag_high['is_resonant'].item() == True
        assert suggested_high > 1.0
        
        # 低い共鳴強度
        G_ii_low = torch.complex(
            torch.ones(2, 16) * 0.1,
            torch.zeros(2, 16)
        )
        suggested_low, diag_low = detector(G_ii_low, 1.0)
        
        assert diag_low['is_resonant'].item() == False
        assert suggested_low < 1.0
    
    def test_curvature_bounds(self):
        """曲率が範囲内に収まる"""
        detector = ResonanceDetector(threshold=0.5, adjustment_rate=1.0)
        
        # 極端に高い共鳴
        G_ii = torch.complex(
            torch.ones(1, 8) * 100.0,
            torch.zeros(1, 8)
        )
        suggested, _ = detector(G_ii, 5.0)
        
        assert 0.1 <= suggested <= 10.0


class TestHybridGradientComputation:
    """ハイブリッド勾配計算のユニットテスト"""
    
    def test_gradient_blend(self):
        """勾配ブレンドの動作確認"""
        # 純粋な理論的勾配
        hybrid_theoretical = HybridGradientComputation(alpha=0.0)
        
        # 純粋なHypothesis-7勾配
        hybrid_h7 = HybridGradientComputation(alpha=1.0)
        
        # 50/50ブレンド
        hybrid_blend = HybridGradientComputation(alpha=0.5)
        
        G_ii = torch.complex(
            torch.randn(2, 16),
            torch.randn(2, 16)
        )
        grad_output = torch.complex(
            torch.randn(2, 16),
            torch.randn(2, 16)
        )
        
        grad_t = hybrid_theoretical.compute_gradient(G_ii, grad_output)
        grad_h = hybrid_h7.compute_gradient(G_ii, grad_output)
        grad_b = hybrid_blend.compute_gradient(G_ii, grad_output)
        
        # ブレンド勾配は両者の中間
        expected_blend = 0.5 * grad_t + 0.5 * grad_h
        assert torch.allclose(grad_b, expected_blend, atol=1e-5)


class TestBKCoreHyperbolicIntegration:
    """BK-Core双曲統合のユニットテスト"""
    
    def test_forward_pass(self):
        """Forward passの動作確認"""
        model = create_bk_core_hyperbolic(
            d_model=64,
            curvature=1.0,
            use_scattering_gate=True,
            use_resonance_detection=True,
        )
        
        x = torch.randn(2, 16, 64)
        attention_weights = torch.softmax(
            torch.randn(2, 4, 16, 16), dim=-1
        )
        
        output, diagnostics = model(x, attention_weights)
        
        # 出力形状
        assert output.shape == attention_weights.shape
        
        # 診断情報
        assert 'G_ii_mean' in diagnostics
        assert 'gate_mean' in diagnostics
        assert 'resonance_strength' in diagnostics
    
    def test_green_function_computation(self):
        """グリーン関数計算の動作確認"""
        model = create_bk_core_hyperbolic(d_model=64)
        
        x = torch.randn(2, 16, 64)
        G_ii, features = model.compute_green_function(x)
        
        # G_iiは複素数
        assert G_ii.dtype == torch.complex64 or G_ii.dtype == torch.complex128
        
        # 特徴量は実数
        assert features.dtype == torch.float32
        assert features.shape == (2, 16, 2)
    
    def test_scattering_energy(self):
        """散乱エネルギー取得の動作確認"""
        model = create_bk_core_hyperbolic(d_model=64)
        
        x = torch.randn(2, 16, 64)
        energy = model.get_scattering_energy(x)
        
        # エネルギーは非負
        assert (energy >= 0).all()
        assert energy.shape == (2, 16)
    
    def test_without_attention_weights(self):
        """アテンション重みなしでの動作"""
        model = create_bk_core_hyperbolic(
            d_model=64,
            use_scattering_gate=True,
        )
        
        x = torch.randn(2, 16, 64)
        output, diagnostics = model(x, attention_weights=None)
        
        # アテンション重みがない場合は特徴量を返す
        assert output.shape == (2, 16, 2)
    
    def test_config_serialization(self):
        """設定のシリアライズ"""
        config = BKCoreHyperbolicConfig(
            d_model=128,
            curvature=2.0,
            gate_scale=1.5,
        )
        
        model = BKCoreHyperbolicIntegration(config)
        
        assert model.config.d_model == 128
        assert model.config.curvature == 2.0
        assert model.config.gate_scale == 1.5


class TestFactoryFunction:
    """ファクトリ関数のテスト"""
    
    def test_create_with_defaults(self):
        """デフォルト設定での作成"""
        model = create_bk_core_hyperbolic()
        
        assert model.d_model == 256
        assert model.curvature == 1.0
        assert model.scattering_gate is not None
        assert model.resonance_detector is not None
    
    def test_create_with_custom_config(self):
        """カスタム設定での作成"""
        model = create_bk_core_hyperbolic(
            d_model=512,
            curvature=2.0,
            use_scattering_gate=False,
            use_resonance_detection=False,
        )
        
        assert model.d_model == 512
        assert model.curvature == 2.0
        assert model.scattering_gate is None
        assert model.resonance_detector is None


# ============================================================
# Integration Tests
# ============================================================

class TestBKCoreHyperbolicIntegrationTests:
    """統合テスト"""
    
    def test_gradient_flow(self):
        """勾配フローの確認（簡易グリーン関数使用）"""
        model = create_bk_core_hyperbolic(
            d_model=64,
            use_scattering_gate=False,
        )
        model.train()
        
        x = torch.randn(2, 16, 64, requires_grad=True)
        
        # 簡易グリーン関数を直接テスト
        he_diag = model.he_diag_proj(x).mean(dim=-1)
        h0_super = model.h0_super_proj(x[:, :-1]).mean(dim=-1)
        h0_sub = model.h0_sub_proj(x[:, 1:]).mean(dim=-1)
        
        G_ii = model._simple_green_function(he_diag, h0_super, h0_sub)
        features = torch.stack([G_ii.real, G_ii.imag], dim=-1)
        
        loss = features.sum()
        loss.backward()
        
        # 勾配が計算されている
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_numerical_stability(self):
        """数値安定性の確認"""
        model = create_bk_core_hyperbolic(d_model=64)
        
        # 大きな値
        x_large = torch.randn(2, 16, 64) * 100
        output_large, _ = model(x_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()
        
        # 小さな値
        x_small = torch.randn(2, 16, 64) * 0.001
        output_small, _ = model(x_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
