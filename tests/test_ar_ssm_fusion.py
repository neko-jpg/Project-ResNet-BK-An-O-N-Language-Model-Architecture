"""
AR-SSM Hyperbolic Fusion Tests

AR-SSM双曲融合モジュールのテスト。
Property-Based TestingとUnit Testを含む。

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5, 21.6
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, '.')

from src.models.phase8.ar_ssm_fusion import (
    ARSSMFusionConfig,
    ARSSMHyperbolicFusion,
    HyperbolicRankGating,
    PhysicsInformedGating,
    AdaptiveRankSSM,
    create_ar_ssm_fusion,
)


# ============================================================
# Property-Based Tests
# ============================================================

class TestARSSMThroughputProperty:
    """
    **Feature: phase8-hyperbolic-transcendence, Property 14: AR-SSM Hyperbolic Throughput**
    **Validates: Requirements 21.5**
    
    AR-SSM Hyperbolic Fusionのスループットが一定の基準を満たすことを検証。
    """
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=16, max_value=64),
        d_model=st.sampled_from([64, 128]),
    )
    @settings(max_examples=20, deadline=60000)
    def test_throughput_positive(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
    ):
        """
        Property: スループットは正の値
        """
        torch.manual_seed(42)
        
        model = create_ar_ssm_fusion(
            d_model=d_model,
            d_state=32,
            max_rank=16,
        )
        model.eval()
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # スループット測定
        metrics = model.compute_throughput_metrics(x, num_iterations=3)
        
        assert metrics['tokens_per_second'] > 0, \
            f"Throughput should be positive: {metrics['tokens_per_second']}"
        assert metrics['elapsed_time'] > 0, \
            f"Elapsed time should be positive: {metrics['elapsed_time']}"


class TestHyperbolicRankGatingProperty:
    """
    双曲ランクゲーティングのプロパティテスト
    
    Requirements: 21.1, 21.2
    """
    
    @given(
        batch_size=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=8, max_value=32),
        d_model=st.sampled_from([64, 128]),
    )
    @settings(max_examples=50, deadline=30000)
    def test_rank_in_valid_range(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
    ):
        """
        Property: 有効ランクは[min_rank, max_rank]の範囲内
        """
        torch.manual_seed(42)
        
        min_rank = 4
        max_rank = 32
        
        gating = HyperbolicRankGating(
            d_model=d_model,
            max_rank=max_rank,
            min_rank=min_rank,
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        with torch.no_grad():
            _, effective_rank, _ = gating(x)
        
        assert (effective_rank >= min_rank).all(), \
            f"Effective rank below min: {effective_rank.min()}"
        assert (effective_rank <= max_rank).all(), \
            f"Effective rank above max: {effective_rank.max()}"
    
    @given(
        scale=st.floats(min_value=0.01, max_value=10.0),
    )
    @settings(max_examples=50, deadline=10000)
    def test_distance_increases_with_norm(self, scale: float):
        """
        Property: ノルムが大きいほど双曲距離も大きい
        """
        assume(not np.isnan(scale))
        
        gating = HyperbolicRankGating(d_model=64)
        
        # 小さいノルムの入力
        x_small = torch.randn(2, 8, 64) * 0.1
        
        # 大きいノルムの入力
        x_large = torch.randn(2, 8, 64) * scale
        
        dist_small = gating.compute_hyperbolic_distance(x_small)
        dist_large = gating.compute_hyperbolic_distance(x_large)
        
        # 大きいノルムの方が距離が大きい（平均で比較）
        if scale > 0.1:
            assert dist_large.mean() >= dist_small.mean() * 0.9, \
                f"Distance should increase with norm: {dist_large.mean()} < {dist_small.mean()}"


# ============================================================
# Unit Tests
# ============================================================

class TestHyperbolicRankGating:
    """双曲ランクゲーティングのユニットテスト"""
    
    def test_output_shapes(self):
        """出力形状の確認"""
        gating = HyperbolicRankGating(d_model=64, max_rank=32, min_rank=4)
        
        x = torch.randn(2, 16, 64)
        rank_weights, effective_rank, diagnostics = gating(x)
        
        assert rank_weights.shape == (2, 16)
        assert effective_rank.shape == (2, 16)
        assert 'distance_mean' in diagnostics
    
    def test_rank_weights_range(self):
        """ランク重みが[0, 1]の範囲内"""
        gating = HyperbolicRankGating(d_model=64)
        
        x = torch.randn(2, 16, 64)
        rank_weights, _, _ = gating(x)
        
        assert (rank_weights >= 0).all()
        assert (rank_weights <= 1).all()


class TestPhysicsInformedGating:
    """物理ベースゲーティングのユニットテスト"""
    
    def test_gate_output_range(self):
        """ゲート出力が[0, 1]の範囲内"""
        gating = PhysicsInformedGating(d_model=64)
        
        G_ii = torch.complex(
            torch.randn(2, 16),
            torch.randn(2, 16)
        )
        
        gate, diagnostics = gating(G_ii)
        
        assert (gate >= 0).all()
        assert (gate <= 1).all()
        assert 'physics_gate_mean' in diagnostics
    
    def test_gate_shape(self):
        """ゲート形状の確認"""
        gating = PhysicsInformedGating(d_model=128)
        
        G_ii = torch.complex(
            torch.randn(3, 32),
            torch.randn(3, 32)
        )
        
        gate, _ = gating(G_ii)
        
        assert gate.shape == (3, 32)


class TestAdaptiveRankSSM:
    """適応的ランクSSMのユニットテスト"""
    
    def test_forward_pass(self):
        """Forward passの動作確認"""
        ssm = AdaptiveRankSSM(d_model=64, d_state=32, max_rank=16)
        
        x = torch.randn(2, 16, 64)
        output, diagnostics = ssm(x)
        
        assert output.shape == x.shape
        assert 'state_norm_mean' in diagnostics
        assert 'A_spectral_norm' in diagnostics
    
    def test_with_rank_weights(self):
        """ランク重み付きの動作確認"""
        ssm = AdaptiveRankSSM(d_model=64, d_state=32, max_rank=16)
        
        x = torch.randn(2, 16, 64)
        rank_weights = torch.rand(2, 16)
        
        output, _ = ssm(x, rank_weights)
        
        assert output.shape == x.shape
    
    def test_spectral_norm_bounded(self):
        """スペクトルノルムが有界"""
        ssm = AdaptiveRankSSM(d_model=64, d_state=32, max_rank=16)
        
        x = torch.randn(2, 16, 64)
        _, diagnostics = ssm(x)
        
        # スペクトルノルムは有限
        assert torch.isfinite(diagnostics['A_spectral_norm'])


class TestARSSMHyperbolicFusion:
    """AR-SSM双曲融合のユニットテスト"""
    
    def test_forward_pass(self):
        """Forward passの動作確認"""
        model = create_ar_ssm_fusion(
            d_model=64,
            d_state=32,
            max_rank=16,
        )
        
        x = torch.randn(2, 16, 64)
        output, diagnostics = model(x)
        
        assert output.shape == x.shape
        assert 'distance_mean' in diagnostics
        assert 'effective_rank_mean' in diagnostics
    
    def test_with_physics_gating(self):
        """物理ベースゲーティング付きの動作確認"""
        model = create_ar_ssm_fusion(
            d_model=64,
            use_physics_gating=True,
        )
        
        x = torch.randn(2, 16, 64)
        G_ii = torch.complex(
            torch.randn(2, 16),
            torch.randn(2, 16)
        )
        
        output, diagnostics = model(x, G_ii)
        
        assert output.shape == x.shape
        assert 'physics_gate_mean' in diagnostics
    
    def test_without_adaptive_rank(self):
        """適応的ランクなしの動作確認"""
        model = create_ar_ssm_fusion(
            d_model=64,
            use_adaptive_rank=False,
        )
        
        x = torch.randn(2, 16, 64)
        output, diagnostics = model(x)
        
        assert output.shape == x.shape
        assert 'distance_mean' not in diagnostics
    
    def test_curvature_adjustment(self):
        """曲率調整の動作確認"""
        model = create_ar_ssm_fusion(
            d_model=64,
            curvature=1.0,
            curvature_adjustment_rate=0.1,
        )
        
        # 高ランクを誘発する入力（大きなノルム）
        x = torch.randn(2, 16, 64) * 10
        
        _, diagnostics = model(x)
        
        # 曲率調整が提案される可能性がある
        # （入力によっては提案されない場合もある）
        if 'suggested_curvature' in diagnostics:
            assert diagnostics['suggested_curvature'] > 0


class TestFactoryFunction:
    """ファクトリ関数のテスト"""
    
    def test_create_with_defaults(self):
        """デフォルト設定での作成"""
        model = create_ar_ssm_fusion()
        
        assert model.d_model == 256
        assert model.curvature == 1.0
        assert model.rank_gating is not None
        assert model.physics_gating is not None
    
    def test_create_with_custom_config(self):
        """カスタム設定での作成"""
        model = create_ar_ssm_fusion(
            d_model=512,
            d_state=128,
            max_rank=64,
            curvature=2.0,
            use_physics_gating=False,
        )
        
        assert model.d_model == 512
        assert model.config.d_state == 128
        assert model.config.max_rank == 64
        assert model.curvature == 2.0
        assert model.physics_gating is None


# ============================================================
# Integration Tests
# ============================================================

class TestARSSMFusionIntegration:
    """統合テスト"""
    
    def test_gradient_flow(self):
        """勾配フローの確認"""
        model = create_ar_ssm_fusion(d_model=64)
        model.train()
        
        x = torch.randn(2, 16, 64, requires_grad=True)
        
        output, _ = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_numerical_stability(self):
        """数値安定性の確認"""
        model = create_ar_ssm_fusion(d_model=64)
        
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
    
    def test_deterministic_output(self):
        """決定論的出力の確認"""
        model = create_ar_ssm_fusion(d_model=64)
        model.eval()
        
        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)
        
        with torch.no_grad():
            output1, _ = model(x)
            output2, _ = model(x)
        
        assert torch.allclose(output1, output2)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
