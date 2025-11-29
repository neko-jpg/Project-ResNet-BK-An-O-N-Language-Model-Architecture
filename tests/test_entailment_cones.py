"""
Entailment Cones Unit Tests and Property Tests

Phase 8のEntailment Conesモジュールのテスト。

テスト項目:
- エンテイルメントスコア範囲 (Property 1)
- アパーチャ単調性 (Property 2)
- 設定ラウンドトリップ (Property 3)
- 論理演算
- 数値安定性

Requirements: 1.1-1.7
"""
import pytest
import torch
import torch.nn.functional as F
import json
import math

# Skip if CUDA not available for GPU tests
cuda_available = torch.cuda.is_available()


class TestEntailmentCones:
    """Entailment Conesの基本テスト"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.device = 'cuda' if cuda_available else 'cpu'
        self.d_model = 64
        self.batch_size = 8
    
    def _create_module(self, **kwargs):
        """モジュール作成"""
        try:
            from src.models.phase8.entailment_cones import (
                EntailmentCones, EntailmentConeConfig
            )
        except ImportError:
            pytest.skip("EntailmentCones not available")
        
        config = EntailmentConeConfig(d_model=self.d_model, **kwargs)
        return EntailmentCones(config).to(self.device)
    
    def _create_vectors(self, batch_size=None):
        """テストベクトル作成"""
        batch_size = batch_size or self.batch_size
        # 原点に近いベクトル（一般的な概念）
        u = torch.randn(batch_size, self.d_model, device=self.device) * 0.3
        # 境界に近いベクトル（具体的な概念）
        v = torch.randn(batch_size, self.d_model, device=self.device) * 0.7
        return u, v
    
    def test_forward_shape(self):
        """出力形状のテスト"""
        module = self._create_module()
        u, v = self._create_vectors()
        
        penalty, aperture = module(u, v)
        
        assert penalty.shape == (self.batch_size,), f"Expected ({self.batch_size},), got {penalty.shape}"
    
    def test_check_entailment_output(self):
        """check_entailmentの出力テスト"""
        module = self._create_module()
        u, v = self._create_vectors()
        
        score, penalty, diagnostics = module.check_entailment(u, v)
        
        assert score.shape == (self.batch_size,)
        assert penalty.shape == (self.batch_size,)
        assert 'theta' in diagnostics
        assert 'aperture' in diagnostics
    
    def test_numerical_stability(self):
        """数値安定性のテスト"""
        module = self._create_module()
        u, v = self._create_vectors()
        
        penalty, aperture = module(u, v)
        
        assert not torch.isnan(penalty).any(), "Penalty contains NaN"
        assert not torch.isinf(penalty).any(), "Penalty contains Inf"
    
    def test_boundary_vectors(self):
        """境界に近いベクトルでの安定性テスト"""
        module = self._create_module()
        
        # 境界に非常に近いベクトル
        u = torch.randn(self.batch_size, self.d_model, device=self.device)
        u = F.normalize(u, dim=-1) * 0.98
        v = torch.randn(self.batch_size, self.d_model, device=self.device)
        v = F.normalize(v, dim=-1) * 0.99
        
        penalty, aperture = module(u, v)
        
        assert not torch.isnan(penalty).any(), "Penalty contains NaN at boundary"
        assert not torch.isinf(penalty).any(), "Penalty contains Inf at boundary"
    
    def test_zero_vectors(self):
        """ゼロベクトル付近での安定性テスト"""
        module = self._create_module()
        
        # 原点に非常に近いベクトル
        u = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.001
        v = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        
        penalty, aperture = module(u, v)
        
        assert not torch.isnan(penalty).any(), "Penalty contains NaN near origin"


class TestLogicalOperations:
    """論理演算のテスト"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.device = 'cuda' if cuda_available else 'cpu'
        self.d_model = 64
        self.batch_size = 8
    
    def _create_module(self):
        """モジュール作成"""
        try:
            from src.models.phase8.entailment_cones import (
                EntailmentCones, EntailmentConeConfig
            )
        except ImportError:
            pytest.skip("EntailmentCones not available")
        
        config = EntailmentConeConfig(d_model=self.d_model)
        return EntailmentCones(config).to(self.device)
    
    def test_logical_and_shape(self):
        """AND演算の出力形状テスト"""
        module = self._create_module()
        
        x = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        y = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        
        result = module.logical_and(x, y)
        
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"
    
    def test_logical_or_shape(self):
        """OR演算の出力形状テスト"""
        module = self._create_module()
        
        x = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        y = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        
        result = module.logical_or(x, y)
        
        assert result.shape == x.shape, f"Expected {x.shape}, got {result.shape}"
    
    def test_logical_and_numerical_stability(self):
        """AND演算の数値安定性テスト"""
        module = self._create_module()
        
        x = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        y = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        
        result = module.logical_and(x, y)
        
        assert not torch.isnan(result).any(), "AND result contains NaN"
        assert not torch.isinf(result).any(), "AND result contains Inf"
    
    def test_logical_or_numerical_stability(self):
        """OR演算の数値安定性テスト"""
        module = self._create_module()
        
        x = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        y = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.5
        
        result = module.logical_or(x, y)
        
        assert not torch.isnan(result).any(), "OR result contains NaN"
        assert not torch.isinf(result).any(), "OR result contains Inf"
    
    def test_logical_or_boundary(self):
        """OR演算が境界を超えないことを確認"""
        module = self._create_module()
        c = module.config.curvature
        max_norm = 1.0 / math.sqrt(c) - 1e-6
        
        x = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.8
        y = torch.randn(self.batch_size, self.d_model, device=self.device) * 0.8
        
        result = module.logical_or(x, y)
        result_norm = result.norm(dim=-1)
        
        assert (result_norm <= max_norm + 1e-5).all(), \
            f"OR result exceeds boundary: max norm = {result_norm.max().item()}"


class TestEntailmentScoreRange:
    """
    Property 1: Entailment Score Range
    
    **Property 1: Entailment Score Range**
    **Validates: Requirements 1.5**
    
    エンテイルメントスコアは常に[0, 1]の範囲内であるべき。
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.device = 'cuda' if cuda_available else 'cpu'
        self.d_model = 64
    
    def _create_module(self):
        """モジュール作成"""
        try:
            from src.models.phase8.entailment_cones import (
                EntailmentCones, EntailmentConeConfig
            )
        except ImportError:
            pytest.skip("EntailmentCones not available")
        
        config = EntailmentConeConfig(d_model=self.d_model)
        return EntailmentCones(config).to(self.device)
    
    def test_score_range_random_vectors(self):
        """
        ランダムベクトルでのスコア範囲テスト
        
        **Property 1: Entailment Score Range**
        **Validates: Requirements 1.5**
        """
        module = self._create_module()
        
        # 多数のランダムベクトルでテスト
        for _ in range(10):
            batch_size = 32
            u = torch.randn(batch_size, self.d_model, device=self.device) * 0.5
            v = torch.randn(batch_size, self.d_model, device=self.device) * 0.5
            
            score, penalty, _ = module.check_entailment(u, v)
            
            assert (score >= 0).all(), f"Score below 0: min = {score.min().item()}"
            assert (score <= 1).all(), f"Score above 1: max = {score.max().item()}"
    
    def test_score_range_extreme_vectors(self):
        """
        極端なベクトルでのスコア範囲テスト
        
        **Property 1: Entailment Score Range**
        **Validates: Requirements 1.5**
        """
        module = self._create_module()
        batch_size = 32
        
        # 原点に近いベクトル
        u_near_origin = torch.randn(batch_size, self.d_model, device=self.device) * 0.01
        # 境界に近いベクトル
        v_near_boundary = torch.randn(batch_size, self.d_model, device=self.device)
        v_near_boundary = F.normalize(v_near_boundary, dim=-1) * 0.98
        
        score, _, _ = module.check_entailment(u_near_origin, v_near_boundary)
        
        assert (score >= 0).all(), f"Score below 0: min = {score.min().item()}"
        assert (score <= 1).all(), f"Score above 1: max = {score.max().item()}"


class TestApertureMonotonicity:
    """
    Property 2: Aperture Monotonicity
    
    **Property 2: Aperture Monotonicity**
    **Validates: Requirements 1.2**
    
    アパーチャネットワーク使用時、原点に近いベクトルほど
    大きなアパーチャを持つべき（より一般的な概念）。
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.device = 'cuda' if cuda_available else 'cpu'
        self.d_model = 64
    
    def _create_module_with_network(self):
        """アパーチャネットワーク付きモジュール作成"""
        try:
            from src.models.phase8.entailment_cones import (
                EntailmentCones, EntailmentConeConfig
            )
        except ImportError:
            pytest.skip("EntailmentCones not available")
        
        config = EntailmentConeConfig(
            d_model=self.d_model,
            use_aperture_network=True,
        )
        return EntailmentCones(config).to(self.device)
    
    def test_aperture_network_output_range(self):
        """
        アパーチャネットワークの出力範囲テスト
        
        **Property 2: Aperture Monotonicity**
        **Validates: Requirements 1.2**
        """
        module = self._create_module_with_network()
        batch_size = 32
        
        x = torch.randn(batch_size, self.d_model, device=self.device) * 0.5
        aperture = module.compute_aperture(x)
        
        assert (aperture >= module.config.aperture_min).all(), \
            f"Aperture below min: {aperture.min().item()}"
        assert (aperture <= module.config.aperture_max).all(), \
            f"Aperture above max: {aperture.max().item()}"


class TestConfigurationRoundTrip:
    """
    Property 3: Configuration Round-Trip
    
    **Property 3: Entailment Cone Configuration Round-Trip**
    **Validates: Requirements 1.7**
    
    設定をJSONにシリアライズしてデシリアライズすると、
    元の設定と同等になるべき。
    """
    
    def test_config_round_trip(self):
        """
        設定のラウンドトリップテスト
        
        **Property 3: Entailment Cone Configuration Round-Trip**
        **Validates: Requirements 1.7**
        """
        try:
            from src.models.phase8.entailment_cones import EntailmentConeConfig
        except ImportError:
            pytest.skip("EntailmentConeConfig not available")
        
        # 元の設定
        original = EntailmentConeConfig(
            d_model=128,
            initial_aperture=0.7,
            aperture_min=0.2,
            aperture_max=1.5,
            curvature=2.0,
            use_learnable_aperture=True,
            use_aperture_network=False,
        )
        
        # シリアライズ→デシリアライズ
        json_str = original.to_json()
        restored = EntailmentConeConfig.from_json(json_str)
        
        # 各フィールドを比較
        assert restored.d_model == original.d_model
        assert restored.initial_aperture == original.initial_aperture
        assert restored.aperture_min == original.aperture_min
        assert restored.aperture_max == original.aperture_max
        assert restored.curvature == original.curvature
        assert restored.use_learnable_aperture == original.use_learnable_aperture
        assert restored.use_aperture_network == original.use_aperture_network
    
    def test_module_round_trip(self):
        """
        モジュールのラウンドトリップテスト
        
        **Property 3: Entailment Cone Configuration Round-Trip**
        **Validates: Requirements 1.7**
        """
        try:
            from src.models.phase8.entailment_cones import (
                EntailmentCones, EntailmentConeConfig
            )
        except ImportError:
            pytest.skip("EntailmentCones not available")
        
        # 元のモジュール
        config = EntailmentConeConfig(
            d_model=64,
            initial_aperture=0.5,
            curvature=1.5,
        )
        original = EntailmentCones(config)
        
        # シリアライズ→デシリアライズ
        json_str = original.to_json()
        restored = EntailmentCones.from_json(json_str)
        
        # 設定を比較
        assert restored.config.d_model == original.config.d_model
        assert restored.config.initial_aperture == original.config.initial_aperture
        assert restored.config.curvature == original.config.curvature
    
    def test_pretty_print(self):
        """pretty_printのテスト"""
        try:
            from src.models.phase8.entailment_cones import EntailmentConeConfig
        except ImportError:
            pytest.skip("EntailmentConeConfig not available")
        
        config = EntailmentConeConfig(d_model=128)
        pretty = config.pretty_print()
        
        assert "EntailmentConeConfig" in pretty
        assert "d_model: 128" in pretty


class TestApertureNetwork:
    """ApertureNetworkのテスト"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """テストセットアップ"""
        self.device = 'cuda' if cuda_available else 'cpu'
        self.d_model = 64
    
    def test_aperture_network_forward(self):
        """ApertureNetworkのforward passテスト"""
        try:
            from src.models.phase8.entailment_cones import ApertureNetwork
        except ImportError:
            pytest.skip("ApertureNetwork not available")
        
        network = ApertureNetwork(
            d_model=self.d_model,
            hidden_dim=32,
            aperture_min=0.1,
            aperture_max=2.0,
        ).to(self.device)
        
        x = torch.randn(8, self.d_model, device=self.device)
        aperture = network(x)
        
        assert aperture.shape == (8,)
        assert (aperture >= 0.1).all()
        assert (aperture <= 2.0).all()
    
    def test_aperture_network_gradient(self):
        """ApertureNetworkの勾配テスト"""
        try:
            from src.models.phase8.entailment_cones import ApertureNetwork
        except ImportError:
            pytest.skip("ApertureNetwork not available")
        
        network = ApertureNetwork(d_model=self.d_model).to(self.device)
        
        x = torch.randn(8, self.d_model, device=self.device, requires_grad=True)
        aperture = network(x)
        loss = aperture.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
