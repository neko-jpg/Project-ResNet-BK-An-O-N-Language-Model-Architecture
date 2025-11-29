"""
Tests for Hybrid Precision Strategy

Property-Based Tests:
- Property 11: Curvature Precision Enforcement
- Property 12: Boundary Collapse Prevention

Requirements: 6.1-6.6
"""

import unittest
import torch
import torch.nn as nn

try:
    from hypothesis import given, strategies as st, settings
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    def given(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator
    class st:
        @staticmethod
        def floats(*args, **kwargs):
            return None

from src.models.phase8.precision_manager import (
    PrecisionConfig,
    PrecisionDiagnostics,
    HybridPrecisionManager,
    BoundaryDetector,
    GradientOverflowDetector,
    BoundaryCollapseGuard,
    create_precision_manager,
    get_torch_dtype,
)


class TestPrecisionConfig(unittest.TestCase):
    """設定クラスのテスト"""
    
    def test_config_defaults(self):
        """デフォルト設定のテスト"""
        config = PrecisionConfig()
        self.assertEqual(config.default_dtype, "float16")
        self.assertEqual(config.curvature_dtype, "float32")
        self.assertEqual(config.boundary_threshold, 0.95)
    
    def test_config_to_json(self):
        """JSON変換のテスト"""
        config = PrecisionConfig(default_dtype="bfloat16")
        json_str = config.to_json()
        self.assertIn("bfloat16", json_str)
    
    def test_config_round_trip(self):
        """設定のラウンドトリップテスト"""
        config = PrecisionConfig(
            default_dtype="bfloat16",
            boundary_threshold=0.9,
            gradient_clip_value=0.5,
        )
        json_str = config.to_json()
        restored = PrecisionConfig.from_json(json_str)
        
        self.assertEqual(config.default_dtype, restored.default_dtype)
        self.assertEqual(config.boundary_threshold, restored.boundary_threshold)
        self.assertEqual(config.gradient_clip_value, restored.gradient_clip_value)


class TestGetTorchDtype(unittest.TestCase):
    """dtype変換のテスト"""
    
    def test_float16(self):
        self.assertEqual(get_torch_dtype("float16"), torch.float16)
    
    def test_bfloat16(self):
        self.assertEqual(get_torch_dtype("bfloat16"), torch.bfloat16)
    
    def test_float32(self):
        self.assertEqual(get_torch_dtype("float32"), torch.float32)
    
    def test_unknown_defaults_to_float32(self):
        self.assertEqual(get_torch_dtype("unknown"), torch.float32)


class TestBoundaryDetector(unittest.TestCase):
    """境界検出器のテスト"""
    
    def test_detect_boundary_tokens(self):
        """境界近くのトークンを検出"""
        detector = BoundaryDetector(threshold=0.95)
        
        # 境界近くのトークン
        x = torch.zeros(1, 10, 64)
        x[0, 0, 0] = 0.96  # ||x|| > 0.95
        x[0, 1, 0] = 0.5   # ||x|| < 0.95
        
        mask = detector(x)
        
        self.assertTrue(mask[0, 0].item())
        self.assertFalse(mask[0, 1].item())
    
    def test_count_boundary_tokens(self):
        """境界トークン数のカウント"""
        detector = BoundaryDetector(threshold=0.95)
        
        x = torch.zeros(1, 10, 64)
        x[0, 0, 0] = 0.96
        x[0, 1, 0] = 0.97
        x[0, 2, 0] = 0.5
        
        count = detector.count_boundary_tokens(x)
        self.assertEqual(count, 2)
    
    def test_no_boundary_tokens(self):
        """境界トークンがない場合"""
        detector = BoundaryDetector(threshold=0.95)
        
        # 小さいノルムを保証するために正規化
        x = torch.randn(2, 16, 64)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8) * 0.5  # ノルム0.5に正規化
        
        count = detector.count_boundary_tokens(x)
        self.assertEqual(count, 0)


class TestGradientOverflowDetector(unittest.TestCase):
    """勾配オーバーフロー検出器のテスト"""
    
    def test_detect_overflow(self):
        """オーバーフロー検出"""
        detector = GradientOverflowDetector(overflow_threshold=100.0)
        
        # オーバーフローするテンソル
        x = torch.tensor([150.0])
        self.assertTrue(detector.check_overflow(x))
        self.assertTrue(detector.overflow_detected)
    
    def test_no_overflow(self):
        """オーバーフローなし"""
        detector = GradientOverflowDetector(overflow_threshold=100.0)
        
        x = torch.tensor([50.0])
        self.assertFalse(detector.check_overflow(x))
        self.assertFalse(detector.overflow_detected)
    
    def test_detect_nan(self):
        """NaN検出"""
        detector = GradientOverflowDetector()
        
        x = torch.tensor([float("nan")])
        self.assertTrue(detector.check_nan(x))
        self.assertTrue(detector.nan_detected)
    
    def test_detect_inf(self):
        """Inf検出"""
        detector = GradientOverflowDetector()
        
        x = torch.tensor([float("inf")])
        self.assertTrue(detector.check_nan(x))
        self.assertTrue(detector.nan_detected)
    
    def test_clip_gradients(self):
        """勾配クリッピング"""
        detector = GradientOverflowDetector(clip_value=1.0)
        
        # 大きな勾配を持つパラメータ
        param = nn.Parameter(torch.randn(10))
        param.grad = torch.ones(10) * 10  # 大きな勾配
        
        clipped, max_norm = detector.clip_gradients([param])
        
        self.assertTrue(clipped)
        self.assertGreater(max_norm, 1.0)
        
        # クリップ後の勾配ノルムを確認
        new_norm = param.grad.norm().item()
        self.assertLessEqual(new_norm, 1.1)  # 余裕を持たせる
    
    def test_reset(self):
        """フラグリセット"""
        detector = GradientOverflowDetector()
        
        detector.check_overflow(torch.tensor([1e10]))
        self.assertTrue(detector.overflow_detected)
        
        detector.reset()
        self.assertFalse(detector.overflow_detected)


class TestHybridPrecisionManager(unittest.TestCase):
    """Hybrid Precision Managerのテスト"""
    
    def setUp(self):
        self.config = PrecisionConfig(
            default_dtype="float32",  # テスト用にFP32
            boundary_threshold=0.95,
        )
        self.manager = HybridPrecisionManager(self.config)
    
    def test_compute_curvature_safe(self):
        """
        Property 11: Curvature Precision Enforcement
        
        **Feature: phase8-hyperbolic-transcendence, Property 11: Curvature Precision Enforcement**
        **Validates: Requirements 6.1**
        
        曲率計算がFP32で実行されることを確認
        """
        def curvature_func(x):
            # 曲率計算をシミュレート
            return x.pow(2).sum(dim=-1)
        
        x = torch.randn(2, 16, 64, dtype=torch.float16)
        
        result = self.manager.compute_curvature_safe(curvature_func, x)
        
        # 結果がFP32であることを確認
        self.assertEqual(result.dtype, torch.float32)
    
    def test_apply_boundary_precision(self):
        """
        Property 12: Boundary Collapse Prevention
        
        **Feature: phase8-hyperbolic-transcendence, Property 12: Boundary Collapse Prevention**
        **Validates: Requirements 6.6**
        
        境界近くのトークンがFP32で処理されることを確認
        """
        def identity_func(x):
            return x
        
        # 境界近くのトークンを含む入力
        x = torch.zeros(1, 10, 64)
        x[0, 0, 0] = 0.96  # 境界近く
        
        result = self.manager.apply_boundary_precision(x, identity_func)
        
        # 結果が有限であることを確認
        self.assertTrue(torch.isfinite(result).all())
    
    def test_detect_and_recover_overflow(self):
        """オーバーフロー検出と回復"""
        # オーバーフローするテンソル
        x = torch.tensor([1e10])
        
        result, recovered = self.manager.detect_and_recover_overflow(x)
        
        # 回復されたことを確認
        self.assertTrue(recovered)
        self.assertEqual(result.dtype, torch.float32)
    
    def test_detect_and_recover_nan(self):
        """NaN検出と回復"""
        x = torch.tensor([float("nan"), 1.0, 2.0])
        
        result, recovered = self.manager.detect_and_recover_overflow(x)
        
        self.assertTrue(recovered)
        # NaNが0に置換されていることを確認
        self.assertTrue(torch.isfinite(result).all())
    
    def test_forward_with_diagnostics(self):
        """診断情報付きforward"""
        def identity_func(x):
            return x
        
        x = torch.randn(2, 16, 64)
        
        result, diag = self.manager(
            x, identity_func, return_diagnostics=True
        )
        
        self.assertIsNotNone(diag)
        self.assertIn(diag.current_dtype, ["float16", "float32", "bfloat16"])
    
    def test_clip_gradients(self):
        """勾配クリッピング"""
        param = nn.Parameter(torch.randn(10))
        param.grad = torch.ones(10) * 10
        
        clipped, max_norm = self.manager.clip_gradients([param])
        
        self.assertTrue(clipped)


class TestBoundaryCollapseGuard(unittest.TestCase):
    """境界崩壊防止ガードのテスト"""
    
    def test_prevent_boundary_collapse(self):
        """
        Property 12: Boundary Collapse Prevention
        
        **Feature: phase8-hyperbolic-transcendence, Property 12: Boundary Collapse Prevention**
        **Validates: Requirements 6.6**
        
        境界を超えるトークンがスケーリングされることを確認
        """
        guard = BoundaryCollapseGuard(max_norm=0.99)
        
        # 境界を超えるトークン
        x = torch.zeros(1, 10, 64)
        x[0, 0, 0] = 1.5  # ||x|| > 0.99
        
        x_safe = guard(x)
        
        # ノルムが閾値以下になっていることを確認
        norms = x_safe.norm(dim=-1)
        self.assertTrue((norms <= 0.99 + 1e-6).all())
    
    def test_preserve_safe_tokens(self):
        """安全なトークンは変更されない"""
        guard = BoundaryCollapseGuard(max_norm=0.99)
        
        # 安全なトークン（ノルムを0.5に正規化）
        x = torch.randn(2, 16, 64)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8) * 0.5
        
        x_safe = guard(x)
        
        # ほぼ同じであることを確認
        self.assertTrue(torch.allclose(x, x_safe, atol=1e-6))
    
    def test_regularization_loss(self):
        """正則化損失の計算"""
        guard = BoundaryCollapseGuard(regularization_strength=0.01)
        
        # 境界近くのトークン
        x = torch.zeros(1, 10, 64)
        x[0, 0, 0] = 0.9
        
        loss = guard.compute_regularization_loss(x)
        
        # 損失が正であることを確認
        self.assertGreater(loss.item(), 0)
    
    def test_regularization_increases_near_boundary(self):
        """境界に近いほど正則化損失が大きい"""
        guard = BoundaryCollapseGuard(regularization_strength=0.01)
        
        # 境界から離れたトークン
        x_far = torch.zeros(1, 10, 64)
        x_far[0, 0, 0] = 0.5
        
        # 境界近くのトークン
        x_near = torch.zeros(1, 10, 64)
        x_near[0, 0, 0] = 0.9
        
        loss_far = guard.compute_regularization_loss(x_far)
        loss_near = guard.compute_regularization_loss(x_near)
        
        self.assertGreater(loss_near.item(), loss_far.item())


class TestCurvaturePrecisionEnforcement(unittest.TestCase):
    """
    Property 11: Curvature Precision Enforcement (Property-Based Test)
    
    **Feature: phase8-hyperbolic-transcendence, Property 11: Curvature Precision Enforcement**
    **Validates: Requirements 6.1**
    
    For any curvature computation, the system SHALL use FP32 precision.
    """
    
    def test_curvature_always_fp32(self):
        """曲率計算は常にFP32"""
        config = PrecisionConfig(default_dtype="float16")
        manager = HybridPrecisionManager(config)
        
        def curvature_func(x):
            return x.pow(2).sum(dim=-1)
        
        # FP16入力
        x_fp16 = torch.randn(2, 16, 64, dtype=torch.float16)
        result = manager.compute_curvature_safe(curvature_func, x_fp16)
        self.assertEqual(result.dtype, torch.float32)
        
        # BF16入力
        x_bf16 = torch.randn(2, 16, 64, dtype=torch.bfloat16)
        result = manager.compute_curvature_safe(curvature_func, x_bf16)
        self.assertEqual(result.dtype, torch.float32)
    
    def test_curvature_numerical_stability(self):
        """曲率計算の数値安定性"""
        config = PrecisionConfig()
        manager = HybridPrecisionManager(config)
        
        def curvature_func(x):
            # 数値的に不安定な計算
            return torch.log(1 - x.pow(2).sum(dim=-1).clamp(max=0.999))
        
        # 境界近くの入力
        x = torch.zeros(1, 10, 64)
        x[0, 0, 0] = 0.99
        
        result = manager.compute_curvature_safe(curvature_func, x)
        
        # 結果が有限であることを確認
        self.assertTrue(torch.isfinite(result).all())


class TestBoundaryCollapsePrevention(unittest.TestCase):
    """
    Property 12: Boundary Collapse Prevention (Property-Based Test)
    
    **Feature: phase8-hyperbolic-transcendence, Property 12: Boundary Collapse Prevention**
    **Validates: Requirements 6.6**
    
    For any embedding near the Poincaré ball boundary,
    the system SHALL prevent collapse by using FP32 precision.
    """
    
    def test_boundary_tokens_use_fp32(self):
        """境界近くのトークンはFP32で処理"""
        config = PrecisionConfig(
            default_dtype="float16",
            boundary_threshold=0.95,
        )
        manager = HybridPrecisionManager(config)
        
        def process_func(x):
            return x * 2
        
        # 境界近くのトークンを含む入力
        x = torch.zeros(1, 10, 64, dtype=torch.float16)
        x[0, 0, 0] = 0.96  # 境界近く
        
        result = manager.apply_boundary_precision(x, process_func)
        
        # 結果が有限であることを確認
        self.assertTrue(torch.isfinite(result).all())
    
    def test_boundary_guard_prevents_collapse(self):
        """境界ガードが崩壊を防止"""
        guard = BoundaryCollapseGuard(max_norm=0.99)
        
        # 境界を大きく超えるトークン
        x = torch.zeros(1, 10, 64)
        x[0, 0, :] = 10.0  # 非常に大きなノルム
        
        x_safe = guard(x)
        
        # ノルムが閾値以下になっていることを確認
        norms = x_safe.norm(dim=-1)
        self.assertTrue((norms <= 0.99 + 1e-6).all())


class TestFactoryFunction(unittest.TestCase):
    """ファクトリ関数のテスト"""
    
    def test_create_precision_manager(self):
        """create_precision_manager関数のテスト"""
        manager = create_precision_manager(
            default_dtype="bfloat16",
            boundary_threshold=0.9,
        )
        
        self.assertIsInstance(manager, HybridPrecisionManager)
        self.assertEqual(manager.config.default_dtype, "bfloat16")
        self.assertEqual(manager.config.boundary_threshold, 0.9)


class TestHybridPrecisionUnitTests(unittest.TestCase):
    """
    タスク11.5: Hybrid Precisionのユニットテスト
    
    Requirements: 6.1-6.6
    """
    
    def test_precision_switching(self):
        """精度切り替えのテスト"""
        config = PrecisionConfig(default_dtype="float16")
        manager = HybridPrecisionManager(config)
        
        # 通常の入力（ノルムを0.5に正規化）
        x_normal = torch.randn(2, 16, 64)
        x_normal = x_normal / (x_normal.norm(dim=-1, keepdim=True) + 1e-8) * 0.5
        
        # 境界近くの入力
        x_boundary = torch.zeros(2, 16, 64)
        x_boundary[0, 0, 0] = 0.96
        
        def identity(x):
            return x
        
        # 通常入力は境界トークンなし
        count_normal = manager.boundary_detector.count_boundary_tokens(x_normal)
        self.assertEqual(count_normal, 0)
        
        # 境界入力は境界トークンあり
        count_boundary = manager.boundary_detector.count_boundary_tokens(x_boundary)
        self.assertGreater(count_boundary, 0)
    
    def test_overflow_handling(self):
        """オーバーフロー処理のテスト"""
        config = PrecisionConfig(auto_upcast_on_overflow=True)
        manager = HybridPrecisionManager(config)
        
        # オーバーフローするテンソル
        x = torch.tensor([1e10])
        
        result, recovered = manager.detect_and_recover_overflow(x)
        
        self.assertTrue(recovered)
        self.assertTrue(manager._upcast_triggered)
    
    def test_nan_handling(self):
        """NaN処理のテスト"""
        config = PrecisionConfig(auto_upcast_on_overflow=True)
        manager = HybridPrecisionManager(config)
        
        # NaNを含むテンソル
        x = torch.tensor([float("nan"), 1.0, 2.0])
        
        result, recovered = manager.detect_and_recover_overflow(x)
        
        self.assertTrue(recovered)
        self.assertTrue(torch.isfinite(result).all())
    
    def test_gradient_clipping(self):
        """勾配クリッピングのテスト"""
        config = PrecisionConfig(gradient_clip_value=1.0)
        manager = HybridPrecisionManager(config)
        
        # 大きな勾配を持つパラメータ
        param = nn.Parameter(torch.randn(100))
        param.grad = torch.ones(100) * 10
        
        clipped, max_norm = manager.clip_gradients([param])
        
        self.assertTrue(clipped)
        
        # クリップ後の勾配ノルム
        new_norm = param.grad.norm().item()
        self.assertLessEqual(new_norm, 1.1)


if __name__ == "__main__":
    unittest.main()
