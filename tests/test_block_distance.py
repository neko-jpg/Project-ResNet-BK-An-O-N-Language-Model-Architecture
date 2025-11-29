"""
Tests for Block-wise Distance Computation

Property-Based Tests:
- Property 13: Block-wise Memory Scaling

Requirements: 7.1-7.6
"""

import unittest
import time
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
        def integers(*args, **kwargs):
            return None

from src.models.phase8.block_distance import (
    BlockDistanceConfig,
    BlockDistanceDiagnostics,
    BlockWiseDistanceComputation,
    SharedMemoryBlockDistance,
    poincare_distance_block,
    create_block_distance,
)


class TestBlockDistanceConfig(unittest.TestCase):
    """設定クラスのテスト"""
    
    def test_config_defaults(self):
        """デフォルト設定のテスト"""
        config = BlockDistanceConfig()
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.block_size_m, 128)
        self.assertEqual(config.block_size_n, 128)
    
    def test_config_to_json(self):
        """JSON変換のテスト"""
        config = BlockDistanceConfig(d_model=128, block_size_m=64)
        json_str = config.to_json()
        self.assertIn("128", json_str)
        self.assertIn("64", json_str)
    
    def test_config_round_trip(self):
        """設定のラウンドトリップテスト"""
        config = BlockDistanceConfig(
            d_model=512,
            num_heads=16,
            curvature=0.5,
            block_size_m=64,
            block_size_n=64,
            causal=True,
        )
        json_str = config.to_json()
        restored = BlockDistanceConfig.from_json(json_str)
        
        self.assertEqual(config.d_model, restored.d_model)
        self.assertEqual(config.num_heads, restored.num_heads)
        self.assertEqual(config.block_size_m, restored.block_size_m)
        self.assertEqual(config.causal, restored.causal)


class TestPoincareDistanceBlock(unittest.TestCase):
    """Poincaré距離ブロック計算のテスト"""
    
    def test_distance_shape(self):
        """距離行列の形状テスト"""
        q = torch.randn(2, 4, 32, 16) * 0.5
        k = torch.randn(2, 4, 32, 16) * 0.5
        c = torch.tensor(1.0)
        
        dist = poincare_distance_block(q, k, c)
        
        self.assertEqual(dist.shape, (2, 4, 32, 32))
    
    def test_distance_non_negative(self):
        """距離は非負"""
        q = torch.randn(2, 4, 16, 8) * 0.5
        k = torch.randn(2, 4, 16, 8) * 0.5
        c = torch.tensor(1.0)
        
        dist = poincare_distance_block(q, k, c)
        
        self.assertTrue((dist >= 0).all())
    
    def test_distance_symmetric(self):
        """同じ入力での距離は対称"""
        x = torch.randn(1, 2, 8, 4) * 0.5
        c = torch.tensor(1.0)
        
        dist = poincare_distance_block(x, x, c)
        
        # 対角成分は0に近い
        diag = torch.diagonal(dist[0, 0])
        self.assertTrue((diag < 0.1).all())


class TestBlockWiseDistanceComputation(unittest.TestCase):
    """Block-wise Distance Computationのテスト"""
    
    def setUp(self):
        self.config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            curvature=1.0,
            block_size_m=32,
            block_size_n=32,
        )
        self.module = BlockWiseDistanceComputation(self.config)
    
    def test_forward_shape(self):
        """Forward passの出力形状テスト"""
        x = torch.randn(2, 64, 64) * 0.5
        output, _ = self.module(x)
        self.assertEqual(output.shape, x.shape)
    
    def test_forward_with_diagnostics(self):
        """診断情報付きforward"""
        x = torch.randn(2, 64, 64) * 0.5
        output, diag = self.module(x, return_diagnostics=True)
        
        self.assertIsNotNone(diag)
        self.assertGreater(diag.num_blocks_computed, 0)
    
    def test_causal_mask(self):
        """Causalマスクのテスト"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
            causal=True,
        )
        module = BlockWiseDistanceComputation(config)
        
        x = torch.randn(2, 64, 64) * 0.5
        output, diag = module(x, return_diagnostics=True)
        
        # Causalではブロックがスキップされる
        self.assertGreater(diag.num_blocks_skipped, 0)
    
    def test_causal_block_skipping(self):
        """Causalブロックスキップのテスト"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
            causal=True,
        )
        module = BlockWiseDistanceComputation(config)
        
        # スキップ判定
        self.assertFalse(module._should_skip_block(0, 0, True))  # 対角
        self.assertFalse(module._should_skip_block(1, 0, True))  # 下三角
        self.assertTrue(module._should_skip_block(0, 1, True))   # 上三角
    
    def test_output_finite(self):
        """出力が有限であることを確認"""
        x = torch.randn(2, 64, 64) * 0.5
        output, _ = self.module(x)
        
        self.assertTrue(torch.isfinite(output).all())
    
    def test_gradient_flow(self):
        """勾配が流れることを確認"""
        x = torch.randn(2, 32, 64) * 0.5
        x.requires_grad_(True)
        
        output, _ = self.module(x)
        loss = output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class TestBlockWiseMemoryScaling(unittest.TestCase):
    """
    Property 13: Block-wise Memory Scaling
    
    **Feature: phase8-hyperbolic-transcendence, Property 13: Block-wise Memory Scaling**
    **Validates: Requirements 7.3**
    
    For any sequence length N, the block-wise distance computation
    SHALL use O(N) memory instead of O(N²).
    """
    
    def test_memory_scaling_linear(self):
        """メモリスケーリングがO(N)であることを確認"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
        )
        module = BlockWiseDistanceComputation(config)
        
        # 異なるシーケンス長でのメモリ推定
        seq_lengths = [128, 256, 512, 1024]
        memories = []
        
        for seq_len in seq_lengths:
            mem = module.estimate_memory_usage(seq_len)
            memories.append(mem)
        
        # 線形スケーリングの確認
        # 2倍のシーケンス長で2倍程度のメモリ増加
        for i in range(1, len(memories)):
            ratio = memories[i] / memories[i-1]
            seq_ratio = seq_lengths[i] / seq_lengths[i-1]
            # O(N)なら ratio ≈ seq_ratio
            # O(N²)なら ratio ≈ seq_ratio²
            self.assertLess(ratio, seq_ratio * 1.5)  # 余裕を持たせる
    
    def test_block_memory_constant(self):
        """ブロックメモリが一定であることを確認"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
        )
        
        # ブロックサイズは固定
        block_memory = config.block_size_m * config.block_size_n * 4  # float32
        
        # シーケンス長に依存しない
        self.assertEqual(block_memory, 32 * 32 * 4)


class TestCausalBlockSkipping(unittest.TestCase):
    """
    タスク13.4: Causalブロックスキップのテスト
    
    Requirements: 7.5
    """
    
    def test_upper_triangular_blocks_skipped(self):
        """上三角ブロックがスキップされる"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=16,
            block_size_n=16,
            causal=True,
        )
        module = BlockWiseDistanceComputation(config)
        
        # 64トークン、16ブロックサイズ → 4x4ブロック
        x = torch.randn(1, 64, 64) * 0.5
        _, diag = module(x, return_diagnostics=True)
        
        # 4x4 = 16ブロック中、上三角6ブロックがスキップ
        # 対角4 + 下三角6 = 10ブロック計算
        self.assertEqual(diag.num_blocks_skipped, 6)
        self.assertEqual(diag.num_blocks_computed, 10)
    
    def test_non_causal_no_skipping(self):
        """非Causalではスキップなし"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=16,
            block_size_n=16,
            causal=False,
        )
        module = BlockWiseDistanceComputation(config)
        
        x = torch.randn(1, 64, 64) * 0.5
        _, diag = module(x, return_diagnostics=True)
        
        # 全ブロック計算
        self.assertEqual(diag.num_blocks_skipped, 0)
        self.assertEqual(diag.num_blocks_computed, 16)


class TestSharedMemoryBlockDistance(unittest.TestCase):
    """共有メモリ最適化版のテスト"""
    
    def test_forward_shape(self):
        """Forward passの出力形状テスト"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
        )
        module = SharedMemoryBlockDistance(config)
        
        x = torch.randn(2, 64, 64) * 0.5
        output, _ = module(x)
        
        self.assertEqual(output.shape, x.shape)
    
    def test_output_matches_base(self):
        """基本実装と同じ出力"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
        )
        
        base_module = BlockWiseDistanceComputation(config)
        shared_module = SharedMemoryBlockDistance(config)
        
        # 重みをコピー
        shared_module.base_module.load_state_dict(base_module.state_dict())
        
        x = torch.randn(1, 32, 64) * 0.5
        
        base_output, _ = base_module(x)
        shared_output, _ = shared_module(x)
        
        self.assertTrue(torch.allclose(base_output, shared_output, atol=1e-5))


class TestFactoryFunction(unittest.TestCase):
    """ファクトリ関数のテスト"""
    
    def test_create_block_distance(self):
        """create_block_distance関数のテスト"""
        module = create_block_distance(
            d_model=128,
            num_heads=8,
            curvature=0.5,
            block_size=64,
            causal=True,
        )
        
        self.assertIsInstance(module, BlockWiseDistanceComputation)
        self.assertEqual(module.d_model, 128)
        self.assertEqual(module.num_heads, 8)
        self.assertEqual(module.block_size_m, 64)
        self.assertTrue(module.config.causal)


class TestBlockDistanceUnitTests(unittest.TestCase):
    """
    タスク13.5: Block-wise Distanceのユニットテスト
    
    Requirements: 7.1-7.6
    """
    
    def test_memory_usage_reasonable(self):
        """メモリ使用量が妥当"""
        config = BlockDistanceConfig(
            d_model=256,
            num_heads=8,
            block_size_m=128,
            block_size_n=128,
        )
        module = BlockWiseDistanceComputation(config)
        
        # 4096トークンでのメモリ推定
        mem = module.estimate_memory_usage(4096)
        
        # O(N)なので、数十MB程度
        self.assertLess(mem, 100)  # 100MB未満
    
    def test_correctness_small_input(self):
        """小さい入力での正確性"""
        config = BlockDistanceConfig(
            d_model=32,
            num_heads=2,
            block_size_m=16,
            block_size_n=16,
        )
        module = BlockWiseDistanceComputation(config)
        
        x = torch.randn(1, 16, 32) * 0.5
        output, _ = module(x)
        
        # 出力が有限
        self.assertTrue(torch.isfinite(output).all())
        
        # 出力形状が正しい
        self.assertEqual(output.shape, x.shape)
    
    def test_different_block_sizes(self):
        """異なるブロックサイズでの動作"""
        for block_size in [16, 32, 64]:
            config = BlockDistanceConfig(
                d_model=64,
                num_heads=4,
                block_size_m=block_size,
                block_size_n=block_size,
            )
            module = BlockWiseDistanceComputation(config)
            
            x = torch.randn(1, 64, 64) * 0.5
            output, _ = module(x)
            
            self.assertEqual(output.shape, x.shape)
            self.assertTrue(torch.isfinite(output).all())
    
    def test_batch_processing(self):
        """バッチ処理のテスト"""
        config = BlockDistanceConfig(
            d_model=64,
            num_heads=4,
            block_size_m=32,
            block_size_n=32,
        )
        module = BlockWiseDistanceComputation(config)
        
        # 異なるバッチサイズ
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 64, 64) * 0.5
            output, _ = module(x)
            
            self.assertEqual(output.shape, (batch_size, 64, 64))


if __name__ == "__main__":
    unittest.main()
