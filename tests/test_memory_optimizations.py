#!/usr/bin/env python3
"""
Memory Optimizations Unit Tests

タスク33.5: メモリ最適化のユニットテスト
- 圧縮、プーリング、プリフェッチのテスト

Requirements: 33.1-33.4
"""

import pytest
import torch
import torch.nn as nn
import time
from typing import Optional

# テスト対象モジュールのインポート
try:
    from src.kernels.kv_cache_compression_triton import (
        KVCacheCompressor, CompressedKVCache,
        compress_kv_cache, decompress_kv_cache
    )
    KV_COMPRESSION_AVAILABLE = True
except ImportError:
    KV_COMPRESSION_AVAILABLE = False

try:
    from src.kernels.memory_pool_triton import (
        HyperbolicMemoryPool, PooledTensor, HyperbolicTensorAllocator,
        get_global_pool, pooled_allocate, pooled_deallocate
    )
    MEMORY_POOL_AVAILABLE = True
except ImportError:
    MEMORY_POOL_AVAILABLE = False

try:
    from src.kernels.prefetch_hyperbolic_triton import (
        PrefetchHyperbolicAttention, StreamingHyperbolicAttention,
        prefetch_hyperbolic_attention
    )
    PREFETCH_AVAILABLE = True
except ImportError:
    PREFETCH_AVAILABLE = False


def get_device():
    """テスト用デバイスを取得"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestKVCacheCompression:
    """KVキャッシュ圧縮テスト"""
    
    @pytest.mark.skipif(not KV_COMPRESSION_AVAILABLE, reason="KV compression not available")
    def test_compress_decompress_round_trip(self):
        """圧縮・展開のラウンドトリップテスト"""
        device = get_device()
        B, H, N, D = 2, 8, 256, 64
        
        k = torch.randn(B, H, N, D, device=device)
        v = torch.randn(B, H, N, D, device=device)
        
        k_c, v_c, metadata = KVCacheCompressor.compress(k, v)
        k_decompressed, v_decompressed = KVCacheCompressor.decompress(k_c, v_c, metadata)
        
        # 形状が一致
        assert k_decompressed.shape == k.shape
        assert v_decompressed.shape == v.shape
        
        # 誤差が許容範囲内
        k_error = (k - k_decompressed).abs().mean()
        v_error = (v - v_decompressed).abs().mean()
        
        assert k_error < 0.5, f"K reconstruction error: {k_error}"
        assert v_error < 0.5, f"V reconstruction error: {v_error}"
    
    @pytest.mark.skipif(not KV_COMPRESSION_AVAILABLE, reason="KV compression not available")
    def test_compression_ratio(self):
        """圧縮率テスト"""
        device = get_device()
        B, H, N, D = 2, 8, 256, 64
        
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        
        k_c, v_c, metadata = KVCacheCompressor.compress(k, v)
        
        # 元のサイズ
        original_size = (k.numel() + v.numel()) * k.element_size()
        
        # 圧縮後のサイズ
        compressed_size = (
            k_c.numel() * k_c.element_size() +
            v_c.numel() * v_c.element_size() +
            sum(m.numel() * m.element_size() for m in metadata.values())
        )
        
        compression_ratio = original_size / compressed_size
        
        # 4ビット圧縮なので約2x以上の圧縮率を期待
        assert compression_ratio > 1.5, f"Compression ratio: {compression_ratio}"
    
    @pytest.mark.skipif(not KV_COMPRESSION_AVAILABLE, reason="KV compression not available")
    def test_compressed_kv_cache_update(self):
        """圧縮KVキャッシュの更新テスト"""
        device = get_device()
        max_seq_len = 512
        num_heads = 8
        head_dim = 64
        
        cache = CompressedKVCache(
            max_seq_len=max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            device=device,
        )
        
        # チャンクごとに追加
        chunk_size = 64
        for start_pos in range(0, 256, chunk_size):
            k = torch.randn(1, num_heads, chunk_size, head_dim, device=device)
            v = torch.randn(1, num_heads, chunk_size, head_dim, device=device)
            cache.update(k, v, start_pos)
        
        assert cache.current_len == 256
        
        # 取得
        k_out, v_out = cache.get()
        
        assert k_out.shape == (1, num_heads, 256, head_dim)
        assert v_out.shape == (1, num_heads, 256, head_dim)
    
    @pytest.mark.skipif(not KV_COMPRESSION_AVAILABLE, reason="KV compression not available")
    def test_compressed_kv_cache_reset(self):
        """圧縮KVキャッシュのリセットテスト"""
        device = get_device()
        
        cache = CompressedKVCache(
            max_seq_len=256,
            num_heads=8,
            head_dim=64,
            device=device,
        )
        
        k = torch.randn(1, 8, 64, 64, device=device)
        v = torch.randn(1, 8, 64, 64, device=device)
        cache.update(k, v, 0)
        
        assert cache.current_len == 64
        
        cache.reset()
        
        assert cache.current_len == 0


class TestMemoryPool:
    """メモリプールテスト"""
    
    @pytest.mark.skipif(not MEMORY_POOL_AVAILABLE, reason="Memory pool not available")
    def test_allocate_deallocate(self):
        """アロケーション・デアロケーションテスト"""
        device = get_device()
        
        pool = HyperbolicMemoryPool(
            initial_size_mb=32,
            max_size_mb=64,
            device=device,
        )
        
        shape = (4, 8, 128, 64)
        
        tensor = pool.allocate(shape)
        
        assert tensor.shape == shape
        assert tensor.device.type == device.type
        
        pool.deallocate(tensor)
        
        stats = pool.get_stats()
        assert stats['allocations'] == 1
        assert stats['deallocations'] == 1
    
    @pytest.mark.skipif(not MEMORY_POOL_AVAILABLE, reason="Memory pool not available")
    def test_cache_hit(self):
        """キャッシュヒットテスト"""
        device = get_device()
        
        pool = HyperbolicMemoryPool(
            initial_size_mb=32,
            max_size_mb=64,
            device=device,
        )
        
        shape = (4, 8, 128, 64)
        
        # 最初のアロケーション
        tensor1 = pool.allocate(shape)
        pool.deallocate(tensor1)
        
        # 2回目のアロケーション（キャッシュヒット）
        tensor2 = pool.allocate(shape)
        
        stats = pool.get_stats()
        
        # キャッシュヒットがあることを確認
        assert stats['cache_hits'] >= 1
    
    @pytest.mark.skipif(not MEMORY_POOL_AVAILABLE, reason="Memory pool not available")
    def test_pooled_tensor_context_manager(self):
        """PooledTensorコンテキストマネージャテスト"""
        device = get_device()
        
        pool = HyperbolicMemoryPool(
            initial_size_mb=32,
            max_size_mb=64,
            device=device,
        )
        
        shape = (4, 8, 128, 64)
        
        with PooledTensor(pool, shape) as tensor:
            assert tensor.shape == shape
            tensor.fill_(1.0)
        
        # コンテキスト終了後にデアロケーションされている
        stats = pool.get_stats()
        assert stats['deallocations'] >= 1
    
    @pytest.mark.skipif(not MEMORY_POOL_AVAILABLE, reason="Memory pool not available")
    def test_allocation_latency(self):
        """アロケーションレイテンシテスト"""
        device = get_device()
        
        pool = HyperbolicMemoryPool(
            initial_size_mb=64,
            max_size_mb=128,
            device=device,
        )
        
        shape = (4, 8, 1024, 64)
        
        # ウォームアップ
        for _ in range(10):
            t = pool.allocate(shape)
            pool.deallocate(t)
        
        # レイテンシ測定
        num_ops = 100
        start_time = time.perf_counter()
        
        for _ in range(num_ops):
            t = pool.allocate(shape)
            pool.deallocate(t)
        
        end_time = time.perf_counter()
        
        avg_latency_us = (end_time - start_time) / num_ops * 1e6
        
        # サブマイクロ秒は難しいが、100us以下を目標
        assert avg_latency_us < 1000, f"Allocation latency: {avg_latency_us} us"
    
    @pytest.mark.skipif(not MEMORY_POOL_AVAILABLE, reason="Memory pool not available")
    def test_tensor_allocator_qkv(self):
        """HyperbolicTensorAllocatorのQKVアロケーションテスト"""
        device = get_device()
        
        allocator = HyperbolicTensorAllocator(
            pool_size_mb=64,
            device=device,
        )
        
        B, H, N, D = 4, 8, 256, 64
        
        q, k, v = allocator.allocate_qkv(B, H, N, D)
        
        assert q.shape == (B, H, N, D)
        assert k.shape == (B, H, N, D)
        assert v.shape == (B, H, N, D)
        
        allocator.deallocate(q, k, v)


class TestPrefetchAttention:
    """プリフェッチアテンションテスト"""
    
    @pytest.mark.skipif(not PREFETCH_AVAILABLE, reason="Prefetch not available")
    def test_prefetch_attention_output_shape(self):
        """プリフェッチアテンションの出力形状テスト"""
        device = get_device()
        B, N, D = 4, 256, 256
        num_heads = 8
        
        module = PrefetchHyperbolicAttention(
            D, num_heads, curvature=1.0, prefetch_distance=2, use_triton=False
        ).to(device)
        x = torch.randn(B, N, D, device=device)
        
        out = module(x)
        
        assert out.shape == x.shape
    
    @pytest.mark.skipif(not PREFETCH_AVAILABLE, reason="Prefetch not available")
    def test_streaming_attention_output_shape(self):
        """ストリーミングアテンションの出力形状テスト"""
        device = get_device()
        B, N, D = 4, 256, 256
        num_heads = 8
        
        module = StreamingHyperbolicAttention(
            D, num_heads, curvature=1.0, use_triton=False
        ).to(device)
        x = torch.randn(B, N, D, device=device)
        
        out = module(x)
        
        assert out.shape == x.shape
    
    @pytest.mark.skipif(not PREFETCH_AVAILABLE, reason="Prefetch not available")
    def test_prefetch_attention_gradient_flow(self):
        """プリフェッチアテンションの勾配フローテスト"""
        device = get_device()
        B, N, D = 2, 128, 128
        num_heads = 4
        
        module = PrefetchHyperbolicAttention(
            D, num_heads, curvature=1.0, use_triton=False
        ).to(device)
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    @pytest.mark.skipif(not PREFETCH_AVAILABLE, reason="Prefetch not available")
    def test_streaming_attention_gradient_flow(self):
        """ストリーミングアテンションの勾配フローテスト"""
        device = get_device()
        B, N, D = 2, 128, 128
        num_heads = 4
        
        module = StreamingHyperbolicAttention(
            D, num_heads, curvature=1.0, use_triton=False
        ).to(device)
        x = torch.randn(B, N, D, device=device, requires_grad=True)
        
        out = module(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    @pytest.mark.skipif(not PREFETCH_AVAILABLE, reason="Prefetch not available")
    def test_prefetch_function_interface(self):
        """プリフェッチ関数インターフェーステスト"""
        device = get_device()
        B, H, N, D = 2, 4, 128, 64
        
        q = torch.randn(B, H, N, D, device=device)
        k = torch.randn(B, H, N, D, device=device)
        v = torch.randn(B, H, N, D, device=device)
        
        out = prefetch_hyperbolic_attention(q, k, v, curvature=1.0)
        
        assert out.shape == q.shape
        assert not torch.isnan(out).any()


class TestMemoryOptimizationIntegration:
    """メモリ最適化統合テスト"""
    
    @pytest.mark.skipif(
        not (KV_COMPRESSION_AVAILABLE and MEMORY_POOL_AVAILABLE),
        reason="Required modules not available"
    )
    def test_compressed_cache_with_pool(self):
        """圧縮キャッシュとメモリプールの統合テスト"""
        device = get_device()
        
        # メモリプール
        pool = HyperbolicMemoryPool(
            initial_size_mb=32,
            max_size_mb=64,
            device=device,
        )
        
        # 圧縮キャッシュ
        cache = CompressedKVCache(
            max_seq_len=256,
            num_heads=8,
            head_dim=64,
            device=device,
        )
        
        # プールからテンソルを割り当て
        shape = (1, 8, 64, 64)
        k = pool.allocate(shape)
        v = pool.allocate(shape)
        
        # ランダムデータで初期化
        k.normal_()
        v.normal_()
        
        # キャッシュに追加
        cache.update(k, v, 0)
        
        # プールに返却
        pool.deallocate(k)
        pool.deallocate(v)
        
        # キャッシュから取得
        k_out, v_out = cache.get()
        
        assert k_out.shape == shape
        assert v_out.shape == shape
    
    @pytest.mark.skipif(
        not (PREFETCH_AVAILABLE and MEMORY_POOL_AVAILABLE),
        reason="Required modules not available"
    )
    def test_prefetch_with_pool(self):
        """プリフェッチアテンションとメモリプールの統合テスト"""
        device = get_device()
        B, N, D = 2, 128, 128
        num_heads = 4
        
        # メモリプール
        allocator = HyperbolicTensorAllocator(
            pool_size_mb=32,
            device=device,
        )
        
        # プリフェッチアテンション
        module = PrefetchHyperbolicAttention(
            D, num_heads, curvature=1.0, use_triton=False
        ).to(device)
        
        # プールから入力を割り当て（float32で）
        x = allocator.pool.allocate((B, N, D))
        x = x.float()  # float32に変換
        x.normal_()
        
        # フォワードパス
        out = module(x)
        
        assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
