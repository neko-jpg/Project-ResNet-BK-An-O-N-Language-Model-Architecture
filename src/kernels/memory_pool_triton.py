#!/usr/bin/env python3
"""
Memory Pool for Hyperbolic Tensors

タスク33.2: 双曲テンソル用メモリプール
- Pre-allocated memory blocks for hyperbolic tensors
- Sub-microsecond allocation latency
- 目標: <20% fragmentation

Requirements: 44.1-44.6
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
import threading
import time
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class MemoryBlock:
    """メモリブロック情報"""
    tensor: torch.Tensor
    size: int
    in_use: bool
    last_used: float
    allocation_count: int


class HyperbolicMemoryPool:
    """
    双曲テンソル用メモリプール
    
    事前割り当てされたメモリブロックを管理し、
    サブマイクロ秒のアロケーションレイテンシを実現する。
    
    Args:
        initial_size_mb: 初期プールサイズ（MB）
        max_size_mb: 最大プールサイズ（MB）
        block_sizes: 事前割り当てするブロックサイズのリスト
        device: デバイス
        dtype: データ型
    """
    
    # 一般的な双曲テンソルサイズ
    DEFAULT_BLOCK_SIZES = [
        (4, 8, 1024, 64),    # Small: B=4, H=8, N=1024, D=64
        (4, 8, 2048, 64),    # Medium
        (4, 8, 4096, 64),    # Large
        (4, 8, 8192, 64),    # XLarge
        (1, 8, 16384, 64),   # Long context
    ]
    
    def __init__(
        self,
        initial_size_mb: float = 256,
        max_size_mb: float = 1024,
        block_sizes: Optional[List[Tuple[int, ...]]] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.initial_size_mb = initial_size_mb
        self.max_size_mb = max_size_mb
        self.block_sizes = block_sizes or self.DEFAULT_BLOCK_SIZES
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        # メモリプール
        self._pools: Dict[Tuple[int, ...], List[MemoryBlock]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # 統計
        self._stats = {
            'allocations': 0,
            'deallocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_allocated_bytes': 0,
            'peak_allocated_bytes': 0,
            'fragmentation_ratio': 0.0,
        }
        
        # 初期化
        self._initialize_pools()
    
    def _initialize_pools(self):
        """プールを初期化"""
        bytes_per_element = torch.tensor([], dtype=self.dtype).element_size()
        target_bytes = int(self.initial_size_mb * 1024 * 1024)
        allocated_bytes = 0
        
        for shape in self.block_sizes:
            # 各サイズに対して複数のブロックを事前割り当て
            block_bytes = 1
            for dim in shape:
                block_bytes *= dim
            block_bytes *= bytes_per_element
            
            # 各サイズに2-4ブロック割り当て
            num_blocks = max(2, min(4, target_bytes // (len(self.block_sizes) * block_bytes)))
            
            for _ in range(num_blocks):
                if allocated_bytes + block_bytes > target_bytes:
                    break
                
                tensor = torch.empty(shape, device=self.device, dtype=self.dtype)
                block = MemoryBlock(
                    tensor=tensor,
                    size=block_bytes,
                    in_use=False,
                    last_used=time.time(),
                    allocation_count=0,
                )
                self._pools[shape].append(block)
                allocated_bytes += block_bytes
        
        self._stats['total_allocated_bytes'] = allocated_bytes
        self._stats['peak_allocated_bytes'] = allocated_bytes
    
    def allocate(
        self,
        shape: Tuple[int, ...],
        zero_fill: bool = False,
    ) -> torch.Tensor:
        """
        テンソルを割り当て
        
        Args:
            shape: テンソル形状
            zero_fill: ゼロ初期化するか
        
        Returns:
            割り当てられたテンソル
        """
        with self._lock:
            self._stats['allocations'] += 1
            
            # 完全一致を探す
            if shape in self._pools:
                for block in self._pools[shape]:
                    if not block.in_use:
                        block.in_use = True
                        block.last_used = time.time()
                        block.allocation_count += 1
                        self._stats['cache_hits'] += 1
                        
                        if zero_fill:
                            block.tensor.zero_()
                        
                        return block.tensor
            
            # 互換性のあるブロックを探す（より大きいサイズ）
            for pool_shape, blocks in self._pools.items():
                if self._is_compatible(pool_shape, shape):
                    for block in blocks:
                        if not block.in_use:
                            block.in_use = True
                            block.last_used = time.time()
                            block.allocation_count += 1
                            self._stats['cache_hits'] += 1
                            
                            # ビューを返す
                            view = block.tensor.view(-1)[:self._numel(shape)].view(shape)
                            
                            if zero_fill:
                                view.zero_()
                            
                            return view
            
            # キャッシュミス: 新しいテンソルを割り当て
            self._stats['cache_misses'] += 1
            
            tensor = torch.empty(shape, device=self.device, dtype=self.dtype)
            
            if zero_fill:
                tensor.zero_()
            
            # プールに追加（最大サイズを超えない場合）
            bytes_per_element = tensor.element_size()
            block_bytes = tensor.numel() * bytes_per_element
            
            if self._stats['total_allocated_bytes'] + block_bytes <= self.max_size_mb * 1024 * 1024:
                block = MemoryBlock(
                    tensor=tensor,
                    size=block_bytes,
                    in_use=True,
                    last_used=time.time(),
                    allocation_count=1,
                )
                self._pools[shape].append(block)
                self._stats['total_allocated_bytes'] += block_bytes
                self._stats['peak_allocated_bytes'] = max(
                    self._stats['peak_allocated_bytes'],
                    self._stats['total_allocated_bytes']
                )
            
            return tensor
    
    def deallocate(self, tensor: torch.Tensor):
        """
        テンソルを解放
        
        Args:
            tensor: 解放するテンソル
        """
        with self._lock:
            self._stats['deallocations'] += 1
            
            # プール内のブロックを探す
            for shape, blocks in self._pools.items():
                for block in blocks:
                    if block.tensor.data_ptr() == tensor.data_ptr():
                        block.in_use = False
                        block.last_used = time.time()
                        return
            
            # プールにない場合は何もしない（GCに任せる）
    
    def _is_compatible(
        self,
        pool_shape: Tuple[int, ...],
        requested_shape: Tuple[int, ...],
    ) -> bool:
        """形状の互換性をチェック"""
        pool_numel = self._numel(pool_shape)
        requested_numel = self._numel(requested_shape)
        return pool_numel >= requested_numel
    
    def _numel(self, shape: Tuple[int, ...]) -> int:
        """要素数を計算"""
        result = 1
        for dim in shape:
            result *= dim
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """統計を取得"""
        with self._lock:
            # フラグメンテーション率を計算
            total_capacity = 0
            used_capacity = 0
            
            for blocks in self._pools.values():
                for block in blocks:
                    total_capacity += block.size
                    if block.in_use:
                        used_capacity += block.size
            
            if total_capacity > 0:
                self._stats['fragmentation_ratio'] = 1.0 - (used_capacity / total_capacity)
            
            return dict(self._stats)
    
    def clear_unused(self, max_age_seconds: float = 60.0):
        """
        未使用のブロックをクリア
        
        Args:
            max_age_seconds: この秒数以上使用されていないブロックをクリア
        """
        with self._lock:
            current_time = time.time()
            
            for shape in list(self._pools.keys()):
                blocks = self._pools[shape]
                new_blocks = []
                
                for block in blocks:
                    if block.in_use or (current_time - block.last_used) < max_age_seconds:
                        new_blocks.append(block)
                    else:
                        self._stats['total_allocated_bytes'] -= block.size
                
                self._pools[shape] = new_blocks
    
    def reset(self):
        """プールをリセット"""
        with self._lock:
            self._pools.clear()
            self._stats = {
                'allocations': 0,
                'deallocations': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'total_allocated_bytes': 0,
                'peak_allocated_bytes': 0,
                'fragmentation_ratio': 0.0,
            }
            self._initialize_pools()


class PooledTensor:
    """
    プール管理されたテンソルのラッパー
    
    コンテキストマネージャとして使用可能。
    """
    
    def __init__(
        self,
        pool: HyperbolicMemoryPool,
        shape: Tuple[int, ...],
        zero_fill: bool = False,
    ):
        self.pool = pool
        self.tensor = pool.allocate(shape, zero_fill)
    
    def __enter__(self) -> torch.Tensor:
        return self.tensor
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.deallocate(self.tensor)
        return False


# グローバルプールインスタンス
_global_pool: Optional[HyperbolicMemoryPool] = None
_pool_lock = threading.Lock()


def get_global_pool(
    initial_size_mb: float = 256,
    max_size_mb: float = 1024,
    device: torch.device = None,
) -> HyperbolicMemoryPool:
    """
    グローバルメモリプールを取得
    
    Args:
        initial_size_mb: 初期サイズ（MB）
        max_size_mb: 最大サイズ（MB）
        device: デバイス
    
    Returns:
        グローバルメモリプール
    """
    global _global_pool
    
    with _pool_lock:
        if _global_pool is None:
            _global_pool = HyperbolicMemoryPool(
                initial_size_mb=initial_size_mb,
                max_size_mb=max_size_mb,
                device=device,
            )
        return _global_pool


def pooled_allocate(
    shape: Tuple[int, ...],
    zero_fill: bool = False,
    device: torch.device = None,
) -> torch.Tensor:
    """
    グローバルプールからテンソルを割り当て
    
    Args:
        shape: テンソル形状
        zero_fill: ゼロ初期化するか
        device: デバイス
    
    Returns:
        割り当てられたテンソル
    """
    pool = get_global_pool(device=device)
    return pool.allocate(shape, zero_fill)


def pooled_deallocate(tensor: torch.Tensor):
    """
    グローバルプールにテンソルを返却
    
    Args:
        tensor: 返却するテンソル
    """
    global _global_pool
    
    if _global_pool is not None:
        _global_pool.deallocate(tensor)


class HyperbolicTensorAllocator(nn.Module):
    """
    双曲テンソルアロケータモジュール
    
    ニューラルネットワークモジュールとして使用可能な
    メモリプール管理クラス。
    
    Args:
        pool_size_mb: プールサイズ（MB）
        device: デバイス
    """
    
    def __init__(
        self,
        pool_size_mb: float = 256,
        device: torch.device = None,
    ):
        super().__init__()
        self.pool = HyperbolicMemoryPool(
            initial_size_mb=pool_size_mb,
            max_size_mb=pool_size_mb * 4,
            device=device,
        )
    
    def allocate_qkv(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Q, K, Vテンソルを割り当て
        
        Args:
            batch_size: バッチサイズ
            num_heads: ヘッド数
            seq_len: シーケンス長
            head_dim: ヘッド次元
        
        Returns:
            (Q, K, V) テンソル
        """
        shape = (batch_size, num_heads, seq_len, head_dim)
        q = self.pool.allocate(shape)
        k = self.pool.allocate(shape)
        v = self.pool.allocate(shape)
        return q, k, v
    
    def allocate_attention_output(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> torch.Tensor:
        """
        アテンション出力テンソルを割り当て
        
        Args:
            batch_size: バッチサイズ
            num_heads: ヘッド数
            seq_len: シーケンス長
            head_dim: ヘッド次元
        
        Returns:
            出力テンソル
        """
        shape = (batch_size, num_heads, seq_len, head_dim)
        return self.pool.allocate(shape)
    
    def allocate_distance_matrix(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        距離行列テンソルを割り当て
        
        Args:
            batch_size: バッチサイズ
            num_heads: ヘッド数
            seq_len: シーケンス長
        
        Returns:
            距離行列テンソル
        """
        shape = (batch_size, num_heads, seq_len, seq_len)
        return self.pool.allocate(shape)
    
    def deallocate(self, *tensors: torch.Tensor):
        """
        テンソルを解放
        
        Args:
            tensors: 解放するテンソル
        """
        for tensor in tensors:
            self.pool.deallocate(tensor)
    
    def get_stats(self) -> Dict[str, Any]:
        """統計を取得"""
        return self.pool.get_stats()
    
    def clear_unused(self, max_age_seconds: float = 60.0):
        """未使用ブロックをクリア"""
        self.pool.clear_unused(max_age_seconds)


# エクスポート
__all__ = [
    'MemoryBlock',
    'HyperbolicMemoryPool',
    'PooledTensor',
    'HyperbolicTensorAllocator',
    'get_global_pool',
    'pooled_allocate',
    'pooled_deallocate',
]
