#!/usr/bin/env python3
"""
Sparse Hyperbolic Attention Triton Kernel

タスク32.3: スパース双曲アテンションカーネル
- Top-k sparsity with LSH for hyperbolic space
- Skip zero blocks entirely
- 目標: 90%スパース性で5xスピードアップ

Requirements: 38.1-38.6
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

# Tritonのインポート
TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass


if TRITON_AVAILABLE:
    @triton.jit
    def _sparse_hyperbolic_attention_kernel(
        Q,  # Query [B, H, N, D]
        K,  # Key [B, H, N, D]
        V,  # Value [B, H, N, D]
        TopK_Indices,  # Top-k indices [B, H, N, K]
        Out,  # Output [B, H, N, D]
        curvature,
        stride_q_b, stride_q_h, stride_q_n, stride_q_d,
        stride_k_b, stride_k_h, stride_k_n, stride_k_d,
        stride_v_b, stride_v_h, stride_v_n, stride_v_d,
        stride_idx_b, stride_idx_h, stride_idx_n, stride_idx_k,
        stride_o_b, stride_o_h, stride_o_n, stride_o_d,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        K_SPARSE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        スパース双曲アテンションカーネル
        
        物理的直観: Top-kのキーのみを使用してアテンションを計算
        これにより、計算量をO(N*K)に削減（K << N）
        """
        # プログラムID
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)
        
        # オフセット
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        
        # Qをロード
        q_ptr = Q + pid_b * stride_q_b + pid_h * stride_q_h + pid_n * stride_q_n
        q = tl.load(q_ptr + offs_d * stride_q_d, mask=mask_d, other=0.0)
        
        # Q norm^2
        q_norm_sq = tl.sum(q * q)
        
        # アキュムレータ
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        sum_weights = 0.0
        max_score = float('-inf')
        
        # Top-kインデックスをイテレート
        for k_idx in range(K_SPARSE):
            # インデックスをロード
            idx_ptr = TopK_Indices + pid_b * stride_idx_b + pid_h * stride_idx_h + pid_n * stride_idx_n
            target_idx = tl.load(idx_ptr + k_idx * stride_idx_k)
            
            # Kをロード
            k_ptr = K + pid_b * stride_k_b + pid_h * stride_k_h + target_idx * stride_k_n
            k_vec = tl.load(k_ptr + offs_d * stride_k_d, mask=mask_d, other=0.0)
            
            # Vをロード
            v_ptr = V + pid_b * stride_v_b + pid_h * stride_v_h + target_idx * stride_v_n
            v_vec = tl.load(v_ptr + offs_d * stride_v_d, mask=mask_d, other=0.0)
            
            # 双曲距離計算
            k_norm_sq = tl.sum(k_vec * k_vec)
            qk_dot = tl.sum(q * k_vec)
            
            # ||q - k||^2
            diff_sq = q_norm_sq + k_norm_sq - 2.0 * qk_dot
            diff_sq = tl.maximum(diff_sq, 0.0)
            
            # 双曲距離
            denom_q = 1.0 - curvature * q_norm_sq
            denom_k = 1.0 - curvature * k_norm_sq
            denom = denom_q * denom_k
            denom = tl.maximum(denom, 1e-6)
            
            cosh_arg = 1.0 + 2.0 * curvature * diff_sq / denom
            cosh_arg = tl.maximum(cosh_arg, 1.0)
            
            # 距離 → スコア（負の距離）
            distance = tl.log(cosh_arg + tl.sqrt(cosh_arg * cosh_arg - 1.0 + 1e-8))
            score = -distance
            
            # オンラインソフトマックス
            new_max = tl.maximum(max_score, score)
            exp_old = tl.exp(max_score - new_max)
            exp_new = tl.exp(score - new_max)
            
            # アキュムレータ更新
            acc = acc * exp_old + v_vec * exp_new
            sum_weights = sum_weights * exp_old + exp_new
            max_score = new_max
        
        # 正規化
        out = acc / (sum_weights + 1e-8)
        
        # 出力を保存
        out_ptr = Out + pid_b * stride_o_b + pid_h * stride_o_h + pid_n * stride_o_n
        tl.store(out_ptr + offs_d * stride_o_d, out, mask=mask_d)


    @triton.jit
    def _lsh_hash_kernel(
        X,  # Input [B, H, N, D]
        Hashes,  # Output hashes [B, H, N, num_hashes]
        RandomVectors,  # Random projection vectors [num_hashes, D]
        stride_x_b, stride_x_h, stride_x_n, stride_x_d,
        stride_h_b, stride_h_h, stride_h_n, stride_h_hash,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        D: tl.constexpr,
        NUM_HASHES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        LSHハッシュ計算カーネル
        
        物理的直観: ランダム射影を使用して類似ベクトルを
        同じバケットにハッシュする
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_n = tl.program_id(2)
        
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        
        # Xをロード
        x_ptr = X + pid_b * stride_x_b + pid_h * stride_x_h + pid_n * stride_x_n
        x = tl.load(x_ptr + offs_d * stride_x_d, mask=mask_d, other=0.0)
        
        # 各ハッシュ関数を計算
        for hash_idx in range(NUM_HASHES):
            # ランダムベクトルをロード
            rv_ptr = RandomVectors + hash_idx * D
            rv = tl.load(rv_ptr + offs_d, mask=mask_d, other=0.0)
            
            # 内積
            dot = tl.sum(x * rv)
            
            # ハッシュ値（符号）
            hash_val = tl.where(dot >= 0, 1, 0)
            
            # 保存
            hash_ptr = Hashes + pid_b * stride_h_b + pid_h * stride_h_h + pid_n * stride_h_n
            tl.store(hash_ptr + hash_idx * stride_h_hash, hash_val)


class LSHHyperbolicIndex:
    """
    双曲空間用LSHインデックス
    
    Locality-Sensitive Hashingを使用して、
    双曲空間で近いベクトルを効率的に検索する。
    """
    
    def __init__(
        self,
        d_model: int,
        num_hashes: int = 8,
        num_buckets: int = 256,
        device: torch.device = None,
    ):
        self.d_model = d_model
        self.num_hashes = num_hashes
        self.num_buckets = num_buckets
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ランダム射影ベクトル
        self.random_vectors = torch.randn(
            num_hashes, d_model, device=self.device
        )
        self.random_vectors = self.random_vectors / self.random_vectors.norm(dim=-1, keepdim=True)
    
    def compute_hashes(self, x: torch.Tensor) -> torch.Tensor:
        """
        LSHハッシュを計算
        
        Args:
            x: 入力テンソル [B, H, N, D]
        
        Returns:
            ハッシュ値 [B, H, N, num_hashes]
        """
        # 内積
        # x: [B, H, N, D], rv: [num_hashes, D]
        # 正しいeinsum: 'bhnd,kd->bhnk' (kはnum_hashes)
        dots = torch.einsum('bhnd,kd->bhnk', x, self.random_vectors.to(x.device))
        
        # 符号をハッシュ値として使用
        hashes = (dots >= 0).int()
        
        return hashes
    
    def find_candidates(
        self,
        q_hashes: torch.Tensor,
        k_hashes: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """
        ハッシュが一致する候補を検索
        
        Args:
            q_hashes: クエリハッシュ [B, H, N, num_hashes]
            k_hashes: キーハッシュ [B, H, N, num_hashes]
            top_k: 返す候補数
        
        Returns:
            候補インデックス [B, H, N, top_k]
        """
        B, H, N, _ = q_hashes.shape
        
        # ハッシュ一致数を計算
        # [B, H, N, 1, num_hashes] vs [B, H, 1, N, num_hashes]
        matches = (q_hashes.unsqueeze(-2) == k_hashes.unsqueeze(-3)).sum(dim=-1)
        # matches: [B, H, N, N]
        
        # Top-k候補を選択
        _, indices = matches.topk(min(top_k, N), dim=-1)
        
        return indices


class SparseHyperbolicAttention(nn.Module):
    """
    スパース双曲アテンションモジュール
    
    LSHを使用してTop-kの関連キーのみを選択し、
    計算量を大幅に削減する。
    
    Args:
        d_model: モデル次元
        num_heads: アテンションヘッド数
        curvature: 双曲空間の曲率
        sparsity_ratio: スパース性の割合（0.9 = 90%スパース）
        num_hashes: LSHハッシュ関数の数
        use_triton: Tritonカーネルを使用するか
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        curvature: float = 1.0,
        sparsity_ratio: float = 0.9,
        num_hashes: int = 8,
        use_triton: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.curvature = curvature
        self.sparsity_ratio = sparsity_ratio
        self.num_hashes = num_hashes
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        # 射影層
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # LSHインデックス
        self.lsh_index = None
    
    def _init_lsh(self, device: torch.device):
        """LSHインデックスを初期化"""
        if self.lsh_index is None or self.lsh_index.device != device:
            self.lsh_index = LSHHyperbolicIndex(
                d_model=self.head_dim,
                num_hashes=self.num_hashes,
                device=device,
            )
    
    def _compute_top_k_indices(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Top-kインデックスを計算"""
        self._init_lsh(q.device)
        
        # LSHハッシュ計算
        q_hashes = self.lsh_index.compute_hashes(q)
        k_hashes = self.lsh_index.compute_hashes(k)
        
        # 候補検索
        indices = self.lsh_index.find_candidates(q_hashes, k_hashes, top_k)
        
        return indices
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            x: 入力テンソル [B, N, D]
            mask: アテンションマスク（未使用、スパースアテンションでは無視）
        
        Returns:
            出力テンソル [B, N, D]
        """
        B, N, D = x.shape
        
        # 射影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # ヘッド分割
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Top-k数を計算
        top_k = max(1, int(N * (1 - self.sparsity_ratio)))
        
        # Top-kインデックスを計算
        indices = self._compute_top_k_indices(q, k, top_k)
        
        # スパースアテンション
        if self.use_triton:
            out = self._triton_sparse_attention(q, k, v, indices)
        else:
            out = self._pytorch_sparse_attention(q, k, v, indices)
        
        # ヘッド結合
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.o_proj(out)
    
    def _triton_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Tritonスパースアテンション"""
        B, H, N, D = q.shape
        K_SPARSE = indices.shape[-1]
        
        out = torch.empty_like(q)
        
        BLOCK_D = min(64, D)
        
        grid = (B, H, N)
        
        _sparse_hyperbolic_attention_kernel[grid](
            q, k, v, indices, out,
            self.curvature,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, H=H, N=N, D=D, K_SPARSE=K_SPARSE, BLOCK_D=BLOCK_D,
        )
        
        return out
    
    def _pytorch_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """PyTorchフォールバック実装"""
        B, H, N, D = q.shape
        K_SPARSE = indices.shape[-1]
        
        # インデックスを使用してK, Vを収集
        # indices: [B, H, N, K_SPARSE]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        
        # K, Vを収集
        k_expanded = k.unsqueeze(2).expand(-1, -1, N, -1, -1)
        v_expanded = v.unsqueeze(2).expand(-1, -1, N, -1, -1)
        
        k_sparse = torch.gather(k_expanded, 3, indices_expanded)  # [B, H, N, K_SPARSE, D]
        v_sparse = torch.gather(v_expanded, 3, indices_expanded)  # [B, H, N, K_SPARSE, D]
        
        # 双曲距離計算
        q_expanded = q.unsqueeze(3)  # [B, H, N, 1, D]
        
        # ||q - k||^2
        diff_sq = ((q_expanded - k_sparse) ** 2).sum(dim=-1)  # [B, H, N, K_SPARSE]
        
        # ノルム
        q_norm_sq = (q ** 2).sum(dim=-1, keepdim=True)  # [B, H, N, 1]
        k_norm_sq = (k_sparse ** 2).sum(dim=-1)  # [B, H, N, K_SPARSE]
        
        # 双曲距離
        denom = (1 - self.curvature * q_norm_sq) * (1 - self.curvature * k_norm_sq)
        denom = torch.clamp(denom, min=1e-6)
        
        cosh_arg = 1 + 2 * self.curvature * diff_sq / denom
        cosh_arg = torch.clamp(cosh_arg, min=1.0)
        
        distance = torch.acosh(cosh_arg)
        
        # スコア（負の距離）
        scores = -distance
        
        # ソフトマックス
        attn = torch.softmax(scores, dim=-1)  # [B, H, N, K_SPARSE]
        
        # 出力
        out = torch.einsum('bhns,bhnsd->bhnd', attn, v_sparse)
        
        return out


def sparse_hyperbolic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    curvature: float = 1.0,
    sparsity_ratio: float = 0.9,
) -> torch.Tensor:
    """
    スパース双曲アテンションの関数インターフェース
    
    Args:
        q: クエリ [B, H, N, D]
        k: キー [B, H, N, D]
        v: バリュー [B, H, N, D]
        curvature: 双曲空間の曲率
        sparsity_ratio: スパース性の割合
    
    Returns:
        出力 [B, H, N, D]
    """
    B, H, N, D = q.shape
    top_k = max(1, int(N * (1 - sparsity_ratio)))
    
    # 簡易的なTop-k選択（内積ベース）
    scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, N]
    _, indices = scores.topk(top_k, dim=-1)  # [B, H, N, top_k]
    
    # PyTorch実装を使用
    module = SparseHyperbolicAttention(
        d_model=D * H,
        num_heads=H,
        curvature=curvature,
        sparsity_ratio=sparsity_ratio,
        use_triton=False,
    )
    
    return module._pytorch_sparse_attention(q, k, v, indices)


# エクスポート
__all__ = [
    'TRITON_AVAILABLE',
    'LSHHyperbolicIndex',
    'SparseHyperbolicAttention',
    'sparse_hyperbolic_attention',
]
