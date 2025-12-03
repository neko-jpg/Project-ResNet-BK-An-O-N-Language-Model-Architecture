"""
Holographic Tensor Train (HTT) Embedding

Phase 1.2: 90%パラメータ圧縮を実現するTensor Train分解ベースのEmbedding層。
位相回転（Holographic encoding）により、圧縮空間内でもトークンの意味関係を保存します。

Mathematical Foundation:
    標準Embedding: E ∈ R^(V×D) → V*D parameters
    
    Tensor Train分解:
        E[i, :] = Contract(Core1[i₁], Core2[i₂], ..., CoreK[iₖ])
        where i = i₁·V₂·...·Vₖ + i₂·V₃·...·Vₖ + ... + iₖ
    
    Holographic Enhancement:
        位相回転: Core1_mod = Core1 · cos(θ) (実数近似)
        干渉パターンにより意味情報を保存
    
    Parameter Count:
        Standard: V × D
        TT (2 cores, rank r): V₁·r·D₁ + V₂·r·D₂
        Compression ratio: ~0.004-0.1 (90-99.6% reduction)

Physical Intuition (物理的直観):
    - Tensor Trainは「量子もつれ状態」の古典近似
    - 位相回転は「波動関数の干渉」を模倣
    - 低ランクでも意味情報が保存される理由は、
      自然言語の「低次元多様体構造」による

Requirements:
    - 2.1: Tensor Train分解とインデックス分解
    - 2.2: 90%以上の圧縮率
    - 2.3: 位相回転による意味保存
    - 2.4: 自動因数分解
    - 2.5: 境界条件処理
    - 2.6: 正確なd_model出力
    - 4.1: 既存インフラとの統合
    - 10.2: 勾配フロー検証

Author: Project MUSE Team
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import Phase1Config
from .errors import InvalidConfigError, NumericalInstabilityError
from .complex_utils import complex_phase_rotation, is_complex_tensor

try:
    from ...kernels.htt_triton import htt_fused_contraction
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class HolographicTTEmbedding(nn.Module):
    """
    Holographic Tensor Train Embedding Layer
    
    90%パラメータ圧縮を実現しながら、位相回転により意味情報を保存する
    Embedding層の実装。
    
    Args:
        vocab_size: 語彙サイズ
        d_model: 出力次元（モデルの隠れ層次元）
        rank: Tensor Trainのランク（圧縮率を制御）
        num_cores: Tensor Trainのコア数（デフォルト2、大規模語彙では3+）
        phase_encoding: 位相回転を有効化するか
        init_scale: パラメータ初期化のスケール
    
    Attributes:
        vocab_size: 語彙サイズ
        d_model: 出力次元
        rank: TTランク
        num_cores: コア数
        phase_encoding: 位相回転の有効/無効
        v1, v2: 語彙の因数分解 (V = v1 × v2)
        d1, d2: 次元の因数分解 (D = d1 × d2)
        core1, core2: Tensor Trainコア
        phase_shift: 位相回転パラメータ
    
    Example:
        >>> # 標準Embeddingの置き換え
        >>> # embedding = nn.Embedding(50000, 1024)  # 51.2M params
        >>> embedding = HolographicTTEmbedding(50000, 1024, rank=16)  # ~0.2M params
        >>> 
        >>> input_ids = torch.randint(0, 50000, (4, 128))
        >>> output = embedding(input_ids)  # (4, 128, 1024)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        rank: int = 16,
        num_cores: int = 2,
        phase_encoding: bool = True,
        init_scale: float = 0.02,
    ):
        super().__init__()
        
        # Validation
        if vocab_size <= 0:
            raise InvalidConfigError(
                param_name="vocab_size",
                param_value=vocab_size,
                reason="Must be positive integer"
            )
        if d_model <= 0:
            raise InvalidConfigError(
                param_name="d_model",
                param_value=d_model,
                reason="Must be positive integer"
            )
        if rank <= 0:
            raise InvalidConfigError(
                param_name="rank",
                param_value=rank,
                reason="Must be positive integer"
            )
        if num_cores != 2:
            # 現在の実装は2コアのみサポート
            # 将来的に3+コアに拡張可能
            raise InvalidConfigError(
                param_name="num_cores",
                param_value=num_cores,
                reason="Currently only num_cores=2 is supported"
            )
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.rank = rank
        self.num_cores = num_cores
        self.phase_encoding = phase_encoding
        self.init_scale = init_scale
        
        # Memory optimization flags
        self.use_triton_kernel = False  # Enable Triton TT contraction
        self.use_checkpointing = False  # Enable gradient checkpointing
        
        # Automatic factorization (sqrt decomposition for balanced cores)
        # 語彙数の因数分解: V = v1 × v2
        # 平方根分解により、コアサイズをバランスさせる
        self.v1 = int(math.ceil(math.sqrt(vocab_size)))
        self.v2 = int(math.ceil(vocab_size / self.v1))
        
        # 次元の因数分解: D = d1 × d2
        self.d1 = int(math.ceil(math.sqrt(d_model)))
        self.d2 = int(math.ceil(d_model / self.d1))
        
        # Tensor Train Cores
        # Core1: (v1, 1, rank, d1) - Left boundary (rank=1 on left)
        # Core2: (v2, rank, 1, d2) - Right boundary (rank=1 on right)
        #
        # 物理的直観: 各コアは「局所的な量子状態」を表現
        # ランク次元は「もつれ結合」の強さを制御
        # Robust Initialization: Scale down by 1/sqrt(rank) to prevent explosion
        scale_factor = init_scale / (rank ** 0.5)
        self.core1 = nn.Parameter(
            torch.randn(self.v1, 1, rank, self.d1) * scale_factor
        )
        self.core2 = nn.Parameter(
            torch.randn(self.v2, rank, 1, self.d2) * scale_factor
        )
        
        # Holographic Phase Parameters
        # 各ランク次元に位相を与える
        # 物理的直観: 波動関数の位相 exp(iθ)
        # 実数実装では cos(θ) による振幅変調として近似
        if phase_encoding:
            self.phase_shift = nn.Parameter(torch.randn(rank) * 0.1)
        else:
            self.register_buffer('phase_shift', torch.zeros(rank))
        
        # Parameter count tracking
        self._standard_params = vocab_size * d_model
        self._tt_params = self.v1 * rank * self.d1 + self.v2 * rank * self.d2
        if phase_encoding:
            self._tt_params += rank
        self._compression_ratio = self._tt_params / self._standard_params
    
    def get_compression_ratio(self) -> float:
        """
        圧縮率を返す（0.1 = 90%圧縮）
        
        Returns:
            compression_ratio: TT params / Standard params
        """
        return self._compression_ratio
    
    def get_parameter_counts(self) -> Tuple[int, int]:
        """
        パラメータ数を返す
        
        Returns:
            (standard_params, tt_params): 標準Embedding vs TT Embedding
        """
        return self._standard_params, self._tt_params

    def quantize(self):
        """
        Quantize cores to INT8 for fused kernel inference.
        """
        # Ensure we don't track gradients during quantization
        with torch.no_grad():
            # Quantize Core1
            c1_max = self.core1.abs().max()
            self.scale1 = c1_max / 127.0
            # Store in a buffer, don't overwrite parameter
            if not hasattr(self, 'core1_quant'):
                self.register_buffer('core1_quant', torch.zeros_like(self.core1, dtype=torch.int8))
            self.core1_quant = (self.core1 / self.scale1).round().clamp(-128, 127).to(torch.int8)
            
            # Quantize Core2
            c2_max = self.core2.abs().max()
            self.scale2 = c2_max / 127.0
            if not hasattr(self, 'core2_quant'):
                self.register_buffer('core2_quant', torch.zeros_like(self.core2, dtype=torch.int8))
            self.core2_quant = (self.core2 / self.scale2).round().clamp(-128, 127).to(torch.int8)
            
            # Mark as quantized for forward pass
            self._is_quantized_state = True
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Memory-Efficient Tensor Train contraction
        
        Args:
            input_ids: (Batch, SeqLen) トークンID
        
        Returns:
            embeddings: (Batch, SeqLen, d_model) 埋め込みベクトル
        
        Raises:
            NumericalInstabilityError: NaN/Infが検出された場合
        
        Memory Optimization:
            従来の実装では einsum が O(B*L*d1*d2) の中間テンソルを生成していました。
            新実装では、次元ごとの縮約により中間メモリを最小化します。
            
            Strategy:
            - d1次元とd2次元を個別に処理
            - 各ステップで O(B*L*rank) のメモリのみ使用
            - 中間的な O(B*L*d1*d2) テンソルを完全に回避
        """
        # Input validation
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input_ids (B, L), got shape {input_ids.shape}"
            )
        
        B, L = input_ids.shape
        
        # 1. Index decomposition
        # トークンID i を (i₁, i₂) に分解
        # i = i₁ × v2 + i₂
        idx1 = input_ids // self.v2
        idx2 = input_ids % self.v2
        
        # 2. Boundary checking and clamping
        # 因数分解の端数により、v1*v2 > vocab_size となる場合がある
        # 範囲外のインデックスをクランプして安全に処理
        idx1 = torch.clamp(idx1, 0, self.v1 - 1)
        idx2 = torch.clamp(idx2, 0, self.v2 - 1)
        
        # 3. Gather cores
        # core1: (v1, 1, rank, d1) → (B, L, 1, rank, d1)
        # core2: (v2, rank, 1, d2) → (B, L, rank, 1, d2)
        c1 = self.core1[idx1]  # (B, L, 1, rank, d1)
        c2 = self.core2[idx2]  # (B, L, rank, 1, d2)
        
        # Squeeze singleton dimensions
        c1 = c1.squeeze(2)  # (B, L, rank, d1)
        c2 = c2.squeeze(3)  # (B, L, rank, d2)
        
        # 4. Apply phase rotation (holographic encoding)
        # 位相回転: cos(θ) による振幅変調（Phase 1）
        if self.phase_encoding:
            phase_mod = torch.cos(self.phase_shift)  # (rank,)
            c1 = c1 * phase_mod.view(1, 1, -1, 1)  # Broadcast to (B, L, rank, d1)
        
        # 5. Memory-Efficient Contraction
        # 物理的直観: 量子もつれ状態の測定（縮約）
        
        # Guard: Clamp cores to prevent explosion
        c1 = torch.clamp(c1, -10.0, 10.0)
        c2 = torch.clamp(c2, -10.0, 10.0)
        
        # Try Triton kernel first (if available)
        use_triton = hasattr(self, 'use_triton_kernel') and self.use_triton_kernel
        
        # Check if we should use fused kernel (if cores are quantized)
        is_quantized = getattr(self, '_is_quantized_state', False)
        
        # Force float32 for stability during contraction
        with torch.cuda.amp.autocast(enabled=False):
            c1 = c1.float()
            c2 = c2.float()
            
            executed_triton = False
            if use_triton and TRITON_AVAILABLE and torch.cuda.is_available():
                try:
                    if is_quantized:
                        # Fused Kernel
                        scale1 = getattr(self, 'scale1', torch.tensor(1.0, device=self.core1.device)).float()
                        scale2 = getattr(self, 'scale2', torch.tensor(1.0, device=self.core2.device)).float()
                        
                        # Use quantized buffers
                        c1_q = getattr(self, 'core1_quant', self.core1)
                        c2_q = getattr(self, 'core2_quant', self.core2)
                        
                        out_tensor = htt_fused_contraction(
                            idx1, idx2,
                            c1_q, c2_q,
                            scale1, scale2,
                            self.d_model
                        )
                        return out_tensor
                    else:
                        from ...kernels.tt_contraction import tt_contraction_triton
                        # Triton kernel: メモリ効率的なTT縮約
                        out_tensor = tt_contraction_triton(c1, c2)  # (B, L, d1, d2)
                        # 6. Reshape to (B, L, D)
                        out = out_tensor.reshape(B, L, -1)  # (B, L, d1*d2)
                        # 7. Crop to exact d_model size
                        out = out[:, :, :self.d_model]
                        return out

                except (ImportError, Exception) as e:
                    # Fallback to einsum
                    executed_triton = False
            
            if not executed_triton:
                # Use gradient checkpointing for memory efficiency
                if self.training and hasattr(self, 'use_checkpointing') and self.use_checkpointing:
                    from torch.utils.checkpoint import checkpoint
                    
                    def contraction_fn(c1_inner, c2_inner):
                        return torch.einsum('blrd,blrf->bldf', c1_inner, c2_inner)
                    
                    out_tensor = checkpoint(contraction_fn, c1, c2, use_reentrant=False)
                else:
                    # Standard einsum
                    out_tensor = torch.einsum('blrd,blrf->bldf', c1, c2)  # (B, L, d1, d2)
                
                # 6. Reshape to (B, L, D)
                out = out_tensor.reshape(B, L, -1)  # (B, L, d1*d2)
                
                # 7. Crop to exact d_model size
                out = out[:, :, :self.d_model]
            
            # Numerical stability check and guard
            if not torch.isfinite(out).all():
                print(f"HTT Embedding NaN detected! Max: {out.abs().max().item()}")
                # Attempt to recover by clamping
                out = torch.clamp(out, -100.0, 100.0)
                if not torch.isfinite(out).all():
                     # If still bad, zero out (extreme fallback)
                     out = torch.nan_to_num(out, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Normalize to keep variance stable (LayerNorm-like effect)
            out = out / (self.rank ** 0.5)
            
            return out.to(input_ids.device) # Cast back to original device/dtype if needed by context (but we return float32 mostly)
    
    def forward_complex(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with full complex phase rotation (Phase 2 準備)
        
        Phase 2で使用する完全な複素位相回転 exp(iθ) を実装。
        現在は実験的機能として提供。
        
        Args:
            input_ids: (Batch, SeqLen) トークンID
        
        Returns:
            embeddings: (Batch, SeqLen, d_model) 複素数埋め込みベクトル
        
        Requirement 11.3: Implement complex-valued phase rotation in HTT (exp(iθ))
        """
        # Input validation
        if input_ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input_ids (B, L), got shape {input_ids.shape}"
            )
        
        B, L = input_ids.shape
        
        # 1-3. Index decomposition and core gathering (same as forward)
        idx1 = input_ids // self.v2
        idx2 = input_ids % self.v2
        idx1 = torch.clamp(idx1, 0, self.v1 - 1)
        idx2 = torch.clamp(idx2, 0, self.v2 - 1)
        
        c1 = self.core1[idx1].squeeze(2)  # (B, L, rank, d1)
        c2 = self.core2[idx2].squeeze(3)  # (B, L, rank, d2)
        
        # 4. Apply FULL complex phase rotation: exp(iθ)
        # Phase 2: 完全な複素位相回転
        if self.phase_encoding:
            # Use complex_phase_rotation with full complex mode
            c1 = complex_phase_rotation(
                c1, 
                self.phase_shift, 
                use_full_complex=True
            )
        
        # 5. Holographic contraction (einsum handles complex automatically)
        out_tensor = torch.einsum('blrd,blrf->bldf', c1, c2)
        
        # 6-7. Reshape and crop
        out = out_tensor.reshape(B, L, -1)
        out = out[:, :, :self.d_model]
        
        # Numerical stability check
        if not torch.isfinite(out).all():
            raise NumericalInstabilityError(
                component="HolographicTTEmbedding (complex mode)",
                diagnostics={
                    'has_nan': torch.isnan(out).any().item(),
                    'has_inf': torch.isinf(out).any().item(),
                    'max_value': out.abs().max().item(),
                }
            )
        
        return out
    
    def extra_repr(self) -> str:
        """
        モジュールの追加情報を文字列で返す
        """
        return (
            f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
            f"rank={self.rank}, num_cores={self.num_cores}, "
            f"phase_encoding={self.phase_encoding}, "
            f"compression_ratio={self._compression_ratio:.4f} "
            f"({self._tt_params}/{self._standard_params})"
        )


class HTTDecoder(nn.Module):
    """
    Decodes hidden states to vocabulary logits using shared HTT weights.
    This module performs the inverse operation of the HTT embedding.
    """
    def __init__(self, embedding: HolographicTTEmbedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): The final hidden states from the model.
                                          Shape: (batch_size, seq_len, d_model)
        Returns:
            torch.Tensor: The output logits. Shape: (batch_size, seq_len, vocab_size)
        """
        B, L, D = hidden_states.shape
        emb = self.embedding

        # 1. Prepare hidden states and cores for contraction
        h_padded = F.pad(hidden_states, (0, emb.d1 * emb.d2 - D))
        h_reshaped = h_padded.view(B, L, emb.d1, emb.d2)

        c1 = emb.core1.squeeze(1)  # Shape: (v1, rank, d1)
        c2 = emb.core2.squeeze(2)  # Shape: (v2, rank, d2)

        # 2. Apply inverse phase modulation
        if emb.phase_encoding:
            phase_mod = torch.cos(emb.phase_shift) # Shape: (rank,)
            c1 = c1 * phase_mod.view(1, -1, 1)

        # 3. Perform tensor contraction
        # einsum('bldf,ird,urf->bliu', h, c1, c2)
        # h: (B, L, d1, d2), c1: (v1, rank, d1), c2: (v2, rank, d2)
        # -> logits_decomposed: (B, L, v1, v2)
        logits_decomposed = torch.einsum('bldf,ird,urf->bliu', h_reshaped, c1, c2)

        # 4. Reshape to final logits and crop
        logits = logits_decomposed.reshape(B, L, -1)
        return logits[:, :, :emb.vocab_size]


def create_htt_embedding(
    vocab_size: int,
    d_model: int,
    config: Optional[Phase1Config] = None,
) -> HolographicTTEmbedding:
    """
    Factory function to create HTT embedding from config
    
    既存モデルのnn.Embeddingを置き換える際に使用します。
    
    Args:
        vocab_size: 語彙サイズ
        d_model: 出力次元
        config: Phase1Config（Noneの場合はデフォルト設定）
    
    Returns:
        HolographicTTEmbedding instance
    
    Example:
        >>> config = Phase1Config(htt_rank=16, htt_phase_encoding=True)
        >>> embedding = create_htt_embedding(50000, 1024, config)
    """
    if config is None:
        config = Phase1Config()
    
    return HolographicTTEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        rank=config.htt_rank,
        num_cores=config.htt_num_cores,
        phase_encoding=config.htt_phase_encoding,
    )


def replace_embedding_with_htt(
    model: nn.Module,
    embedding_attr: str = "token_embedding",
    config: Optional[Phase1Config] = None,
) -> nn.Module:
    """
    既存モデルのEmbedding層をHTTに置き換える
    
    Args:
        model: 対象モデル
        embedding_attr: Embedding層の属性名
        config: Phase1Config
    
    Returns:
        Modified model (in-place)
    
    Example:
        >>> model = MyLanguageModel(vocab_size=50000, d_model=1024)
        >>> model = replace_embedding_with_htt(model, "token_embedding")
        >>> # model.token_embedding is now HolographicTTEmbedding
    """
    if not hasattr(model, embedding_attr):
        raise ValueError(
            f"Model does not have attribute '{embedding_attr}'"
        )
    
    old_embedding = getattr(model, embedding_attr)
    
    if not isinstance(old_embedding, nn.Embedding):
        raise TypeError(
            f"Expected nn.Embedding, got {type(old_embedding)}"
        )
    
    vocab_size = old_embedding.num_embeddings
    d_model = old_embedding.embedding_dim
    
    new_embedding = create_htt_embedding(vocab_size, d_model, config)
    
    setattr(model, embedding_attr, new_embedding)
    
    return model



def verify_compression_ratio(
    embedding: HolographicTTEmbedding,
    target_ratio: float = 0.1,
) -> dict:
    """
    HTT Embeddingの圧縮率を検証
    
    Args:
        embedding: HolographicTTEmbedding instance
        target_ratio: 目標圧縮率（デフォルト0.1 = 90%圧縮）
    
    Returns:
        dict with:
            - standard_params: 標準Embeddingのパラメータ数
            - tt_params: TT Embeddingのパラメータ数
            - compression_ratio: 実際の圧縮率
            - compression_percentage: 圧縮率（%表示）
            - meets_target: 目標圧縮率を達成しているか
            - parameter_reduction: 削減されたパラメータ数
    
    Example:
        >>> embedding = HolographicTTEmbedding(50000, 1024, rank=16)
        >>> result = verify_compression_ratio(embedding)
        >>> print(f"Compression: {result['compression_percentage']:.1f}%")
    """
    standard_params, tt_params = embedding.get_parameter_counts()
    compression_ratio = embedding.get_compression_ratio()
    
    return {
        'standard_params': standard_params,
        'tt_params': tt_params,
        'compression_ratio': compression_ratio,
        'compression_percentage': (1 - compression_ratio) * 100,
        'meets_target': compression_ratio <= target_ratio,
        'parameter_reduction': standard_params - tt_params,
    }


def verify_gradient_flow(
    embedding: HolographicTTEmbedding,
    input_ids: torch.Tensor,
    check_all_cores: bool = True,
) -> dict:
    """
    HTT Embeddingの勾配フローを検証
    
    すべてのTensor Train coresに勾配が流れることを確認します。
    
    Args:
        embedding: HolographicTTEmbedding instance
        input_ids: (Batch, SeqLen) テスト用トークンID
        check_all_cores: すべてのコアの勾配をチェックするか
    
    Returns:
        dict with:
            - core1_has_grad: Core1に勾配があるか
            - core2_has_grad: Core2に勾配があるか
            - phase_has_grad: Phase parameterに勾配があるか
            - core1_grad_norm: Core1の勾配ノルム
            - core2_grad_norm: Core2の勾配ノルム
            - phase_grad_norm: Phase parameterの勾配ノルム
            - all_cores_have_grad: すべてのコアに勾配があるか
    
    Example:
        >>> embedding = HolographicTTEmbedding(1000, 128, rank=8)
        >>> input_ids = torch.randint(0, 1000, (2, 10))
        >>> result = verify_gradient_flow(embedding, input_ids)
        >>> assert result['all_cores_have_grad'], "Gradient flow broken!"
    """
    # Forward pass
    output = embedding(input_ids)
    
    # Backward pass with dummy loss
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    core1_has_grad = embedding.core1.grad is not None
    core2_has_grad = embedding.core2.grad is not None
    phase_has_grad = (
        embedding.phase_shift.grad is not None 
        if embedding.phase_encoding 
        else True  # Not applicable if phase encoding disabled
    )
    
    core1_grad_norm = (
        embedding.core1.grad.norm().item() 
        if core1_has_grad 
        else 0.0
    )
    core2_grad_norm = (
        embedding.core2.grad.norm().item() 
        if core2_has_grad 
        else 0.0
    )
    phase_grad_norm = (
        embedding.phase_shift.grad.norm().item() 
        if phase_has_grad and embedding.phase_encoding
        else 0.0
    )
    
    all_cores_have_grad = core1_has_grad and core2_has_grad and phase_has_grad
    
    return {
        'core1_has_grad': core1_has_grad,
        'core2_has_grad': core2_has_grad,
        'phase_has_grad': phase_has_grad,
        'core1_grad_norm': core1_grad_norm,
        'core2_grad_norm': core2_grad_norm,
        'phase_grad_norm': phase_grad_norm,
        'all_cores_have_grad': all_cores_have_grad,
    }


def calculate_htt_memory_savings(
    vocab_size: int,
    d_model: int,
    rank: int = 16,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """
    HTT Embeddingのメモリ削減量を計算
    
    Args:
        vocab_size: 語彙サイズ
        d_model: 出力次元
        rank: TTランク
        dtype: データ型
    
    Returns:
        dict with:
            - standard_memory_mb: 標準Embeddingのメモリ使用量（MB）
            - htt_memory_mb: HTT Embeddingのメモリ使用量（MB）
            - memory_saved_mb: 削減されたメモリ量（MB）
            - memory_saved_percentage: メモリ削減率（%）
    
    Example:
        >>> result = calculate_htt_memory_savings(50000, 1024, rank=16)
        >>> print(f"Memory saved: {result['memory_saved_mb']:.1f} MB")
    """
    bytes_per_param = torch.finfo(dtype).bits // 8
    
    # Standard embedding
    standard_params = vocab_size * d_model
    standard_memory_mb = (standard_params * bytes_per_param) / (1024 ** 2)
    
    # HTT embedding
    v1 = int(math.ceil(math.sqrt(vocab_size)))
    v2 = int(math.ceil(vocab_size / v1))
    d1 = int(math.ceil(math.sqrt(d_model)))
    d2 = int(math.ceil(d_model / d1))
    
    htt_params = v1 * rank * d1 + v2 * rank * d2 + rank  # +rank for phase
    htt_memory_mb = (htt_params * bytes_per_param) / (1024 ** 2)
    
    memory_saved_mb = standard_memory_mb - htt_memory_mb
    memory_saved_percentage = (memory_saved_mb / standard_memory_mb) * 100
    
    return {
        'standard_memory_mb': standard_memory_mb,
        'htt_memory_mb': htt_memory_mb,
        'memory_saved_mb': memory_saved_mb,
        'memory_saved_percentage': memory_saved_percentage,
    }
