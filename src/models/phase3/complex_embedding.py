"""
Complex Embedding Layer for Phase 3

このモジュールは、Token EmbeddingとPosition Embeddingを複素数化し、
Phase 2のZetaEmbeddingを継承して統合します。

Physical Intuition:
    複素数埋め込みは、トークンの意味（実部）と文脈（虚部）を独立に学習します。
    - 実部: トークンの基本的な意味表現
    - 虚部: 文脈依存の位相情報（否定、皮肉、多義性など）

Memory Efficiency:
    complex32（float16 × 2）を使用することで、complex64の半分のメモリで動作します。

Requirements:
    - Requirement 1.13: Token Embeddingを複素数化（実部と虚部を独立に学習）
    - Requirement 1.14: Phase 2のZetaEmbeddingを継承
"""

import torch
import torch.nn as nn
from typing import Optional
import warnings

from .complex_tensor import ComplexTensor
from ..phase2.zeta_init import ZetaEmbedding, ZetaInitializer


class ComplexEmbedding(nn.Module):
    """
    Complex Embedding Layer
    
    Token EmbeddingとPosition Embeddingを複素数化し、統合します。
    Phase 2のZetaEmbeddingを継承して、ゼータ零点ベースの位置埋め込みを使用します。
    
    Architecture:
        Input: Token IDs (B, N)
        ↓
        [Token Embedding (Real)] → real_emb (B, N, D)
        [Token Embedding (Imag)] → imag_emb (B, N, D)
        ↓
        [Zeta Position Embedding] → pos_emb (B, N, D) [real only]
        ↓
        Output: ComplexTensor(real_emb + pos_emb, imag_emb) (B, N, D)
    
    Args:
        vocab_size (int): 語彙サイズ
        d_model (int): モデル次元
        max_seq_len (int): 最大シーケンス長（デフォルト: 2048）
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        zeta_scale (float): Zeta初期化のスケール（デフォルト: 1.0）
        trainable_pos (bool): 位置埋め込みを学習可能にするか（デフォルト: False）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
    
    Examples:
        >>> # 基本的な使用方法
        >>> embedding = ComplexEmbedding(vocab_size=50000, d_model=512)
        >>> input_ids = torch.randint(0, 50000, (4, 128))
        >>> z = embedding(input_ids)  # ComplexTensor(4, 128, 512)
        
        >>> # Phase 2互換モード（complex64）
        >>> embedding = ComplexEmbedding(vocab_size=50000, d_model=512, use_complex32=False)
        >>> z = embedding(input_ids)  # torch.complex64
    
    Physical Interpretation:
        - Token Embedding (Real): トークンの基本的な意味ベクトル
        - Token Embedding (Imag): トークンの文脈依存の位相情報
        - Position Embedding: ゼータ零点ベースの位置情報（実部のみ）
        
        複素数表現により、以下の言語現象をモデリング可能:
        - 否定形: 位相反転（π回転）
        - 皮肉: 位相のずれ
        - 多義語: 複数の位相成分の重ね合わせ
    
    Requirements: 1.13, 1.14
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        use_complex32: bool = True,
        zeta_scale: float = 1.0,
        trainable_pos: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_complex32 = use_complex32
        self.zeta_scale = zeta_scale
        self.trainable_pos = trainable_pos
        
        # データ型の設定
        self.dtype = torch.float16 if use_complex32 else torch.float32
        
        # ========================================
        # Token Embedding (Complex)
        # ========================================
        
        # 実部: トークンの基本的な意味表現
        self.token_embedding_real = nn.Embedding(vocab_size, d_model)
        
        # 虚部: トークンの文脈依存の位相情報
        self.token_embedding_imag = nn.Embedding(vocab_size, d_model)
        
        # Zeta初期化を適用（データ型変換前）
        try:
            ZetaInitializer.initialize_embedding_zeta(
                self.token_embedding_real, 
                scale=zeta_scale
            )
            ZetaInitializer.initialize_embedding_zeta(
                self.token_embedding_imag, 
                scale=zeta_scale * 0.5  # 虚部は実部の半分のスケールで初期化
            )
        except Exception as e:
            warnings.warn(f"Zeta initialization failed: {e}. Using default initialization.")
        
        # データ型を変換
        self.token_embedding_real.weight.data = self.token_embedding_real.weight.data.to(self.dtype)
        self.token_embedding_imag.weight.data = self.token_embedding_imag.weight.data.to(self.dtype)
        
        # ========================================
        # Position Embedding (Zeta-based)
        # ========================================
        
        # Phase 2のZetaEmbeddingを継承
        self.position_embedding = ZetaEmbedding(
            max_len=max_seq_len,
            d_model=d_model,
            trainable=trainable_pos,
            scale=zeta_scale
        )
        
        # ========================================
        # Dropout
        # ========================================
        
        self.dropout = nn.Dropout(dropout)
        
        # ========================================
        # 統計情報（デバッグ用）
        # ========================================
        
        self.register_buffer('_call_count', torch.tensor(0, dtype=torch.long))
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        initial_phase: Optional[torch.Tensor] = None
    ) -> ComplexTensor:
        """
        Forward pass
        
        Args:
            input_ids (torch.Tensor): トークンID (B, N)
            positions (torch.Tensor, optional): 位置インデックス (B, N)
                Noneの場合、自動的に [0, 1, 2, ..., N-1] を使用
            initial_phase (torch.Tensor, optional): 初期位相シフト (B, N)
                "Sentiment Phase Shifting" に使用。
                各トークンの複素数ベクトルを指定された角度だけ回転させます。
        
        Returns:
            ComplexTensor: 複素数埋め込み (B, N, D)
                - use_complex32=True: ComplexTensor形式
                - use_complex32=False: torch.complex64形式
        
        Raises:
            ValueError: input_idsの形状が不正な場合
            ValueError: シーケンス長がmax_seq_lenを超える場合
        """
        # 入力検証
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D (B, N), got shape {input_ids.shape}"
            )
        
        batch_size, seq_len = input_ids.shape
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        
        # 統計情報の更新
        self._call_count += 1
        
        # ========================================
        # 1. Token Embedding (Complex)
        # ========================================
        
        # 実部: トークンの基本的な意味表現
        token_emb_real = self.token_embedding_real(input_ids)  # (B, N, D)
        
        # 虚部: トークンの文脈依存の位相情報
        token_emb_imag = self.token_embedding_imag(input_ids)  # (B, N, D)
        
        # データ型の変換（complex32の場合）
        if self.use_complex32:
            token_emb_real = token_emb_real.half()
            token_emb_imag = token_emb_imag.half()
        
        # ========================================
        # 2. Position Embedding (Zeta-based)
        # ========================================
        
        # 位置インデックスの生成
        if positions is None:
            positions = torch.arange(
                seq_len, 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)  # (B, N)
        
        # Zeta Position Embedding（実部のみ）
        pos_emb = self.position_embedding(positions)  # (B, N, D)
        
        # データ型の変換（complex32の場合）
        if self.use_complex32:
            pos_emb = pos_emb.half()
        
        # ========================================
        # 3. 統合: Token + Position
        # ========================================
        
        # 実部: Token Embedding (Real) + Position Embedding
        real_part = token_emb_real + pos_emb
        
        # 虚部: Token Embedding (Imag) のみ
        # 位置情報は実部にのみ加算（物理的直観: 位置は実空間の情報）
        imag_part = token_emb_imag
        
        # ========================================
        # 3.5. Sentiment Phase Shifting (LOGOS Layer 1)
        # ========================================

        if initial_phase is not None:
            # 位相回転を適用
            # z' = z * e^(i * theta)
            #    = (x + iy) * (cos(theta) + i*sin(theta))
            #    = (x*cos - y*sin) + i(x*sin + y*cos)

            # initial_phase: (B, N) -> (B, N, 1)
            theta = initial_phase.unsqueeze(-1)
            cos_theta = torch.cos(theta).to(real_part.dtype)
            sin_theta = torch.sin(theta).to(real_part.dtype)

            # 元の実部と虚部
            x = real_part
            y = imag_part

            # 回転後の実部と虚部
            real_part = x * cos_theta - y * sin_theta
            imag_part = x * sin_theta + y * cos_theta

        # ========================================
        # 4. Dropout
        # ========================================
        
        real_part = self.dropout(real_part)
        imag_part = self.dropout(imag_part)
        
        # ========================================
        # 5. ComplexTensor形式で返す
        # ========================================
        
        if self.use_complex32:
            # ComplexTensor形式（メモリ効率優先）
            return ComplexTensor(real_part, imag_part)
        else:
            # PyTorch complex64形式（Phase 2互換）
            return torch.complex(real_part, imag_part)
    
    def get_embedding_weight(self, complex_part: str = 'real') -> torch.Tensor:
        """
        埋め込み重みの取得（デバッグ用）
        
        Args:
            complex_part (str): 'real' または 'imag'
        
        Returns:
            torch.Tensor: 埋め込み重み (vocab_size, d_model)
        """
        if complex_part == 'real':
            return self.token_embedding_real.weight.data
        elif complex_part == 'imag':
            return self.token_embedding_imag.weight.data
        else:
            raise ValueError(f"complex_part must be 'real' or 'imag', got {complex_part}")
    
    def get_statistics(self) -> dict:
        """
        統計情報の取得（デバッグ用）
        
        Returns:
            dict: 統計情報
                - 'call_count': forward呼び出し回数
                - 'real_norm': 実部の重みのノルム
                - 'imag_norm': 虚部の重みのノルム
                - 'real_mean': 実部の重みの平均
                - 'imag_mean': 虚部の重みの平均
                - 'real_std': 実部の重みの標準偏差
                - 'imag_std': 虚部の重みの標準偏差
        """
        real_weight = self.token_embedding_real.weight.data
        imag_weight = self.token_embedding_imag.weight.data
        
        return {
            'call_count': self._call_count.item(),
            'real_norm': real_weight.norm().item(),
            'imag_norm': imag_weight.norm().item(),
            'real_mean': real_weight.mean().item(),
            'imag_mean': imag_weight.mean().item(),
            'real_std': real_weight.std().item(),
            'imag_std': imag_weight.std().item(),
        }
    
    def extra_repr(self) -> str:
        """モジュールの追加情報を返す"""
        return (
            f'vocab_size={self.vocab_size}, d_model={self.d_model}, '
            f'max_seq_len={self.max_seq_len}, use_complex32={self.use_complex32}, '
            f'zeta_scale={self.zeta_scale}, trainable_pos={self.trainable_pos}'
        )


# ========================================
# ユーティリティ関数
# ========================================

def convert_phase2_embedding_to_complex(
    phase2_embedding: nn.Embedding,
    use_complex32: bool = True
) -> ComplexEmbedding:
    """
    Phase 2のEmbeddingをPhase 3のComplexEmbeddingに変換
    
    Args:
        phase2_embedding (nn.Embedding): Phase 2のEmbedding層
        use_complex32 (bool): complex32を使用するか
    
    Returns:
        ComplexEmbedding: Phase 3のComplexEmbedding
    
    Implementation:
        - 実部: Phase 2の重みをそのまま使用
        - 虚部: ゼロで初期化（学習により獲得）
    
    Examples:
        >>> phase2_emb = nn.Embedding(50000, 512)
        >>> phase3_emb = convert_phase2_embedding_to_complex(phase2_emb)
    """
    vocab_size, d_model = phase2_embedding.weight.shape
    
    # ComplexEmbeddingを作成
    complex_emb = ComplexEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        use_complex32=use_complex32
    )
    
    # 実部: Phase 2の重みをコピー
    with torch.no_grad():
        complex_emb.token_embedding_real.weight.copy_(phase2_embedding.weight)
        
        # 虚部: ゼロで初期化（学習により獲得）
        complex_emb.token_embedding_imag.weight.zero_()
    
    return complex_emb


def analyze_complex_embedding_interference(
    embedding: ComplexEmbedding,
    token_ids: torch.Tensor
) -> dict:
    """
    複素数埋め込みの干渉効果を分析
    
    Args:
        embedding (ComplexEmbedding): 分析するComplexEmbedding
        token_ids (torch.Tensor): トークンID (N,)
    
    Returns:
        dict: 分析結果
            - 'magnitude': 各トークンの振幅 (N,)
            - 'phase': 各トークンの位相 (N,)
            - 'interference': トークン間の干渉強度 (N, N)
    
    Physical Interpretation:
        - 振幅: トークンの意味の強さ
        - 位相: トークンの文脈依存性
        - 干渉: トークン間の意味的な相互作用
    
    Examples:
        >>> embedding = ComplexEmbedding(vocab_size=50000, d_model=512)
        >>> token_ids = torch.tensor([10, 20, 30])
        >>> analysis = analyze_complex_embedding_interference(embedding, token_ids)
        >>> print(analysis['magnitude'])  # 各トークンの振幅
    """
    with torch.no_grad():
        # 埋め込みを取得
        z = embedding(token_ids.unsqueeze(0))  # (1, N, D)
        
        if isinstance(z, ComplexTensor):
            # ComplexTensor形式
            magnitude = z.abs().squeeze(0)  # (N, D)
            phase = z.angle().squeeze(0)  # (N, D)
        else:
            # complex64形式
            magnitude = torch.abs(z).squeeze(0)  # (N, D)
            phase = torch.angle(z).squeeze(0)  # (N, D)
        
        # トークン間の干渉強度を計算
        # 干渉 = |z_i + z_j|² - |z_i|² - |z_j|²
        N = token_ids.shape[0]
        interference = torch.zeros(N, N)
        
        for i in range(N):
            for j in range(i+1, N):
                if isinstance(z, ComplexTensor):
                    z_i = ComplexTensor(z.real[0, i:i+1, :], z.imag[0, i:i+1, :])
                    z_j = ComplexTensor(z.real[0, j:j+1, :], z.imag[0, j:j+1, :])
                    z_sum = z_i + z_j
                    
                    interference_ij = (
                        z_sum.abs_squared().sum() - 
                        z_i.abs_squared().sum() - 
                        z_j.abs_squared().sum()
                    ).item()
                else:
                    z_i = z[0, i:i+1, :]
                    z_j = z[0, j:j+1, :]
                    z_sum = z_i + z_j
                    
                    interference_ij = (
                        (torch.abs(z_sum) ** 2).sum() - 
                        (torch.abs(z_i) ** 2).sum() - 
                        (torch.abs(z_j) ** 2).sum()
                    ).item()
                
                interference[i, j] = interference_ij
                interference[j, i] = interference_ij
        
        return {
            'magnitude': magnitude.mean(dim=-1),  # (N,) 次元平均
            'phase': phase.mean(dim=-1),  # (N,) 次元平均
            'interference': interference  # (N, N)
        }
