"""
Phase 3 Stage 1 Model - Complex Dynamics Foundation

このモジュールは、Phase 3の最初のステージであるComplex Dynamics Foundationを実装します。

Stage 1の目的:
    複素数ニューラルネットワークの基盤を構築し、メモリ効率50%削減を実証する。

Architecture:
    Input: Token IDs (B, N)
    ↓
    [ComplexEmbedding] → z (B, N, D) [complex]
    ↓
    [Phase3Stage1Block] × L layers
        ├─ ComplexLayerNorm
        ├─ ComplexLinear (Self-Attention風の変換)
        ├─ ModReLU
        ├─ Residual Connection
        └─ ComplexLayerNorm
    ↓
    [Output Head] → logits (B, N, vocab_size)

Stage 1完了条件:
    - Perplexity: WikiText-2で Phase 2比 +3%以内
    - VRAM削減: Phase 2比 52%以下
    - 数値安定性: ランダム入力100回試行で NaN発生率 0%
    - 勾配健全性: 全層の勾配ノルムが 1e-6以上、1e3以下
    - メモリレイアウト: ComplexTensorが Planar形式 で実装されていること

Requirements:
    - Requirement 1.15: ComplexEmbedding → ComplexLinear × N → Output の流れ
    - Requirement 1.16: Phase 2モデルの重みをComplexTensorに変換
    - Requirement 1.17: Forward/Backward passが正常に動作

Physical Intuition:
    複素数ニューラルネットワークは、実部（振幅）と虚部（位相）を独立かつ相互作用させながら処理します。
    - 実部: 情報の「強さ」（意味の明確さ）
    - 虚部: 情報の「方向性」（文脈、ニュアンス）
    
    これにより、否定形・皮肉・多義語などの干渉効果を量子力学的にモデリングできます。

Author: Project MUSE Team
Date: 2025-01-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union
import warnings

from .complex_tensor import ComplexTensor
from .complex_ops import ComplexLinear, ModReLU, ComplexLayerNorm
from .complex_embedding import ComplexEmbedding


class Phase3Stage1Config:
    """
    Phase 3 Stage 1モデルの設定クラス
    
    Args:
        vocab_size (int): 語彙サイズ
        d_model (int): モデル次元（デフォルト: 512）
        n_layers (int): レイヤー数（デフォルト: 6）
        n_seq (int): 最大シーケンス長（デフォルト: 2048）
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
        zeta_scale (float): Zeta初期化のスケール（デフォルト: 1.0）
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_seq: int = 2048,
        use_complex32: bool = True,
        dropout: float = 0.1,
        zeta_scale: float = 1.0
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = n_seq  # n_seqをmax_seq_lenにマッピング
        self.use_complex32 = use_complex32
        self.dropout = dropout
        self.zeta_scale = zeta_scale


class Phase3Stage1Block(nn.Module):
    """
    Phase 3 Stage 1の単一ブロック
    
    構造:
        x → [ComplexLayerNorm] → [ComplexLinear] → [ModReLU] → [Residual] → x
    
    Physical Interpretation:
        - ComplexLayerNorm: 複素平面上で正規化（振幅と位相の両方を正規化）
        - ComplexLinear: 複素線形変換（意味と文脈の両方を変換）
        - ModReLU: 振幅フィルタリング + 位相保存（弱い信号を抑制、方向性を保持）
        - Residual: 情報の流れを保証（勾配消失を防ぐ）
    
    Args:
        d_model (int): モデル次元
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
    
    Examples:
        >>> block = Phase3Stage1Block(d_model=512)
        >>> z = ComplexTensor(torch.randn(4, 128, 512, dtype=torch.float16),
        ...                   torch.randn(4, 128, 512, dtype=torch.float16))
        >>> z_out = block(z)
        >>> print(z_out.shape)  # torch.Size([4, 128, 512])
    
    Requirements: 1.15
    """
    
    def __init__(
        self,
        d_model: int,
        use_complex32: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_complex32 = use_complex32
        
        # 1. ComplexLayerNorm（前正規化）
        self.norm1 = ComplexLayerNorm(d_model)
        
        # 2. ComplexLinear（Self-Attention風の変換）
        # Q, K, V の代わりに、単一の線形変換を使用（Stage 1では簡略化）
        self.linear = ComplexLinear(d_model, d_model, use_complex32=use_complex32)
        
        # 3. ModReLU（位相保存活性化関数）
        self.activation = ModReLU(d_model, use_half=use_complex32)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 5. ComplexLayerNorm（後正規化）
        self.norm2 = ComplexLayerNorm(d_model)
    
    def forward(
        self,
        z: Union[ComplexTensor, torch.Tensor],
        return_diagnostics: bool = False
    ) -> Union[ComplexTensor, torch.Tensor, Tuple]:
        """
        順伝播（Forward Pass）
        
        Args:
            z: 複素入力（ComplexTensor or complex64）
                - Shape: (B, N, D)
            return_diagnostics: 診断情報を返すか（デフォルト: False）
        
        Returns:
            複素出力（入力と同じ型）
                - Shape: (B, N, D)
            
            return_diagnostics=Trueの場合:
                (output, diagnostics) のタプル
                diagnostics: dict
                    - 'magnitude_mean': 振幅の平均
                    - 'magnitude_std': 振幅の標準偏差
                    - 'phase_mean': 位相の平均
                    - 'phase_std': 位相の標準偏差
        
        Requirements: 1.15
        """
        # 残差接続のための入力を保存
        residual = z
        
        # 1. 前正規化
        z = self.norm1(z)
        
        # 2. 複素線形変換
        z = self.linear(z)
        
        # 3. ModReLU活性化
        z = self.activation(z)
        
        # 4. Dropout（実部と虚部に独立に適用）
        if isinstance(z, ComplexTensor):
            z = ComplexTensor(
                self.dropout(z.real),
                self.dropout(z.imag)
            )
        else:
            # complex64の場合
            z = torch.complex(
                self.dropout(z.real),
                self.dropout(z.imag)
            )
        
        # 5. 残差接続
        z = z + residual
        
        # 6. 後正規化
        z = self.norm2(z)
        
        # 診断情報の収集
        if return_diagnostics:
            diagnostics = self._collect_diagnostics(z)
            return z, diagnostics
        
        return z
    
    def _collect_diagnostics(self, z: Union[ComplexTensor, torch.Tensor]) -> Dict:
        """
        診断情報の収集
        
        Args:
            z: 複素テンソル
        
        Returns:
            dict: 診断情報
        """
        with torch.no_grad():
            if isinstance(z, ComplexTensor):
                magnitude = z.abs()
                phase = z.angle()
            else:
                magnitude = torch.abs(z)
                phase = torch.angle(z)
            
            return {
                'magnitude_mean': magnitude.mean().item(),
                'magnitude_std': magnitude.std().item(),
                'phase_mean': phase.mean().item(),
                'phase_std': phase.std().item(),
            }


class Phase3Stage1Model(nn.Module):
    """
    Phase 3 Stage 1 統合モデル
    
    このモデルは、Complex Dynamics Foundationを実装し、
    Phase 2モデルを複素数化します。
    
    Architecture:
        Input: Token IDs (B, N)
        ↓
        [ComplexEmbedding] → z (B, N, D) [complex]
        ↓
        [Phase3Stage1Block] × L layers
        ↓
        [Output Head] → logits (B, N, vocab_size)
    
    Args:
        vocab_size (int): 語彙サイズ
        d_model (int): モデル次元
        n_layers (int): レイヤー数（デフォルト: 6）
        max_seq_len (int): 最大シーケンス長（デフォルト: 2048）
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
        zeta_scale (float): Zeta初期化のスケール（デフォルト: 1.0）
    
    Examples:
        >>> # 基本的な使用方法
        >>> model = Phase3Stage1Model(vocab_size=50000, d_model=512, n_layers=6)
        >>> input_ids = torch.randint(0, 50000, (4, 128))
        >>> logits = model(input_ids)
        >>> print(logits.shape)  # torch.Size([4, 128, 50000])
        
        >>> # 診断情報付き
        >>> logits, diagnostics = model(input_ids, return_diagnostics=True)
        >>> print(diagnostics.keys())  # ['layer_0', 'layer_1', ..., 'output']
    
    Requirements: 1.15, 1.16, 1.17
    """
    
    def __init__(
        self,
        config: Optional[Phase3Stage1Config] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        n_layers: int = 6,
        max_seq_len: int = 2048,
        use_complex32: bool = True,
        dropout: float = 0.1,
        zeta_scale: float = 1.0
    ):
        super().__init__()
        
        # Configオブジェクトが渡された場合、それを使用
        if config is not None:
            self.vocab_size = config.vocab_size
            self.d_model = config.d_model
            self.n_layers = config.n_layers
            self.max_seq_len = config.max_seq_len
            self.use_complex32 = config.use_complex32
            dropout = config.dropout
            zeta_scale = config.zeta_scale
        else:
            # 個別のパラメータを使用
            if vocab_size is None or d_model is None:
                raise ValueError("Either config or (vocab_size and d_model) must be provided")
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_layers = n_layers
            self.max_seq_len = max_seq_len
            self.use_complex32 = use_complex32
        
        # ========================================
        # 1. ComplexEmbedding
        # ========================================
        
        self.embedding = ComplexEmbedding(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            use_complex32=self.use_complex32,
            zeta_scale=zeta_scale,
            dropout=dropout
        )
        
        # ========================================
        # 2. Phase3Stage1Block × L layers
        # ========================================
        
        self.blocks = nn.ModuleList([
            Phase3Stage1Block(
                d_model=self.d_model,
                use_complex32=self.use_complex32,
                dropout=dropout
            )
            for _ in range(self.n_layers)
        ])
        
        # ========================================
        # 3. Output Head
        # ========================================
        
        # 最終正規化
        self.final_norm = ComplexLayerNorm(self.d_model)
        
        # Complex → Real 射影
        # 複素数を実数に変換するため、実部と虚部を結合
        # Option 1: 実部のみを使用（シンプル）
        # Option 2: 実部と虚部を結合（情報を保持）
        # ここではOption 1を採用（Stage 1では簡略化）
        # dtype: use_complex32の場合はfloat16、そうでない場合はfloat32
        dtype = torch.float16 if self.use_complex32 else torch.float32
        self.output_proj = nn.Linear(self.d_model, self.vocab_size)
        # 重みをdtypeに変換
        self.output_proj.weight.data = self.output_proj.weight.data.to(dtype)
        if self.output_proj.bias is not None:
            self.output_proj.bias.data = self.output_proj.bias.data.to(dtype)
        
        # 統計情報（デバッグ用）
        self.register_buffer('_forward_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_nan_count', torch.tensor(0, dtype=torch.long))
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        順伝播（Forward Pass）
        
        Args:
            input_ids (torch.Tensor): トークンID (B, N)
            positions (torch.Tensor, optional): 位置インデックス (B, N)
            return_diagnostics (bool): 診断情報を返すか（デフォルト: False）
        
        Returns:
            torch.Tensor: logits (B, N, vocab_size)
            
            return_diagnostics=Trueの場合:
                (logits, diagnostics) のタプル
                diagnostics: dict
                    - 'layer_{i}': 各層の診断情報
                    - 'output': 出力の診断情報
                    - 'nan_detected': NaNが検出されたか
        
        Requirements: 1.15, 1.17
        """
        # 統計情報の更新
        self._forward_count += 1
        
        diagnostics = {} if return_diagnostics else None
        
        # ========================================
        # 1. ComplexEmbedding
        # ========================================
        
        z = self.embedding(input_ids, positions)  # (B, N, D) [complex]
        
        if return_diagnostics:
            diagnostics['embedding'] = self._collect_diagnostics(z)
        
        # ========================================
        # 2. Phase3Stage1Block × L layers
        # ========================================
        
        for i, block in enumerate(self.blocks):
            if return_diagnostics:
                z, block_diag = block(z, return_diagnostics=True)
                diagnostics[f'layer_{i}'] = block_diag
            else:
                z = block(z)
            
            # NaN検出
            if self._check_nan(z):
                self._nan_count += 1
                warnings.warn(
                    f"NaN detected in layer {i}. "
                    f"Total NaN count: {self._nan_count.item()}/{self._forward_count.item()}"
                )
                if return_diagnostics:
                    diagnostics['nan_detected'] = True
                    diagnostics['nan_layer'] = i
        
        # ========================================
        # 3. 最終正規化
        # ========================================
        
        z = self.final_norm(z)
        
        # ========================================
        # 4. Complex → Real 射影
        # ========================================
        
        # 実部のみを使用（Stage 1では簡略化）
        if isinstance(z, ComplexTensor):
            x = z.real  # (B, N, D)
        else:
            x = z.real  # (B, N, D)
        
        # ========================================
        # 5. Output Head
        # ========================================
        
        logits = self.output_proj(x)  # (B, N, vocab_size)
        
        if return_diagnostics:
            diagnostics['output'] = {
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item(),
                'logits_min': logits.min().item(),
                'logits_max': logits.max().item(),
            }
            
            return logits, diagnostics
        
        return logits
    
    def _collect_diagnostics(self, z: Union[ComplexTensor, torch.Tensor]) -> Dict:
        """診断情報の収集"""
        with torch.no_grad():
            if isinstance(z, ComplexTensor):
                magnitude = z.abs()
                phase = z.angle()
            else:
                magnitude = torch.abs(z)
                phase = torch.angle(z)
            
            return {
                'magnitude_mean': magnitude.mean().item(),
                'magnitude_std': magnitude.std().item(),
                'phase_mean': phase.mean().item(),
                'phase_std': phase.std().item(),
            }
    
    def _check_nan(self, z: Union[ComplexTensor, torch.Tensor]) -> bool:
        """NaN検出"""
        if isinstance(z, ComplexTensor):
            return torch.isnan(z.real).any() or torch.isnan(z.imag).any()
        else:
            return torch.isnan(z).any()
    
    def get_statistics(self) -> Dict:
        """
        統計情報の取得
        
        Returns:
            dict: 統計情報
                - 'forward_count': forward呼び出し回数
                - 'nan_count': NaN検出回数
                - 'nan_rate': NaN発生率
        """
        return {
            'forward_count': self._forward_count.item(),
            'nan_count': self._nan_count.item(),
            'nan_rate': self._nan_count.item() / max(self._forward_count.item(), 1),
        }
    
    def reset_statistics(self):
        """統計情報のリセット"""
        self._forward_count.zero_()
        self._nan_count.zero_()


# ========================================
# ユーティリティ関数
# ========================================

def create_phase3_stage1_model(
    vocab_size: int,
    d_model: int = 512,
    n_layers: int = 6,
    max_seq_len: int = 2048,
    use_complex32: bool = True,
    dropout: float = 0.1,
    zeta_scale: float = 1.0
) -> Phase3Stage1Model:
    """
    Phase 3 Stage 1モデルのファクトリー関数
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元（デフォルト: 512）
        n_layers: レイヤー数（デフォルト: 6）
        max_seq_len: 最大シーケンス長（デフォルト: 2048）
        use_complex32: complex32を使用するか（デフォルト: True）
        dropout: ドロップアウト率（デフォルト: 0.1）
        zeta_scale: Zeta初期化のスケール（デフォルト: 1.0）
    
    Returns:
        Phase3Stage1Model: 初期化されたモデル
    
    Examples:
        >>> model = create_phase3_stage1_model(vocab_size=50000)
        >>> print(model)
    """
    return Phase3Stage1Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        use_complex32=use_complex32,
        dropout=dropout,
        zeta_scale=zeta_scale
    )



# ========================================
# Phase 2互換性関数
# ========================================

def convert_phase2_to_complex(
    phase2_model: nn.Module,
    use_complex32: bool = True
) -> Phase3Stage1Model:
    """
    Phase 2モデルをPhase 3 Stage 1モデルに変換
    
    この関数は、Phase 2モデルの重みをComplexTensorに変換し、
    Phase 3 Stage 1モデルを初期化します。
    
    Conversion Strategy:
        1. Embedding層:
            - 実部: Phase 2の重みをそのまま使用
            - 虚部: ゼロで初期化（学習により獲得）
        
        2. 中間層:
            - Phase 2のLinear層 → Phase 3のComplexLinear層
            - 実部: Phase 2の重みをコピー
            - 虚部: 小さなランダム値で初期化（学習の多様性を確保）
        
        3. Output層:
            - Phase 2の重みをそのまま使用（実部のみを使用するため）
    
    Args:
        phase2_model (nn.Module): Phase 2モデル
            - Phase2IntegratedModel または互換性のあるモデル
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
    
    Returns:
        Phase3Stage1Model: 変換されたPhase 3 Stage 1モデル
    
    Examples:
        >>> # Phase 2モデルをロード
        >>> from src.models.phase2.factory import create_phase2_model
        >>> phase2_model = create_phase2_model(vocab_size=50000, d_model=512)
        >>> 
        >>> # Phase 3 Stage 1モデルに変換
        >>> phase3_model = convert_phase2_to_complex(phase2_model)
        >>> 
        >>> # 推論
        >>> input_ids = torch.randint(0, 50000, (4, 128))
        >>> logits = phase3_model(input_ids)
    
    Requirements: 1.16
    """
    # Phase 2モデルの設定を取得
    try:
        # Phase2IntegratedModelの場合（token_embeddingとposition_embeddingを直接持つ）
        if hasattr(phase2_model, 'token_embedding') and hasattr(phase2_model, 'position_embedding'):
            vocab_size = phase2_model.token_embedding.num_embeddings
            d_model = phase2_model.token_embedding.embedding_dim
            n_layers = len(phase2_model.blocks)
            max_seq_len = phase2_model.position_embedding.max_len
        # zeta_embedding属性を使用する形式
        elif hasattr(phase2_model, 'zeta_embedding'):
            vocab_size = phase2_model.zeta_embedding.token_embedding.num_embeddings
            d_model = phase2_model.zeta_embedding.token_embedding.embedding_dim
            n_layers = len(phase2_model.blocks)
            max_seq_len = phase2_model.zeta_embedding.max_len
        # embedding属性を使用する形式
        elif hasattr(phase2_model, 'embedding'):
            vocab_size = phase2_model.embedding.token_embedding.num_embeddings
            d_model = phase2_model.embedding.token_embedding.embedding_dim
            n_layers = len(phase2_model.blocks)
            max_seq_len = phase2_model.embedding.max_len
        else:
            raise AttributeError("No embedding layer found")
    except (AttributeError, TypeError) as e:
        # 互換性のないモデルの場合
        raise ValueError(
            f"phase2_model must be a Phase2IntegratedModel or compatible model. "
            f"Expected attributes: token_embedding, zeta_embedding, or embedding. Error: {e}"
        )
    
    # Phase 3 Stage 1モデルを作成
    phase3_model = Phase3Stage1Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        use_complex32=use_complex32
    )
    
    # ========================================
    # 1. Embedding層の変換
    # ========================================
    
    with torch.no_grad():
        # Token Embedding
        # 実部: Phase 2の重みをコピー
        phase3_model.embedding.token_embedding_real.weight.copy_(
            phase2_model.token_embedding.weight
        )
        
        # 虚部: ゼロで初期化（学習により獲得）
        phase3_model.embedding.token_embedding_imag.weight.zero_()
        
        # Position Embedding
        # Phase 2のZetaEmbeddingをそのまま使用
        if hasattr(phase2_model, 'position_embedding'):
            phase3_model.embedding.position_embedding.load_state_dict(
                phase2_model.position_embedding.state_dict()
            )
    
    # ========================================
    # 2. 中間層の変換
    # ========================================
    
    # Phase 2のブロックからLinear層を抽出して変換
    # 注意: Phase 2とPhase 3のアーキテクチャが異なるため、
    # 完全な変換は不可能。ここでは近似的な変換を行う。
    
    with torch.no_grad():
        for i, (phase2_block, phase3_block) in enumerate(zip(phase2_model.blocks, phase3_model.blocks)):
            # Phase 2のFFN層から重みを抽出
            # Phase 2: FFN = Linear(d_model, ffn_dim) → GELU → Linear(ffn_dim, d_model)
            # Phase 3: ComplexLinear(d_model, d_model)
            
            # 簡略化のため、Phase 2のFFNの最初の層の重みを使用
            if hasattr(phase2_block, 'ffn') and len(phase2_block.ffn) > 0:
                # FFNの最初のLinear層
                ffn_linear = phase2_block.ffn[0]
                
                # Phase 3のComplexLinearに変換
                # 実部: Phase 2の重みを射影（次元が異なる場合）
                if ffn_linear.out_features == d_model:
                    # 次元が一致する場合、そのままコピー
                    phase3_block.linear.weight_real.copy_(
                        ffn_linear.weight[:d_model, :d_model]
                    )
                else:
                    # 次元が異なる場合、平均プーリングで射影
                    # (ffn_dim, d_model) → (d_model, d_model)
                    weight_phase2 = ffn_linear.weight  # (ffn_dim, d_model)
                    # 簡略化: 最初のd_model行を使用
                    phase3_block.linear.weight_real.copy_(
                        weight_phase2[:d_model, :]
                    )
                
                # バイアスのコピー
                if ffn_linear.bias is not None:
                    phase3_block.linear.bias_real.copy_(
                        ffn_linear.bias[:d_model]
                    )
                
                # 虚部: 小さなランダム値で初期化（学習の多様性を確保）
                phase3_block.linear.weight_imag.normal_(0, 0.01)
                if phase3_block.linear.bias_imag is not None:
                    phase3_block.linear.bias_imag.zero_()
    
    # ========================================
    # 3. Output層の変換
    # ========================================
    
    with torch.no_grad():
        # Phase 2のOutput層の重みをコピー
        if hasattr(phase2_model, 'output_proj'):
            phase3_model.output_proj.weight.copy_(
                phase2_model.output_proj.weight
            )
            if phase2_model.output_proj.bias is not None:
                phase3_model.output_proj.bias.copy_(
                    phase2_model.output_proj.bias
                )
    
    print(f"✓ Phase 2モデルをPhase 3 Stage 1モデルに変換しました")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - d_model: {d_model}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - max_seq_len: {max_seq_len}")
    print(f"  - use_complex32: {use_complex32}")
    
    return phase3_model


def load_phase2_checkpoint_and_convert(
    checkpoint_path: str,
    use_complex32: bool = True,
    device: str = 'cpu'
) -> Phase3Stage1Model:
    """
    Phase 2のチェックポイントをロードしてPhase 3 Stage 1モデルに変換
    
    Args:
        checkpoint_path (str): Phase 2チェックポイントのパス
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        device (str): デバイス（'cpu' or 'cuda'）
    
    Returns:
        Phase3Stage1Model: 変換されたPhase 3 Stage 1モデル
    
    Examples:
        >>> phase3_model = load_phase2_checkpoint_and_convert(
        ...     'checkpoints/phase2_best.pt',
        ...     use_complex32=True,
        ...     device='cuda'
        ... )
    
    Requirements: 1.16
    """
    # チェックポイントのロード
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Phase 2モデルの再構築
    from src.models.phase2.factory import create_phase2_model
    
    # チェックポイントから設定を取得
    if 'config' in checkpoint:
        config = checkpoint['config']
        phase2_model = create_phase2_model(**config)
    else:
        # 設定がない場合、デフォルト値を使用
        warnings.warn(
            "Checkpoint does not contain 'config'. Using default configuration."
        )
        phase2_model = create_phase2_model(
            vocab_size=50000,
            d_model=512,
            n_layers=6
        )
    
    # 重みのロード
    if 'model_state_dict' in checkpoint:
        phase2_model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        phase2_model.load_state_dict(checkpoint['state_dict'])
    else:
        phase2_model.load_state_dict(checkpoint)
    
    # Phase 3 Stage 1モデルに変換
    phase3_model = convert_phase2_to_complex(phase2_model, use_complex32=use_complex32)
    
    # デバイスへの移動
    phase3_model = phase3_model.to(device)
    
    print(f"✓ Phase 2チェックポイント '{checkpoint_path}' をロードし、Phase 3 Stage 1モデルに変換しました")
    
    return phase3_model


def compare_phase2_phase3_outputs(
    phase2_model: nn.Module,
    phase3_model: Phase3Stage1Model,
    input_ids: torch.Tensor,
    tolerance: float = 0.1
) -> Dict:
    """
    Phase 2とPhase 3の出力を比較
    
    この関数は、Phase 2モデルとPhase 3 Stage 1モデルの出力を比較し、
    変換が正しく行われたかを検証します。
    
    Args:
        phase2_model (nn.Module): Phase 2モデル
        phase3_model (Phase3Stage1Model): Phase 3 Stage 1モデル
        input_ids (torch.Tensor): 入力トークンID (B, N)
        tolerance (float): 許容誤差（デフォルト: 0.1）
    
    Returns:
        dict: 比較結果
            - 'mse': 平均二乗誤差
            - 'max_diff': 最大差分
            - 'mean_diff': 平均差分
            - 'is_similar': 出力が類似しているか（MSE < tolerance）
    
    Examples:
        >>> input_ids = torch.randint(0, 50000, (4, 128))
        >>> comparison = compare_phase2_phase3_outputs(phase2_model, phase3_model, input_ids)
        >>> print(f"MSE: {comparison['mse']:.6f}")
        >>> print(f"Is similar: {comparison['is_similar']}")
    
    Requirements: 1.16
    """
    with torch.no_grad():
        # Phase 2の出力
        phase2_output = phase2_model(input_ids)
        
        # Phase 3の出力
        phase3_output = phase3_model(input_ids)
        
        # 差分の計算
        diff = phase2_output - phase3_output
        mse = (diff ** 2).mean().item()
        max_diff = diff.abs().max().item()
        mean_diff = diff.abs().mean().item()
        
        # 類似性の判定
        is_similar = mse < tolerance
        
        return {
            'mse': mse,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'is_similar': is_similar,
            'phase2_mean': phase2_output.mean().item(),
            'phase3_mean': phase3_output.mean().item(),
            'phase2_std': phase2_output.std().item(),
            'phase3_std': phase3_output.std().item(),
        }
