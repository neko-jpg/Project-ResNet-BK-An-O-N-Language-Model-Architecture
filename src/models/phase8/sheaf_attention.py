"""
Sheaf Attention Module for Phase 8.

物理的直観:
- 各ヘッドは局所的な「視点」を持つ
- 重複する領域で矛盾する情報は除外
- 整合する情報のみが最終出力に寄与

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
import math


@dataclass
class SheafAttentionConfig:
    """Configuration for Sheaf Attention module."""
    d_model: int = 256
    num_heads: int = 8
    agreement_threshold: float = 0.1
    use_cohomology: bool = True
    dropout: float = 0.1
    
    def to_json(self) -> str:
        """Serialize configuration to JSON."""
        return json.dumps({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'agreement_threshold': self.agreement_threshold,
            'use_cohomology': self.use_cohomology,
            'dropout': self.dropout,
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SheafAttentionConfig':
        """Parse configuration from JSON."""
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class SheafDiagnostics:
    """Diagnostics from Sheaf Attention computation."""
    agreement_matrix: torch.Tensor  # (B, H, H) ヘッド間の整合性
    consensus_weights: torch.Tensor  # (B, H) 各ヘッドの重み
    cohomology_obstruction: Optional[torch.Tensor] = None  # 大域的障害
    filtered_heads: Optional[torch.Tensor] = None  # フィルタされたヘッド数
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'agreement_matrix': self.agreement_matrix.tolist() if isinstance(self.agreement_matrix, torch.Tensor) else self.agreement_matrix,
            'consensus_weights': self.consensus_weights.tolist() if isinstance(self.consensus_weights, torch.Tensor) else self.consensus_weights,
            'cohomology_obstruction': self.cohomology_obstruction.tolist() if self.cohomology_obstruction is not None and isinstance(self.cohomology_obstruction, torch.Tensor) else self.cohomology_obstruction,
            'filtered_heads': self.filtered_heads.tolist() if self.filtered_heads is not None and isinstance(self.filtered_heads, torch.Tensor) else self.filtered_heads,
        }


class SheafAttentionModule(nn.Module):
    """
    Sheaf Attention for Structural Consistency.
    
    物理的直観:
    - 各ヘッドは局所的な「視点」を持つ
    - 重複する領域で矛盾する情報は除外
    - 整合する情報のみが最終出力に寄与
    
    Args:
        config_or_d_model: SheafAttentionConfigまたはモデル次元
        num_heads: アテンションヘッド数（configを使用しない場合）
        agreement_threshold: 整合性閾値（デフォルト: 0.1）
        use_cohomology: コホモロジー計算を使用するか
        dropout: ドロップアウト率
    """
    
    def __init__(
        self,
        config_or_d_model = None,
        num_heads: int = 8,
        agreement_threshold: float = 0.1,
        use_cohomology: bool = True,
        dropout: float = 0.1,
        *,
        d_model: int = None,
    ):
        super().__init__()
        
        # Configオブジェクトまたは個別パラメータをサポート
        if isinstance(config_or_d_model, SheafAttentionConfig):
            config = config_or_d_model
            d_model = config.d_model
            num_heads = config.num_heads
            agreement_threshold = config.agreement_threshold
            use_cohomology = config.use_cohomology
            dropout = config.dropout
        elif config_or_d_model is not None:
            d_model = config_or_d_model
        elif d_model is None:
            d_model = 256
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.agreement_threshold = agreement_threshold
        self.use_cohomology = use_cohomology
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Restriction maps between heads
        # 各ペア(i, j)に対してi→jへの写像を学習
        num_pairs = num_heads * (num_heads - 1) // 2
        self.restriction_maps = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False)
            for _ in range(num_pairs)
        ])
        
        # 整合性スコアの学習可能なスケール
        self.agreement_scale = nn.Parameter(torch.tensor(1.0))
        
        self.dropout = nn.Dropout(dropout)
        
        # 診断情報の保存
        self._last_agreement_matrix = None
        self._last_consensus_weights = None
    
    def _get_pair_index(self, i: int, j: int) -> int:
        """Get the index of the restriction map for pair (i, j)."""
        if i > j:
            i, j = j, i
        # Index in upper triangular matrix (excluding diagonal)
        return i * self.num_heads - i * (i + 1) // 2 + j - i - 1
    
    def compute_restriction_maps(
        self,
        head_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        ヘッド間のRestriction Mapを計算。
        
        Args:
            head_outputs: (B, H, N, D/H) 各ヘッドの出力
            
        Returns:
            agreement_matrix: (B, H, H) ヘッド間の整合性
        """
        B, H, N, D_h = head_outputs.shape
        device = head_outputs.device
        
        agreement = torch.zeros(B, H, H, device=device)
        
        for i in range(H):
            for j in range(i + 1, H):
                idx = self._get_pair_index(i, j)
                
                # Restriction map適用: head_i → head_j の空間へ写像
                restricted_i = self.restriction_maps[idx](head_outputs[:, i])  # (B, N, D_h)
                
                # 整合性計算: 写像後の差のノルム
                diff = (restricted_i - head_outputs[:, j]).norm(dim=-1).mean(dim=-1)  # (B,)
                
                # 整合性スコア: 差が小さいほど高い
                score = torch.exp(-diff * self.agreement_scale / self.agreement_threshold)
                
                agreement[:, i, j] = score
                agreement[:, j, i] = score
        
        # 対角成分は1（自己整合性）
        agreement[:, range(H), range(H)] = 1.0
        
        return agreement
    
    def consensus_aggregation(
        self,
        head_outputs: torch.Tensor,
        agreement_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        整合性に基づく重み付け集約。
        
        **Property 6: Sheaf Consensus Consistency**
        整合性の低いヘッドの情報をフィルタリング。
        
        Args:
            head_outputs: (B, H, N, D/H) 各ヘッドの出力
            agreement_matrix: (B, H, H) 整合性行列
            
        Returns:
            output: (B, N, D) 集約された出力
            consensus_weights: (B, H) 各ヘッドの重み
        """
        B, H, N, D_h = head_outputs.shape
        
        # 各ヘッドの整合性スコア（他ヘッドとの平均整合性）
        # 対角成分を除外して計算
        mask = ~torch.eye(H, dtype=torch.bool, device=agreement_matrix.device)
        masked_agreement = agreement_matrix * mask.unsqueeze(0)
        consensus_scores = masked_agreement.sum(dim=-1) / (H - 1)  # (B, H)
        
        # 閾値以下のヘッドをフィルタ
        filter_mask = consensus_scores > self.agreement_threshold
        
        # Softmax重み（フィルタされたヘッドは重みが低くなる）
        filtered_scores = consensus_scores * filter_mask.float()
        consensus_weights = F.softmax(filtered_scores + 1e-10, dim=-1)  # (B, H)
        
        # 重み付け集約
        weighted_heads = head_outputs * consensus_weights[:, :, None, None]  # (B, H, N, D_h)
        output = weighted_heads.sum(dim=1)  # (B, N, D_h)
        
        # 次元を戻す（H個のヘッドを連結）
        # ここでは単純に繰り返すのではなく、out_projで処理
        output = output.repeat(1, 1, H)  # (B, N, D)
        
        return output, consensus_weights
    
    def compute_cohomology(
        self,
        agreement_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        層コホモロジーを計算して大域的障害を検出。
        
        簡略化された実装: 整合性行列の固有値分解を使用。
        完全な整合性 → 全固有値が1に近い
        障害あり → 小さい固有値が存在
        
        Args:
            agreement_matrix: (B, H, H) 整合性行列
            
        Returns:
            obstruction: (B,) 障害スコア [0, 1]
        """
        B, H, _ = agreement_matrix.shape
        
        # 固有値を計算
        eigenvalues = torch.linalg.eigvalsh(agreement_matrix)  # (B, H)
        
        # 最小固有値が障害の指標
        # 完全整合なら全固有値≈1、障害があれば小さい固有値が存在
        min_eigenvalue = eigenvalues.min(dim=-1).values  # (B,)
        
        # 障害スコア: 1 - min_eigenvalue（正規化）
        obstruction = 1.0 - min_eigenvalue.clamp(0, 1)
        
        return obstruction

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, Optional[SheafDiagnostics]]:
        """
        Sheaf Attentionの順伝播。
        
        Args:
            x: (B, N, D) 入力
            mask: (B, N, N) アテンションマスク（オプション）
            return_diagnostics: 診断情報を返すか
            
        Returns:
            output: (B, N, D) 出力
            diagnostics: SheafDiagnostics（オプション）
        """
        B, N, D = x.shape
        H = self.num_heads
        D_h = self.head_dim
        
        # Q, K, V projection
        q = self.q_proj(x).view(B, N, H, D_h).transpose(1, 2)  # (B, H, N, D_h)
        k = self.k_proj(x).view(B, N, H, D_h).transpose(1, 2)  # (B, H, N, D_h)
        v = self.v_proj(x).view(B, N, H, D_h).transpose(1, 2)  # (B, H, N, D_h)
        
        # Scaled dot-product attention per head
        scale = math.sqrt(D_h)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, N, N)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Attention output per head
        head_outputs = torch.matmul(attn_weights, v)  # (B, H, N, D_h)
        
        # Compute restriction maps and agreement
        agreement_matrix = self.compute_restriction_maps(head_outputs)
        
        # Consensus aggregation
        aggregated, consensus_weights = self.consensus_aggregation(head_outputs, agreement_matrix)
        
        # Output projection
        output = self.out_proj(aggregated)
        
        # Store for diagnostics
        self._last_agreement_matrix = agreement_matrix
        self._last_consensus_weights = consensus_weights
        
        diagnostics = None
        if return_diagnostics:
            cohomology_obstruction = None
            if self.use_cohomology:
                cohomology_obstruction = self.compute_cohomology(agreement_matrix)
            
            # Count filtered heads
            filtered_heads = (consensus_weights < self.agreement_threshold).sum(dim=-1).float()
            
            diagnostics = SheafDiagnostics(
                agreement_matrix=agreement_matrix,
                consensus_weights=consensus_weights,
                cohomology_obstruction=cohomology_obstruction,
                filtered_heads=filtered_heads,
            )
        
        return output, diagnostics
    
    def get_last_diagnostics(self) -> Optional[SheafDiagnostics]:
        """Get diagnostics from the last forward pass."""
        if self._last_agreement_matrix is None:
            return None
        
        cohomology_obstruction = None
        if self.use_cohomology:
            cohomology_obstruction = self.compute_cohomology(self._last_agreement_matrix)
        
        filtered_heads = (self._last_consensus_weights < self.agreement_threshold).sum(dim=-1).float()
        
        return SheafDiagnostics(
            agreement_matrix=self._last_agreement_matrix,
            consensus_weights=self._last_consensus_weights,
            cohomology_obstruction=cohomology_obstruction,
            filtered_heads=filtered_heads,
        )


class SheafSection:
    """
    Represents a section of the sheaf over the input sequence.
    
    Used for serialization and analysis.
    """
    
    def __init__(
        self,
        head_index: int,
        data: torch.Tensor,
        restriction_to: Optional[Dict[int, torch.Tensor]] = None
    ):
        self.head_index = head_index
        self.data = data
        self.restriction_to = restriction_to or {}
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'head_index': self.head_index,
            'data_shape': list(self.data.shape),
            'data_norm': self.data.norm().item(),
            'restrictions': {
                str(k): v.norm().item() for k, v in self.restriction_to.items()
            }
        }


def serialize_sheaf_structure(
    module: SheafAttentionModule,
    head_outputs: torch.Tensor
) -> str:
    """
    Serialize the sheaf structure to JSON.
    
    Args:
        module: SheafAttentionModule
        head_outputs: (B, H, N, D_h) head outputs
        
    Returns:
        JSON string representing the sheaf structure
    """
    B, H, N, D_h = head_outputs.shape
    
    structure = {
        'num_heads': H,
        'head_dim': D_h,
        'sequence_length': N,
        'batch_size': B,
        'sections': [],
        'restriction_maps': [],
    }
    
    # Add section information
    for h in range(H):
        section = {
            'head_index': h,
            'data_norm': head_outputs[:, h].norm(dim=-1).mean().item(),
        }
        structure['sections'].append(section)
    
    # Add restriction map information
    for i in range(H):
        for j in range(i + 1, H):
            idx = module._get_pair_index(i, j)
            restriction = {
                'from_head': i,
                'to_head': j,
                'map_index': idx,
                'weight_norm': module.restriction_maps[idx].weight.norm().item(),
            }
            structure['restriction_maps'].append(restriction)
    
    return json.dumps(structure, indent=2)


def create_sheaf_attention(
    d_model: int = 256,
    num_heads: int = 8,
    agreement_threshold: float = 0.1,
    **kwargs
) -> SheafAttentionModule:
    """
    Factory function for creating SheafAttentionModule.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        agreement_threshold: Agreement threshold
        **kwargs: Additional arguments
        
    Returns:
        SheafAttentionModule instance
    """
    return SheafAttentionModule(
        d_model=d_model,
        num_heads=num_heads,
        agreement_threshold=agreement_threshold,
        **kwargs
    )
