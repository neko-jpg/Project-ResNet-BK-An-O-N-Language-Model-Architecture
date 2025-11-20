"""
Phase 2 Integrated Model - Breath of Life

This module implements the complete Phase 2 architecture that integrates:
1. Non-Hermitian Forgetting (散逸的忘却)
2. Dissipative Hebbian Dynamics (散逸的Hebbian動力学)
3. SNR-based Memory Selection (SNRベースの記憶選択)
4. Memory Resonance (記憶共鳴)
5. Zeta Initialization (ゼータ初期化)

Architecture:
    Input → ZetaEmbedding → Phase2Block × N → Output
    
    Phase2Block:
        x → NonHermitian → BK-Core → DissipativeHebbian → SNRFilter → MemoryResonance → FFN → x

Physical Interpretation:
    Phase 2 transforms the static Phase 1 Hamiltonian into a dynamic system where:
    - Memory state M influences potential V(x, M)
    - Natural forgetting through dissipation Γ
    - Adaptive memory selection via SNR
    - Resonance-based memory organization

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
Author: Project MUSE Team
Date: 2025-01-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, Any
from dataclasses import dataclass
import warnings

from .non_hermitian import DissipativeBKLayer, NonHermitianPotential
from .dissipative_hebbian import DissipativeHebbianLayer, LyapunovStabilityMonitor
from .memory_selection import SNRMemoryFilter, MemoryImportanceEstimator
from .memory_resonance import MemoryResonanceLayer, ZetaBasisTransform
from .zeta_init import ZetaEmbedding, ZetaInitializer


class Phase2Block(nn.Module):
    """
    Phase 2の単一ブロック
    
    構造:
        x → [LN] → NonHermitian+BK-Core → [Residual]
          → [LN] → DissipativeHebbian → SNRFilter → MemoryResonance → [Residual]
          → [LN] → FFN → [Residual]
    
    データフロー:
        1. NonHermitian: 複素ポテンシャル V - iΓ を生成
        2. BK-Core: 三重対角行列の逆行列対角要素を計算
        3. DissipativeHebbian: Fast Weightsを更新 (W_new = exp(-Γ*dt)*W_old + η*(k^T v))
        4. SNRFilter: 重要な記憶を選択的に保持
        5. MemoryResonance: ゼータ基底で対角化し、共鳴する記憶を強化
        6. FFN: 標準的なフィードフォワード層
    
    Args:
        d_model: モデル次元
        n_seq: シーケンス長
        num_heads: ヘッド数 (default: 8)
        head_dim: ヘッド次元 (default: 64)
        use_triton: Tritonカーネルを使用するか (default: True)
        ffn_dim: FFN中間次元 (default: 4 * d_model)
        dropout: ドロップアウト率 (default: 0.1)
        **kwargs: 各コンポーネントへの追加引数
    
    Requirements: 6.1, 6.2, 6.3
    """
    
    def __init__(
        self,
        d_model: int,
        n_seq: int,
        num_heads: int = 8,
        head_dim: int = 64,
        use_triton: bool = True,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_seq = n_seq
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        if ffn_dim is None:
            ffn_dim = 4 * d_model
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        
        # 1. Non-Hermitian + BK-Core (Requirement 6.1)
        self.dissipative_bk = DissipativeBKLayer(
            d_model=d_model,
            n_seq=n_seq,
            use_triton=use_triton,
            **kwargs.get('non_hermitian_kwargs', {})
        )
        
        # BK特徴を元の次元に射影
        # BK-Coreは(B, N, 2)を出力するので、d_modelに拡張
        self.bk_proj = nn.Linear(2, d_model)
        
        # 2. Dissipative Hebbian (Requirement 6.1)
        self.hebbian = DissipativeHebbianLayer(
            d_model=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            **kwargs.get('hebbian_kwargs', {})
        )
        
        # 3. SNR Filter (Requirement 6.1)
        self.snr_filter = SNRMemoryFilter(
            **kwargs.get('snr_kwargs', {})
        )
        
        # 4. Memory Resonance (Requirement 6.1)
        self.resonance = MemoryResonanceLayer(
            d_model=d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            **kwargs.get('resonance_kwargs', {})
        )
        
        # 5. FFN (Requirement 6.1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        
        # Fast Weight State（推論時の状態保持用）
        self.fast_weight_state = None
        
        # ドロップアウト
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through Phase2Block
        
        Args:
            x: (B, N, D) 入力テンソル
            return_diagnostics: 診断情報を返すか (Requirement 6.2)
        
        Returns:
            output: (B, N, D) 出力テンソル
            diagnostics: 診断情報の辞書 (return_diagnostics=True の場合)
        
        Requirements: 6.2, 6.3
        """
        B, N, D = x.shape
        
        # 診断情報の初期化 (Requirement 6.2)
        diagnostics = {} if return_diagnostics else None
        
        # ===== 1. Non-Hermitian + BK-Core =====
        x_norm1 = self.ln1(x)
        
        # BK-Core計算
        bk_features, V_complex = self.dissipative_bk(
            x_norm1, 
            return_potential=return_diagnostics
        )  # (B, N, 2), (B, N) complex
        
        # BK特徴を元の次元に射影
        bk_proj = self.bk_proj(bk_features)  # (B, N, D)
        
        # 残差接続 (Requirement 6.3)
        x = x + self.dropout(bk_proj)
        
        # Γ（減衰率）を取得
        gamma = self.dissipative_bk.get_gamma(x_norm1)  # (B, N)
        
        if return_diagnostics:
            diagnostics['gamma'] = gamma.detach()
            diagnostics['v_complex'] = V_complex.detach() if V_complex is not None else None
            diagnostics['bk_features'] = bk_features.detach()
        
        # ===== 2. Dissipative Hebbian =====
        x_norm2 = self.ln2(x)
        
        hebbian_out, self.fast_weight_state, potential_feedback = self.hebbian(
            x_norm2,
            gamma,
            state=self.fast_weight_state,
            return_potential_feedback=return_diagnostics,
        )  # (B, N, D), (B, H, D_h, D_h), (B, N) or None
        
        # 残差接続 (Requirement 6.3)
        x = x + self.dropout(hebbian_out)
        
        if return_diagnostics:
            diagnostics['hebbian_output'] = hebbian_out.detach()
            diagnostics['fast_weight_energy'] = torch.norm(self.fast_weight_state).item() if self.fast_weight_state is not None else 0.0
            diagnostics['potential_feedback'] = potential_feedback.detach() if potential_feedback is not None else None
        
        # ===== 3. SNR Filter =====
        if self.fast_weight_state is not None:
            # SNRに基づいてΓとηを調整
            # gamma is (B, N), but SNR filter expects (B,) - use mean across sequence
            gamma_mean = gamma.mean(dim=1)  # (B,)
            adjusted_gamma, adjusted_eta = self.snr_filter(
                self.fast_weight_state,
                gamma_mean,
                self.hebbian.eta
            )
            # Expand adjusted_gamma back to (B, N)
            adjusted_gamma = adjusted_gamma.unsqueeze(1).expand(-1, N)  # (B, N)
            
            # 次回のために調整値を保存
            # Note: etaは次のforward時に使用される
            self.hebbian.eta = adjusted_eta
            
            if return_diagnostics:
                diagnostics['snr_stats'] = self.snr_filter.get_statistics()
                diagnostics['adjusted_gamma'] = adjusted_gamma.detach()
                diagnostics['adjusted_eta'] = adjusted_eta
        else:
            if return_diagnostics:
                diagnostics['snr_stats'] = {}
                diagnostics['adjusted_gamma'] = gamma.detach()
                diagnostics['adjusted_eta'] = self.hebbian.eta
        
        # ===== 4. Memory Resonance =====
        if self.fast_weight_state is not None:
            filtered_weights, resonance_info = self.resonance(
                self.fast_weight_state, x
            )
            self.fast_weight_state = filtered_weights
            
            if return_diagnostics:
                diagnostics['resonance_info'] = resonance_info
        else:
            if return_diagnostics:
                diagnostics['resonance_info'] = {}
        
        # ===== 5. FFN =====
        x_norm3 = self.ln3(x)
        ffn_out = self.ffn(x_norm3)
        
        # 残差接続 (Requirement 6.3)
        x = x + ffn_out
        
        if return_diagnostics:
            diagnostics['ffn_output'] = ffn_out.detach()
        
        # ===== 安定性チェック =====
        if return_diagnostics and self.fast_weight_state is not None:
            # Lyapunov安定性メトリクス
            decay = torch.exp(-gamma.mean() * self.hebbian.dt)
            decay_expanded = decay.view(1, 1, 1, 1)
            update_dummy = torch.zeros_like(self.fast_weight_state)
            
            stability_metrics = self.hebbian.stability_monitor.check(
                self.fast_weight_state,
                decay_expanded,
                update_dummy
            )
            diagnostics['stability'] = stability_metrics
        
        if return_diagnostics:
            return x, diagnostics
        return x
    
    def reset_state(self):
        """Fast Weight状態をリセット"""
        self.fast_weight_state = None
        self.hebbian.reset_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """ブロック全体の統計情報を取得"""
        stats = {
            'hebbian': self.hebbian.get_statistics(),
            'snr': self.snr_filter.get_statistics(),
            'non_hermitian': self.dissipative_bk.potential.get_statistics(),
        }
        return stats


class Phase2IntegratedModel(nn.Module):
    """
    Phase 2統合モデル - 生命の息吹
    
    アーキテクチャ:
        Input → Token Embedding + Zeta Position Embedding
              → Phase2Block × N
              → Layer Norm
              → LM Head
              → Output Logits
    
    Phase2Block:
        x → NonHermitian → BK-Core → DissipativeHebbian → SNRFilter → MemoryResonance → FFN → x
    
    物理的解釈:
        Phase 2は静的なPhase 1ハミルトニアンを動的システムに変換:
        - 記憶状態Mがポテンシャル V(x, M) に影響
        - 散逸Γによる自然な忘却
        - SNRによる適応的記憶選択
        - 共鳴ベースの記憶組織化
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元 (default: 512)
        n_layers: レイヤー数 (default: 6)
        n_seq: シーケンス長 (default: 1024)
        num_heads: ヘッド数 (default: 8)
        head_dim: ヘッド次元 (default: 64)
        use_triton: Tritonカーネルを使用するか (default: True)
        ffn_dim: FFN中間次元 (default: None = 4 * d_model)
        dropout: ドロップアウト率 (default: 0.1)
        zeta_embedding_trainable: Zeta埋め込みを学習可能にするか (default: False)
        phase1_config: Phase 1設定（互換性のため、オプション）
        **kwargs: 各コンポーネントへの追加引数
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_seq: int = 1024,
        num_heads: int = 8,
        head_dim: int = 64,
        use_triton: bool = True,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        zeta_embedding_trainable: bool = False,
        phase1_config: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_seq = n_seq
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_triton = use_triton
        
        if ffn_dim is None:
            ffn_dim = 4 * d_model
        
        # Phase 1互換性 (Requirement 6.2, 6.4)
        self.phase1_config = phase1_config
        
        # ===== Subtask 9.1: Embeddingレイヤーの実装 =====
        # Token Embedding (Requirement 6.1)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position Embedding (Zeta) (Requirement 6.1)
        self.position_embedding = ZetaEmbedding(
            max_len=n_seq,
            d_model=d_model,
            trainable=zeta_embedding_trainable,
            scale=1.0
        )
        
        # Embedding Dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # ===== Phase2Block × N を積み重ねる =====
        # (Requirement 6.1)
        self.blocks = nn.ModuleList([
            Phase2Block(
                d_model=d_model,
                n_seq=n_seq,
                num_heads=num_heads,
                head_dim=head_dim,
                use_triton=use_triton,
                ffn_dim=ffn_dim,
                dropout=dropout,
                **kwargs
            )
            for _ in range(n_layers)
        ])
        
        # ===== 出力層 =====
        # Final Layer Norm (Requirement 6.1)
        self.ln_f = nn.LayerNorm(d_model)
        
        # LM Head (Requirement 6.1)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying（オプション）
        # Token EmbeddingとLM Headで重みを共有
        # self.lm_head.weight = self.token_embedding.weight
        
        # ===== Subtask 9.2: モデル初期化の実装 =====
        # ゼータ初期化を適用 (Requirement 6.1)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        ゼータ初期化を適用
        
        実装詳細:
            1. Token Embeddingにゼータ初期化
            2. すべてのLinear層にゼータ初期化
            3. Position Embeddingは既にZetaEmbeddingで初期化済み
        
        Requirements: 6.1
        """
        # Token Embeddingにゼータ初期化
        ZetaInitializer.initialize_embedding_zeta(
            self.token_embedding,
            scale=0.02  # 小さめのスケールで初期化
        )
        
        # すべてのLinear層にゼータ初期化
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # LM Headは除外（大きすぎる可能性があるため）
                if 'lm_head' not in name:
                    try:
                        ZetaInitializer.initialize_linear_zeta(
                            module,
                            scale=0.02
                        )
                    except Exception as e:
                        # 初期化に失敗した場合は警告を出して続行
                        warnings.warn(
                            f"Failed to apply Zeta initialization to {name}: {e}. "
                            f"Using default initialization.",
                            UserWarning
                        )
        
        # LM Headは標準的な初期化
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through Phase2IntegratedModel
        
        Args:
            input_ids: (B, N) トークンID
            attention_mask: (B, N) アテンションマスク（オプション、現在未使用）
            return_diagnostics: 診断情報を返すか (Requirement 6.2, 6.5)
        
        Returns:
            logits: (B, N, V) 出力ロジット
            diagnostics: 診断情報の辞書 (return_diagnostics=True の場合)
        
        Requirements: 6.2, 6.5
        """
        B, N = input_ids.shape
        
        # 診断情報の初期化 (Requirement 6.2)
        diagnostics = {
            'layer_outputs': [],
            'gamma_values': [],
            'snr_stats': [],
            'resonance_info': [],
            'stability_metrics': [],
        } if return_diagnostics else None
        
        # ===== Embedding =====
        # Token Embedding
        x = self.token_embedding(input_ids)  # (B, N, D)
        
        # Position Embedding (Zeta)
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)  # (B, N)
        pos_emb = self.position_embedding(positions)  # (B, N, D)
        
        # Embedding結合
        x = x + pos_emb
        x = self.emb_dropout(x)
        
        if return_diagnostics:
            diagnostics['input_embeddings'] = x.detach()
        
        # ===== Phase2Block × N =====
        for i, block in enumerate(self.blocks):
            if return_diagnostics:
                x, block_diag = block(x, return_diagnostics=True)
                
                # 診断情報を収集 (Requirement 6.2, 6.5)
                diagnostics['layer_outputs'].append(x.detach())
                diagnostics['gamma_values'].append(block_diag.get('gamma'))
                diagnostics['snr_stats'].append(block_diag.get('snr_stats', {}))
                diagnostics['resonance_info'].append(block_diag.get('resonance_info', {}))
                diagnostics['stability_metrics'].append(block_diag.get('stability', {}))
            else:
                x = block(x)
        
        # ===== 出力層 =====
        # Final Layer Norm
        x = self.ln_f(x)
        
        # LM Head
        logits = self.lm_head(x)  # (B, N, V)
        
        if return_diagnostics:
            diagnostics['final_hidden_states'] = x.detach()
            diagnostics['logits'] = logits.detach()
            return logits, diagnostics
        
        return logits
    
    def reset_state(self) -> None:
        """
        すべてのブロックのFast Weight状態をリセット
        
        使用例:
            >>> model.reset_state()  # 新しいシーケンスの開始時
        """
        for block in self.blocks:
            block.reset_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        モデル全体の統計情報を取得
        
        Returns:
            stats: 統計情報の辞書
                - 'num_parameters': パラメータ数
                - 'num_layers': レイヤー数
                - 'block_stats': 各ブロックの統計
        """
        # パラメータ数を計算
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 各ブロックの統計
        block_stats = []
        for i, block in enumerate(self.blocks):
            block_stats.append({
                'layer': i,
                **block.get_statistics()
            })
        
        return {
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable_params,
            'num_layers': self.n_layers,
            'd_model': self.d_model,
            'vocab_size': self.vocab_size,
            'n_seq': self.n_seq,
            'block_stats': block_stats,
        }
    
    def extra_repr(self) -> str:
        """モジュールの追加情報を返す"""
        return (
            f'vocab_size={self.vocab_size}, d_model={self.d_model}, '
            f'n_layers={self.n_layers}, n_seq={self.n_seq}, '
            f'num_heads={self.num_heads}, head_dim={self.head_dim}, '
            f'use_triton={self.use_triton}'
        )


# Export
__all__ = [
    'Phase2Block',
    'Phase2IntegratedModel',
]
