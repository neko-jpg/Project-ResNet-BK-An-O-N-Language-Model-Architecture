"""
Phase 3 Stage 2 Model - Hamiltonian ODE Integration

このモジュールは、Phase 3の第2ステージであるHamiltonian ODE Integrationを実装します。

Stage 2の目的:
    エネルギー保存思考機構を追加し、O(1)メモリ学習を実証する。

Architecture:
    Input: Token IDs (B, N)
    ↓
    [ComplexEmbedding] → z (B, N, D) [complex]
    ↓
    [Phase3Stage2Block] × L layers
        ├─ ComplexLayerNorm
        ├─ Complex → Real変換
        ├─ HamiltonianNeuralODE (思考プロセス)
        ├─ Real → Complex変換
        ├─ Residual Connection
        └─ ComplexLayerNorm
    ↓
    [Output Head] → logits (B, N, vocab_size)

Stage 2完了条件:
    - Perplexity: WikiText-2で Stage 1比 +2%以内
    - Energy Drift: 100ステップ積分で < 5e-5
    - VRAM制約: Batch=2, Seq=2048で < 7.5GB
    - 再構成誤差: Symplectic Adjoint使用時 < 8e-6
    - フォールバック動作: 再構成誤差 > 1e-5の時、自動的にCheckpointingモードに切り替わる

Requirements:
    - Requirement 2.18: ComplexEmbedding → HamiltonianODE → ComplexLinear × N → Output
    - Requirement 2.19: Complex → Real変換、Real → Complex変換
    - Requirement 2.20: エネルギー保存とフォールバックの検証

Physical Intuition:
    Hamiltonian ODEは、エネルギー保存則に従う思考プロセスをシミュレートします。
    - 位置q: 現在の状態（意味表現）
    - 運動量p: 変化の勢い（文脈の流れ）
    - エネルギー保存: 論理的一貫性を保証

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
from .hamiltonian_ode import HamiltonianNeuralODE
from .stage1_model import Phase3Stage1Config


class Phase3Stage2Config(Phase3Stage1Config):
    """
    Phase 3 Stage 2モデルの設定クラス
    
    Stage 1の設定を継承し、Hamiltonian ODE関連のパラメータを追加します。
    
    Args:
        vocab_size (int): 語彙サイズ
        d_model (int): モデル次元（デフォルト: 512）
        n_layers (int): レイヤー数（デフォルト: 6）
        n_seq (int): 最大シーケンス長（デフォルト: 2048）
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
        zeta_scale (float): Zeta初期化のスケール（デフォルト: 1.0）
        ode_dt (float): ODE時間刻み（デフォルト: 0.1）
        ode_steps (int): ODE積分ステップ数（デフォルト: 10）
        potential_type (str): ポテンシャルネットワークの種類（デフォルト: 'mlp'）
        recon_threshold (float): 再構成誤差の閾値（デフォルト: 1e-5）
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_seq: int = 2048,
        use_complex32: bool = True,
        dropout: float = 0.1,
        zeta_scale: float = 1.0,
        ode_dt: float = 0.1,
        ode_steps: int = 10,
        potential_type: str = 'mlp',
        recon_threshold: float = 1e-5
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_seq=n_seq,
            use_complex32=use_complex32,
            dropout=dropout,
            zeta_scale=zeta_scale
        )
        
        # Hamiltonian ODE関連のパラメータ
        self.ode_dt = ode_dt
        self.ode_steps = ode_steps
        self.potential_type = potential_type
        self.recon_threshold = recon_threshold


class Phase3Stage2Block(nn.Module):
    """
    Phase 3 Stage 2の単一ブロック
    
    構造:
        x → [ComplexLayerNorm] → [Complex→Real] → [HamiltonianODE] → [Real→Complex] → [Residual] → x
    
    Physical Interpretation:
        - ComplexLayerNorm: 複素平面上で正規化
        - Complex→Real: 複素数を実数ベクトルに変換（ODEは実数で処理）
        - HamiltonianODE: エネルギー保存則に従う思考プロセス
        - Real→Complex: 実数ベクトルを複素数に変換
        - Residual: 情報の流れを保証
    
    Args:
        d_model (int): モデル次元
        ode_dt (float): ODE時間刻み（デフォルト: 0.1）
        ode_steps (int): ODE積分ステップ数（デフォルト: 10）
        potential_type (str): ポテンシャルネットワークの種類（デフォルト: 'mlp'）
        recon_threshold (float): 再構成誤差の閾値（デフォルト: 1e-5）
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
    
    Requirements: 2.18, 2.19
    """
    
    def __init__(
        self,
        d_model: int,
        ode_dt: float = 0.1,
        ode_steps: int = 10,
        potential_type: str = 'mlp',
        recon_threshold: float = 1e-5,
        use_complex32: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.ode_dt = ode_dt
        self.ode_steps = ode_steps
        self.use_complex32 = use_complex32
        
        # 1. ComplexLayerNorm（前正規化）
        self.norm1 = ComplexLayerNorm(d_model)
        
        # 2. Hamiltonian Neural ODE
        # Requirement 2.18: HamiltonianODEを各ブロックで実行
        self.hamiltonian_ode = HamiltonianNeuralODE(
            d_model=d_model,
            potential_type=potential_type,
            dt=ode_dt,
            recon_threshold=recon_threshold
        )
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 4. ComplexLayerNorm（後正規化）
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
        
        Requirements: 2.18, 2.19
        """
        # 残差接続のための入力を保存
        residual = z
        
        # 1. 前正規化
        z = self.norm1(z)
        
        # 2. Complex → Real変換（Requirement 2.19）
        # ODEは実数で処理するため、ComplexTensorを実部のみ使用
        # （虚部は別途処理するか、または実部のみで思考プロセスを実行）
        if isinstance(z, ComplexTensor):
            # ComplexTensor: (B, N, D) → Real: (B, N, D)
            # 実部のみを使用（簡略化）
            x_real = z.real
            original_dtype = torch.float16
        else:
            # complex64: (B, N, D) → Real: (B, N, D)
            x_real = z.real
            original_dtype = torch.float32
        
        # 3. Hamiltonian ODE（思考プロセス）
        # Requirement 2.18: 各ブロックでODEを実行
        # 初期状態: [q₀, p₀] = [x_real, 0]
        # 位置q: 現在の状態
        # 運動量p: ゼロで初期化（静止状態から開始）
        B, N, D = x_real.shape
        
        # 位置qと運動量pを準備
        # ODEはfloat32で処理するため、float32に変換
        q = x_real.float()  # (B, N, D) float32
        p = torch.zeros_like(q)  # (B, N, D) float32
        
        # Phase space: [q, p]
        phase_space = torch.cat([q, p], dim=-1)  # (B, N, 2D) float32
        
        # ODE積分
        t_span = (0, self.ode_dt * self.ode_steps)
        phase_space_final = self.hamiltonian_ode(phase_space, t_span=t_span)
        
        # 最終状態から位置qのみを取り出す
        # phase_space_final: (B, N, 2D) → q_final: (B, N, D)
        q_final = phase_space_final[..., :D]
        
        # 元のdtypeに戻す
        if original_dtype == torch.float16:
            q_final = q_final.half()
        
        # 4. Real → Complex変換（Requirement 2.19）
        # Real: (B, N, D) → ComplexTensor: (B, N, D)
        # 実部を更新、虚部は元のまま保持
        if isinstance(z, ComplexTensor):
            real_part = q_final
            imag_part = z.imag  # 虚部は元のまま
        else:
            real_part = q_final
            imag_part = z.imag  # 虚部は元のまま
        
        if isinstance(z, ComplexTensor):
            z_out = ComplexTensor(real_part, imag_part)
        else:
            z_out = torch.complex(real_part, imag_part)
        
        # 5. Dropout
        if isinstance(z_out, ComplexTensor):
            z_out = ComplexTensor(
                self.dropout(z_out.real),
                self.dropout(z_out.imag)
            )
        else:
            z_out = torch.complex(
                self.dropout(z_out.real),
                self.dropout(z_out.imag)
            )
        
        # 6. 残差接続
        z_out = z_out + residual
        
        # 7. 後正規化
        z_out = self.norm2(z_out)
        
        # 診断情報の収集
        if return_diagnostics:
            diagnostics = self._collect_diagnostics(z_out, phase_space_final)
            return z_out, diagnostics
        
        return z_out
    
    def _collect_diagnostics(
        self,
        z: Union[ComplexTensor, torch.Tensor],
        phase_space: torch.Tensor
    ) -> Dict:
        """
        診断情報の収集
        
        Args:
            z: 複素テンソル
            phase_space: Phase space状態
        
        Returns:
            dict: 診断情報
        """
        with torch.no_grad():
            # 複素数の統計
            if isinstance(z, ComplexTensor):
                magnitude = z.abs()
                phase = z.angle()
            else:
                magnitude = torch.abs(z)
                phase = torch.angle(z)
            
            # エネルギーの計算
            D = phase_space.shape[-1] // 2
            q = phase_space[..., :D]
            p = phase_space[..., D:]
            
            # 運動エネルギー: T = ½|p|²
            kinetic_energy = 0.5 * (p ** 2).sum(dim=-1).mean()
            
            # ポテンシャルエネルギー: V(q)（簡略化: ½|q|²）
            potential_energy = 0.5 * (q ** 2).sum(dim=-1).mean()
            
            # 全エネルギー: H = T + V
            total_energy = kinetic_energy + potential_energy
            
            return {
                'magnitude_mean': magnitude.mean().item(),
                'magnitude_std': magnitude.std().item(),
                'phase_mean': phase.mean().item(),
                'phase_std': phase.std().item(),
                'kinetic_energy': kinetic_energy.item(),
                'potential_energy': potential_energy.item(),
                'total_energy': total_energy.item(),
                'ode_mode': self.hamiltonian_ode.mode,
            }


class Phase3Stage2Model(nn.Module):
    """
    Phase 3 Stage 2 統合モデル
    
    このモデルは、Stage 1モデルにHamiltonian ODEを追加します。
    
    Architecture:
        Input: Token IDs (B, N)
        ↓
        [ComplexEmbedding] → z (B, N, D) [complex]
        ↓
        [Phase3Stage2Block] × L layers
        ↓
        [Output Head] → logits (B, N, vocab_size)
    
    Args:
        config (Phase3Stage2Config, optional): 設定オブジェクト
        vocab_size (int, optional): 語彙サイズ
        d_model (int, optional): モデル次元
        n_layers (int): レイヤー数（デフォルト: 6）
        max_seq_len (int): 最大シーケンス長（デフォルト: 2048）
        use_complex32 (bool): complex32を使用するか（デフォルト: True）
        dropout (float): ドロップアウト率（デフォルト: 0.1）
        zeta_scale (float): Zeta初期化のスケール（デフォルト: 1.0）
        ode_dt (float): ODE時間刻み（デフォルト: 0.1）
        ode_steps (int): ODE積分ステップ数（デフォルト: 10）
        potential_type (str): ポテンシャルネットワークの種類（デフォルト: 'mlp'）
        recon_threshold (float): 再構成誤差の閾値（デフォルト: 1e-5）
    
    Requirements: 2.18, 2.19, 2.20
    """
    
    def __init__(
        self,
        config: Optional[Phase3Stage2Config] = None,
        vocab_size: Optional[int] = None,
        d_model: Optional[int] = None,
        n_layers: int = 6,
        max_seq_len: int = 2048,
        use_complex32: bool = True,
        dropout: float = 0.1,
        zeta_scale: float = 1.0,
        ode_dt: float = 0.1,
        ode_steps: int = 10,
        potential_type: str = 'mlp',
        recon_threshold: float = 1e-5
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
            ode_dt = config.ode_dt
            ode_steps = config.ode_steps
            potential_type = config.potential_type
            recon_threshold = config.recon_threshold
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
        # 2. Phase3Stage2Block × L layers
        # ========================================
        
        self.blocks = nn.ModuleList([
            Phase3Stage2Block(
                d_model=self.d_model,
                ode_dt=ode_dt,
                ode_steps=ode_steps,
                potential_type=potential_type,
                recon_threshold=recon_threshold,
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
        dtype = torch.float16 if self.use_complex32 else torch.float32
        self.output_proj = nn.Linear(self.d_model, self.vocab_size)
        self.output_proj.weight.data = self.output_proj.weight.data.to(dtype)
        if self.output_proj.bias is not None:
            self.output_proj.bias.data = self.output_proj.bias.data.to(dtype)
        
        # 統計情報（デバッグ用）
        self.register_buffer('_forward_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_nan_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('_fallback_count', torch.tensor(0, dtype=torch.long))
    
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
        
        Requirements: 2.18, 2.19
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
        # 2. Phase3Stage2Block × L layers
        # ========================================
        
        for i, block in enumerate(self.blocks):
            if return_diagnostics:
                z, block_diag = block(z, return_diagnostics=True)
                diagnostics[f'layer_{i}'] = block_diag
                
                # フォールバック回数の記録
                if block_diag.get('ode_mode') != 'symplectic_adjoint':
                    self._fallback_count += 1
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
        
        # 実部のみを使用
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
                - 'fallback_count': フォールバック回数
        """
        return {
            'forward_count': self._forward_count.item(),
            'nan_count': self._nan_count.item(),
            'nan_rate': self._nan_count.item() / max(self._forward_count.item(), 1),
            'fallback_count': self._fallback_count.item(),
        }
    
    def reset_statistics(self):
        """統計情報のリセット"""
        self._forward_count.zero_()
        self._nan_count.zero_()
        self._fallback_count.zero_()
    
    def reset_ode_to_symplectic(self):
        """
        全てのODEをSymplectic Adjointモードにリセット
        
        Usage:
            エポック開始時に呼び出すことで、再度Symplectic Adjointを試行する。
        """
        for block in self.blocks:
            block.hamiltonian_ode.reset_to_symplectic()


# ========================================
# ユーティリティ関数
# ========================================

def create_phase3_stage2_model(
    vocab_size: int,
    d_model: int = 512,
    n_layers: int = 6,
    max_seq_len: int = 2048,
    use_complex32: bool = True,
    dropout: float = 0.1,
    zeta_scale: float = 1.0,
    ode_dt: float = 0.1,
    ode_steps: int = 10,
    potential_type: str = 'mlp',
    recon_threshold: float = 1e-5
) -> Phase3Stage2Model:
    """
    Phase 3 Stage 2モデルのファクトリー関数
    
    Args:
        vocab_size: 語彙サイズ
        d_model: モデル次元（デフォルト: 512）
        n_layers: レイヤー数（デフォルト: 6）
        max_seq_len: 最大シーケンス長（デフォルト: 2048）
        use_complex32: complex32を使用するか（デフォルト: True）
        dropout: ドロップアウト率（デフォルト: 0.1）
        zeta_scale: Zeta初期化のスケール（デフォルト: 1.0）
        ode_dt: ODE時間刻み（デフォルト: 0.1）
        ode_steps: ODE積分ステップ数（デフォルト: 10）
        potential_type: ポテンシャルネットワークの種類（デフォルト: 'mlp'）
        recon_threshold: 再構成誤差の閾値（デフォルト: 1e-5）
    
    Returns:
        Phase3Stage2Model: 初期化されたモデル
    """
    return Phase3Stage2Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        use_complex32=use_complex32,
        dropout=dropout,
        zeta_scale=zeta_scale,
        ode_dt=ode_dt,
        ode_steps=ode_steps,
        potential_type=potential_type,
        recon_threshold=recon_threshold
    )


def convert_stage1_to_stage2(
    stage1_model,
    ode_dt: float = 0.1,
    ode_steps: int = 10,
    potential_type: str = 'mlp',
    recon_threshold: float = 1e-5
) -> Phase3Stage2Model:
    """
    Stage 1モデルをStage 2モデルに変換
    
    この関数は、Stage 1モデルの重みをStage 2モデルに転送します。
    
    Conversion Strategy:
        1. Embedding層: Stage 1の重みをそのまま使用
        2. 中間層: Stage 1のComplexLinear層は使用せず、新しいODEブロックを初期化
        3. Output層: Stage 1の重みをそのまま使用
    
    Args:
        stage1_model: Stage 1モデル
        ode_dt: ODE時間刻み
        ode_steps: ODE積分ステップ数
        potential_type: ポテンシャルネットワークの種類
        recon_threshold: 再構成誤差の閾値
    
    Returns:
        Phase3Stage2Model: 変換されたStage 2モデル
    """
    # Stage 1モデルの設定を取得
    vocab_size = stage1_model.vocab_size
    d_model = stage1_model.d_model
    n_layers = stage1_model.n_layers
    max_seq_len = stage1_model.max_seq_len
    use_complex32 = stage1_model.use_complex32
    
    # Stage 2モデルを作成
    stage2_model = Phase3Stage2Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        use_complex32=use_complex32,
        ode_dt=ode_dt,
        ode_steps=ode_steps,
        potential_type=potential_type,
        recon_threshold=recon_threshold
    )
    
    # ========================================
    # 1. Embedding層の転送
    # ========================================
    
    with torch.no_grad():
        stage2_model.embedding.load_state_dict(stage1_model.embedding.state_dict())
    
    # ========================================
    # 2. Output層の転送
    # ========================================
    
    with torch.no_grad():
        stage2_model.output_proj.weight.copy_(stage1_model.output_proj.weight)
        if stage1_model.output_proj.bias is not None:
            stage2_model.output_proj.bias.copy_(stage1_model.output_proj.bias)
    
    print(f"✓ Stage 1モデルをStage 2モデルに変換しました")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - d_model: {d_model}")
    print(f"  - n_layers: {n_layers}")
    print(f"  - max_seq_len: {max_seq_len}")
    print(f"  - ode_dt: {ode_dt}")
    print(f"  - ode_steps: {ode_steps}")
    print(f"  - potential_type: {potential_type}")
    
    return stage2_model
