"""
Koopman Operator for Phase 3: Physics Transcendence

このモジュールは、非線形力学系を線形化するKoopman作用素を実装します。

物理的直観:
    Koopman作用素は、非線形な状態遷移 x_{t+1} = f(x_t) を、
    高次元の「観測可能空間」g = Ψ(x) において線形化します:
    g_{t+1} = K · g_t
    
    これにより、多段階予測が行列累乗 K^n で高速に計算できます。

主要コンポーネント:
    1. Observable Encoder Ψ: 物理状態 x → Koopman空間 g への射影
    2. Koopman Operator K: 線形発展作用素
    3. Observable Decoder Ψ⁻¹: Koopman空間 g → 物理状態 x への逆射影
    4. Residual Correction R: 線形近似の限界を補う非線形項

数学的定式化:
    x_{t+1} = Ψ⁻¹(K · Ψ(x_t)) + R(Ψ(x_t))
    
    線形性誤差: ||Ψ(f(x)) - K·Ψ(x)||
    目標: < 5e-4

References:
    - Koopman, B. O. (1931). "Hamiltonian Systems and Transformation in Hilbert Space"
    - Williams, M. O., et al. (2015). "A Data-Driven Approximation of the Koopman Operator"
    - Lusch, B., et al. (2018). "Deep learning for universal linear embeddings of nonlinear dynamics"

Author: Project MUSE Team
Date: 2025-11-21
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanOperator(nn.Module):
    """
    Koopman Operator for Global Linearization
    
    非線形力学系を線形化し、多段階予測を高速化します。
    
    Architecture:
        x → Ψ(x) → K·g → Ψ⁻¹(g') + R(g) → x'
        
    Args:
        d_model: 入力次元（物理状態の次元）
        d_koopman: Koopman空間の次元（通常は d_model の 2-4倍）
        use_residual: 残差補正を使用するか（デフォルト: True）
        
    Attributes:
        psi: Observable Encoder（MLP）
        K: Koopman線形作用素（単位行列で初期化）
        psi_inv: Observable Decoder（MLP）
        residual_net: 残差補正ネットワーク（オプション）
        
    Example:
        >>> koopman = KoopmanOperator(d_model=512, d_koopman=1024)
        >>> x = torch.randn(4, 100, 512)
        >>> x_pred, g, g_next = koopman(x)
        >>> print(x_pred.shape)  # (4, 100, 512)
        
        # 多段階予測（高速）
        >>> predictions = koopman.multi_step_prediction(x, steps=10)
        >>> print(predictions.shape)  # (4, 10, 100, 512)
    """
    
    def __init__(
        self,
        d_model: int,
        d_koopman: int,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.d_koopman = d_koopman
        self.use_residual = use_residual
        
        # Observable Encoder Ψ: x → g
        # 物理的意味: 非線形な物理状態を、線形発展する「観測可能量」に変換
        self.psi = nn.Sequential(
            nn.Linear(d_model, d_koopman * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_koopman * 2, d_koopman)
        )
        
        # Koopman Operator K: g_t → g_{t+1}
        # 物理的意味: 観測可能量の線形発展を記述する作用素
        # 初期化: 単位行列 + 小さなノイズ（恒等写像から学習開始）
        self.K = nn.Linear(d_koopman, d_koopman, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.K.weight)
            self.K.weight.data += torch.randn_like(self.K.weight.data) * 0.01
        
        # Observable Decoder Ψ⁻¹: g → x
        # 物理的意味: 観測可能量から物理状態を復元
        self.psi_inv = nn.Sequential(
            nn.Linear(d_koopman, d_koopman * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_koopman * 2, d_model)
        )
        
        # Residual Correction R(g)
        # 物理的意味: 有限次元Koopman空間では完全な線形化は不可能
        #            残差項で非線形性を補正する
        if use_residual:
            self.residual_net = nn.Sequential(
                nn.Linear(d_koopman, d_koopman),
                nn.GELU(),
                nn.Linear(d_koopman, d_model)
            )
        else:
            self.residual_net = None
    
    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: 1ステップの予測
        
        Args:
            x: 入力状態 (B, N, D)
            
        Returns:
            x_pred: 予測状態 (B, N, D)
            g: Koopman空間の現在状態 (B, N, K)
            g_next: Koopman空間の次状態 (B, N, K)
            
        物理的プロセス:
            1. Lift: x → g (非線形射影)
            2. Evolve: g → K·g (線形発展)
            3. Decode: K·g → x' (逆射影)
            4. Correct: x' + R(g) (残差補正)
        """
        # 1. Lift to Koopman space
        # 物理状態を観測可能量に変換
        g = self.psi(x)  # (B, N, K)
        
        # 2. Linear evolution
        # Koopman空間での線形発展（これが高速化の鍵）
        g_next = self.K(g)  # (B, N, K)
        
        # 3. Decode to physical space
        # 観測可能量を物理状態に戻す
        x_pred = self.psi_inv(g_next)  # (B, N, D)
        
        # 4. Residual correction
        # 線形近似の誤差を非線形項で補正
        if self.use_residual and self.residual_net is not None:
            residual = self.residual_net(g)  # (B, N, D)
            x_pred = x_pred + residual
        
        return x_pred, g, g_next
    
    def multi_step_prediction(
        self,
        x: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        多段階予測（高速推論）
        
        物理的直観:
            通常の自己回帰: x → f(x) → f(f(x)) → ... (Nステップ)
            Koopman予測: x → Ψ(x) → K^n·Ψ(x) → Ψ⁻¹(K^n·Ψ(x)) (1ステップ相当)
            
            線形発展 K^n は行列累乗で高速に計算できるため、
            多段階予測が劇的に高速化されます。
        
        Args:
            x: 初期状態 (B, N, D)
            steps: 予測ステップ数
            
        Returns:
            predictions: 各ステップの予測 (B, steps, N, D)
            
        計算量:
            通常: O(steps × D²)
            Koopman: O(K² + steps × K × D) ≈ O(steps × K × D)
            高速化率: 約 D/K 倍（通常 K < D なので高速）
        """
        # Lift to Koopman space
        g = self.psi(x)  # (B, N, K)
        
        predictions = []
        
        # Koopman行列の累乗を事前計算（オプション）
        # 注: 大規模な場合はメモリを消費するため、逐次計算も可能
        K_weight = self.K.weight  # (K, K)
        
        for step in range(steps):
            # Linear evolution in Koopman space
            g = F.linear(g, K_weight)  # (B, N, K)
            
            # Decode to physical space
            x_pred = self.psi_inv(g)  # (B, N, D)
            
            # Residual correction
            if self.use_residual and self.residual_net is not None:
                x_pred = x_pred + self.residual_net(g)
            
            predictions.append(x_pred)
        
        return torch.stack(predictions, dim=1)  # (B, steps, N, D)
    
    def compute_eigenspectrum(self) -> Dict[str, torch.Tensor]:
        """
        Koopman作用素の固有値スペクトルを計算
        
        物理的意味:
            固有値は「概念の安定性」を表します:
            - |λ| < 1: 減衰モード（忘却される概念）
            - |λ| = 1: 保存モード（記憶される概念）
            - |λ| > 1: 発散モード（不安定、要注意）
            
            位相 arg(λ) は「概念の周期性」を表します。
        
        Returns:
            dict:
                'eigenvalues': 固有値（複素数） (K,)
                'magnitude': 固有値の大きさ (K,)
                'phase': 固有値の位相 (K,)
                'stable_ratio': 安定な固有値の割合（|λ| ≤ 1）
                
        可視化例:
            >>> spectrum = koopman.compute_eigenspectrum()
            >>> plt.scatter(spectrum['eigenvalues'].real, spectrum['eigenvalues'].imag)
            >>> plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False))
            >>> plt.title('Koopman Eigenspectrum')
        """
        K_matrix = self.K.weight.detach().cpu()
        eigenvalues = torch.linalg.eigvals(K_matrix)
        
        magnitude = torch.abs(eigenvalues)
        phase = torch.angle(eigenvalues)
        
        # 安定性解析
        stable_ratio = (magnitude <= 1.0).float().mean()
        
        return {
            'eigenvalues': eigenvalues,
            'magnitude': magnitude,
            'phase': phase,
            'stable_ratio': stable_ratio.item(),
            'max_magnitude': magnitude.max().item(),
            'mean_magnitude': magnitude.mean().item()
        }
    
    def get_linearity_error(
        self,
        x_t: torch.Tensor,
        x_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        線形性誤差の計算
        
        定義:
            error = ||Ψ(x_{t+1}) - K·Ψ(x_t)|| / ||Ψ(x_{t+1})||
            
        物理的意味:
            Koopman空間での線形発展が、実際の非線形ダイナミクスを
            どれだけ正確に近似できているかを測定します。
            
        目標: < 5e-4
        
        Args:
            x_t: 現在の状態 (B, N, D)
            x_t1: 次の状態（真値） (B, N, D)
            
        Returns:
            error: 相対線形性誤差 (スカラー)
        """
        with torch.no_grad():
            # Lift both states
            g_t = self.psi(x_t)
            g_t1_true = self.psi(x_t1)
            
            # Linear prediction
            g_t1_pred = self.K(g_t)
            
            # Relative error
            numerator = (g_t1_pred - g_t1_true).norm()
            denominator = g_t1_true.norm()
            
            error = numerator / (denominator + 1e-8)
        
        return error


def koopman_linearity_loss(
    koopman: KoopmanOperator,
    x_t: torch.Tensor,
    x_t1: torch.Tensor
) -> torch.Tensor:
    """
    線形性を強制する補助損失
    
    定義:
        L = ||Ψ(x_{t+1}) - K·Ψ(x_t)||²
        
    物理的意味:
        Koopman空間での線形発展が、実際の状態遷移と一致するように
        Observable EncoderとKoopman Operatorを学習します。
        
    使用方法:
        >>> loss = main_loss + 0.1 * koopman_linearity_loss(model.koopman, x_t, x_t1)
        
    Args:
        koopman: KoopmanOperatorモジュール
        x_t: 現在の状態 (B, N, D)
        x_t1: 次の状態（真値） (B, N, D)
        
    Returns:
        loss: 線形性損失（スカラー）
    """
    # Lift both states
    g_t = koopman.psi(x_t)
    g_t1_true = koopman.psi(x_t1)
    
    # Linear prediction
    g_t1_pred = koopman.K(g_t)
    
    # MSE loss
    loss = F.mse_loss(g_t1_pred, g_t1_true)
    
    return loss


class KoopmanTrainingScheduler:
    """
    Koopman Operatorの段階的学習スケジューラ
    
    戦略:
        Phase 1 (Epoch 0-N): Residual Rをfreeze、Kのみ学習
            → 線形性を獲得する期間
            → Residualが全てを説明してしまうのを防ぐ
            
        Phase 2 (Epoch N-M): Residual Rをunfreeze、両方学習
            → 線形近似の限界を残差で補正
            
        Phase 3 (Epoch M+): 正則化項を追加
            → モード崩壊を防止
            → 固有値が単位円に近いことを奨励
    
    Args:
        koopman_module: KoopmanOperatorモジュール
        freeze_residual_epochs: Residualをfreezeするエポック数（デフォルト: 10）
        
    Example:
        >>> scheduler = KoopmanTrainingScheduler(model.koopman, freeze_residual_epochs=10)
        >>> for epoch in range(total_epochs):
        ...     scheduler.step_epoch(epoch)
        ...     # Training loop
        ...     reg_loss = scheduler.get_regularization_loss()
        ...     total_loss = main_loss + reg_loss
    """
    
    def __init__(
        self,
        koopman_module: KoopmanOperator,
        freeze_residual_epochs: int = 10
    ):
        self.koopman = koopman_module
        self.freeze_residual_epochs = freeze_residual_epochs
        self.current_epoch = 0
    
    def step_epoch(self, epoch: int):
        """
        エポック開始時に呼び出す
        
        Args:
            epoch: 現在のエポック番号
        """
        self.current_epoch = epoch
        
        if epoch < self.freeze_residual_epochs:
            # Phase 1: Residualをfreeze
            if self.koopman.residual_net is not None:
                for param in self.koopman.residual_net.parameters():
                    param.requires_grad = False
                print(f"Epoch {epoch}: Koopman Phase 1 - Training K only (R frozen)")
        else:
            # Phase 2: Residualをunfreeze
            if self.koopman.residual_net is not None:
                for param in self.koopman.residual_net.parameters():
                    param.requires_grad = True
                print(f"Epoch {epoch}: Koopman Phase 2 - Training K and R")
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        正則化損失（Phase 3で使用）
        
        目的:
            1. Residual Rが全てを説明してしまうのを防ぐ
            2. Koopman行列の固有値が単位円に近いことを奨励
            
        Returns:
            reg_loss: 正則化損失（スカラー）
        """
        if self.current_epoch < self.freeze_residual_epochs:
            return torch.tensor(0.0, device=self.koopman.K.weight.device)
        
        device = self.koopman.K.weight.device
        
        # 1. Residualの大きさにペナルティ
        residual_norm = torch.tensor(0.0, device=device)
        if self.koopman.residual_net is not None:
            for param in self.koopman.residual_net.parameters():
                residual_norm = residual_norm + param.norm()
        
        # 2. Koopman行列の固有値が単位円に近いことを奨励
        K_matrix = self.koopman.K.weight
        eigenvalues = torch.linalg.eigvals(K_matrix)
        eigenvalue_penalty = ((torch.abs(eigenvalues) - 1.0) ** 2).mean()
        
        # 総正則化損失
        reg_loss = 0.01 * residual_norm + 0.1 * eigenvalue_penalty
        
        return reg_loss


class KoopmanLinearityMonitor:
    """
    Koopman線形性の監視クラス
    
    目的:
        学習中にKoopman空間での線形性を監視し、
        線形性誤差が閾値を超えた場合に警告を発します。
        
    Args:
        threshold: 線形性誤差の閾値（デフォルト: 5e-4）
        
    Example:
        >>> monitor = KoopmanLinearityMonitor(threshold=5e-4)
        >>> for batch in dataloader:
        ...     x_t, x_t1 = batch
        ...     is_linear = monitor.check(koopman, x_t, x_t1)
        ...     if not is_linear:
        ...         print("Warning: Linearity violated!")
    """
    
    def __init__(self, threshold: float = 5e-4):
        self.threshold = threshold
        self.error_history = []
    
    def check(
        self,
        koopman: KoopmanOperator,
        x_t: torch.Tensor,
        x_t1: torch.Tensor
    ) -> bool:
        """
        線形性誤差をチェック
        
        Args:
            koopman: KoopmanOperatorモジュール
            x_t: 現在の状態 (B, N, D)
            x_t1: 次の状態（真値） (B, N, D)
            
        Returns:
            is_linear: 線形性が保たれているか（True/False）
        """
        error = koopman.get_linearity_error(x_t, x_t1)
        self.error_history.append(error.item())
        
        if error > self.threshold:
            warnings.warn(
                f"Koopman linearity error: {error:.2e} > threshold {self.threshold:.2e}"
            )
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """
        線形性誤差の統計情報を取得
        
        Returns:
            dict:
                'mean_error': 平均誤差
                'max_error': 最大誤差
                'min_error': 最小誤差
                'current_error': 最新の誤差
        """
        if not self.error_history:
            return {
                'mean_error': 0.0,
                'max_error': 0.0,
                'min_error': 0.0,
                'current_error': 0.0
            }
        
        return {
            'mean_error': sum(self.error_history) / len(self.error_history),
            'max_error': max(self.error_history),
            'min_error': min(self.error_history),
            'current_error': self.error_history[-1]
        }
