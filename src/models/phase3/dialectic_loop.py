"""
Dialectic Loop for Phase 3: Physics Transcendence

このモジュールは、生成AIと批判AIの弁証法的自己進化ループを実装します。
HamiltonianNeuralODEを用いて論理的矛盾（エネルギー分散）を計算し、
Gumbel-Softmaxにより微分可能な形で仮説を修正します。

Requirements:
    - Requirement 6.1: Hypothesis Generation (Gumbel-Softmax)
    - Requirement 6.2: Hypothesis Critique (Hamiltonian Energy Variance)
    - Requirement 6.3: Synthesis (Contradiction Minimization)
    - Requirement 6.4: Temperature Annealing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from src.models.phase3.hamiltonian_ode import HamiltonianNeuralODE


class DialecticLoop(nn.Module):
    """
    Dialectic Resonance Loop (弁証法的共鳴ループ)

    Architecture:
        1. Hypothesis Generation (Thesis):
           - 高温度で多様な仮説を生成
           - Gumbel-Softmaxで離散トークンを微分可能に近似

        2. Critique (Antithesis):
           - Hamiltonian Neural ODEで論理的整合性を検証
           - エネルギー分散 Var(E(t)) を矛盾スコアとして計算

        3. Synthesis (Aufheben):
           - 矛盾スコアを最小化するように勾配を逆伝播
           - 自己進化による論理的超越

    Args:
        d_model (int): モデル次元
        vocab_size (int): 語彙サイズ
        hamiltonian_ode (HamiltonianNeuralODE): 批判用Hamiltonian ODE
        initial_temperature (float): 初期温度（デフォルト: 2.0）
        min_temperature (float): 最小温度（デフォルト: 0.1）
        annealing_rate (float): アニーリング率（デフォルト: 0.99）

    Reference:
        Hegel's Dialectics: Thesis -> Antithesis -> Synthesis
        Physics: Energy Conservation as Logical Consistency
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        hamiltonian_ode: HamiltonianNeuralODE,
        initial_temperature: float = 2.0,
        min_temperature: float = 0.1,
        annealing_rate: float = 0.99
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.hamiltonian_ode = hamiltonian_ode

        # Temperature Annealing parameters
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.annealing_rate = annealing_rate

        # Generator components (Thesis)
        # 入力状態からロジットを生成する層
        self.generator_head = nn.Linear(d_model, vocab_size)

        # Embedding layer (for converting sampled tokens back to vectors for critique)
        # NOTE: 実際にはPhase 3のComplexEmbeddingを使用するが、
        # ここでは微分可能なパスを作るために重みを共有または射影を使用する
        self.embedding_projection = nn.Linear(vocab_size, d_model, bias=False)

        # 統計情報
        self.register_buffer('contradiction_history', torch.tensor([], dtype=torch.float32))

    def generate_hypothesis(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        18.1 仮説生成 (Thesis)

        Gumbel-Softmaxを使用して、微分可能な形で仮説（トークン分布）をサンプリングします。

        Args:
            x (torch.Tensor): 入力状態 (B, N, D)
            temperature (float, optional): サンプリング温度

        Returns:
            soft_tokens (torch.Tensor): One-hot近似ベクトル (B, N, V)
            logits (torch.Tensor): ロジット (B, N, V)
        """
        temp = temperature if temperature is not None else self.temperature

        # Logits計算
        logits = self.generator_head(x)  # (B, N, V)

        # Gumbel-Softmax Sampling
        # hard=False: 微分可能なSoftmax分布を返す
        # hard=True: One-hotを返すが、勾配はStraight-Through EstimatorでSoftmaxを通る
        if self.training:
            soft_tokens = F.gumbel_softmax(logits, tau=temp, hard=False, dim=-1)
        else:
            # 推論時はargmax相当（低温極限）
            soft_tokens = F.softmax(logits / temp, dim=-1)

        return soft_tokens, logits

    def critique_hypothesis(
        self,
        soft_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        18.2 仮説批判 (Antithesis)

        生成された仮説をベクトル空間に射影し、Hamiltonian ODEで時間発展させ、
        エネルギー保存則の破れ（分散）を矛盾スコアとして計算します。

        Args:
            soft_tokens (torch.Tensor): 生成された仮説ベクトル (B, N, V)

        Returns:
            contradiction_score (torch.Tensor): 矛盾スコア (B,)

        Physics Intuition:
            論理的に整合した思考は、エネルギー保存則（一貫性）を満たす。
            エネルギーが変動する場合、それは論理的矛盾または幻覚を示唆する。
        """
        # 1. Vector Projection: (B, N, V) @ (V, D) -> (B, N, D)
        # nn.Linear(V, D).weight has shape (D, V)
        # We need (B, N, V) @ (V, D). The weight should be transposed if we use it directly as matrix.
        # nn.Linear(x) does x @ weight.T.
        # embedding_projection is Linear(vocab_size, d_model).
        # weight shape is (d_model, vocab_size).
        # soft_tokens is (B, N, vocab_size).
        # soft_tokens @ weight.T -> (B, N, d_model).

        # Use the linear layer directly
        hypothesis_vec = self.embedding_projection(soft_tokens)

        # 2. Phase Space Construction
        # Hamiltonian ODEは (q, p) を必要とするため、入力を分割または拡張
        # ここでは仮説ベクトルを位置qとし、運動量pはゼロまたは学習可能なパラメータで初期化
        # Phase 3設計に従い、入力次元の半分をq、半分をpとするか、embeddingを拡張する
        # ここでは単純化のため、hypothesis_vecを複製して (q, p) とする
        # (B, N, D) -> (B, N, 2D)
        state_q = hypothesis_vec
        state_p = torch.zeros_like(hypothesis_vec) # 初期運動量はゼロと仮定
        state_0 = torch.cat([state_q, state_p], dim=-1)

        # 3. Hamiltonian Time Evolution
        # 時間発展 (t=0 to t=1)
        # gradient checkpoiningなどのフォールバックはHamiltonianNeuralODEが処理
        state_t = self.hamiltonian_ode(state_0, t_span=(0, 1))

        # 4. Energy Calculation & Variance
        # 軌跡上のエネルギーを計算する代わりに、始点と終点のエネルギー差（Drift）を使用
        # より厳密には、trajectory全体の分散を計算すべきだが、計算コスト削減のため端点のみ使用

        # エネルギー計算 H(q, p)
        # HamiltonianFunction expects (t, x) where x is concatenated [q, p]
        E_start = self.hamiltonian_ode.h_func(0.0, state_0) # (B, N, 1)

        # state_t is already [q_end, p_end]
        E_end = self.hamiltonian_ode.h_func(1.0, state_t) # (B, N, 1)

        # 矛盾スコア = エネルギー変動の二乗平均 (MSE)
        # (B, N, 1) -> (B,)
        # 論理的矛盾 = エネルギー保存則の破れ
        energy_drift = (E_end - E_start) ** 2
        contradiction_score = energy_drift.mean(dim=1).squeeze(-1) # (B,)

        return contradiction_score

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        18.3 Synthesis (Aufheben)

        仮説生成と批判を統合し、矛盾最小化損失を計算します。

        Args:
            x (torch.Tensor): 入力コンテキスト (B, N, D)

        Returns:
            logits (torch.Tensor): 生成されたロジット (B, N, V)
            contradiction_loss (torch.Tensor): 矛盾損失 (scalar)
            diagnostics (dict): 診断情報
        """
        # 1. Generate Hypothesis
        soft_tokens, logits = self.generate_hypothesis(x)

        # 2. Critique Hypothesis
        contradiction_scores = self.critique_hypothesis(soft_tokens)

        # 3. Calculate Loss
        # 矛盾スコア自体を損失とする
        contradiction_loss = contradiction_scores.mean()

        # 診断情報
        diagnostics = {
            'contradiction_score': contradiction_loss.item(),
            'temperature': self.temperature,
            'energy_drift_max': contradiction_scores.max().item(),
            'entropy': -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1).mean().item()
        }

        return logits, contradiction_loss, diagnostics

    def anneal_temperature(self):
        """
        18.4 温度アニーリング

        学習が進むにつれて温度を下げ、探索から活用へ移行させる。
        High Temp (Thesis): 多様な仮説生成（カオス）
        Low Temp (Synthesis): 確信的な論理構築（秩序）
        """
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.annealing_rate
        )
