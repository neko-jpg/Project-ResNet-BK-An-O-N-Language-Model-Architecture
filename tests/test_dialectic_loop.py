"""
Unit Tests for Dialectic Loop

Requirements:
    - Requirement 6.1: Hypothesis Generation
    - Requirement 6.2: Hypothesis Critique
    - Requirement 6.3: Synthesis
    - Requirement 6.4: Temperature Annealing
    - Requirement 6.5: Gradient Propagation
"""

import torch
import torch.nn as nn
import pytest
from typing import Tuple

from src.models.phase3.dialectic_loop import DialecticLoop
from src.models.phase3.hamiltonian_ode import HamiltonianNeuralODE


class TestDialecticLoop:
    @pytest.fixture
    def dialectic_setup(self):
        d_model = 32
        vocab_size = 100
        dt = 0.1

        # Mock Hamiltonian ODE for testing
        hamiltonian_ode = HamiltonianNeuralODE(
            d_model=d_model,
            potential_type='mlp',
            potential_hidden_dim=64,
            dt=dt
        )

        dialectic = DialecticLoop(
            d_model=d_model,
            vocab_size=vocab_size,
            hamiltonian_ode=hamiltonian_ode,
            initial_temperature=1.0,
            annealing_rate=0.9
        )

        return dialectic, d_model, vocab_size

    def test_initialization(self, dialectic_setup):
        """初期化のテスト"""
        dialectic, d_model, vocab_size = dialectic_setup

        assert dialectic.d_model == d_model
        assert dialectic.vocab_size == vocab_size
        assert dialectic.temperature == 1.0
        assert isinstance(dialectic.hamiltonian_ode, HamiltonianNeuralODE)

    def test_hypothesis_generation(self, dialectic_setup):
        """18.1 仮説生成のテスト (Req 6.1)"""
        dialectic, d_model, vocab_size = dialectic_setup
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, d_model)

        soft_tokens, logits = dialectic.generate_hypothesis(x)

        assert soft_tokens.shape == (batch_size, seq_len, vocab_size)
        assert logits.shape == (batch_size, seq_len, vocab_size)

        # Gumbel-Softmaxの出力は確率分布（和が1）
        assert torch.allclose(soft_tokens.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-5)

    def test_hypothesis_critique(self, dialectic_setup):
        """18.2 仮説批判のテスト (Req 6.2)"""
        dialectic, d_model, vocab_size = dialectic_setup
        batch_size, seq_len = 4, 10

        # ダミーのsoft_tokens
        soft_tokens = torch.randn(batch_size, seq_len, vocab_size).softmax(dim=-1)

        contradiction_score = dialectic.critique_hypothesis(soft_tokens)

        assert contradiction_score.shape == (batch_size,)
        assert (contradiction_score >= 0).all(), "Contradiction score should be non-negative (MSE)"

    def test_forward_synthesis(self, dialectic_setup):
        """18.3 Synthesisのテスト (Req 6.3)"""
        dialectic, d_model, vocab_size = dialectic_setup
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, d_model)

        logits, loss, diagnostics = dialectic(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0 # Scalar

        assert 'contradiction_score' in diagnostics
        assert 'temperature' in diagnostics
        assert 'entropy' in diagnostics

    def test_temperature_annealing(self, dialectic_setup):
        """18.4 温度アニーリングのテスト (Req 6.4)"""
        dialectic, _, _ = dialectic_setup
        initial_temp = dialectic.temperature

        dialectic.anneal_temperature()

        assert dialectic.temperature < initial_temp
        assert dialectic.temperature == initial_temp * dialectic.annealing_rate

    def test_gradient_propagation(self, dialectic_setup):
        """18.5 勾配伝播のテスト (Req 6.5)"""
        dialectic, d_model, vocab_size = dialectic_setup
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        logits, loss, _ = dialectic(x)

        # 損失を逆伝播
        loss.backward()

        # 入力xに勾配が伝播しているか確認
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert x.grad.norm() > 0, "Gradient should flow back to input"

        # ジェネレータの重みに勾配があるか確認
        assert dialectic.generator_head.weight.grad is not None

        # Hamiltonian ODEのパラメータにも勾配が流れるべき（批判を通じて生成を改善するため）
        # ただし、Hamiltonian Neural ODEの実装によっては、パラメータ勾配のチェックが必要
        # ここでは少なくとも微分可能であることを確認

    def test_integration_with_training_loop(self, dialectic_setup):
        """学習ループでの統合テストシミュレーション"""
        dialectic, d_model, _ = dialectic_setup
        optimizer = torch.optim.Adam(dialectic.parameters(), lr=0.01)
        x = torch.randn(4, 10, d_model)

        # 初期状態
        _, initial_loss, _ = dialectic(x)

        # 1ステップ更新
        optimizer.zero_grad()
        initial_loss.backward()
        optimizer.step()

        # 更新後の損失
        _, new_loss, _ = dialectic(x)

        # 損失が変化していることを確認（必ずしも下がるとは限らないが、パラメータは更新される）
        assert initial_loss.item() != new_loss.item()

    def test_numerical_stability(self, dialectic_setup):
        """数値安定性のテスト"""
        dialectic, d_model, _ = dialectic_setup
        x = torch.randn(4, 100, d_model) * 10.0 # 大きな入力

        logits, loss, _ = dialectic(x)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(loss)
