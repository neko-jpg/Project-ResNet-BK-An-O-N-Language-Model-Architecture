"""
Phase 3 Integrated Model: Physics Transcendence

このモジュールは、Phase 3の全コンポーネントを統合した完全なモデルを実装します。

Components:
    1. Complex Embedding & Phase 2 Compatibility
    2. MERA Router (Global Context)
    3. Phase 3 Block (Complex LayerNorm -> Hamiltonian ODE -> Koopman -> Residual)
    4. Dialectic Loop (Self-Correction)

Requirements:
    - Requirement 7: Phase 3 Integrated Model
    - Requirement 19: Phase 3 Block
    - Requirement 20: Phase 3 Integrated Model Implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List

# Phase 3 Components
from src.models.phase3.complex_embedding import ComplexEmbedding
from src.models.phase3.mera import MERARouter
from src.models.phase3.hamiltonian_ode import HamiltonianNeuralODE
from src.models.phase3.koopman import KoopmanOperator
from src.models.phase3.dialectic_loop import DialecticLoop
from src.models.phase3.complex_ops import ComplexLayerNorm, ComplexLinear
from src.models.phase3.complex_tensor import ComplexTensor

class Phase3Block(nn.Module):
    """
    Phase 3 Building Block (Task 19)

    Architecture:
        Input (Complex)
        ↓
        Complex LayerNorm
        ↓
        Hamiltonian Neural ODE (Time Evolution)
        ↓
        Koopman Operator (Global Linearization)
        ↓
        Residual Connection
        ↓
        Output

    Args:
        d_model: モデル次元
        d_koopman: Koopman次元
        potential_type: Hamiltonianポテンシャルの種類
    """
    def __init__(
        self,
        d_model: int,
        d_koopman: int,
        potential_type: str = 'bk_core'
    ):
        super().__init__()

        # 1. Complex LayerNorm
        # 複素数空間での正規化（Requirement 7.1）
        self.norm = ComplexLayerNorm(d_model)

        # 2. Hamiltonian Neural ODE
        # エネルギー保存則に従う時間発展（Requirement 7.2）
        # ComplexTensorの振幅（実部）と位相（虚部）を(q, p)に対応させるか、
        # あるいはComplexTensorを実数(2D)に展開してHamiltonianに入力する
        # ここでは展開方式を採用: Complex(B, N, D) -> Real(B, N, 2D)
        self.hamiltonian = HamiltonianNeuralODE(
            d_model=d_model, # Input position dim
            potential_type=potential_type
        )

        # 3. Koopman Operator
        # 非線形ダイナミクスの線形化（Requirement 7.3）
        # Hamiltonianの出力（非線形）を線形空間で補正・加速
        # Hamiltonianは(q, p)を出力するので、入力次元は2*d_model
        self.koopman = KoopmanOperator(
            d_model=2 * d_model,
            d_koopman=d_koopman
        )

        # Output projection back to d_model size if needed
        # Koopman output is 2*d_model (state space), we might want to keep it or project
        # The block input is Complex(D), output should be Complex(D).
        # State space is (2D).

    def forward(
        self,
        x: ComplexTensor,
        return_diagnostics: bool = False
    ) -> Tuple[ComplexTensor, Optional[Dict[str, Any]]]:
        """
        Args:
            x: Input ComplexTensor (B, N, D)

        Returns:
            out: Output ComplexTensor (B, N, D)
            diagnostics: 診断情報
        """
        residuals = x

        # 1. LayerNorm
        normed_x = self.norm(x)

        # 2. Complex to Real (Phase Space) Conversion
        # Real part -> q (Position), Imag part -> p (Momentum)
        q = normed_x.real.float()
        p = normed_x.imag.float()
        # Combine to state vector (B, N, 2D)
        state_in = torch.cat([q, p], dim=-1)

        # 3. Hamiltonian ODE Time Evolution
        # t=0 -> t=1
        state_evolved = self.hamiltonian(state_in, t_span=(0, 1))

        # 4. Koopman Linearization & Correction
        state_koopman, _, _ = self.koopman(state_evolved)

        # 5. Real to Complex Conversion
        # Split back to q, p
        q_out, p_out = torch.chunk(state_koopman, 2, dim=-1)

        # Convert to ComplexTensor
        # Note: float32 -> float16 inside ComplexTensor.from_real if needed
        out_complex = ComplexTensor.from_real(q_out, p_out)

        # 6. Residual Connection
        out = residuals + out_complex

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'hamiltonian_diagnostics': self.hamiltonian.get_diagnostics(),
                # Energy check
                'energy_in': self.hamiltonian.h_func(0.0, state_in).mean().item(),
                'energy_out': self.hamiltonian.h_func(1.0, state_evolved).mean().item()
            }

        return out, diagnostics


class Phase3IntegratedModel(nn.Module):
    """
    Phase 3 Integrated Model (Task 20)

    Architecture:
        Input IDs
        ↓
        Complex Embedding
        ↓
        MERA Router (Extract Global Context)
        ↓
        Broadcast Global Context
        ↓
        Phase 3 Blocks x N
        ↓
        Dialectic Loop (Output Generation & Critique)

    Args:
        config: モデル設定オブジェクト
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # 1. Complex Embedding
        self.embedding = ComplexEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=getattr(config, 'max_seq_len', 2048),
            use_complex32=getattr(config, 'use_complex32', True)
        )

        # 2. MERA Router
        self.mera = MERARouter(
            d_model=config.d_model,
            max_seq_len=getattr(config, 'max_seq_len', 2048)
        )

        # 3. Phase 3 Blocks
        self.layers = nn.ModuleList([
            Phase3Block(
                d_model=config.d_model,
                d_koopman=getattr(config, 'd_koopman', config.d_model * 2),
                potential_type=getattr(config, 'potential_type', 'bk_core')
            )
            for _ in range(config.n_layers)
        ])

        # 4. Dialectic Loop (Output Head)
        # Hamiltonian for Dialectic Loop (reuse specific one or new one)
        # Typically Dialectic Loop needs its own critique mechanism
        critique_ode = HamiltonianNeuralODE(
            d_model=config.d_model,
            potential_type='mlp', # Lightweight critique
            potential_hidden_dim=config.d_model * 2
        )

        self.dialectic = DialecticLoop(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            hamiltonian_ode=critique_ode
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: (B, N)
            labels: (B, N) Optional
            return_diagnostics: bool

        Returns:
            output_dict: {
                'logits': (B, N, V),
                'loss': scalar,
                'diagnostics': dict
            }
        """
        # 1. Embedding
        x = self.embedding(input_ids) # ComplexTensor(B, N, D)

        # 2. MERA Routing (Real part mainly used for routing logic usually,
        # but we can project absolute value)
        # Project to real for MERA input
        x_real_proj = x.abs().float() # (B, N, D)
        global_context, mera_hierarchy = self.mera(x_real_proj)

        # Broadcast context and add to x (as Real part addition or Complex addition)
        # Assuming Global Context is "Real" semantic summary
        # We add it to the Real part of embedding
        global_context_expanded = self.mera.broadcast(global_context, x.shape[1]) # (B, N, D)

        # Add to real part of ComplexTensor
        # Note: ComplexTensor is immutable-ish structure, need to create new
        x = ComplexTensor(
            x.real + global_context_expanded.half(), # Assuming x is half
            x.imag
        )

        # 3. Phase 3 Layers
        layer_diagnostics = []
        for layer in self.layers:
            x, diag = layer(x, return_diagnostics=return_diagnostics)
            if diag:
                layer_diagnostics.append(diag)

        # 4. Output (Dialectic Loop)
        # Dialectic Loop expects Real Tensor input (Synthesis of complex state)
        # We take the magnitude or real part?
        # Ideally, we project Complex State to Real State for generation
        # Let's use Real part + Imag part (Interference)
        # Or just Real part.
        # Requirement 7.5: Complex -> Real conversion
        # Implementation: |x| or Real(x)
        # Let's use Real part for simple projection, but maybe magnitude captures energy best.
        # However, sign matters for logits.
        # Let's use x.real for now.
        x_final_real = x.real.float() # (B, N, D)

        logits, contradiction_loss, dialectic_diag = self.dialectic(x_final_real)

        loss = contradiction_loss

        # Language Modeling Loss
        if labels is not None:
            # Standard Cross Entropy
            lm_loss = nn.CrossEntropyLoss()(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
            # Total Loss = LM Loss + Contradiction Loss
            # Contradiction acts as regularization
            loss = lm_loss + 0.1 * contradiction_loss

        return {
            'logits': logits,
            'loss': loss,
            'diagnostics': {
                'mera_hierarchy': len(mera_hierarchy),
                'layer_diagnostics': layer_diagnostics,
                'dialectic_diagnostics': dialectic_diag
            } if return_diagnostics else None
        }
