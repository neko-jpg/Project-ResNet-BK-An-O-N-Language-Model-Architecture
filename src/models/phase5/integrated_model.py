import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple

from src.models.resnet_bk import LanguageModel
from src.models.phase5.monad.state_monad import ConsciousnessMonad
from src.models.phase5.ethics.sheaf_ethics import SheafEthics
from src.models.phase5.quantum.process_matrix import QuantumProcessMatrix
from src.models.phase5.quantum.superposition_state import SuperpositionState
from src.models.phase5.physics.adaptive_lyapunov import AdaptiveLyapunovControl
from src.models.phase6.curriculum.pacing_controller import PacingController

class Phase5IntegratedModel(nn.Module):
    """
    MUSE Phase 5: The "Consciousness" Monad & Topological Ethics.

    Integrates:
    - ResNet-BK (Physics Core)
    - Consciousness Monad (State, Writer, Reflector)
    - Sheaf Ethics (Inconsistency detection)
    - Quantum Process Matrix (Parallel Causal Paths)
    - Phase 6: Curriculum Pacing (Fatigue, Temperature)
    """

    def __init__(
        self,
        base_model: LanguageModel,
        d_model: int,
        vocab_size: int,
        max_seq_len: int = 128,
        beam_width: int = 2
    ):
        super().__init__()
        self.base_model = base_model
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Phase 5 Components
        self.monad = ConsciousnessMonad(d_model=d_model)
        self.ethics = SheafEthics(d_model=d_model, max_nodes=max_seq_len)
        self.quantum = QuantumProcessMatrix(beam_width=beam_width)
        self.physics_ctrl = AdaptiveLyapunovControl()

        # Phase 6 Components
        self.pacing = PacingController()

    def forward(
        self,
        input_ids: torch.Tensor, # (B, N)
        initial_state: Optional[Dict] = None,
        use_quantum: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with Monad integration.
        """
        if use_quantum:
            return self.forward_quantum(input_ids)

        # 1. Standard Forward through Physics Core
        x = input_ids
        batch_size, n_seq = x.shape

        tok_emb = self.base_model.token_embedding(x)
        pos = torch.arange(0, n_seq, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.base_model.position_embedding(pos)
        h = tok_emb + pos_emb

        # Pass through blocks
        for block in self.base_model.blocks:
            h = block(h)

        h = self.base_model.layer_norm_final(h) # (B, N, D)
        logits = self.base_model.lm_head(h)     # (B, N, V)

        # 2. Phase 5 & 6 Logic

        # A. Sheaf Ethics Check
        adj = torch.eye(n_seq, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        energy, ethics_diag = self.ethics(h, adj) # (B,)

        # B. Monad Update (Reflector)
        current_state = h[:, -1, :] # (B, D)
        thoughts, thought_embeds = self.monad.get_inner_voice()

        # --- Task 4 Integration: Pain Signal ---
        physics_diag = self.base_model.get_stability_diagnostics()
        cond_num = physics_diag.get('max_condition_number', 1.0)
        unitarity = self.base_model.blocks[-1].bk_layer.check_unitarity_violation().item() if hasattr(self.base_model.blocks[-1], 'bk_layer') else 0.0

        pain_signal = torch.log10(torch.tensor(max(1.0, cond_num))).item() + unitarity

        if pain_signal > 1.0:
            self.monad.log_thought(f"System instability detected. Pain level: {pain_signal:.2f}")

        new_params = self.monad.update_physics(current_state, thought_embeds)

        # C. ALC Update (Lyapunov)
        norm_in = (tok_emb + pos_emb).norm(dim=-1).mean()
        norm_out = h.norm(dim=-1).mean()
        local_lambda = torch.log((norm_out + 1e-9) / (norm_in + 1e-9)).item()
        gamma_target = self.physics_ctrl.compute_gamma(local_lambda)

        # D. Phase 6: Curriculum Pacing
        pacing_mode = self.pacing.step(logits, h, pain_signal)

        # Log pacing status
        if pacing_mode['status'] == 'exhausted':
            self.monad.log_thought("I am exhausted. Switching to low precision mode.")

        diagnostics = {
            'ethics': ethics_diag,
            'physics_params': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in new_params.items()},
            'lyapunov': local_lambda,
            'target_gamma': gamma_target,
            'pain_signal': pain_signal,
            'condition_number': cond_num,
            'pacing': pacing_mode
        }

        return logits, diagnostics

    def forward_quantum(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantum Process Matrix Execution.
        """
        logits, diagnostics = self.forward(input_ids, use_quantum=False)

        energy = diagnostics['ethics']['sheaf_energy']
        if isinstance(energy, float):
             energy = torch.tensor([energy], device=input_ids.device).expand(input_ids.shape[0])

        diagnostics['quantum_mode'] = True
        diagnostics['action_potential'] = energy

        return logits, diagnostics
