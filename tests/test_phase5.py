import unittest
import torch
import torch.nn as nn
from src.models.phase5.monad.writer_monad import WriterMonad
from src.models.phase5.reflector.meta_network import Reflector
from src.models.phase5.monad.state_monad import ConsciousnessMonad
from src.models.phase5.ethics.sheaf_ethics import SheafEthics
from src.models.phase5.quantum.process_matrix import QuantumProcessMatrix
from src.models.phase5.quantum.superposition_state import SuperpositionState
from src.models.phase5.physics.adaptive_lyapunov import AdaptiveLyapunovControl

class TestPhase5(unittest.TestCase):

    def test_writer_monad(self):
        writer = WriterMonad()
        writer.tell("Test log", torch.tensor([0.1, 0.2]))
        logs, embeds = writer.listen()
        self.assertEqual(len(logs), 1)
        self.assertEqual(embeds.shape, (1, 2))
        writer.flush()
        logs, _ = writer.listen()
        self.assertEqual(len(logs), 0)

    def test_reflector(self):
        d_model = 32
        reflector = Reflector(d_model=d_model)
        h = torch.randn(2, d_model)
        params = reflector(h)
        self.assertEqual(params.shape, (2, 4)) # gamma, bump, decay, gain

    def test_consciousness_monad(self):
        d_model = 32
        monad = ConsciousnessMonad(d_model=d_model)
        h = torch.randn(2, d_model)

        # Test physics update
        params = monad.update_physics(h)
        self.assertIn('gamma', params)
        self.assertIn('bump_scale', params)

    def test_sheaf_ethics(self):
        d_model = 32
        seq_len = 10
        ethics = SheafEthics(d_model=d_model, max_nodes=seq_len)
        x = torch.randn(2, seq_len, d_model)
        adj = torch.rand(2, seq_len, seq_len)

        energy, diag = ethics(x, adj)
        self.assertEqual(energy.shape, (2,))
        self.assertIn('sheaf_energy', diag)

    def test_quantum_process(self):
        d_model = 16
        qpm = QuantumProcessMatrix(beam_width=2)
        seed = SuperpositionState(
            token_ids=[1],
            hidden_state=torch.zeros(1, d_model),
            cumulative_log_prob=0.0
        )
        qpm.initialize_superposition(seed)

        logits = torch.randn(1, 100)
        h = torch.randn(1, d_model)
        energies = torch.tensor([0.0])

        candidates = qpm.expand_superposition(logits, h, energies)
        self.assertEqual(len(candidates), 2)

    def test_adaptive_lyapunov(self):
        alc = AdaptiveLyapunovControl(base_gamma=0.01)
        # Mock growth
        local_lambda = 0.5 # Chaotic
        gamma = alc.compute_gamma(local_lambda)
        # Should increase gamma (damping) to counteract chaos
        self.assertGreater(gamma, 0.01)

if __name__ == '__main__':
    unittest.main()
