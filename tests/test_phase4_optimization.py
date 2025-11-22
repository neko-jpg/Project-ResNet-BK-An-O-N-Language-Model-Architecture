
import torch
import unittest
from unittest.mock import MagicMock
from src.models.phase4.integrated_model import Phase4IntegratedModel

class TestPhase4Optimization(unittest.TestCase):
    def test_lazy_evaluation(self):
        """
        Verify that heavy components are NOT called when return_diagnostics=False.
        """
        # Mock Phase 3 Model
        mock_phase3 = MagicMock()
        mock_phase3.d_model = 64
        mock_phase3.config = MagicMock()
        # Set config attributes explicitly to avoid MagicMock being passed to torch functions
        mock_phase3.config.max_seq_len = 20
        mock_phase3.config.vocab_size = 100

        # Return a dummy dict from forward
        mock_phase3.return_value = {
            'logits': torch.randn(2, 10, 100),
            'loss': torch.tensor(1.0),
            'diagnostics': {}
        }

        # Initialize Phase 4 Model
        model = Phase4IntegratedModel(
            phase3_model=mock_phase3,
            enable_emotion=True,
            enable_dream=False,
            enable_holographic=True,
            enable_quantum=True,
            enable_topological=False,
            enable_ethics=False,
            enable_meta=True,
            enable_boundary=True
        )

        # Helper to create mock modules
        class MockModule(torch.nn.Module):
            def __init__(self, return_val):
                super().__init__()
                self.mock_call = MagicMock(return_value=return_val)
            def forward(self, *args, **kwargs):
                return self.mock_call(*args, **kwargs)

        # Mock the components with MockModules
        model.emotion_detector = MockModule({'state': 'CALM'})
        model.bulk_generator = MockModule((None, {}))
        model.quantum_observer = MockModule((None, {'entropy_reduction': torch.tensor([0.0])}))

        # Boundary Core and Meta Commentary are not nn.Modules, so MagicMock is fine if they are just objects
        # But integrated_model expects them to be assigned.
        # boundary_core is MockDocumentFetcher (object). meta is MetaCommentary (object).
        # But PyTorch might not complain if they are not submodules?
        # Let's check how they are assigned.
        # self.boundary_core = MockDocumentFetcher()
        # If they are not nn.Modules, they are just attributes.
        model.boundary_core = MagicMock()
        model.meta_commentary = MagicMock()

        # Run with return_diagnostics=False
        input_ids = torch.randint(0, 100, (2, 10))
        _ = model(input_ids, return_diagnostics=False)

        # Assert NOT called
        model.emotion_detector.mock_call.assert_not_called()
        model.bulk_generator.mock_call.assert_not_called()
        model.quantum_observer.mock_call.assert_not_called()
        model.boundary_core.fetch.assert_not_called()
        model.meta_commentary.generate_commentary.assert_not_called()

        print("\nLazy Evaluation Verified: Components were NOT called.")

        # Run with return_diagnostics=True
        _ = model(input_ids, return_diagnostics=True)

        # Assert called
        model.emotion_detector.mock_call.assert_called()
        model.bulk_generator.mock_call.assert_called()
        model.quantum_observer.mock_call.assert_called()
        model.boundary_core.fetch.assert_called()
        model.meta_commentary.generate_commentary.assert_called()

        print("Active Evaluation Verified: Components WERE called.")

if __name__ == '__main__':
    unittest.main()
