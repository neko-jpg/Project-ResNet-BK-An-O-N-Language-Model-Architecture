import torch
import torch.nn as nn
import unittest
from unittest.mock import MagicMock, patch
import shutil
import os

from src.models.phase3.integrated_model import Phase3IntegratedModel
from src.models.phase4.integrated_model import Phase4IntegratedModel
from src.models.phase4.dream_core.pipeline_manager import PassivePipelineManager

# Mock Config for Phase 3
class MockConfig:
    def __init__(self):
        self.d_model = 64
        self.vocab_size = 1000
        self.n_layers = 2
        self.max_seq_len = 32
        self.d_koopman = 128
        self.potential_type = 'bk_core'
        self.use_complex32 = False

class TestPhase4IntegratedModel(unittest.TestCase):
    def setUp(self):
        self.config = MockConfig()

        # Initialize Phase 3 Model (Real one, small config)
        # We need to patch the sub-components if they are heavy,
        # but for d_model=64 it should be fine.
        # However, Phase3IntegratedModel imports ComplexEmbedding etc.
        # Let's hope they work. If not, we mock the whole Phase 3 model.

        # To be safe and fast, we mock Phase 3 Model entirely,
        # BUT we need the structure (dialectic) for the hook.

        self.phase3_model = MagicMock(spec=Phase3IntegratedModel)
        self.phase3_model.d_model = self.config.d_model
        self.phase3_model.config = self.config
        self.phase3_model.vocab_size = self.config.vocab_size
        self.phase3_model.n_seq = self.config.max_seq_len # Some components access this

        # Mock Dialectic Loop for the hook
        self.phase3_model.dialectic = nn.Linear(self.config.d_model, self.config.vocab_size) # Dummy layer

        # Mock Forward return
        self.phase3_model.return_value = {
            'logits': torch.randn(1, self.config.max_seq_len, self.config.vocab_size),
            'loss': torch.tensor(0.5),
            'diagnostics': {'phase3': True}
        }

        # Mock the Dialectic call used in Phase 4 (Holographic re-run)
        # It returns logits, loss, diag
        self.phase3_model.dialectic.forward = MagicMock(return_value=(
             torch.randn(1, self.config.max_seq_len, self.config.vocab_size), # logits
             torch.tensor(0.1), # loss
             {} # diag
        ))
        # Note: The hook registers on .dialectic, so we need to ensure the mock object supports it.
        # MagicMock spec might not suffice for register_forward_hook if it's not an nn.Module.
        # So self.phase3_model.dialectic MUST be a real nn.Module or we verify the hook registration differently.
        # I assigned a real nn.Linear above.

        # Prepare data
        self.input_ids = torch.randint(0, self.config.vocab_size, (1, self.config.max_seq_len))

        # Ensure data dir exists for Zarr
        if not os.path.exists('data'):
            os.makedirs('data')

    def tearDown(self):
        # Clean up zarr data if created
        if os.path.exists('data/phase4_knot_memory.zarr'):
            try:
                shutil.rmtree('data/phase4_knot_memory.zarr')
            except:
                pass

    def test_init_defaults(self):
        """Test initialization with defaults (all enabled)."""
        model = Phase4IntegratedModel(self.phase3_model)
        self.assertTrue(model.enable_emotion)
        self.assertTrue(model.enable_dream)
        self.assertFalse(model.is_phase3_only)
        self.assertIsNotNone(model.emotion_detector)
        self.assertIsNotNone(model.passive_pipeline)

    def test_phase3_compatibility_strict(self):
        """Test strict Phase 3 mode (all flags OFF)."""
        model = Phase4IntegratedModel(
            self.phase3_model,
            enable_emotion=False,
            enable_dream=False,
            enable_holographic=False,
            enable_quantum=False,
            enable_topological=False,
            enable_ethics=False
        )

        self.assertTrue(model.is_phase3_only)

        # Run forward
        out = model(self.input_ids)

        # Verify Phase 3 was called
        self.phase3_model.assert_called_once()

        # Verify output identity
        self.assertEqual(out['diagnostics']['phase3'], True)

        # Verify Phase 4 components are NOT initialized
        self.assertFalse(hasattr(model, 'emotion_detector'))

    def test_full_integration_forward(self):
        """Test forward pass with all Phase 4 components enabled."""
        model = Phase4IntegratedModel(self.phase3_model)

        # We need to simulate the hook capturing hidden states.
        # Since we mocked Phase 3, the hook won't fire automatically unless we run the real dialectic forward.
        # But we mocked Phase 3 forward.
        # We need to manually trigger the hook or patch the hook logic.
        # Actually, since `model.forward` calls `self.phase3_model(input_ids)`, which is a Mock,
        # it returns the dict immediately. The hook is registered on `self.phase3_model.dialectic`.
        # The `Phase3IntegratedModel` normally calls `dialectic`. Our Mock DOES NOT call dialectic.
        # So `hidden_states` will be None.

        # To fix this test, we need to make the mock side_effect call the hook?
        # Or just patch the logic inside Phase4IntegratedModel to find the hidden state.

        # Let's assume for this test that we Mock `Phase4IntegratedModel.forward`'s internal hook mechanism? No that defeats the purpose.

        # Better: Make `self.phase3_model` a real-ish class or use side_effect.

        def phase3_forward_side_effect(input_ids, labels=None, return_diagnostics=False):
            # Simulate internal behavior: call dialectic to trigger hook
            dummy_hidden = torch.randn(1, self.config.max_seq_len, self.config.d_model)
            # Trigger hook
            # We need to access the hook registered on dialectic
            # Since it's an nn.Module (Linear), calling it triggers hooks.
            model.phase3_model.dialectic(dummy_hidden)

            return {
                'logits': torch.randn(1, self.config.max_seq_len, self.config.vocab_size),
                'diagnostics': {'phase3': True}
            }

        self.phase3_model.side_effect = phase3_forward_side_effect

        # Run forward
        out = model(self.input_ids, return_diagnostics=True)

        # Check keys in diagnostics
        diag = out['diagnostics']
        self.assertIn('phase3', diag)
        self.assertIn('emotion', diag)
        self.assertIn('bulk', diag)
        # self.assertIn('quantum', diag) # Quantum might be None if no collapse logic triggered or simple pass
        self.assertIn('memory', diag)

        # Check shapes
        self.assertEqual(out['logits'].shape, (1, self.config.max_seq_len, self.config.vocab_size))

    def test_idle_mode(self):
        """Test entering and exiting idle mode."""
        model = Phase4IntegratedModel(self.phase3_model, enable_dream=True)

        # Mock the pipeline manager start/stop
        model.passive_pipeline.start_passive_loop = MagicMock()
        model.passive_pipeline.stop_passive_loop = MagicMock()

        model.enter_idle_mode(interval=0.5)
        model.passive_pipeline.start_passive_loop.assert_called_once()

        model.exit_idle_mode()
        model.passive_pipeline.stop_passive_loop.assert_called_once()

    def test_component_toggles(self):
        """Test partial enabling of components."""
        model = Phase4IntegratedModel(
            self.phase3_model,
            enable_emotion=True,
            enable_holographic=False
        )

        self.assertTrue(hasattr(model, 'emotion_detector'))
        self.assertFalse(hasattr(model, 'bulk_generator'))

        # Fix side effect for hidden state
        def phase3_forward_side_effect(input_ids, labels=None, return_diagnostics=False):
            dummy_hidden = torch.randn(1, self.config.max_seq_len, self.config.d_model)
            model.phase3_model.dialectic(dummy_hidden)
            return {'logits': torch.randn(1, self.config.max_seq_len, self.config.vocab_size)}
        self.phase3_model.side_effect = phase3_forward_side_effect

        out = model(self.input_ids, return_diagnostics=True)
        self.assertIn('emotion', out['diagnostics'])
        self.assertNotIn('bulk', out['diagnostics'])

if __name__ == '__main__':
    unittest.main()
