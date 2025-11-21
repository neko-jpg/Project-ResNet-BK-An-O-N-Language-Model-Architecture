import pytest
import torch
import os
import shutil
import time
from src.models.phase4.dream_core.inverse_diffusion import DreamCore
from src.models.phase4.dream_core.pipeline_manager import PassivePipelineManager
from src.models.phase4.dream_core.visualization import visualize_dream_as_text, visualize_dream_as_knots

class MockEthicalFilter:
    def __init__(self, should_pass=True):
        self.should_pass = should_pass

    def check(self, concept):
        return self.should_pass

class MockTopologicalMemory:
    def __init__(self):
        self.knots = []

    def add_knot(self, concept, metadata):
        self.knots.append((concept, metadata))

class TestDreamCore:
    def test_dream_generation(self):
        d_model = 64
        model = DreamCore(d_model, n_fragments=5, diffusion_steps=10)

        fragments = torch.randn(5, d_model)

        concept, diagnostics = model(fragments)

        assert concept.shape == (d_model,)
        assert not torch.isnan(concept).any()
        assert 'trajectory' in diagnostics
        assert 'final_energy' in diagnostics
        assert diagnostics['trajectory'].shape[0] == 11 # 1 initial + 10 steps

    def test_gradient_flow(self):
        d_model = 64
        model = DreamCore(d_model, n_fragments=5, diffusion_steps=5)
        fragments = torch.randn(5, d_model, requires_grad=True)

        model.train() # Enable checkpointing logic if implemented
        concept, _ = model(fragments)

        loss = concept.sum()
        loss.backward()

        assert fragments.grad is not None

class TestPassivePipeline:
    def test_pipeline_execution(self):
        d_model = 64
        dream_core = DreamCore(d_model, n_fragments=5, diffusion_steps=5)
        topo_mem = MockTopologicalMemory()
        ethical_filter = MockEthicalFilter(should_pass=True)

        manager = PassivePipelineManager(dream_core, topo_mem, ethical_filter)

        fragments = torch.randn(5, d_model)

        # Manual trigger
        concept = manager.generate_dream(fragments)

        assert concept is not None
        assert len(topo_mem.knots) == 1
        assert topo_mem.knots[0][1]['source'] == 'dream'

    def test_ethical_rejection(self):
        d_model = 64
        dream_core = DreamCore(d_model)
        topo_mem = MockTopologicalMemory()
        ethical_filter = MockEthicalFilter(should_pass=False)

        manager = PassivePipelineManager(dream_core, topo_mem, ethical_filter)

        fragments = torch.randn(10, d_model)
        concept = manager.generate_dream(fragments)

        assert concept is None
        assert len(topo_mem.knots) == 0

    def test_async_loop(self):
        # Test loop mechanism using a counter in provider
        d_model = 64
        dream_core = DreamCore(d_model)
        topo_mem = MockTopologicalMemory()
        ethical_filter = MockEthicalFilter(should_pass=True)
        manager = PassivePipelineManager(dream_core, topo_mem, ethical_filter)

        count = 0
        def provider():
            nonlocal count
            if count < 3:
                count += 1
                return torch.randn(10, d_model)
            return None

        manager.start_passive_loop(provider, interval=0.1)
        time.sleep(0.5) # Wait for a few loops
        manager.stop_passive_loop()

        assert count == 3
        assert len(topo_mem.knots) >= 1

class TestDreamVisualization:
    def test_text_generation(self):
        d_model = 64
        concept = torch.randn(d_model)
        text = visualize_dream_as_text(concept, diagnostics={'final_energy': 0.5})
        assert isinstance(text, str)
        assert len(text) > 0
        assert "dreamt" in text

    def test_knot_visualization(self):
        d_model = 64
        concept = torch.randn(d_model)
        save_path = "test_dream_knot.png"

        try:
            visualize_dream_as_knots(concept, save_path)
            assert os.path.exists(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
