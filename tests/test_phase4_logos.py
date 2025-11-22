"""
Unit tests for LOGOS Components (Phase 4 extension)
"""

import pytest
import torch
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.phase4.logos_tokenizer import ComplexTokenizer
from src.models.phase4.topological_memory.factuality_knots import FactualityKnots
from src.models.phase4.meta_commentary import MetaCommentary

class TestComplexTokenizer:
    def test_phase_shift_calculation(self):
        tokenizer = ComplexTokenizer(base_tokenizer=None)

        # Test neutral
        assert tokenizer.get_phase_shift("Hello world.") == 0.0

        # Test emphasis
        assert tokenizer.get_phase_shift("Hello world!") == pytest.approx(3.14159 / 4.0, 0.001)

        # Test question
        assert tokenizer.get_phase_shift("Hello world?") == pytest.approx(3.14159 / 2.0, 0.001)

        # Test mixed (additive in current logic)
        assert tokenizer.get_phase_shift("Really?!") == pytest.approx(3.14159 * 0.75, 0.001)

class TestFactualityKnots:
    def test_contradiction_detection(self):
        knots = FactualityKnots() # Load defaults

        # True statement
        is_violation, _ = knots.check_contradiction("France capital is Paris")
        assert not is_violation

        # Explicit contradiction
        is_violation, info = knots.check_contradiction("France capital is London")
        assert is_violation
        assert info['energy_penalty'] == float('inf')

        # Another trap
        is_violation, _ = knots.check_contradiction("I am vegetarian so I love steak")
        assert is_violation

class TestMetaCommentaryLogos:
    def test_consistency_check(self):
        meta = MetaCommentary()

        # Normal drift
        diag_normal = {'hamiltonian_drift': 0.01}
        comment_normal = meta.generate_commentary(diag_normal)
        assert "Self-Correction" not in comment_normal

        # High drift
        diag_high = {'hamiltonian_drift': 5.0}
        comment_high = meta.generate_commentary(diag_high)
        assert "Self-Correction" in comment_high
        assert "dH/dt" in comment_high

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
