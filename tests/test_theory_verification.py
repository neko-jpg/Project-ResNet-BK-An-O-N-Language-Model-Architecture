# tests/test_theory_verification.py

"""
Tests for verifying the mathematical and theoretical claims of the model.

This test suite is designed to bridge the gap between the mathematical theory
in `riemann_hypothesis_main.tex` and the PyTorch implementation. Each test
focuses on a specific theoretical property and verifies its numerical behavior
on a small scale.
"""

import torch
import pytest
from src.models.birman_schwinger_core import BirmanSchwingerCore

@pytest.mark.unit
class TestTheoreticalProperties:
    """
    Test suite for core theoretical properties of the ResNet-BK model.
    """

    def test_mourre_estimate_verification(self):
        """
        Tests the numerical verification of the Mourre estimate.

        Theory Reference: `riemann_hypothesis_main.tex`, Theorem mourre-H0.

        The Mourre estimate, `[H_0, iA] = I`, provides a fundamental guarantee
        for the stability of the system. The `verify_mourre_estimate` method
        in `BirmanSchwingerCore` implements a numerical check of this identity
        for the discretized operators. This test ensures that the check passes.
        """
        n_seq = 128  # A reasonable sequence length for a unit test

        # Instantiate the core with Mourre estimate verification enabled
        core = BirmanSchwingerCore(
            n_seq=n_seq,
            use_mourre=True,
            use_lap=False  # Isolate the Mourre test
        )

        # The method should return True if the commutator is close to identity
        is_mourre_verified = core.verify_mourre_estimate()

        # Retrieve the actual error for more informative assertion
        # (This requires a slight modification of the core method, for now we assume it exists)
        # For now, we just assert the boolean return value.
        # In a future step, we could make `verify_mourre_estimate` return the error value.

        assert is_mourre_verified, \
            "The numerical Mourre estimate verification failed. " \
            "The commutator [H_0, iA] is not close to the identity matrix. " \
            "This indicates a potential discrepancy between the discretized operators " \
            "and their continuous counterparts in the theory."
