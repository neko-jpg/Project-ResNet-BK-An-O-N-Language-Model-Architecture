"""
Unit tests for the EpsilonScheduler.
"""
import torch
import torch.nn as nn
import pytest
import math

from src.training.epsilon_scheduler import EpsilonScheduler
from src.models.phase7.hyperbolic_attention import HyperbolicMultiHeadAttention

T_MAX = 1000
EPSILON_MAX = 0.5
EPSILON_MIN = 0.01

@pytest.fixture
def scheduler():
    """
    Returns an instance of the EpsilonScheduler.
    """
    return EpsilonScheduler(t_max=T_MAX, epsilon_max=EPSILON_MAX, epsilon_min=EPSILON_MIN)

def test_scheduler_initialization(scheduler):
    """
    Tests that the scheduler is initialized correctly.
    """
    assert scheduler.t_max == T_MAX
    assert scheduler.epsilon_max == EPSILON_MAX
    assert scheduler.epsilon_min == EPSILON_MIN

def test_get_epsilon_bounds(scheduler):
    """
    Tests that get_epsilon returns the correct values at the boundaries.
    """
    assert scheduler.get_epsilon(0) == pytest.approx(EPSILON_MAX)
    assert scheduler.get_epsilon(T_MAX) == pytest.approx(EPSILON_MIN)

def test_get_epsilon_midpoint(scheduler):
    """
    Tests that get_epsilon returns the correct value at the midpoint.
    """
    mid_step = T_MAX // 2
    expected_epsilon = EPSILON_MIN + 0.5 * (EPSILON_MAX - EPSILON_MIN) * (1 + math.cos(math.pi * mid_step / T_MAX))
    assert scheduler.get_epsilon(mid_step) == pytest.approx(expected_epsilon)

def test_update_model_curvature():
    """
    Tests that the scheduler correctly updates the log_c parameter in a model.
    """
    # Create a dummy model with a log_c parameter
    model = HyperbolicMultiHeadAttention(d_model=64, num_heads=4, use_triton_kernel=False)

    scheduler = EpsilonScheduler(t_max=T_MAX)

    # Check initial value
    initial_log_c = model.log_c.item()

    # Update curvature at step 0
    scheduler.update_model_curvature(model, 0)
    log_c_at_start = model.log_c.item()

    # Expected value: -log(epsilon_max)
    expected_log_c_start = -math.log(scheduler.epsilon_max)
    assert log_c_at_start == pytest.approx(expected_log_c_start)

    # Update curvature at last step
    scheduler.update_model_curvature(model, T_MAX)
    log_c_at_end = model.log_c.item()

    # Expected value: -log(epsilon_min)
    expected_log_c_end = -math.log(scheduler.epsilon_min)
    assert log_c_at_end == pytest.approx(expected_log_c_end)

    # Check that the value has changed
    assert log_c_at_start != initial_log_c
    assert log_c_at_end != log_c_at_start
