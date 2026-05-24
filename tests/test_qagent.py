"""Tests for the QL-CAVO tabular Q-learning agent."""
import sys; sys.path.insert(0, '.')

import numpy as np

from lcavo_sim import QAgent, carbon_prof


def test_agent_q_table_shape():
    """Paper specifies a 6×4×3 state grid and 4 actions (one per region)."""
    agent = QAgent()
    assert agent.Q.shape == (6, 4, 3, 4)


def test_agent_action_in_range():
    """ε-greedy must return a valid region index 0..3."""
    agent = QAgent(seed=1)
    for h in range(0, 24, 3):
        for mc in (50, 220, 380, 600):
            for q in (0.0, 3.0, 20.0):
                assert agent.act(h, mc, q) in (0, 1, 2, 3)


def test_state_discretisation_buckets():
    """State indices must respect the documented bucket bounds."""
    agent = QAgent()
    h_idx, c_idx, q_idx = agent._s(23, 700, 100.0)
    assert 0 <= h_idx < 6
    assert 0 <= c_idx < 4
    assert 0 <= q_idx < 3
    # Hour 0 / very-clean grid / empty queue → first bucket of each axis
    assert agent._s(0, 50, 0.0) == (0, 0, 0)


def test_pretrain_modifies_q_table():
    """300 pre-training episodes on historical profiles must touch the table."""
    agent = QAgent(seed=11)
    before = agent.Q.copy()
    agent.pretrain(carbon_prof(), ep=20)
    assert not np.allclose(agent.Q, before)


def test_weights_disfavour_preferred_region():
    """weights(a) returns a low weight (=cheap) for the preferred region."""
    agent = QAgent()
    w = agent.weights(1)  # prefer region B
    assert w["B"] == min(w.values())


def test_epsilon_decays_below_initial():
    """Exploration rate must monotonically decay during updates."""
    agent = QAgent(seed=3)
    initial = agent.eps
    for _ in range(50):
        agent.update(0, 100, 0.0, 0, -1.0, 1, 100, 0.0)
    assert agent.eps < initial
    assert agent.eps >= 0.02  # documented floor
