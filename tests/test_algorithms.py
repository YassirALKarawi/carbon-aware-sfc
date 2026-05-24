"""Tests for placement algorithms."""
import sys; sys.path.insert(0, '.')
from lcavo_sim import sim

def test_lcavo_completes():
    """L-CAVO should complete on small run."""
    result = sim("NSFNET", "Low", "L-CAVO", ns=1)
    assert result is not None

def test_random_completes():
    """Random baseline should complete."""
    result = sim("NSFNET", "Low", "Random", ns=1)
    assert result is not None

def test_energy_aware_completes():
    """Energy-aware baseline should complete."""
    result = sim("NSFNET", "Low", "Energy-aware", ns=1)
    assert result is not None
