"""End-to-end tests for the per-scenario simulation loop."""
import sys; sys.path.insert(0, '.')

import numpy as np

from lcavo_sim import T, metrics, sim


REQUIRED_METRIC_KEYS = {
    "carbon", "power", "delay", "accept", "active",
    "rpct", "nadm", "ntot", "dlist", "rt", "t", "Qt",
}


def test_sim_returns_one_run_per_seed():
    """sim() should produce *ns* outer lists, each containing T per-slot dicts."""
    seeds = sim("NSFNET", "Low", "L-CAVO", ns=2)
    assert len(seeds) == 2
    for run in seeds:
        assert len(run) == T


def test_metrics_keys_present():
    """Every per-slot metric dict must carry the keys the figure code reads."""
    seeds = sim("NSFNET", "Low", "Energy-aware", ns=1)
    for slot in seeds[0]:
        missing = REQUIRED_METRIC_KEYS - set(slot)
        assert not missing, f"missing metric keys: {missing}"


def test_accept_rate_within_bounds():
    """Acceptance rate is reported as a percentage in [0, 100]."""
    seeds = sim("NSFNET", "Medium", "L-CAVO", ns=1)
    for slot in seeds[0]:
        assert 0.0 <= slot["accept"] <= 100.0


def test_carbon_and_power_non_negative():
    """No baseline should ever record negative emissions or power."""
    seeds = sim("NSFNET", "Low", "Carbon-greedy", ns=1)
    for slot in seeds[0]:
        assert slot["carbon"] >= 0
        assert slot["power"] >= 0


def test_sim_is_reproducible():
    """Same scenario + same default seed → identical aggregate carbon."""
    a = sim("NSFNET", "Low", "L-CAVO", ns=1)
    b = sim("NSFNET", "Low", "L-CAVO", ns=1)
    ca = sum(s["carbon"] for s in a[0])
    cb = sum(s["carbon"] for s in b[0])
    assert abs(ca - cb) < 1e-9


def test_qlcavo_updates_virtual_queue():
    """QL-CAVO shares the virtual-queue mechanism with L-CAVO; Qt must be ≥0."""
    seeds = sim("NSFNET", "Medium", "QL-CAVO", ns=1)
    for slot in seeds[0]:
        assert slot["Qt"] >= 0.0


def test_geant_low_load_runs():
    """The 23-node GÉANT topology must complete a Low-load simulation."""
    seeds = sim("GEANT", "Low", "Latency-aware", ns=1)
    assert len(seeds) == 1
    assert len(seeds[0]) == T
