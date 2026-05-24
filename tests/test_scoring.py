"""Tests for per-method scoring functions and the L-CAVO Lyapunov term."""
import sys; sys.path.insert(0, '.')

import numpy as np

from lcavo_sim import (
    SS,
    _lp,
    _nsf,
    _np,
    sc_cg,
    sc_ea,
    sc_la,
    sc_lc,
    sc_rf,
)


def _state():
    G, reg, _, _ = _nsf()
    rng = np.random.default_rng(7)
    np0 = _np(G, rng)
    lp0 = _lp(G, rng)
    st = SS(G, np0, lp0)
    ci = {"A": 100.0, "B": 250.0, "C": 400.0, "D": 600.0}
    return G, reg, st, ci


def test_sc_la_returns_propagation_delay_only():
    """Latency-aware scorer ignores everything except propagation delay."""
    _, reg, st, ci = _state()
    assert sc_la(0, 10, st, reg, ci, pd=7.5) == 7.5
    assert sc_la(5, 1, st, reg, ci, pd=0.0) == 0.0


def test_sc_rf_is_constant_zero():
    """Random baseline scorer must be flat so np.argmin is uniform after shuffle."""
    _, reg, st, ci = _state()
    for n in range(5):
        assert sc_rf(n, 10, st, reg, ci, pd=n) == 0


def test_sc_ea_prefers_lower_power():
    """Energy-aware scorer for a fresh node should be the linear-power increment."""
    _, reg, st, ci = _state()
    s_small = sc_ea(0, 10, st, reg, ci, pd=5.0)
    s_large = sc_ea(0, 100, st, reg, ci, pd=5.0)
    assert s_large > s_small


def test_sc_cg_grows_with_carbon_intensity():
    """Carbon-greedy on the same node should grow as the region's CI rises."""
    _, reg, st, _ = _state()
    ci_low = {"A": 50.0, "B": 50.0, "C": 50.0, "D": 50.0}
    ci_high = {"A": 800.0, "B": 800.0, "C": 800.0, "D": 800.0}
    assert sc_cg(0, 20, st, reg, ci_high, pd=1.0) > sc_cg(0, 20, st, reg, ci_low, pd=1.0)


def test_sc_lc_matches_paper_formula():
    """L-CAVO score equals V·ΔCO₂ + 0.5·delay + 0.3·util + 0.05·Q (paper Eq.)."""
    from lcavo_sim import PUE, co2, pw
    G, reg, st, ci = _state()
    V, Qt = 50.0, 4.0
    n, c, pd = 0, 20, 3.0
    scorer = sc_lc(V, Qt)
    actual = scorer(n, c, st, reg, ci, pd)
    # Recompute the expected value the same way the function does
    old_u = st.u(n)
    new_u = (st.ld[n] + c) / st.np[n]["C"]
    dp = pw(new_u, st.np[n]["Pi"], st.np[n]["Pm"]) - pw(old_u, st.np[n]["Pi"], st.np[n]["Pm"])
    expected = V * co2(dp, ci[reg[n]]) + 0.5 * pd + 0.3 * new_u + 0.05 * Qt
    assert abs(actual - expected) < 1e-9


def test_sc_lc_queue_term_is_linear_in_Q():
    """Holding V and node state fixed, score must rise by exactly 0.05·ΔQ."""
    _, reg, st, ci = _state()
    f0 = sc_lc(50.0, 0.0)(0, 10, st, reg, ci, pd=2.0)
    f1 = sc_lc(50.0, 100.0)(0, 10, st, reg, ci, pd=2.0)
    assert abs((f1 - f0) - 0.05 * 100.0) < 1e-9
