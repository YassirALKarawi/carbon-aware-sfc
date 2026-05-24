"""Tests for SFC request generation (Poisson arrivals + per-chain VNF spec)."""
import sys; sys.path.insert(0, '.')

import numpy as np

from lcavo_sim import _nsf, gen_reqs


REQUIRED_REQUEST_KEYS = {"s", "q", "K", "cpu", "bw", "Dmax", "pd"}


def test_gen_reqs_returns_list():
    _, _, edge_nodes, _ = _nsf()
    reqs = gen_reqs(edge_nodes, 10, np.random.default_rng(42))
    assert isinstance(reqs, list)


def test_gen_reqs_not_empty_under_high_lambda():
    _, _, edge_nodes, _ = _nsf()
    reqs = gen_reqs(edge_nodes, 20, np.random.default_rng(42))
    assert len(reqs) > 0


def test_gen_reqs_reproducible():
    _, _, edge_nodes, _ = _nsf()
    reqs1 = gen_reqs(edge_nodes, 10, np.random.default_rng(42))
    reqs2 = gen_reqs(edge_nodes, 10, np.random.default_rng(42))
    assert len(reqs1) == len(reqs2)
    for r1, r2 in zip(reqs1, reqs2):
        assert r1["s"] == r2["s"] and r1["q"] == r2["q"] and r1["K"] == r2["K"]


def test_each_request_has_all_six_fields():
    """Paper §V-B: a request specifies six parameters (s, q, K, cpu, bw, Dmax)."""
    _, _, edge_nodes, _ = _nsf()
    for r in gen_reqs(edge_nodes, 20, np.random.default_rng(7)):
        missing = REQUIRED_REQUEST_KEYS - set(r)
        assert not missing, f"missing keys: {missing}"


def test_source_and_destination_differ():
    """A request must terminate at an edge node distinct from its source."""
    _, _, edge_nodes, _ = _nsf()
    for r in gen_reqs(edge_nodes, 30, np.random.default_rng(0)):
        assert r["s"] != r["q"]


def test_chain_length_within_paper_range():
    """Paper §VII-C: K ∈ {2, 3, 4}."""
    _, _, edge_nodes, _ = _nsf()
    for r in gen_reqs(edge_nodes, 50, np.random.default_rng(1)):
        assert 2 <= r["K"] <= 4
        assert len(r["cpu"]) == r["K"]


def test_vnf_cpu_within_paper_range():
    """Paper §VII-C: per-VNF CPU demand is sampled from U[8, 20]."""
    _, _, edge_nodes, _ = _nsf()
    for r in gen_reqs(edge_nodes, 50, np.random.default_rng(2)):
        for c in r["cpu"]:
            assert 8 <= c <= 20


def test_bandwidth_and_dmax_within_paper_range():
    """Paper §VII-C: b_r ∈ [10, 20] Mbps, D_max ∈ [15, 30] ms."""
    _, _, edge_nodes, _ = _nsf()
    for r in gen_reqs(edge_nodes, 50, np.random.default_rng(3)):
        assert 10 <= r["bw"] <= 20
        assert 15 <= r["Dmax"] <= 30
