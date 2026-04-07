"""Tests for SFC request generation."""
import numpy as np
import sys; sys.path.insert(0, '.')
from lcavo_sim import gen_reqs, _nsf

def test_gen_reqs_returns_list():
    G, _, edge_nodes, _ = _nsf()[:4]
    rng = np.random.default_rng(42)
    reqs = gen_reqs(edge_nodes, 10, rng)
    assert isinstance(reqs, list)

def test_gen_reqs_not_empty():
    G, _, edge_nodes, _ = _nsf()[:4]
    rng = np.random.default_rng(42)
    reqs = gen_reqs(edge_nodes, 20, rng)
    assert len(reqs) > 0

def test_gen_reqs_reproducible():
    _, _, edge_nodes, _ = _nsf()[:4]
    reqs1 = gen_reqs(edge_nodes, 10, np.random.default_rng(42))
    reqs2 = gen_reqs(edge_nodes, 10, np.random.default_rng(42))
    assert len(reqs1) == len(reqs2)
