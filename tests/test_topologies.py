"""Tests for network topologies (NSFNET, GÉANT) — must match the paper."""
import sys; sys.path.insert(0, '.')
import networkx as nx
from lcavo_sim import _nsf, _gea


def test_nsfnet_has_14_nodes():
    G = _nsf()[0]
    assert G.number_of_nodes() == 14


def test_nsfnet_has_21_links():
    """Paper §VII-A: NSFNET has 21 undirected links."""
    G = _nsf()[0]
    assert G.number_of_edges() == 21


def test_geant_has_23_nodes():
    G = _gea()[0]
    assert G.number_of_nodes() == 23


def test_geant_has_37_links():
    """Paper §VII-A: GÉANT has 37 undirected links."""
    G = _gea()[0]
    assert G.number_of_edges() == 37


def test_nsfnet_is_connected():
    G = _nsf()[0]
    assert nx.is_connected(G)


def test_geant_is_connected():
    G = _gea()[0]
    assert nx.is_connected(G)


def test_nsfnet_returns_tuple():
    result = _nsf()
    assert isinstance(result, tuple)
    assert len(result) >= 3


def test_nsfnet_has_four_regions():
    """Paper splits NSFNET into 4 grid regions A–D."""
    _, regions, _, _ = _nsf()
    assert set(regions.values()) == {"A", "B", "C", "D"}


def test_geant_has_four_regions():
    _, regions, _, _ = _gea()
    assert set(regions.values()) == {"A", "B", "C", "D"}


def test_edge_nodes_are_in_topology():
    """Every declared edge node must actually exist in the graph."""
    for build in (_nsf, _gea):
        G, _, edge_nodes, _ = build()
        for n in edge_nodes:
            assert n in G.nodes()
