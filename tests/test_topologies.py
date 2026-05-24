"""Tests for network topologies."""
import sys; sys.path.insert(0, '.')
import networkx as nx
from lcavo_sim import _nsf, _gea

def test_nsfnet_has_14_nodes():
    G = _nsf()[0]
    assert G.number_of_nodes() == 14

def test_geant_has_23_nodes():
    G = _gea()[0]
    assert G.number_of_nodes() == 23

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
