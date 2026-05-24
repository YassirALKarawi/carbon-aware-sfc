"""Tests for placement algorithms and the canonical methods list."""
import sys; sys.path.insert(0, '.')

import pytest

from lcavo_sim import DEFAULT_METHODS, METHOD_ORDER, sim


PAPER_METHODS = {
    "MILP-OPT",
    "L-CAVO",
    "QL-CAVO",
    "Energy-aware",
    "Latency-aware",
    "Carbon-greedy",
    "Random",
}


def test_method_order_matches_paper():
    """The paper compares exactly seven methods; METHOD_ORDER must list them all."""
    assert set(METHOD_ORDER) == PAPER_METHODS


def test_default_methods_excludes_only_milp():
    """DEFAULT_METHODS is the online subset (everything except offline MILP)."""
    assert "MILP-OPT" not in DEFAULT_METHODS
    assert set(DEFAULT_METHODS) == PAPER_METHODS - {"MILP-OPT"}


@pytest.mark.parametrize(
    "method",
    ["L-CAVO", "QL-CAVO", "Energy-aware", "Latency-aware", "Carbon-greedy", "Random"],
)
def test_every_online_method_completes(method):
    """Each online method must finish a 1-seed Low-load NSFNET run."""
    result = sim("NSFNET", "Low", method, ns=1)
    assert result is not None and len(result) == 1
