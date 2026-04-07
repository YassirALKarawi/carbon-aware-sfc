"""Tests for carbon intensity profiles."""
import numpy as np
import sys; sys.path.insert(0, '.')
from lcavo_sim import carbon_prof

def test_carbon_profile_has_regions():
    cp = carbon_prof()
    assert isinstance(cp, dict)
    assert len(cp) >= 2

def test_carbon_profile_24_hours():
    cp = carbon_prof()
    for region, profile in cp.items():
        assert len(profile) == 24, f"Region {region} has {len(profile)} hours"

def test_carbon_values_positive():
    cp = carbon_prof()
    for region, profile in cp.items():
        assert all(v > 0 for v in profile), f"Region {region} has non-positive values"

def test_regions_differ():
    cp = carbon_prof()
    regions = list(cp.keys())
    if len(regions) >= 2:
        assert not np.allclose(cp[regions[0]], cp[regions[1]])
