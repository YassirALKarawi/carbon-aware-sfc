"""Tests for the 24-hour regional carbon-intensity profiles."""
import sys; sys.path.insert(0, '.')

import numpy as np

from lcavo_sim import carbon_prof


def test_carbon_profile_has_four_regions():
    """Paper §VII-B defines four grid regions A–D."""
    cp = carbon_prof()
    assert set(cp.keys()) == {"A", "B", "C", "D"}


def test_carbon_profile_24_hours():
    cp = carbon_prof()
    for region, profile in cp.items():
        assert len(profile) == 24, f"Region {region} has {len(profile)} hours"


def test_carbon_values_positive():
    cp = carbon_prof()
    for region, profile in cp.items():
        assert all(v > 0 for v in profile), f"Region {region} has non-positive values"


def test_regions_differ():
    """Carbon profiles must not collapse onto a single curve."""
    cp = carbon_prof()
    assert not np.allclose(cp["A"], cp["D"])


def test_region_a_cleaner_than_region_d():
    """Region A is the cleanest grid in the paper; D is the dirtiest."""
    cp = carbon_prof()
    assert np.mean(cp["A"]) < np.mean(cp["D"])


def test_sigma_zero_collapses_regions():
    """sigma=0 must collapse all regions to the global mean (variance reduction)."""
    cp0 = carbon_prof(sigma=0.0)
    means = [np.mean(cp0[r]) for r in "ABCD"]
    assert max(means) - min(means) < 1e-6


def test_larger_sigma_increases_spread():
    """Spread between cleanest and dirtiest region must grow with sigma."""
    cp_small = carbon_prof(sigma=0.5)
    cp_large = carbon_prof(sigma=2.0)
    spread_small = np.mean(cp_small["D"]) - np.mean(cp_small["A"])
    spread_large = np.mean(cp_large["D"]) - np.mean(cp_large["A"])
    assert spread_large > spread_small


def test_carbon_values_within_documented_range():
    """All intensities must fit the [10, 900] gCO₂eq/kWh clip from the model."""
    cp = carbon_prof()
    for region, profile in cp.items():
        assert profile.min() >= 10
        assert profile.max() <= 900
