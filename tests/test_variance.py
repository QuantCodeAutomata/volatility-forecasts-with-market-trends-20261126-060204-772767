"""Unit tests for src/variance.py (paper equation 2, EWMA variance)."""

import numpy as np
import pytest

from src import variance


def test_ewma_recursion_matches_definition():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(200)
    alpha = 0.94
    sigma2 = variance.ewma_variance(r, alpha=alpha)
    expected = np.empty(200)
    prev = 0.0
    for t in range(200):
        expected[t] = alpha * prev + (1 - alpha) * r[t] ** 2
        prev = expected[t]
    np.testing.assert_allclose(sigma2, expected, atol=1e-10)


def test_ewma_variance_is_nonnegative():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(500) * 3
    sigma2 = variance.ewma_variance(r, alpha=0.94)
    assert np.all(sigma2 >= 0)


def test_ewma_variance_reacts_to_shock_magnitude():
    """A larger shock at t must raise sigma2 at t relative to a smaller one,
    all else equal (paper eq. 2: sigma_t^2 increasing in r_t^2)."""
    r_small = np.zeros(10)
    r_small[5] = 0.5
    r_large = np.zeros(10)
    r_large[5] = 3.0
    s_small = variance.ewma_variance(r_small, alpha=0.9)
    s_large = variance.ewma_variance(r_large, alpha=0.9)
    assert s_large[5] > s_small[5]


def test_ewma_variance_handles_nan_input_without_crashing():
    r = np.array([0.1, np.nan, -0.2, 0.3])
    sigma2 = variance.ewma_variance(r, alpha=0.94)
    assert np.all(np.isfinite(sigma2))
