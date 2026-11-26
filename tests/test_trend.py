"""Unit tests for src/trend.py (paper section 2.2)."""

import numpy as np
import pandas as pd
import pytest

from src import config, trend


def test_normalization_factor_matches_definition_numerically():
    """sum_n w_T(n)^2 should equal 1 for the closed-form M_T (paper section 2.2)."""
    for t_days in (2, 8, 64, 256):
        m_t = trend.normalization_factor(t_days)
        rho = np.exp(-2.0 / t_days)
        n = np.arange(0, 20000)
        w = m_t * n * rho ** n
        assert np.sum(w ** 2) == pytest.approx(1.0, abs=1e-4)


def test_trend_strength_recursive_matches_direct_convolution():
    """The recursive two-filter form must equal the direct w_T(n) convolution."""
    rng = np.random.default_rng(0)
    r_hat = rng.standard_normal(500)
    t_days = 16
    rho = np.exp(-2.0 / t_days)
    m_t = trend.normalization_factor(t_days)
    n = np.arange(len(r_hat))
    w = m_t * n * rho ** n
    direct = np.array([np.sum(w[:t + 1] * r_hat[t::-1]) for t in range(len(r_hat))])
    recursive = trend.trend_strength(r_hat, t_days)
    np.testing.assert_allclose(recursive, direct, atol=1e-8)


def test_trend_strength_sign_follows_direction_of_trend():
    """A sustained upward move must yield a positive trend strength, and a
    sustained downward move a negative one (paper: phi measures trend
    direction and strength)."""
    up = np.full(300, 0.01)
    down = -up
    phi_up = trend.trend_strength(up, 32)
    phi_down = trend.trend_strength(down, 32)
    assert phi_up[-1] > 0
    assert phi_down[-1] < 0
    assert phi_up[-1] == pytest.approx(-phi_down[-1])


def test_all_trend_strengths_are_capped(synthetic_close):
    out = trend.all_trend_strengths(synthetic_close)
    for k in config.TREND_HORIZONS_K:
        col = out[f"phi_{k}"]
        assert col.max() <= config.TREND_CAP + 1e-9
        assert col.min() >= -config.TREND_CAP - 1e-9


def test_normalized_returns_have_unit_variance_and_zero_mean(synthetic_close):
    r_norm = trend.normalized_returns(synthetic_close)
    assert r_norm.mean() == pytest.approx(0.0, abs=1e-8)
    assert r_norm.std(ddof=1) == pytest.approx(1.0, abs=1e-8)


def test_normalized_returns_rejects_constant_price():
    constant = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=50))
    with pytest.raises(AssertionError):
        trend.normalized_returns(constant)


def test_aggregate_trend_basic_vs_enhanced_horizon_counts(synthetic_close):
    df = trend.all_trend_strengths(synthetic_close)
    agg_basic, agg2_basic = trend.aggregate_trend(df, config.AGG_HORIZONS_BASIC)
    agg_enh, agg2_enh = trend.aggregate_trend(df, config.AGG_HORIZONS_ENHANCED)
    assert len(config.AGG_HORIZONS_BASIC) == 8
    assert len(config.AGG_HORIZONS_ENHANCED) == 6
    assert agg_basic.shape == agg_enh.shape == (len(synthetic_close),)
    assert np.all(agg2_basic >= 0) and np.all(agg2_enh >= 0)
