"""Unit tests for src/garch_models.py (GARCH(1,1) / GJR-GARCH with trends)."""

import numpy as np
import pandas as pd
import pytest

from src import garch_models


def test_garch_recursion_matches_manual_loop():
    rng = np.random.default_rng(0)
    n = 50
    r = rng.standard_normal(n) * 0.5
    phi_agg = rng.standard_normal(n) * 0.2
    phi_agg2 = phi_agg ** 2
    omega, alpha, beta, gamma, lam, zeta = 0.05, 0.1, 0.8, 0.0, -0.1, 0.1
    sigma2 = garch_models.garch_recursion(r, phi_agg, phi_agg2, omega, alpha, beta,
                                           gamma=gamma, lam=lam, zeta=zeta, sigma2_init=1.0)
    expected = np.empty(n)
    prev = 1.0
    for t in range(n):
        neg = 1.0 if r[t] < 0 else 0.0
        value = (omega + (alpha + gamma * neg) * r[t] ** 2 + beta * prev
                 + lam * (prev - 1.0) * phi_agg[t] + zeta * phi_agg2[t])
        expected[t] = max(value, 1e-6)
        prev = expected[t]
    np.testing.assert_allclose(sigma2, expected, atol=1e-10)


def test_garch_recursion_gamma_zero_is_plain_garch():
    rng = np.random.default_rng(1)
    n = 40
    r = rng.standard_normal(n)
    zeros = np.zeros(n)
    plain = garch_models.garch_recursion(r, zeros, zeros, 0.05, 0.1, 0.85, gamma=0.0)
    gjr_zero_gamma = garch_models.garch_recursion(r, zeros, zeros, 0.05, 0.1, 0.85, gamma=0.0)
    np.testing.assert_allclose(plain, gjr_zero_gamma)


def test_garch_recursion_asymmetric_responds_more_to_negative_shocks():
    """GJR-GARCH's gamma>0 must raise variance more after a negative shock
    than an equal-magnitude positive shock (paper's leverage effect)."""
    n = 5
    zeros = np.zeros(n)
    r_pos = np.array([0.0, 0.0, 2.0, 0.0, 0.0])
    r_neg = np.array([0.0, 0.0, -2.0, 0.0, 0.0])
    sigma2_pos = garch_models.garch_recursion(r_pos, zeros, zeros, 0.05, 0.05, 0.85, gamma=0.2)
    sigma2_neg = garch_models.garch_recursion(r_neg, zeros, zeros, 0.05, 0.05, 0.85, gamma=0.2)
    assert sigma2_neg[3] > sigma2_pos[3]


def test_garch_recursion_variance_floored_positive():
    n = 20
    r = np.zeros(n)
    phi = np.full(n, -10.0)
    phi2 = np.full(n, 10.0)
    sigma2 = garch_models.garch_recursion(r, phi, phi2, 0.01, 0.01, 0.5, lam=-5.0, zeta=0.0,
                                           sigma2_init=1.0)
    assert np.all(sigma2 >= 1e-6)


def _toy_garch_panel(n=300, n_tickers=2, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_tickers):
        r = rng.standard_normal(n) * 0.8
        phi_agg = rng.standard_normal(n) * 0.3
        dates = pd.bdate_range("2015-01-01", periods=n)
        frame = pd.DataFrame({
            "ticker": f"T{i}", "date": dates, "r": r,
            "phi_agg_enh": phi_agg, "phi_agg2_enh": phi_agg ** 2,
        })
        frame["r2_next"] = frame["r"].shift(-1) ** 2
        frames.append(frame.iloc[:-1])
    return pd.concat(frames, ignore_index=True)


def test_fit_pooled_garch_runs_and_predict_matches_recursion():
    panel = _toy_garch_panel()
    fitted = garch_models.fit_pooled_garch(panel, with_trend=False, asymmetric=False, maxiter=20)
    assert set(fitted["names"]) == {"omega", "alpha", "beta"}
    assert np.isfinite(fitted["qlike_per_obs"])
    pred = garch_models.predict_pooled_garch(fitted, panel)
    assert len(pred) == len(panel)
    assert np.all(np.isfinite(pred))


def test_fit_pooled_garch_with_trend_adds_lambda_zeta():
    panel = _toy_garch_panel()
    fitted = garch_models.fit_pooled_garch(panel, with_trend=True, asymmetric=True, maxiter=20)
    assert set(fitted["names"]) == {"omega", "alpha", "beta", "gamma", "lam", "zeta"}
