"""Unit tests for exp/exp_4.py (Heston / Leverage-SV models with trend
factors), using only synthetic data (a few tickers, a few hundred rows
each). No network access and no ``load_universe`` call.
"""

import numpy as np
import pandas as pd
import pytest

from src import config, sv_models


def _synthetic_panel_by_ticker(n: int = 200, n_tickers: int = 3, seed: int = 0) -> dict:
    """{ticker: DataFrame[r, phi_agg, phi_agg2]}, the input format
    ``sv_models.fit_heston``/``fit_lsv``/``predict_sv`` expect."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_tickers):
        phi_agg = rng.standard_normal(n) * 0.3
        sigma = 0.8 + 0.2 * np.sin(np.linspace(0, 6, n))  # mild, bounded volatility clustering
        r = sigma * rng.standard_normal(n)
        out[f"SYN{i}"] = pd.DataFrame({"r": r, "phi_agg": phi_agg, "phi_agg2": phi_agg ** 2})
    return out


@pytest.fixture
def synthetic_panel_by_ticker() -> dict:
    return _synthetic_panel_by_ticker()


@pytest.mark.parametrize("with_trend", [False, True])
def test_fit_heston_runs_and_returns_bounded_finite_params(synthetic_panel_by_ticker, with_trend):
    fitted = sv_models.fit_heston(synthetic_panel_by_ticker, with_trend=with_trend, maxiter=10)
    expected_names = {"kappa", "theta", "xi"} | ({"lam", "zeta"} if with_trend else set())
    assert set(fitted["names"]) == expected_names
    assert fitted["kind"] == "heston"
    assert np.isfinite(fitted["neg_loglik"])
    assert fitted["n_obs"] == sum(len(g) for g in synthetic_panel_by_ticker.values())
    for name, value in fitted["params"].items():
        lo, hi = config.SV_PARAM_BOUNDS[name]
        assert np.isfinite(value)
        assert lo - 1e-6 <= value <= hi + 1e-6
    assert fitted["params"]["xi"] > 0.0


@pytest.mark.parametrize("with_trend", [False, True])
def test_fit_lsv_runs_and_returns_bounded_finite_params(synthetic_panel_by_ticker, with_trend):
    fitted = sv_models.fit_lsv(synthetic_panel_by_ticker, with_trend=with_trend, maxiter=10)
    expected_names = {"mu", "beta", "xi"} | ({"lam", "zeta"} if with_trend else set())
    assert set(fitted["names"]) == expected_names
    assert fitted["kind"] == "lsv"
    assert np.isfinite(fitted["neg_loglik"])
    assert fitted["n_obs"] == sum(len(g) for g in synthetic_panel_by_ticker.values())
    for name, value in fitted["params"].items():
        lo, hi = config.SV_PARAM_BOUNDS[name]
        assert np.isfinite(value)
        assert lo - 1e-6 <= value <= hi + 1e-6
    assert fitted["params"]["xi"] > 0.0


def test_predict_sv_heston_returns_positive_variance_of_right_shape(synthetic_panel_by_ticker):
    fitted = sv_models.fit_heston(synthetic_panel_by_ticker, with_trend=True, maxiter=10)
    preds = sv_models.predict_sv(fitted, synthetic_panel_by_ticker)
    assert set(preds.keys()) == set(synthetic_panel_by_ticker.keys())
    for ticker, sigma2_hat in preds.items():
        assert isinstance(sigma2_hat, np.ndarray)
        assert len(sigma2_hat) == len(synthetic_panel_by_ticker[ticker])
        assert np.all(np.isfinite(sigma2_hat))
        assert np.all(sigma2_hat > 0.0)


def test_predict_sv_lsv_returns_positive_variance_of_right_shape(synthetic_panel_by_ticker):
    fitted = sv_models.fit_lsv(synthetic_panel_by_ticker, with_trend=True, maxiter=10)
    preds = sv_models.predict_sv(fitted, synthetic_panel_by_ticker)
    assert set(preds.keys()) == set(synthetic_panel_by_ticker.keys())
    for ticker, sigma2_hat in preds.items():
        assert isinstance(sigma2_hat, np.ndarray)
        assert len(sigma2_hat) == len(synthetic_panel_by_ticker[ticker])
        assert np.all(np.isfinite(sigma2_hat))
        assert np.all(sigma2_hat > 0.0)


def test_leverage_correlation_is_finite_and_in_unit_interval(synthetic_panel_by_ticker):
    g = synthetic_panel_by_ticker["SYN0"]
    fitted = sv_models.fit_heston(synthetic_panel_by_ticker, with_trend=True, maxiter=10)
    sigma2_hat = sv_models.predict_sv(fitted, synthetic_panel_by_ticker)["SYN0"]
    rho = sv_models.leverage_correlation(g["r"].to_numpy(), sigma2_hat)
    assert isinstance(rho, float)
    assert np.isfinite(rho)
    assert -1.0 - 1e-9 <= rho <= 1.0 + 1e-9


def test_leverage_correlation_nan_for_degenerate_input():
    r = np.ones(10)
    sigma2_hat = np.ones(10)
    rho = sv_models.leverage_correlation(r, sigma2_hat)
    assert np.isnan(rho)
