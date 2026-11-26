"""Unit tests for src/regression_models.py (paper eqs. 5, 6, 8)."""

import numpy as np
import pandas as pd
import pytest

from src import config, regression_models


def _toy_panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sigma2 = np.abs(rng.standard_normal(n)) + 0.5
    phi_agg = rng.standard_normal(n) * 0.5
    phi_agg2 = phi_agg ** 2
    true_a, true_d, true_e, true_f = 0.05, 0.7, -0.2, 0.3
    mean = true_a + true_d * sigma2 + true_e * phi_agg + true_f * phi_agg2
    r2_next = np.abs(mean + rng.standard_normal(n) * 0.1)
    return pd.DataFrame({
        "sigma2": sigma2, "phi_agg": phi_agg, "phi_agg2": phi_agg2,
        "phi_agg_enh": phi_agg, "phi_agg2_enh": phi_agg2, "r2_next": r2_next,
    }), (true_a, true_d, true_e, true_f)


def test_fit_ols_recovers_known_coefficients():
    panel, (a, d, e, f) = _toy_panel()
    fitted = regression_models.fit_ols(panel, variant="full")
    params = fitted["params"]
    assert params["sigma2"] == pytest.approx(d, abs=0.05)
    assert params["phi_agg"] == pytest.approx(e, abs=0.05)
    assert params["phi_agg2"] == pytest.approx(f, abs=0.05)


def test_fit_ols_variants_change_design_columns():
    panel, _ = _toy_panel()
    full = regression_models.fit_ols(panel, variant="full")
    trend_only = regression_models.fit_ols(panel, variant="trend_only")
    variance_only = regression_models.fit_ols(panel, variant="variance_only")
    assert "sigma2" not in trend_only["design"].columns
    assert "phi_agg" not in variance_only["design"].columns
    assert set(full["design"].columns) == {"sigma2", "phi_agg", "phi_agg2"}


def test_fit_ols_rejects_unknown_variant():
    panel, _ = _toy_panel()
    with pytest.raises(ValueError):
        regression_models.fit_ols(panel, variant="bogus")


def test_qlike_is_minimized_at_true_variance():
    rng = np.random.default_rng(3)
    sigma2_true = 1.5
    r2 = (rng.standard_normal(5000) * np.sqrt(sigma2_true)) ** 2
    candidates = np.linspace(0.5, 3.0, 50)
    losses = [regression_models.qlike(np.full_like(r2, c), r2) for c in candidates]
    best = candidates[int(np.argmin(losses))]
    assert best == pytest.approx(sigma2_true, abs=0.15)


def test_fit_qmle_converges_and_improves_qlike_over_ols():
    panel, _ = _toy_panel(n=600, seed=5)
    qmle = regression_models.fit_qmle(panel, variant="full")
    ols = regression_models.fit_ols(panel, variant="full")
    ols_sigma2_hat = np.clip(ols["fitted"], 1e-6, None)
    ols_qlike = regression_models.qlike(ols_sigma2_hat, panel["r2_next"].to_numpy()) / len(panel)
    # QMLE directly minimizes QLIKE, so it should not be much worse than OLS's QLIKE.
    assert qmle["qlike_per_obs"] <= ols_qlike + 0.5


def test_fit_enhanced_ols_uses_variance_interaction_design():
    panel, _ = _toy_panel()
    fitted = regression_models.fit_enhanced_ols(panel)
    assert set(fitted["design"].columns) == {"sigma2", "phi_sigma", "phi_agg2"}


def test_longer_horizon_target_sums_future_squared_returns():
    panel = pd.DataFrame({
        "ticker": ["A"] * 6,
        "r": [0.1, 0.2, -0.1, 0.05, -0.2, 0.3],
    })
    target = regression_models.longer_horizon_target(panel, horizon_days=2)
    r2 = panel["r"].to_numpy() ** 2
    expected0 = r2[1] + r2[2]
    assert target.iloc[0] == pytest.approx(expected0)
    assert pd.isna(target.iloc[-1]) and pd.isna(target.iloc[-2])


def test_predict_matches_manual_linear_combination():
    panel, _ = _toy_panel()
    fitted = regression_models.fit_ols(panel, variant="full")
    pred = regression_models.predict(fitted["params"], fitted["design"])
    np.testing.assert_allclose(pred, fitted["fitted"], atol=1e-8)
