"""Unit tests for src/range_models.py (paper section on range-based models)."""

import numpy as np
import pandas as pd
import pytest

from src import range_models


def test_normalized_range_has_unit_mean():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(2000) * 0.01))
    high = close * (1 + np.abs(rng.standard_normal(2000)) * 0.01)
    low = close * (1 - np.abs(rng.standard_normal(2000)) * 0.01)
    range2 = range_models.normalized_range(high, low)
    assert range2.mean() == pytest.approx(1.0, abs=1e-8)
    assert np.all(range2 >= 0)


def test_squared_range_rejects_low_above_high():
    high = np.array([10.0, 11.0])
    low = np.array([10.5, 9.0])
    with pytest.raises(AssertionError):
        range_models.squared_range(high, low)


def test_har_lag_features_weekly_monthly_match_rolling_means():
    """paper eq. (9): beta_w uses the trailing 5-day mean, beta_m the
    trailing 22-day mean of the squared range, ending at t (no look-ahead)."""
    n = 60
    dates = pd.bdate_range("2020-01-01", periods=n)
    high = pd.Series(np.linspace(101, 101 + n, n))
    low = pd.Series(np.linspace(99, 99 + n, n))
    panel = pd.DataFrame({"ticker": "A", "date": dates, "high": high, "low": low})
    panel["range2"] = range_models.normalized_range(panel["high"].to_numpy(), panel["low"].to_numpy())
    out = range_models.har_lag_features(panel)
    manual_w = panel["range2"].rolling(5).mean()
    manual_m = panel["range2"].rolling(22).mean()
    merged = out.merge(panel[["date", "range2"]].assign(w=manual_w, m=manual_m), on="date")
    np.testing.assert_allclose(merged["range2_w"], merged["w"], atol=1e-8)
    np.testing.assert_allclose(merged["range2_m"], merged["m"], atol=1e-8)
    # next-day target is just the next row's range2, never today's own value
    np.testing.assert_allclose(
        out["range2_next"].to_numpy()[:-1], out["range2"].to_numpy()[1:], atol=1e-8
    )


def test_fit_har_and_fit_range_garch_run_on_a_small_panel(synthetic_universe):
    from src import panel as panel_mod

    per_ticker, universe = synthetic_universe
    full_panel = panel_mod.build_panel(per_ticker, universe)
    full_panel = panel_mod.add_range_columns(full_panel)

    har_no_trend = range_models.fit_har(full_panel, with_trend=False)
    har_trend = range_models.fit_har(full_panel, with_trend=True)
    assert "phi_agg" not in har_no_trend["design"].columns
    assert "phi_agg" in har_trend["design"].columns
    pred = range_models.predict_har(har_trend, full_panel)
    assert len(pred) == len(full_panel)

    fitted = range_models.fit_range_garch(full_panel, with_trend=False, maxiter=15)
    pred_garch = range_models.predict_range_garch(fitted, full_panel)
    assert np.all(np.isfinite(pred_garch))
    assert np.all(pred_garch > 0)
