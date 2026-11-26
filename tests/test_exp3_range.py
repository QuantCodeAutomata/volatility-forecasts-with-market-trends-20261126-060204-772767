"""Unit tests for exp/exp_3.py (range-based GARCH(1,1) and HAR with trend
factors), using only synthetic data (a handful of tickers, a few hundred
rows each). No network access and no ``load_universe`` call.
"""

import numpy as np
import pandas as pd
import pytest

from src import evaluation, range_models


def _synthetic_range_panel(n: int = 300, n_tickers: int = 4, seed: int = 0) -> pd.DataFrame:
    """A tiny pooled panel with the columns ``fit_range_garch`` /
    ``predict_range_garch`` / ``fit_har`` / ``predict_har`` /
    ``cross_validate`` need: ticker, date, range2, range2_next, range2_w,
    range2_m, phi_agg, phi_agg2, phi_agg_enh, phi_agg2_enh.

    ``range_models.fit_range_garch`` delegates to
    ``garch_models.fit_pooled_garch``, which (hardcoded there, same as the
    real return-based panel) reads the *_enh aggregate columns when
    ``with_trend=True``, while ``range_models.fit_har``'s with-trend design
    uses the plain (non-enhanced) aggregate columns (paper's simplest-model
    (6) factors, per ``fit_har``'s docstring) -- so both pairs of columns
    are populated here, independently seeded as they would be in the real
    panel (basic vs. enhanced aggregate horizons differ).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_tickers):
        dates = pd.bdate_range("2018-01-01", periods=n)
        raw_range = rng.chisquare(df=2, size=n) * 0.5
        range2 = raw_range / raw_range.mean()
        phi_agg = rng.standard_normal(n) * 0.25
        phi_agg_enh = rng.standard_normal(n) * 0.25
        frame = pd.DataFrame({
            "ticker": f"SYN{i}", "date": dates, "range2": range2,
            "phi_agg": phi_agg, "phi_agg2": phi_agg ** 2,
            "phi_agg_enh": phi_agg_enh, "phi_agg2_enh": phi_agg_enh ** 2,
        })
        frame["range2_w"] = frame["range2"].rolling(5, min_periods=5).mean()
        frame["range2_m"] = frame["range2"].rolling(22, min_periods=22).mean()
        frame["range2_next"] = frame["range2"].shift(-1)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["range2_w", "range2_m", "range2_next"]).reset_index(drop=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture
def synthetic_range_panel() -> pd.DataFrame:
    return _synthetic_range_panel()


@pytest.mark.parametrize("with_trend", [False, True])
def test_fit_range_garch_runs_without_raising(synthetic_range_panel, with_trend):
    fitted = range_models.fit_range_garch(synthetic_range_panel, with_trend=with_trend, maxiter=20)
    expected_names = {"omega", "alpha", "beta"}
    if with_trend:
        expected_names.update({"lam", "zeta"})
    assert set(fitted["names"]) == expected_names
    assert np.isfinite(fitted["qlike_per_obs"])
    assert fitted["n_obs"] == len(synthetic_range_panel)


def test_predict_range_garch_returns_positive_array_of_right_length(synthetic_range_panel):
    fitted = range_models.fit_range_garch(synthetic_range_panel, with_trend=True, maxiter=20)
    pred = range_models.predict_range_garch(fitted, synthetic_range_panel)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(synthetic_range_panel)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0.0)


@pytest.mark.parametrize("with_trend", [False, True])
def test_fit_har_and_predict_har_return_finite_predictions(synthetic_range_panel, with_trend):
    fitted = range_models.fit_har(synthetic_range_panel, with_trend=with_trend)
    pred = range_models.predict_har(fitted, synthetic_range_panel)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(synthetic_range_panel)
    assert np.all(np.isfinite(pred))
    if with_trend:
        assert "phi_agg" in fitted["design"].columns
    else:
        assert "phi_agg" not in fitted["design"].columns


def test_cross_validate_range_garch_returns_finite_r2_adj(synthetic_range_panel):
    def fit_fn(tr: pd.DataFrame) -> dict:
        return range_models.fit_range_garch(tr, with_trend=True, maxiter=10)

    cv = evaluation.cross_validate(synthetic_range_panel, fit_fn, range_models.predict_range_garch,
                                    n_params=5, n_folds=3, y_col="range2_next")
    assert isinstance(cv["r2_adj"], float)
    assert np.isfinite(cv["r2_adj"])
    assert cv["n_obs"] == len(synthetic_range_panel)


def test_cross_validate_har_returns_finite_r2_adj(synthetic_range_panel):
    def fit_fn(tr: pd.DataFrame) -> dict:
        return range_models.fit_har(tr, with_trend=True)

    cv = evaluation.cross_validate(synthetic_range_panel, fit_fn, range_models.predict_har,
                                    n_params=5, n_folds=3, y_col="range2_next")
    assert isinstance(cv["r2_adj"], float)
    assert np.isfinite(cv["r2_adj"])
    assert cv["n_obs"] == len(synthetic_range_panel)
