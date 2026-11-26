"""Unit tests for exp/exp_2.py (GARCH(1,1) / GJR-GARCH with trend factors),
using only synthetic data (a handful of tickers, a few hundred rows each).
No network access and no ``load_universe`` call.
"""

import numpy as np
import pandas as pd
import pytest

from src import evaluation, garch_models


def _synthetic_panel(n: int = 300, n_tickers: int = 4, seed: int = 0) -> pd.DataFrame:
    """A tiny pooled panel with the columns ``fit_pooled_garch`` /
    ``predict_pooled_garch`` / ``cross_validate`` need: ticker, date, r,
    phi_agg_enh, phi_agg2_enh, r2_next."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_tickers):
        r = rng.standard_normal(n) * 0.7
        phi_agg = rng.standard_normal(n) * 0.25
        dates = pd.bdate_range("2018-01-01", periods=n)
        frame = pd.DataFrame({
            "ticker": f"SYN{i}", "date": dates, "r": r,
            "phi_agg_enh": phi_agg, "phi_agg2_enh": phi_agg ** 2,
        })
        frame["r2_next"] = frame["r"].shift(-1) ** 2
        frames.append(frame.iloc[:-1])
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    return _synthetic_panel()


@pytest.mark.parametrize("with_trend,asymmetric", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_fit_pooled_garch_runs_without_raising(synthetic_panel, with_trend, asymmetric):
    fitted = garch_models.fit_pooled_garch(synthetic_panel, with_trend=with_trend,
                                            asymmetric=asymmetric, maxiter=20)
    expected_names = {"omega", "alpha", "beta"}
    if asymmetric:
        expected_names.add("gamma")
    if with_trend:
        expected_names.update({"lam", "zeta"})
    assert set(fitted["names"]) == expected_names
    assert np.isfinite(fitted["qlike_per_obs"])
    assert fitted["n_obs"] == len(synthetic_panel)


def test_predict_pooled_garch_returns_positive_array_of_right_length(synthetic_panel):
    fitted = garch_models.fit_pooled_garch(synthetic_panel, with_trend=True, asymmetric=True, maxiter=20)
    pred = garch_models.predict_pooled_garch(fitted, synthetic_panel)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(synthetic_panel)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0.0)


def test_cross_validate_returns_finite_r2_adj(synthetic_panel):
    def fit_fn(tr: pd.DataFrame) -> dict:
        return garch_models.fit_pooled_garch(tr, with_trend=True, asymmetric=False, maxiter=10)

    cv = evaluation.cross_validate(synthetic_panel, fit_fn, garch_models.predict_pooled_garch,
                                    n_params=5, n_folds=3)
    assert isinstance(cv["r2_adj"], float)
    assert np.isfinite(cv["r2_adj"])
    assert cv["n_obs"] == len(synthetic_panel)


def test_bootstrap_errors_block_bootstrap_on_synthetic_panel(synthetic_panel):
    order = ["omega", "alpha", "beta", "lam", "zeta"]

    def stat_fn(p: pd.DataFrame) -> np.ndarray:
        fitted = garch_models.fit_pooled_garch(p, with_trend=True, asymmetric=False, maxiter=10)
        return fitted["params"].reindex(order).to_numpy()

    boot = evaluation.bootstrap_errors(synthetic_panel, stat_fn, n_samples=3, block=True,
                                        n_blocks=2, block_length=50)
    assert boot["estimates"].shape == (3, len(order))
    assert np.all(np.isfinite(boot["error"]))
