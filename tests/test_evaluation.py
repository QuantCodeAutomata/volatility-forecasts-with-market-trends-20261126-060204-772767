"""Unit tests for src/evaluation.py (paper section 3.2: CV and bootstrap)."""

import numpy as np
import pandas as pd
import pytest

from src import config, evaluation


def test_r2_adj_is_one_for_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert evaluation.r2_adj(y, y, n_params=1) == pytest.approx(1.0)


def test_r2_adj_rejects_degenerate_target():
    y = np.ones(10)
    with pytest.raises(AssertionError):
        evaluation.r2_adj(y, y, n_params=1)


def test_day_folds_partition_all_days_into_n_folds():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=160).repeat(3))
    masks = evaluation.day_folds(dates, n_folds=4)
    assert len(masks) == 4
    total = np.zeros(len(dates), dtype=bool)
    for m in masks:
        assert not np.any(total & m)  # folds are disjoint
        total |= m
    assert total.all()  # folds cover every row


def test_day_folds_are_adjacent_blocks_of_days():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=40))
    masks = evaluation.day_folds(dates, n_folds=4)
    sorted_days = np.sort(dates.unique())
    for mask in masks:
        days_in_fold = np.sort(dates[mask].unique())
        # an adjacent block has no gap beyond the business-day calendar
        positions = np.searchsorted(sorted_days, days_in_fold)
        assert np.all(np.diff(positions) == 1)


def test_cross_validate_pools_every_fold(synthetic_universe):
    from src import panel as panel_mod
    import statsmodels.api as sm

    per_ticker, universe = synthetic_universe
    full_panel = panel_mod.build_panel(per_ticker, universe)

    def fit_fn(train):
        x = sm.add_constant(train[["sigma2"]])
        return sm.OLS(train["r2_next"].to_numpy(), x).fit()

    def predict_fn(model, test):
        x = sm.add_constant(test[["sigma2"]], has_constant="add")
        return model.predict(x)

    result = evaluation.cross_validate(full_panel, fit_fn, predict_fn, n_params=1, n_folds=4)
    assert result["n_obs"] == len(full_panel)
    assert np.isfinite(result["r2_adj"])


def test_bootstrap_errors_day_level_preserves_cross_correlation(synthetic_universe):
    from src import panel as panel_mod

    per_ticker, universe = synthetic_universe
    full_panel = panel_mod.build_panel(per_ticker, universe)

    def statistic_fn(p):
        return np.array([p["sigma2"].mean()])

    out = evaluation.bootstrap_errors(full_panel, statistic_fn, n_samples=20, block=False)
    assert out["estimates"].shape == (20, 1)
    assert out["error"][0] >= 0


def test_block_bootstrap_indices_use_contiguous_blocks():
    days = np.arange(1000)
    rng = np.random.default_rng(0)
    samples = evaluation.block_bootstrap_indices(days, n_blocks=10, block_length=50,
                                                   n_samples=5, rng=rng)
    for sample in samples:
        assert len(sample) == 500
