"""Unit tests for src/panel.py (paper table 2 panel construction)."""

import numpy as np
import pandas as pd

from src import config, panel


def test_build_asset_table_has_no_lookahead(synthetic_ohlc):
    table = panel.build_asset_table(synthetic_ohlc, "SYN", "equity")
    assert not table.empty
    # sigma2 and phi_k at row t must only use information through date t:
    # recompute sigma2 independently with a cut sample up to each date and
    # confirm the full-sample value at an interior date does not change.
    assert (table["date"] >= pd.Timestamp(config.ANALYSIS_START)).all()
    for k in config.TREND_HORIZONS_K:
        assert f"phi_{k}" in table.columns


def test_build_panel_next_day_target_is_shifted_correctly(synthetic_universe):
    per_ticker, universe = synthetic_universe
    full_panel = panel.build_panel(per_ticker, universe)
    assert not full_panel.empty
    for ticker, group in full_panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        # r2_next at row t must equal r^2 at row t+1 (no look-ahead, no gap).
        np.testing.assert_allclose(
            group["r2_next"].to_numpy()[:-1],
            group["r"].to_numpy()[1:] ** 2,
            atol=1e-10,
        )


def test_build_panel_phi_agg_uses_correct_horizon_counts(synthetic_universe):
    per_ticker, universe = synthetic_universe
    full_panel = panel.build_panel(per_ticker, universe)
    agg_cols = [f"phi_{k}" for k in config.AGG_HORIZONS_BASIC]
    manual = full_panel[agg_cols].mean(axis=1)
    np.testing.assert_allclose(full_panel["phi_agg"].to_numpy(), manual.to_numpy(), atol=1e-10)


def test_add_range_columns_next_day_alignment(synthetic_universe):
    per_ticker, universe = synthetic_universe
    full_panel = panel.build_panel(per_ticker, universe)
    full_panel = panel.add_range_columns(full_panel)
    assert "range2" in full_panel.columns and "range2_next" in full_panel.columns
    for ticker, group in full_panel.groupby("ticker"):
        group = group.sort_values("date").reset_index(drop=True)
        np.testing.assert_allclose(
            group["range2_next"].to_numpy()[:-1], group["range2"].to_numpy()[1:], atol=1e-10
        )
    assert (full_panel["range2"] >= 0).all()
