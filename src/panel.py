"""Build the (day, market) data panel used by every model (paper table 2).

One row per (day, market) with the next-day squared return (prediction
target), the EWMA variance sigma_t^2, and the 8 trend strengths; mirrors the
layout described in paper section 2.2 / table 2.
"""

import numpy as np
import pandas as pd

from src import config, trend, variance


def build_asset_table(ohlc: pd.DataFrame, ticker: str, asset_class: str,
                       alpha: float = config.EWMA_ALPHA) -> pd.DataFrame:
    """One asset's table: date, ticker, asset_class, r (next-day return used
    as the realized-variance target), sigma2 (today's EWMA variance),
    phi_1..phi_8 (today's trend strengths), high/low (for range models).

    Warmup rows before config.ANALYSIS_START (paper: "earlier data used only
    for computing initial trend strengths and variances") are dropped.
    """
    close = ohlc["close"].dropna()
    if len(close) < 50:
        return pd.DataFrame()
    trends = trend.all_trend_strengths(close)
    sigma2 = variance.ewma_variance(trends["r_norm"].to_numpy(), alpha=alpha)

    table = pd.DataFrame(index=close.index)
    table["r"] = trends["r_norm"].to_numpy()          # today's normalized log-return
    table["sigma2"] = sigma2                           # today's EWMA variance (uses r up to today)
    for k in config.TREND_HORIZONS_K:
        table[f"phi_{k}"] = trends[f"phi_{k}"].to_numpy()
    table["ticker"] = ticker
    table["asset_class"] = asset_class
    table = table.join(ohlc[["high", "low"]], how="left")

    # Paper section 2.1: data before ANALYSIS_START is used only to warm up
    # the recursive trend/variance filters; the reported sample starts there.
    table = table[table.index >= pd.Timestamp(config.ANALYSIS_START)]
    table = table.dropna(subset=["r", "sigma2"])
    return table.reset_index().rename(columns={"index": "date"})


def build_panel(per_ticker: dict, universe: dict,
                 alpha: float = config.EWMA_ALPHA) -> pd.DataFrame:
    """Stack every asset's table (paper table 2) and add the next-day
    squared return r_{t+1}^2, the prediction target of equations (6)/(8).
    """
    frames = []
    for ticker, ohlc in per_ticker.items():
        asset_class = universe[ticker]
        table = build_asset_table(ohlc, ticker, asset_class, alpha=alpha)
        if table.empty:
            continue
        table = table.sort_values("date").reset_index(drop=True)
        table["r_next"] = table["r"].shift(-1)
        table["r2_next"] = table["r_next"] ** 2
        table = table.iloc[:-1]  # drop the last day, which has no next-day return
        frames.append(table)
    panel = pd.concat(frames, ignore_index=True)
    agg_cols = [f"phi_{k}" for k in config.TREND_HORIZONS_K]
    phi = panel[agg_cols].to_numpy()
    basic_k = [config.TREND_HORIZONS_K.index(k) for k in config.AGG_HORIZONS_BASIC]
    enh_k = [config.TREND_HORIZONS_K.index(k) for k in config.AGG_HORIZONS_ENHANCED]
    panel["phi_agg"] = phi[:, basic_k].mean(axis=1)
    panel["phi_agg2"] = (phi[:, basic_k] ** 2).mean(axis=1)
    panel["phi_agg_enh"] = phi[:, enh_k].mean(axis=1)
    panel["phi_agg2_enh"] = (phi[:, enh_k] ** 2).mean(axis=1)
    return panel


def add_range_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add the squared-range column ``range2`` (paper section 4.3: R_t^2 =
    N*((H_t-L_t)/(H_t+L_t))^2, N chosen per asset so E[R^2]=1) plus the
    next-day target ``range2_next`` and the HAR weekly/monthly trailing
    averages, computed independently per ticker so no normalization or
    rolling window mixes assets. Drops rows without a complete HAR window or
    a next-day target (paper eq. (9))."""
    from src import range_models  # local import: avoids a module import cycle

    out = []
    for _, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        g["range2"] = range_models.normalized_range(g["high"].to_numpy(), g["low"].to_numpy())
        out.append(g)
    panel = pd.concat(out, ignore_index=True)
    panel = range_models.har_lag_features(panel)
    return panel
