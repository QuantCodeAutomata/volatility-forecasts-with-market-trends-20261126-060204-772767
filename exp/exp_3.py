"""Experiment 3: Range-based Volatility Models (GARCH and HAR) with Trend
Factors (paper section on range-based models, exp_3 methodology steps 4-8).

Tests whether adding trend factors improves a range-based GARCH(1,1) model
(squared daily high-low range R_t^2 in place of squared returns r_t^2,
paper section 4.3) and a HAR (Heterogeneous Autoregressive) model of R_t^2
(paper eq. (9)), each with vs. without trend factors, and reports their
out-of-sample R^2_adj alongside a textual comparison to the return-based
models of experiments 1/2 (paper table 9 vs. tables 3/8).

Run: python -m exp.exp_3 [--full]
``--full`` uses the paper's cross-validation/bootstrap/block-bootstrap
repetition counts (config.CV_FOLDS_PAPER, config.BOOTSTRAP_SAMPLES_PAPER,
config.BLOCK_BOOTSTRAP_BLOCKS_PAPER, config.BLOCK_LENGTH_DAYS_PAPER); by
default the reduced counts in src/config.py are used (see
results/RESULTS.md, "Deviations from the paper").

Sample start date: the paper's range-based analysis covers a shorter period
than the return-based models because daily High/Low only becomes available
from the data path used here starting April 1995 (vs. May 1991 for close-to-
close returns, config.ANALYSIS_START). src/config.py has no
``RANGE_SAMPLE_START`` constant, so this module defines ``RANGE_SAMPLE_START
= "1995-04-01"`` locally (not by editing src/config.py) and passes it to
``load_universe`` as ``start_date``. Because the Dataloader cache key
(data/data_loader.py) hashes the full request including ``start_date``, this
is a different cache entry from experiment 1/2's "1990-01-01" load; it will
download fresh OHLC data on first run (high/low are already part of the
same OHLCV response the other experiments use, so no new endpoint/field is
needed) and cache it under its own key for subsequent runs.

Runtime note (same reasoning as exp_2's module docstring): the pooled
range-GARCH recursion (src/garch_models.garch_recursion, reused by
src/range_models.fit_range_garch) is a per-row Python loop over a
multi-decade, multi-market panel, so a quasi-Newton fit to full convergence
is too slow to run inside the time budget for four models x several CV
folds x bootstrap resamples. The ``maxiter`` values below are kept small,
exactly as in exp_2, and ``converged`` is reported honestly (often False)
rather than hidden. The HAR model is a one-shot OLS fit (no iterative
recursion) and is always fit to full convergence.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_universe
from src import config, evaluation, panel as panel_mod, range_models

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# See the module docstring: no config.RANGE_SAMPLE_START exists, so the
# range-appropriate start date is defined here, inside exp_3.py only.
RANGE_SAMPLE_START = "1995-04-01"

RANGE_GARCH_COEF_ORDER = ["omega", "alpha", "beta", "lam", "zeta"]
HAR_NO_TREND_COEF_ORDER = ["const", "daily", "weekly", "monthly"]
HAR_TREND_COEF_ORDER = ["const", "daily", "weekly", "monthly", "phi_agg", "phi_agg2"]

# See the runtime note in the module docstring: these maxiter values trade
# optimizer convergence for a bounded total runtime, exactly as in exp_2.
# Same convergence benchmark as exp_2 (whole-panel GARCH-with-trend fit
# reaches scipy.optimize.minimize's own ``converged=True`` at maxiter=40 on
# this panel, vs. ``converged=False`` at maxiter=8); raised accordingly so
# every reported fit is a genuine local optimum within the time budget.
GARCH_FULL_MAXITER = 40    # whole-panel fit (step 2), 2 variants (with/without trend)
GARCH_CV_MAXITER = 20      # per-fold fit inside cross_validate (step 4), 2 variants x n_folds
GARCH_BOOT_MAXITER = 50    # bootstrap resample fit (step 7): resample is small, see exp_2's docstring
GARCH_EQUITY_FULL_MAXITER = 40  # whole-equity-subset fit (step 6)
GARCH_EQUITY_CV_MAXITER = 20    # per-fold fit on the equity subset (step 6)


def _har_n_params(with_trend: bool) -> int:
    return len(HAR_TREND_COEF_ORDER) - 1 if with_trend else len(HAR_NO_TREND_COEF_ORDER) - 1


def _garch_n_params(with_trend: bool) -> int:
    return 5 if with_trend else 3


def fit_range_garch_table(panel: pd.DataFrame, maxiter: int) -> tuple:
    """Methodology step 2: fit the range-GARCH(1,1) recursion with/without
    trend factors over the whole panel and report its coefficients,
    convergence flag and QLIKE."""
    rows = []
    fitted_by_trend = {}
    for with_trend in (False, True):
        fitted = range_models.fit_range_garch(panel, with_trend=with_trend, maxiter=maxiter)
        row = {"model": "range_garch", "with_trend": with_trend,
               "converged": bool(fitted["converged"]), "qlike_per_obs": float(fitted["qlike_per_obs"]),
               "n_obs": int(fitted["n_obs"])}
        row.update({c: float(fitted["params"].get(c, np.nan)) for c in RANGE_GARCH_COEF_ORDER})
        rows.append(row)
        fitted_by_trend[with_trend] = fitted
    return pd.DataFrame(rows), fitted_by_trend


def fit_har_table(panel: pd.DataFrame) -> tuple:
    """Methodology step 3: fit the HAR regression with/without trend factors
    over the whole panel and report its coefficients and in-sample R^2 /
    R^2_adj (statsmodels), as a reference point alongside the out-of-sample
    number computed separately in step 4."""
    rows = []
    fitted_by_trend = {}
    for with_trend in (False, True):
        fitted = range_models.fit_har(panel, with_trend=with_trend)
        row = {"model": "har", "with_trend": with_trend, "n_obs": int(len(fitted["y"])),
               "r2_in_sample": float(fitted["model"].rsquared),
               "r2_adj_in_sample_statsmodels": float(fitted["model"].rsquared_adj)}
        row.update({c: float(fitted["params"].get(c, np.nan)) for c in HAR_TREND_COEF_ORDER})
        rows.append(row)
        fitted_by_trend[with_trend] = fitted
    return pd.DataFrame(rows), fitted_by_trend


def oos_r2_table(panel: pd.DataFrame, n_folds: int, garch_maxiter: int) -> pd.DataFrame:
    """Methodology step 4 (and, restricted to the equity subset, step 6):
    out-of-sample R^2_adj for the four models (range-GARCH, HAR; each
    with/without trend) via n_folds adjacent-day cross-validation on
    ``range2_next``, with with-trend-minus-without-trend deltas."""
    rows = []
    for with_trend in (False, True):
        def fit_fn(tr: pd.DataFrame, wt: bool = with_trend) -> dict:
            return range_models.fit_range_garch(tr, with_trend=wt, maxiter=garch_maxiter)

        cv = evaluation.cross_validate(panel, fit_fn, range_models.predict_range_garch,
                                        n_params=_garch_n_params(with_trend), n_folds=n_folds,
                                        y_col="range2_next")
        rows.append({"model": "range_garch", "with_trend": with_trend,
                     "r2_adj_oos": cv["r2_adj"], "n_obs": cv["n_obs"]})

    for with_trend in (False, True):
        def fit_fn(tr: pd.DataFrame, wt: bool = with_trend) -> dict:
            return range_models.fit_har(tr, with_trend=wt)

        cv = evaluation.cross_validate(panel, fit_fn, range_models.predict_har,
                                        n_params=_har_n_params(with_trend), n_folds=n_folds,
                                        y_col="range2_next")
        rows.append({"model": "har", "with_trend": with_trend,
                     "r2_adj_oos": cv["r2_adj"], "n_obs": cv["n_obs"]})

    table = pd.DataFrame(rows)
    table["delta_r2_adj_oos"] = np.nan
    for model in ("range_garch", "har"):
        sub = table[table["model"] == model]
        base = sub.loc[~sub["with_trend"], "r2_adj_oos"].iloc[0]
        trend_idx = sub.index[sub["with_trend"]][0]
        table.loc[trend_idx, "delta_r2_adj_oos"] = table.loc[trend_idx, "r2_adj_oos"] - base
    return table


def full_and_equity_oos_table(panel: pd.DataFrame, n_folds: int, garch_maxiter: int) -> pd.DataFrame:
    """Combines step 4 (full panel) and step 6 (equity-only subset, paper:
    equity R^2 can approach 50%) into one table with a ``subset`` column."""
    full = oos_r2_table(panel, n_folds, garch_maxiter)
    full.insert(0, "subset", "full")

    equity = panel[panel["asset_class"] == "equity"]
    equity_folds = min(n_folds, 4)
    equity_table = oos_r2_table(equity, equity_folds, garch_maxiter)
    equity_table.insert(0, "subset", "equity")

    return pd.concat([full, equity_table], ignore_index=True)


def bootstrap_se_table(panel: pd.DataFrame, fitted_garch_trend: dict, fitted_har_trend: dict,
                        n_boot: int, n_blocks: int, block_length: int, garch_maxiter: int) -> pd.DataFrame:
    """Methodology step 7: bootstrap standard errors for the range-GARCH-
    with-trend and HAR-with-trend coefficient vectors.

    Both use the block bootstrap (``block=True``): it is required for
    range-GARCH because its variance is an iterative recursion that plain
    day-resampling would break (same reasoning as exp_2's GARCH bootstrap),
    and it is also valid (if not strictly necessary) for the HAR model,
    which is a plain one-shot OLS regression like exp_1's basic model; using
    the same block-bootstrap routine for both avoids maintaining two
    bootstrap code paths for this experiment.
    """
    rows = []

    estimate_garch = fitted_garch_trend["params"].reindex(RANGE_GARCH_COEF_ORDER)

    def stat_fn_garch(p: pd.DataFrame) -> np.ndarray:
        fitted = range_models.fit_range_garch(p, with_trend=True, maxiter=garch_maxiter)
        return fitted["params"].reindex(RANGE_GARCH_COEF_ORDER).to_numpy()

    boot_garch = evaluation.bootstrap_errors(panel, stat_fn_garch, n_samples=n_boot, block=True,
                                              n_blocks=n_blocks, block_length=block_length)
    for i, coef in enumerate(RANGE_GARCH_COEF_ORDER):
        rows.append({"model": "range_garch_trend", "coefficient": coef,
                     "estimate": float(estimate_garch[coef]), "bootstrap_se": float(boot_garch["error"][i])})

    estimate_har = fitted_har_trend["params"].reindex(HAR_TREND_COEF_ORDER)

    def stat_fn_har(p: pd.DataFrame) -> np.ndarray:
        fitted = range_models.fit_har(p, with_trend=True)
        return fitted["params"].reindex(HAR_TREND_COEF_ORDER).to_numpy()

    boot_har = evaluation.bootstrap_errors(panel, stat_fn_har, n_samples=n_boot, block=True,
                                            n_blocks=n_blocks, block_length=block_length)
    for i, coef in enumerate(HAR_TREND_COEF_ORDER):
        rows.append({"model": "har_trend", "coefficient": coef,
                     "estimate": float(estimate_har[coef]), "bootstrap_se": float(boot_har["error"][i])})

    return pd.DataFrame(rows)


def main(full: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    n_folds = config.CV_FOLDS_PAPER if full else config.CV_FOLDS_RUN
    n_boot = config.BOOTSTRAP_SAMPLES_PAPER if full else config.BOOTSTRAP_SAMPLES_RUN
    n_blocks = config.BLOCK_BOOTSTRAP_BLOCKS_PAPER if full else config.BLOCK_BOOTSTRAP_BLOCKS_RUN
    block_length = config.BLOCK_LENGTH_DAYS_PAPER if full else config.BLOCK_LENGTH_DAYS_RUN

    t0 = time.time()
    per_ticker, meta = load_universe(config.UNIVERSE_RUN, RANGE_SAMPLE_START, config.SAMPLE_END,
                                      identity_date=config.IDENTITY_DATE, use_cache=True)
    panel = panel_mod.build_panel(per_ticker, config.UNIVERSE_RUN)
    panel = panel_mod.add_range_columns(panel)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    load_elapsed = time.time() - t0
    print(f"start_date used: {RANGE_SAMPLE_START} (no config.RANGE_SAMPLE_START; High/Low data "
          f"path coverage starts ~1995-04 vs. {config.ANALYSIS_START} for return-based models)")
    print(f"panel built in {load_elapsed:.1f}s: {len(panel)} rows, "
          f"{panel['ticker'].nunique()} tickers, {panel['date'].min()} to {panel['date'].max()}")

    garch_coef_table, fitted_garch_by_trend = fit_range_garch_table(panel, GARCH_FULL_MAXITER)
    garch_coef_table.to_csv(RESULTS_DIR / "exp3_range_garch_coefficients.csv", index=False)

    har_coef_table, fitted_har_by_trend = fit_har_table(panel)
    har_coef_table.to_csv(RESULTS_DIR / "exp3_har_coefficients.csv", index=False)

    # Methodology step 5 (textual, not computed here): the paper reports
    # range-based out-of-sample R^2_adj roughly 0.19-0.21 (table 9) vs. about
    # 0.12 for the return-based basic/GARCH models (tables 3/8); exp_1/exp_2
    # are not re-run here, this is only a reference point for the numbers
    # printed below.
    print("Comparison point (paper table 9 vs. tables 3/8, not recomputed here): "
          "range-based R^2_adj ~0.19-0.21 vs. ~0.12 for return-based models.")

    oos_table = full_and_equity_oos_table(panel, n_folds, GARCH_CV_MAXITER)
    oos_table.to_csv(RESULTS_DIR / "exp3_oos_r2.csv", index=False)

    boot_table = bootstrap_se_table(panel, fitted_garch_by_trend[True], fitted_har_by_trend[True],
                                     n_boot, n_blocks, block_length, GARCH_BOOT_MAXITER)
    boot_table.to_csv(RESULTS_DIR / "exp3_bootstrap_se.csv", index=False)

    elapsed = time.time() - t0
    data_log = {
        "n_obs": int(len(panel)), "n_tickers": int(panel["ticker"].nunique()),
        "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()),
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "failed": meta.get("failed", {}), "non_massive_source": meta.get("non_massive_source", {}),
        "cv_folds": n_folds, "bootstrap_samples": n_boot,
        "block_bootstrap_blocks": n_blocks, "block_length_days": block_length,
        "start_date_used": RANGE_SAMPLE_START,
        "start_date_reason": "no config.RANGE_SAMPLE_START; daily High/Low coverage on the data "
                              "path used here starts ~1995-04 vs. config.ANALYSIS_START "
                              f"({config.ANALYSIS_START}) for return-based models",
        "garch_full_maxiter": GARCH_FULL_MAXITER, "garch_cv_maxiter": GARCH_CV_MAXITER,
        "garch_boot_maxiter": GARCH_BOOT_MAXITER,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp3_data_log.json", "w") as f:
        json.dump(data_log, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 4))
    full_rows = oos_table[oos_table["subset"] == "full"]
    labels = ["range-GARCH\n(no trend)", "range-GARCH\n(+ trend)", "HAR\n(no trend)", "HAR\n(+ trend)"]
    values = [
        full_rows.loc[(full_rows["model"] == "range_garch") & (~full_rows["with_trend"]), "r2_adj_oos"].iloc[0],
        full_rows.loc[(full_rows["model"] == "range_garch") & (full_rows["with_trend"]), "r2_adj_oos"].iloc[0],
        full_rows.loc[(full_rows["model"] == "har") & (~full_rows["with_trend"]), "r2_adj_oos"].iloc[0],
        full_rows.loc[(full_rows["model"] == "har") & (full_rows["with_trend"]), "r2_adj_oos"].iloc[0],
    ]
    colors = ["#4c72b0", "#dd8452", "#4c72b0", "#dd8452"]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("out-of-sample R^2_adj")
    ax.set_title("Range-GARCH / HAR: out-of-sample R^2_adj with vs. without trends")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_oos_r2_comparison.png", dpi=150)
    plt.close(fig)

    print("\n== Range-GARCH coefficients (whole-panel pooled fit) ==")
    print(garch_coef_table.to_string(index=False))
    print("\n== HAR coefficients (whole-panel fit) ==")
    print(har_coef_table.to_string(index=False))
    print("\n== Out-of-sample R^2_adj (full panel and equity-only subset) ==")
    print(oos_table.to_string(index=False))
    print("\n== Bootstrap standard errors (block bootstrap) ==")
    print(boot_table.to_string(index=False))
    print(f"\ntotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's CV/bootstrap repetition counts instead of the reduced run configuration")
    args = parser.parse_args()
    main(full=args.full)
