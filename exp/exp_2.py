"""Experiment 2: GARCH(1,1) and GJR-GARCH Models with Trend Factors.

Reproduces the paper's test (paper section on GARCH models, exp_2
methodology steps 4-7) of whether adding trend factors to the GARCH(1,1)
and GJR-GARCH variance equations improves next-day variance forecasts
(out-of-sample R^2_adj), comparing the two variants against each other and
(via experiment 1's results) against the EWMA model.

Run: python -m exp.exp_2 [--full]
``--full`` uses the paper's cross-validation/bootstrap/block-bootstrap
repetition counts (config.CV_FOLDS_PAPER, config.BOOTSTRAP_SAMPLES_PAPER,
config.BLOCK_BOOTSTRAP_BLOCKS_PAPER, config.BLOCK_LENGTH_DAYS_PAPER); by
default the reduced counts in src/config.py are used (see
results/RESULTS.md, "Deviations from the paper").

Runtime note (not a formula change): the pooled GARCH recursion
(src/garch_models.garch_recursion) is a per-row Python loop, and the
realized panel here has ~139k (day, market) rows (24 tickers, 1993-2026),
not the "a few thousand rows" of a single market. A quasi-Newton fit with a
numerically-differentiated gradient costs roughly (2*n_params+1) objective
evaluations per iteration, each a full pass over the panel
(~0.1-0.2s/evaluation, benchmarked directly on this panel). Fitting the
whole panel or a cross-validation training fold (which keeps nearly all of
the panel) to a very tight gradient tolerance would take much longer. The
``maxiter`` values below were set from direct timing on this panel:
``scipy.optimize.minimize``'s own ``converged`` flag turns ``True`` for the
whole-panel GARCH-with-trend fit at ``maxiter=40`` (about 21s per fit,
vs. ``converged=False`` at ``maxiter=8``), so every reported full-panel and
CV-fold fit below is set to reach that same convergence criterion within
the available time budget (4 models x 6 CV folds plus the whole-panel and
equity-subset fits). ``converged`` is still reported for every fit rather
than assumed. The block-bootstrap resamples (step 4) are, by construction of
``evaluation.block_bootstrap_indices`` with the reduced block count/length,
drawn from a small fixed window of the panel, so those fits are cheap and
use a larger ``maxiter``.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_universe
from src import config, evaluation, garch_models, panel as panel_mod

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

COEF_ORDER = ["omega", "alpha", "beta", "gamma", "lam", "zeta"]
GARCH_TREND_COEF_ORDER = ["omega", "alpha", "beta", "lam", "zeta"]
GJR_TREND_COEF_ORDER = ["omega", "alpha", "beta", "gamma", "lam", "zeta"]

MODELS = [
    {"name": "GARCH", "with_trend": False, "asymmetric": False},
    {"name": "GARCH_trend", "with_trend": True, "asymmetric": False},
    {"name": "GJR-GARCH", "with_trend": False, "asymmetric": True},
    {"name": "GJR-GARCH_trend", "with_trend": True, "asymmetric": True},
]

# See the runtime note in the module docstring: these maxiter values trade
# optimizer convergence for a bounded total runtime on the ~139k-row panel.
# Benchmarked directly on the full panel: maxiter=40 reaches
# scipy.optimize.minimize's own convergence criterion (``converged=True``)
# for the pooled GARCH-with-trend fit in ~21s, vs. ``converged=False`` at
# the earlier maxiter=8 (~12s); the values below were raised accordingly.
FULL_FIT_MAXITER = 40    # whole-panel fit (step 2), 4 models
CV_FIT_MAXITER = 20      # per-fold fit inside cross_validate (step 3), 4 models x n_folds
BOOT_FIT_MAXITER = 50    # bootstrap resample fit (step 4): resample is small, see docstring
EQUITY_FULL_MAXITER = 40  # whole-equity-subset fit (step 5)
EQUITY_CV_MAXITER = 20   # per-fold fit on the equity subset (step 5)


def _n_params(with_trend: bool, asymmetric: bool) -> int:
    """Number of fitted parameters for a (with_trend, asymmetric) variant,
    matching ``garch_models.fit_pooled_garch``'s ``names`` list, without
    needing an extra fit just to read ``len(fitted["names"])``."""
    return 3 + int(asymmetric) + 2 * int(with_trend)


def fit_coefficients_table(panel: pd.DataFrame, maxiter: int) -> tuple:
    """Methodology step 2: fit the four pooled models over the whole panel
    and report their coefficients, convergence flag and QLIKE."""
    rows = []
    fitted_by_name = {}
    for spec in MODELS:
        fitted = garch_models.fit_pooled_garch(
            panel, with_trend=spec["with_trend"], asymmetric=spec["asymmetric"], maxiter=maxiter)
        row = {"name": spec["name"], "with_trend": spec["with_trend"], "asymmetric": spec["asymmetric"],
               "converged": bool(fitted["converged"]), "qlike_per_obs": float(fitted["qlike_per_obs"]),
               "n_obs": int(fitted["n_obs"])}
        row.update({c: float(fitted["params"].get(c, np.nan)) for c in COEF_ORDER})
        rows.append(row)
        fitted_by_name[spec["name"]] = fitted
    return pd.DataFrame(rows), fitted_by_name


def oos_r2_table(panel: pd.DataFrame, n_folds: int, maxiter: int) -> pd.DataFrame:
    """Methodology step 3: out-of-sample R^2_adj for the four pooled models
    via n_folds adjacent-day cross-validation, with with-trend-minus-
    without-trend deltas for GARCH and GJR-GARCH."""
    rows = []
    for spec in MODELS:
        with_trend, asymmetric = spec["with_trend"], spec["asymmetric"]

        def fit_fn(tr: pd.DataFrame, wt: bool = with_trend, asym: bool = asymmetric) -> dict:
            return garch_models.fit_pooled_garch(tr, with_trend=wt, asymmetric=asym, maxiter=maxiter)

        n_params = _n_params(with_trend, asymmetric)
        cv = evaluation.cross_validate(panel, fit_fn, garch_models.predict_pooled_garch,
                                        n_params=n_params, n_folds=n_folds)
        rows.append({"name": spec["name"], "with_trend": with_trend, "asymmetric": asymmetric,
                     "r2_adj_oos": cv["r2_adj"], "n_obs": cv["n_obs"]})
    table = pd.DataFrame(rows)
    table["delta_r2_adj_oos"] = np.nan
    for asymmetric in (False, True):
        sub = table[table["asymmetric"] == asymmetric]
        base = sub.loc[~sub["with_trend"], "r2_adj_oos"].iloc[0]
        trend_idx = sub.index[sub["with_trend"]][0]
        table.loc[trend_idx, "delta_r2_adj_oos"] = table.loc[trend_idx, "r2_adj_oos"] - base
    return table


def bootstrap_se_table(panel: pd.DataFrame, fitted_by_name: dict, n_boot: int,
                        n_blocks: int, block_length: int, maxiter: int) -> pd.DataFrame:
    """Methodology step 4: block-bootstrap standard errors for the
    GARCH-with-trend and GJR-GARCH-with-trend coefficient vectors (paper
    section 3.2: 18 contiguous blocks, whole blocks resampled to preserve
    the recursion's serial structure; block=True is required here, unlike
    experiment 1's day-level bootstrap, because GARCH variance is an
    iterative recursion that day-resampling would break)."""
    rows = []
    specs = [("GARCH_trend", False, GARCH_TREND_COEF_ORDER),
             ("GJR-GARCH_trend", True, GJR_TREND_COEF_ORDER)]
    for name, asymmetric, order in specs:
        estimate = fitted_by_name[name]["params"].reindex(order)

        def stat_fn(p: pd.DataFrame, asym: bool = asymmetric, order: list = order) -> np.ndarray:
            fitted = garch_models.fit_pooled_garch(p, with_trend=True, asymmetric=asym, maxiter=maxiter)
            return fitted["params"].reindex(order).to_numpy()

        boot = evaluation.bootstrap_errors(panel, stat_fn, n_samples=n_boot, block=True,
                                            n_blocks=n_blocks, block_length=block_length)
        for i, coef in enumerate(order):
            rows.append({"model": name, "coefficient": coef,
                         "estimate": float(estimate[coef]), "bootstrap_se": float(boot["error"][i])})
    return pd.DataFrame(rows)


def equity_subset_table(panel: pd.DataFrame, n_folds: int, full_maxiter: int, cv_maxiter: int) -> pd.DataFrame:
    """Methodology step 5: GARCH with vs. without trends restricted to the
    equity asset class (paper reports equities reaching R^2_adj ~18-19.5%).
    Uses out-of-sample CV (smaller fold count) when the equity subsample has
    enough distinct trading days per fold, else falls back to in-sample
    R^2_adj."""
    equity = panel[panel["asset_class"] == "equity"]
    cv_folds = min(n_folds, 4)
    use_cv = equity["date"].nunique() >= cv_folds * 50

    rows = []
    for with_trend in (False, True):
        fitted = garch_models.fit_pooled_garch(equity, with_trend=with_trend, asymmetric=False,
                                                maxiter=full_maxiter)
        n_params = _n_params(with_trend, False)
        if use_cv:
            def fit_fn(tr: pd.DataFrame, wt: bool = with_trend) -> dict:
                return garch_models.fit_pooled_garch(tr, with_trend=wt, asymmetric=False, maxiter=cv_maxiter)

            cv = evaluation.cross_validate(equity, fit_fn, garch_models.predict_pooled_garch,
                                            n_params=n_params, n_folds=cv_folds)
            r2, n_obs, method = cv["r2_adj"], cv["n_obs"], "oos"
        else:
            pred = garch_models.predict_pooled_garch(fitted, equity)
            r2 = evaluation.r2_adj(equity["r2_next"].to_numpy(), pred, n_params)
            n_obs, method = len(equity), "in_sample"
        rows.append({"asset_class": "equity", "with_trend": with_trend, "method": method,
                     "r2_adj": r2, "n_obs": n_obs, "converged": bool(fitted["converged"])})
    table = pd.DataFrame(rows)
    base = table.loc[~table["with_trend"], "r2_adj"].iloc[0]
    table["delta_r2_adj"] = np.nan
    table.loc[table["with_trend"], "delta_r2_adj"] = table.loc[table["with_trend"], "r2_adj"] - base
    return table


def main(full: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    n_folds = config.CV_FOLDS_PAPER if full else config.CV_FOLDS_RUN
    n_boot = config.BOOTSTRAP_SAMPLES_PAPER if full else config.BOOTSTRAP_SAMPLES_RUN
    n_blocks = config.BLOCK_BOOTSTRAP_BLOCKS_PAPER if full else config.BLOCK_BOOTSTRAP_BLOCKS_RUN
    block_length = config.BLOCK_LENGTH_DAYS_PAPER if full else config.BLOCK_LENGTH_DAYS_RUN

    t0 = time.time()
    per_ticker, meta = load_universe(config.UNIVERSE_RUN, config.SAMPLE_START, config.SAMPLE_END,
                                      identity_date=config.IDENTITY_DATE, use_cache=True)
    panel = panel_mod.build_panel(per_ticker, config.UNIVERSE_RUN)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    load_elapsed = time.time() - t0
    print(f"panel built in {load_elapsed:.1f}s: {len(panel)} rows, "
          f"{panel['ticker'].nunique()} tickers, {panel['date'].min()} to {panel['date'].max()}")

    coef_table, fitted_by_name = fit_coefficients_table(panel, FULL_FIT_MAXITER)
    coef_table.to_csv(RESULTS_DIR / "exp2_garch_coefficients.csv", index=False)

    oos_table = oos_r2_table(panel, n_folds, CV_FIT_MAXITER)
    oos_table.to_csv(RESULTS_DIR / "exp2_oos_r2.csv", index=False)

    boot_table = bootstrap_se_table(panel, fitted_by_name, n_boot, n_blocks, block_length, BOOT_FIT_MAXITER)
    boot_table.to_csv(RESULTS_DIR / "exp2_bootstrap_se.csv", index=False)

    equity_table = equity_subset_table(panel, n_folds, EQUITY_FULL_MAXITER, EQUITY_CV_MAXITER)
    equity_table.to_csv(RESULTS_DIR / "exp2_equity_subset.csv", index=False)

    elapsed = time.time() - t0
    data_log = {
        "n_obs": int(len(panel)), "n_tickers": int(panel["ticker"].nunique()),
        "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()),
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "failed": meta.get("failed", {}), "non_massive_source": meta.get("non_massive_source", {}),
        "cv_folds": n_folds, "bootstrap_samples": n_boot,
        "block_bootstrap_blocks": n_blocks, "block_length_days": block_length,
        "full_fit_maxiter": FULL_FIT_MAXITER, "cv_fit_maxiter": CV_FIT_MAXITER,
        "bootstrap_fit_maxiter": BOOT_FIT_MAXITER,
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp2_data_log.json", "w") as f:
        json.dump(data_log, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["GARCH\n(no trend)", "GARCH\n(+ trend)", "GJR-GARCH\n(no trend)", "GJR-GARCH\n(+ trend)"]
    colors = ["#4c72b0", "#dd8452", "#4c72b0", "#dd8452"]
    ax.bar(labels, oos_table["r2_adj_oos"].to_numpy(), color=colors)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("out-of-sample R^2_adj")
    ax.set_title("GARCH / GJR-GARCH: out-of-sample R^2_adj with vs. without trends")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_coefficients_with_without_trend.png", dpi=150)
    plt.close(fig)

    print("\n== Coefficients (whole-panel pooled fit) ==")
    print(coef_table.to_string(index=False))
    print("\n== Out-of-sample R^2_adj ==")
    print(oos_table.to_string(index=False))
    print("\n== Bootstrap standard errors (block bootstrap) ==")
    print(boot_table.to_string(index=False))
    print("\n== Equity-only subset ==")
    print(equity_table.to_string(index=False))
    print(f"\ntotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's CV/bootstrap repetition counts instead of the reduced run configuration")
    args = parser.parse_args()
    main(full=args.full)
