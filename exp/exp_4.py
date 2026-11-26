"""Experiment 4: Stochastic Volatility Models (Heston and Leverage SV) with
Trend Factors (paper section on stochastic volatility models, exp_4
methodology steps 4-11).

Tests whether adding trend factors to a Heston-type (level-variance) and a
Leverage-SV / LSV (log-variance) stochastic volatility model improves
predictive power, measured by out-of-sample quasi-likelihood per
observation ``1 - QLIKE/N`` (QLIKE = sum_i[log(sigma_hat_i^2) +
r_i^2/sigma_hat_i^2], see ``_qlike_total`` below / ``src.garch_models
._qlike_sum`` which is reused for the summation).

Run: python -m exp.exp_4 [--full]

Runtime note (performance-driven simplification, NOT a formula change):
``sv_models.fit_heston``/``fit_lsv`` run a pure-Python Kalman-filter
recursion over every (day, market) row, once per objective-function
evaluation, and ``scipy.optimize.minimize`` needs roughly
``maxiter * (n_params + 1)`` such evaluations (numerically-differentiated
gradient, L-BFGS-B). Measured directly on this panel (24 tickers, ~139k
rows): one ``fit_heston(with_trend=False, maxiter=20)`` call (the paper's
illustrative ``maxiter=100`` scaled down) took about 60-65 seconds (see the
printed timing measurement at the start of every run). Four whole-panel
fits (Heston / LSV, with/without trend) plus a cross-validation (n_folds
refits x 4 models) plus a block-bootstrap (n_samples refits x 2 with-trend
models) at the paper's full settings (``maxiter=100``, 16 CV folds, 300
bootstrap samples, all 24 tickers everywhere) would take many hours, far
beyond this experiment's time budget. The reduced-run constants below
(``SV_MAXITER_RUN``, ``SV_CV_FOLDS_RUN``, ``SV_CV_MAXITER``,
``SV_BOOTSTRAP_*``, ``SV_CV_TICKER_SUBSET``) trade optimizer convergence
and resampling granularity for a bounded total runtime (~200-250s), chosen
from the measured per-call cost above; ``converged`` is reported honestly
(often False at this reduced ``maxiter``) rather than hidden, and up to one
alternate starting point (``x0``) is tried for every whole-panel fit before
accepting a non-converged result. ``--full`` selects the paper's repetition
counts (``config.CV_FOLDS_PAPER``, ``config.BOOTSTRAP_SAMPLES_PAPER``,
``config.BLOCK_BOOTSTRAP_BLOCKS_PAPER``, ``config.BLOCK_LENGTH_DAYS_PAPER``),
``maxiter=100`` everywhere, and the full 24-ticker universe for
cross-validation and bootstrap too (not just the whole-panel fit); this path
is expected to be slow and is not exercised by the default run.
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
from src import config, evaluation, panel as panel_mod, sv_models
from src.garch_models import _qlike_sum

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

HESTON_COEF_ORDER = ["kappa", "theta", "xi", "lam", "zeta"]
LSV_COEF_ORDER = ["mu", "beta", "xi", "lam", "zeta"]
MODEL_SPECS = [
    ("Heston", "heston", False),
    ("Heston_trend", "heston", True),
    ("LSV", "lsv", False),
    ("LSV_trend", "lsv", True),
]

# --- Performance-driven reductions for the default (non-``--full``) run ---
# See the module docstring's "Runtime note" for the measured cost these are
# derived from. Only the repetition counts / iteration caps are reduced; the
# fitting method and QLIKE formula are unchanged.
MEASURE_MAXITER = 20          # step (a): one-off calibration call, always run
SV_MAXITER_RUN = 25           # whole-panel fits (step 2), all 4 models
SV_CV_FOLDS_RUN = 2           # adjacent day-folds for out-of-sample QLIKE (step 3)
SV_CV_MAXITER = 8             # per-fold refit iterations (step 3)
SV_BOOTSTRAP_SAMPLES_RUN = 12       # block-bootstrap resamples (step 4)
SV_BOOTSTRAP_MAXITER = 15          # per-resample refit iterations (step 4)
SV_BOOTSTRAP_BLOCKS_RUN = 4        # contiguous day-blocks per resample
SV_BOOTSTRAP_BLOCK_LENGTH_RUN = 30  # days per block
# Deterministic 2-per-asset-class subset (preserves config.UNIVERSE_RUN's
# insertion order), used for cross-validation and the bootstrap only (step
# 2's whole-panel fit always uses the full 24-ticker universe).
SV_CV_TICKER_SUBSET = ["SPY", "EWC", "SHV", "SHY", "C:CADUSD", "C:EURUSD", "USO", "UNG"]

FULL_MAXITER = 100  # --full: paper's illustrative maxiter for every fit

# Alternate x0 starting points (second attempt if the first does not
# converge), chosen within config.SV_PARAM_BOUNDS and away from the default
# x0 baked into sv_models.fit_heston/fit_lsv.
HESTON_X0_ALT = {"kappa": 0.2, "theta": 0.5, "xi": 0.5, "lam": -0.05, "zeta": -0.02}
LSV_X0_ALT = {"mu": -1.0, "beta": 0.1, "xi": 0.5, "lam": -0.05, "zeta": -0.02}


def _panel_by_ticker(df: pd.DataFrame) -> dict:
    """{ticker: DataFrame[r, phi_agg, phi_agg2]} sorted by date, matching
    ``sv_models.fit_heston``/``fit_lsv``'s expected input (and mirroring
    ``garch_models.fit_pooled_garch``'s internal ``g.sort_values("date")``,
    which this project's existing bootstrap/cross-validation machinery
    already relies on for recursive models built from resampled rows)."""
    cols = ["r", "phi_agg", "phi_agg2"]
    return {t: g.sort_values("date")[cols].reset_index(drop=True)
            for t, g in df.groupby("ticker", sort=False)}


def _fit_model(kind: str, with_trend: bool, panel_by_ticker: dict, maxiter: int,
               x0_alt: dict = None) -> tuple:
    """Fit Heston (``kind="heston"``) or LSV (``kind="lsv"``), retrying once
    with an alternate starting point if the first attempt does not converge
    (methodology requirement: "try at least 2 different x0 starting points
    before giving up"). Returns ``(fitted, n_x0_attempts)``, keeping whichever
    attempt has the lower negative log-likelihood if neither converges."""
    fit_fn = sv_models.fit_heston if kind == "heston" else sv_models.fit_lsv
    first = fit_fn(panel_by_ticker, with_trend=with_trend, maxiter=maxiter)
    if first["converged"] or x0_alt is None:
        return first, 1
    second = fit_fn(panel_by_ticker, with_trend=with_trend, x0=x0_alt, maxiter=maxiter)
    if second["converged"] or second["neg_loglik"] < first["neg_loglik"]:
        return second, 2
    return first, 2


def fit_coefficients_table(panel_by_ticker_full: dict, maxiter: int) -> tuple:
    """Methodology step 2: fit Heston/LSV with/without trend on the full
    (24-ticker) panel and report coefficients, convergence and neg_loglik."""
    rows = []
    fitted_by_name = {}
    order_by_kind = {"heston": HESTON_COEF_ORDER, "lsv": LSV_COEF_ORDER}
    for name, kind, with_trend in MODEL_SPECS:
        x0_alt = HESTON_X0_ALT if kind == "heston" else LSV_X0_ALT
        fitted, n_attempts = _fit_model(kind, with_trend, panel_by_ticker_full, maxiter, x0_alt)
        row = {"name": name, "kind": kind, "with_trend": with_trend,
               "converged": bool(fitted["converged"]), "neg_loglik": float(fitted["neg_loglik"]),
               "n_obs": int(fitted["n_obs"]), "n_x0_attempts": n_attempts, "maxiter": maxiter}
        row.update({c: float(fitted["params"].get(c, np.nan)) for c in order_by_kind[kind]})
        rows.append(row)
        fitted_by_name[name] = fitted
    return pd.DataFrame(rows), fitted_by_name


def leverage_correlation_table(panel_by_ticker_full: dict, fitted_by_name: dict) -> dict:
    """Methodology step 5: cross-ticker average leverage correlation rho for
    the with-trend Heston and LSV fits only (diagnostic, compared against
    the paper's reported rho ~ -0.29 / -0.25)."""
    out = {}
    for name in ("Heston_trend", "LSV_trend"):
        preds = sv_models.predict_sv(fitted_by_name[name], panel_by_ticker_full)
        rhos = [sv_models.leverage_correlation(panel_by_ticker_full[t]["r"].to_numpy(), sigma2_hat)
                for t, sigma2_hat in preds.items()]
        rhos = [r for r in rhos if np.isfinite(r)]
        out[name] = float(np.mean(rhos)) if rhos else float("nan")
    return out


def oos_quasi_likelihood_table(panel: pd.DataFrame, ticker_subset: list, n_folds: int,
                                cv_maxiter: int) -> pd.DataFrame:
    """Methodology step 3: out-of-sample quasi-likelihood via ``n_folds``
    adjacent day-folds over the pooled (ticker-subset-restricted) panel's
    dates: train each of the 4 models per fold, predict the held-out fold
    via ``sv_models.predict_sv``, accumulate QLIKE over every held-out
    (ticker, day) observation, and report ``1 - QLIKE_total/N_total`` plus a
    with-trend-minus-without-trend ``delta`` for Heston and for LSV."""
    subset_panel = panel[panel["ticker"].isin(ticker_subset)].reset_index(drop=True)
    masks = evaluation.day_folds(subset_panel["date"], n_folds)
    totals = {name: {"qlike": 0.0, "n": 0} for name, _, _ in MODEL_SPECS}
    for mask in masks:
        train_df, test_df = subset_panel.loc[~mask], subset_panel.loc[mask]
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        train_by_ticker = _panel_by_ticker(train_df)
        test_by_ticker = _panel_by_ticker(test_df)
        for name, kind, with_trend in MODEL_SPECS:
            fit_fn = sv_models.fit_heston if kind == "heston" else sv_models.fit_lsv
            fitted = fit_fn(train_by_ticker, with_trend=with_trend, maxiter=cv_maxiter)
            preds = sv_models.predict_sv(fitted, test_by_ticker)
            for ticker, sigma2_hat in preds.items():
                r2 = test_by_ticker[ticker]["r"].to_numpy() ** 2
                totals[name]["qlike"] += _qlike_sum(sigma2_hat, r2)
                totals[name]["n"] += len(r2)

    rows = []
    for name, kind, with_trend in MODEL_SPECS:
        n, qlike = totals[name]["n"], totals[name]["qlike"]
        score = 1.0 - qlike / n if n > 0 else float("nan")
        rows.append({"name": name, "kind": kind, "with_trend": with_trend,
                     "oos_1_minus_qlike_over_n": score, "n_obs": n,
                     "n_tickers_cv": len(ticker_subset)})
    table = pd.DataFrame(rows)
    table["delta"] = np.nan
    for kind in ("heston", "lsv"):
        sub = table[table["kind"] == kind]
        base = sub.loc[~sub["with_trend"], "oos_1_minus_qlike_over_n"].iloc[0]
        idx = sub.index[sub["with_trend"]][0]
        table.loc[idx, "delta"] = table.loc[idx, "oos_1_minus_qlike_over_n"] - base
    return table


def bootstrap_se_table(panel: pd.DataFrame, ticker_subset: list, fitted_by_name: dict,
                        n_boot: int, n_blocks: int, block_length: int, maxiter: int) -> pd.DataFrame:
    """Methodology step 4: block-bootstrap standard errors for the
    Heston-with-trend and LSV-with-trend coefficient vectors, resampling
    contiguous day-blocks (``evaluation.block_bootstrap_indices``, same
    whole-day/whole-block unit as experiments 1-2) of the ticker-subset
    panel; point estimates are the whole-(24-ticker)-panel fits from step 2."""
    subset_panel = panel[panel["ticker"].isin(ticker_subset)].reset_index(drop=True)
    rows = []
    specs = [("Heston_trend", "heston", HESTON_COEF_ORDER),
             ("LSV_trend", "lsv", LSV_COEF_ORDER)]
    for name, kind, order in specs:
        fit_fn = sv_models.fit_heston if kind == "heston" else sv_models.fit_lsv
        estimate = fitted_by_name[name]["params"].reindex(order)

        def stat_fn(p: pd.DataFrame, fit_fn=fit_fn, order=order) -> np.ndarray:
            fitted = fit_fn(_panel_by_ticker(p), with_trend=True, maxiter=maxiter)
            return fitted["params"].reindex(order).to_numpy()

        boot = evaluation.bootstrap_errors(subset_panel, stat_fn, n_samples=n_boot, block=True,
                                            n_blocks=n_blocks, block_length=block_length)
        for i, coef in enumerate(order):
            rows.append({"model": name, "coefficient": coef,
                         "estimate": float(estimate[coef]), "bootstrap_se": float(boot["error"][i])})
    return pd.DataFrame(rows)


def main(full: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    if full:
        n_folds = config.CV_FOLDS_PAPER
        n_boot = config.BOOTSTRAP_SAMPLES_PAPER
        n_blocks = config.BLOCK_BOOTSTRAP_BLOCKS_PAPER
        block_length = config.BLOCK_LENGTH_DAYS_PAPER
        maxiter_full = maxiter_cv = maxiter_boot = FULL_MAXITER
    else:
        n_folds = SV_CV_FOLDS_RUN
        n_boot = SV_BOOTSTRAP_SAMPLES_RUN
        n_blocks = SV_BOOTSTRAP_BLOCKS_RUN
        block_length = SV_BOOTSTRAP_BLOCK_LENGTH_RUN
        maxiter_full = SV_MAXITER_RUN
        maxiter_cv = SV_CV_MAXITER
        maxiter_boot = SV_BOOTSTRAP_MAXITER

    t0 = time.time()
    per_ticker, meta = load_universe(config.UNIVERSE_RUN, config.SAMPLE_START, config.SAMPLE_END,
                                      identity_date=config.IDENTITY_DATE, use_cache=True)
    panel = panel_mod.build_panel(per_ticker, config.UNIVERSE_RUN)
    load_elapsed = time.time() - t0
    print(f"panel built in {load_elapsed:.1f}s: {len(panel)} rows, "
          f"{panel['ticker'].nunique()} tickers, {panel['date'].min()} to {panel['date'].max()}")

    ticker_subset = sorted(panel["ticker"].unique().tolist()) if full else SV_CV_TICKER_SUBSET
    panel_by_ticker_full = _panel_by_ticker(panel)

    # Step (a): measure wall-clock cost of ONE fit_heston call on the full
    # panel, at the paper's illustrative maxiter=20-scale-down; this is a
    # one-off diagnostic (always run, regardless of --full) that calibrates
    # the reduced-run constants documented in the module docstring.
    t_measure = time.time()
    sv_models.fit_heston(panel_by_ticker_full, with_trend=False, maxiter=MEASURE_MAXITER)
    measure_elapsed = time.time() - t_measure
    n_obs_full = sum(len(g) for g in panel_by_ticker_full.values())
    print(f"timing measurement: fit_heston(with_trend=False, maxiter={MEASURE_MAXITER}) on "
          f"{len(panel_by_ticker_full)} tickers / {n_obs_full} rows took {measure_elapsed:.1f}s")

    t_stage = time.time()
    coef_table, fitted_by_name = fit_coefficients_table(panel_by_ticker_full, maxiter_full)
    leverage = leverage_correlation_table(panel_by_ticker_full, fitted_by_name)
    coef_table["avg_leverage_correlation"] = coef_table["name"].map(leverage)
    coef_table.to_csv(RESULTS_DIR / "exp4_sv_coefficients.csv", index=False)
    coef_elapsed = time.time() - t_stage
    print(f"step 2 (whole-panel coefficients + leverage) took {coef_elapsed:.1f}s")

    t_stage = time.time()
    oos_table = oos_quasi_likelihood_table(panel, ticker_subset, n_folds, maxiter_cv)
    oos_table.to_csv(RESULTS_DIR / "exp4_oos_quasi_likelihood.csv", index=False)
    oos_elapsed = time.time() - t_stage
    print(f"step 3 (out-of-sample quasi-likelihood, {n_folds} folds) took {oos_elapsed:.1f}s")

    t_stage = time.time()
    boot_table = bootstrap_se_table(panel, ticker_subset, fitted_by_name, n_boot, n_blocks,
                                     block_length, maxiter_boot)
    boot_table.to_csv(RESULTS_DIR / "exp4_bootstrap_se.csv", index=False)
    boot_elapsed = time.time() - t_stage
    print(f"step 4 (block bootstrap, {n_boot} samples) took {boot_elapsed:.1f}s")

    elapsed = time.time() - t0
    data_log = {
        "n_obs": int(len(panel)), "n_tickers": int(panel["ticker"].nunique()),
        "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()),
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "failed": meta.get("failed", {}), "non_massive_source": meta.get("non_massive_source", {}),
        "full_run": full,
        "timing_measurement": {
            "call": f"fit_heston(with_trend=False, maxiter={MEASURE_MAXITER})",
            "n_tickers": len(panel_by_ticker_full), "n_obs": n_obs_full,
            "elapsed_seconds": measure_elapsed,
        },
        "reductions": {
            "reason": (
                "the per-row Python Kalman-filter recursion in src/sv_models.py "
                f"costs ~{measure_elapsed:.1f}s for one maxiter={MEASURE_MAXITER} "
                "whole-panel fit (measured above); the paper's maxiter=100 / "
                "16 CV folds / 300 bootstrap samples / 24-ticker CV+bootstrap "
                "would take far longer than this experiment's time budget "
                "(target ~260s total, this run)."
            ),
            "whole_panel_fit_maxiter": maxiter_full,
            "cv_folds": n_folds, "cv_fit_maxiter": maxiter_cv,
            "bootstrap_samples": n_boot, "bootstrap_fit_maxiter": maxiter_boot,
            "bootstrap_blocks": n_blocks, "bootstrap_block_length_days": block_length,
            "cv_bootstrap_ticker_subset": None if full else ticker_subset,
            "n_tickers_cv_bootstrap": len(ticker_subset),
        },
        "stage_elapsed_seconds": {
            "load_and_build_panel": load_elapsed, "timing_measurement": measure_elapsed,
            "step2_whole_panel_coefficients_and_leverage": coef_elapsed,
            "step3_oos_quasi_likelihood": oos_elapsed, "step4_block_bootstrap": boot_elapsed,
        },
        "elapsed_seconds": elapsed,
    }
    with open(RESULTS_DIR / "exp4_data_log.json", "w") as f:
        json.dump(data_log, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Heston\n(no trend)", "Heston\n(+ trend)", "LSV\n(no trend)", "LSV\n(+ trend)"]
    colors = ["#4c72b0", "#dd8452", "#4c72b0", "#dd8452"]
    ax.bar(labels, oos_table["oos_1_minus_qlike_over_n"].to_numpy(), color=colors)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("out-of-sample 1 - QLIKE/N")
    ax.set_title("Heston / LSV: out-of-sample quasi-likelihood with vs. without trends")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4_oos_quasi_likelihood.png", dpi=150)
    plt.close(fig)

    print("\n== Coefficients (whole-panel pooled fit) ==")
    print(coef_table.to_string(index=False))
    print("\n== Out-of-sample 1 - QLIKE/N ==")
    print(oos_table.to_string(index=False))
    print("\n== Bootstrap standard errors (block bootstrap) ==")
    print(boot_table.to_string(index=False))
    print(f"\ntotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's maxiter/CV-fold/bootstrap repetition counts and the "
                              "full 24-ticker universe for cross-validation/bootstrap too (slow)")
    args = parser.parse_args()
    main(full=args.full)
