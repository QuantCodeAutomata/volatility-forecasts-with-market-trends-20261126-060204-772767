"""Experiment 1: EWMA-based Volatility Model with Trend Factors.

Reproduces the paper's basic next-day variance model (eq. 6), its
quasi-maximum-likelihood fit (eq. 5), the coefficient breakdown by trend
horizon and asset class, the 8-year rolling-window stability check, the
variance-dependent enhanced model (eq. 8) and the longer-horizon (up to
3-month) forecasts of section 3.6.

Run: python -m exp.exp_1 [--full]
``--full`` uses the paper's cross-validation/bootstrap repetition counts
(config.CV_FOLDS_PAPER, config.BOOTSTRAP_SAMPLES_PAPER); by default the
reduced counts in src/config.py are used (see results/RESULTS.md,
"Deviations from the paper").
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
from src import config, evaluation, panel as panel_mod, regression_models

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BASIC_COLS = ["const", "sigma2", "phi_agg", "phi_agg2"]
ENHANCED_COLS = ["const", "sigma2", "phi_sigma", "phi_agg2"]


def _params_row(params: pd.Series, columns: list) -> dict:
    return {c: float(params.get(c, np.nan)) for c in columns}


def _oos_r2(panel: pd.DataFrame, variant: str, n_folds: int) -> tuple:
    n_params = len(regression_models.basic_design(panel, variant).columns)
    cv = evaluation.cross_validate(
        panel,
        fit_fn=lambda tr: regression_models.fit_ols(tr, variant=variant),
        predict_fn=lambda fitted, te: regression_models.predict(
            fitted["params"], regression_models.basic_design(te, variant)),
        n_params=n_params, n_folds=n_folds,
    )
    return cv["r2_adj"], cv["n_obs"]


def run_basic_and_qmle(panel: pd.DataFrame, n_folds: int, n_boot: int) -> dict:
    """Methodology steps 6-8: basic model (full/trend-only/variance-only) by
    OLS, its out-of-sample R^2_adj, and the QMLE fit of the full model."""
    rows = []
    for variant in ["full", "trend_only", "variance_only"]:
        ols = regression_models.fit_ols(panel, variant=variant)
        in_sample_r2 = evaluation.r2_adj(ols["y"], ols["fitted"], len(ols["design"].columns))
        oos_r2, oos_n = _oos_r2(panel, variant, n_folds)
        row = {"variant": variant, "method": "OLS", "n_obs": len(ols["y"]),
               "r2_adj_in_sample": in_sample_r2, "r2_adj_oos": oos_r2, "oos_n_obs": oos_n}
        row.update(_params_row(ols["params"], BASIC_COLS))
        rows.append(row)

    def stat_fn(p):
        return regression_models.fit_ols(p, variant="full")["params"].reindex(BASIC_COLS).to_numpy()

    boot = evaluation.bootstrap_errors(panel, stat_fn, n_samples=n_boot, block=False)
    se = dict(zip(BASIC_COLS, boot["error"]))

    qmle = regression_models.fit_qmle(panel, variant="full")
    qmle_row = {"variant": "full", "method": "QMLE", "n_obs": len(qmle["y"]),
                "r2_adj_in_sample": np.nan, "r2_adj_oos": np.nan, "oos_n_obs": np.nan,
                "qlike_per_obs": qmle["qlike_per_obs"]}
    qmle_row.update(_params_row(qmle["params"], BASIC_COLS))
    rows.append(qmle_row)

    table = pd.DataFrame(rows)
    for c in BASIC_COLS:
        table.loc[table["variant"].eq("full") & table["method"].eq("OLS"), f"{c}_boot_se"] = se[c]
    return {"table": table, "bootstrap_estimates": boot["estimates"]}


def run_by_horizon(panel: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Methodology step 12: repeat the basic regression with a single trend
    horizon k (phi_k, phi_k^2) replacing the 8-horizon aggregate."""
    rows = []
    for k in config.TREND_HORIZONS_K:
        single = panel.copy()
        single["phi_agg"] = single[f"phi_{k}"]
        single["phi_agg2"] = single[f"phi_{k}"] ** 2
        ols = regression_models.fit_ols(single, variant="full")
        in_sample_r2 = evaluation.r2_adj(ols["y"], ols["fitted"], len(ols["design"].columns))
        row = {"k": k, "horizon_days": config.TREND_HORIZON_DAYS[k], "n_obs": len(ols["y"]),
               "r2_adj_in_sample": in_sample_r2}
        row.update(_params_row(ols["params"], BASIC_COLS))
        rows.append(row)
    return pd.DataFrame(rows)


def run_by_asset_class(panel: pd.DataFrame) -> pd.DataFrame:
    """Methodology step 11: repeat the basic regression separately per asset
    class (equities, rates, currencies, commodities)."""
    rows = []
    for asset_class, group in panel.groupby("asset_class"):
        if len(group) < 50:
            continue
        ols = regression_models.fit_ols(group, variant="full")
        in_sample_r2 = evaluation.r2_adj(ols["y"], ols["fitted"], len(ols["design"].columns))
        row = {"asset_class": asset_class, "n_obs": len(ols["y"]), "r2_adj_in_sample": in_sample_r2}
        row.update(_params_row(ols["params"], BASIC_COLS))
        rows.append(row)
    return pd.DataFrame(rows)


def run_rolling_stability(panel: pd.DataFrame, window_years: int = 8, step_years: int = 4) -> pd.DataFrame:
    """Methodology step 13: 8-year rolling windows overlapping by 4 years
    (paper section 3.4). Window boundaries are derived from the realized
    sample's date range (start of 8-year windows advanced in 4-year steps
    until the window would end after the last available date)."""
    start = panel["date"].min()
    end = panel["date"].max()
    rows = []
    window_start = start
    while window_start + pd.DateOffset(years=window_years) <= end + pd.Timedelta(days=1):
        window_end = window_start + pd.DateOffset(years=window_years)
        mask = (panel["date"] >= window_start) & (panel["date"] < window_end)
        group = panel.loc[mask]
        if len(group) > 200:
            ols = regression_models.fit_ols(group, variant="full")
            row = {"window_start": str(window_start.date()), "window_end": str(window_end.date()),
                   "n_obs": len(ols["y"])}
            row.update(_params_row(ols["params"], BASIC_COLS))
            rows.append(row)
        window_start = window_start + pd.DateOffset(years=step_years)
    return pd.DataFrame(rows)


def run_enhanced_model(panel: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Methodology step 9: enhanced model (eq. 8) with the variance-dependent
    trend interaction, compared in-sample and out-of-sample against the
    basic model, both restricted to the k<=6 trend horizons used by the
    enhanced model (paper section 3.5)."""
    basic_enh = panel.copy()
    basic_enh["phi_agg"] = basic_enh["phi_agg_enh"]
    basic_enh["phi_agg2"] = basic_enh["phi_agg2_enh"]
    ols = regression_models.fit_ols(basic_enh, variant="full")
    basic_in = evaluation.r2_adj(ols["y"], ols["fitted"], len(ols["design"].columns))
    basic_oos, _ = _oos_r2(basic_enh, "full", n_folds)

    enh = regression_models.fit_enhanced_ols(panel)
    enh_in = evaluation.r2_adj(enh["y"], enh["fitted"], len(enh["design"].columns))

    def enh_fit_fn(tr):
        return regression_models.fit_enhanced_ols(tr)

    def enh_predict_fn(fitted, te):
        from src.regression_models import _design_enhanced
        design = _design_enhanced(te)
        return regression_models.predict(fitted["params"], design)

    cv = evaluation.cross_validate(panel, enh_fit_fn, enh_predict_fn,
                                    n_params=len(enh["design"].columns), n_folds=n_folds)
    rows = [
        {"model": "basic (k<=6)", "r2_adj_in_sample": basic_in, "r2_adj_oos": basic_oos},
        {"model": "enhanced", "r2_adj_in_sample": enh_in, "r2_adj_oos": cv["r2_adj"]},
    ]
    table = pd.DataFrame(rows)
    table["delta_r2_adj_oos"] = table["r2_adj_oos"] - table.loc[0, "r2_adj_oos"]
    return table


def run_long_horizon(panel: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    """Methodology step 10 / section 3.6: predictions of the T=2^k-day
    realized variance for k=1..6, with and without trend factors, out-of-
    sample R^2_adj and the percentage improvement from adding trends."""
    rows = []
    for k in range(1, 7):
        long_panel = panel_mod.build_long_horizon_panel(panel, k)
        no_trend, _ = _oos_r2(long_panel, "variance_only", n_folds)
        with_trend, _ = _oos_r2(long_panel, "full", n_folds)
        pct = (with_trend - no_trend) / abs(no_trend) * 100.0 if no_trend != 0 else np.nan
        rows.append({
            "k": k, "horizon_days": config.TREND_HORIZON_DAYS[k],
            "r2_adj_oos_no_trend": no_trend, "r2_adj_oos_with_trend": with_trend,
            "pct_improvement": pct, "n_obs": len(long_panel),
        })
    return pd.DataFrame(rows)


def main(full: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    n_folds = config.CV_FOLDS_PAPER if full else config.CV_FOLDS_RUN
    n_boot = config.BOOTSTRAP_SAMPLES_PAPER if full else config.BOOTSTRAP_SAMPLES_RUN

    t0 = time.time()
    per_ticker, meta = load_universe(config.UNIVERSE_RUN, config.SAMPLE_START, config.SAMPLE_END,
                                      identity_date=config.IDENTITY_DATE, use_cache=True)
    panel = panel_mod.build_panel(per_ticker, config.UNIVERSE_RUN)
    load_elapsed = time.time() - t0
    print(f"panel built in {load_elapsed:.1f}s: {len(panel)} rows, "
          f"{panel['ticker'].nunique()} tickers, {panel['date'].min()} to {panel['date'].max()}")

    data_log = {
        "n_obs": int(len(panel)), "n_tickers": int(panel["ticker"].nunique()),
        "date_min": str(panel["date"].min().date()), "date_max": str(panel["date"].max().date()),
        "tickers": sorted(panel["ticker"].unique().tolist()),
        "failed": meta.get("failed", {}), "non_massive_source": meta.get("non_massive_source", {}),
        "cv_folds": n_folds, "bootstrap_samples": n_boot,
    }

    basic = run_basic_and_qmle(panel, n_folds, n_boot)
    basic["table"].to_csv(RESULTS_DIR / "exp1_basic_model.csv", index=False)

    by_horizon = run_by_horizon(panel, n_folds)
    by_horizon.to_csv(RESULTS_DIR / "exp1_by_horizon.csv", index=False)

    by_class = run_by_asset_class(panel)
    by_class.to_csv(RESULTS_DIR / "exp1_by_asset_class.csv", index=False)

    rolling = run_rolling_stability(panel)
    rolling.to_csv(RESULTS_DIR / "exp1_rolling_stability.csv", index=False)

    enhanced = run_enhanced_model(panel, n_folds)
    enhanced.to_csv(RESULTS_DIR / "exp1_enhanced_model.csv", index=False)

    long_horizon = run_long_horizon(panel, n_folds)
    long_horizon.to_csv(RESULTS_DIR / "exp1_long_horizon.csv", index=False)

    data_log["elapsed_seconds"] = time.time() - t0
    with open(RESULTS_DIR / "exp1_data_log.json", "w") as f:
        json.dump(data_log, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(by_horizon["horizon_days"], by_horizon["phi_agg"], marker="o", label="e (linear)")
    ax.plot(by_horizon["horizon_days"], by_horizon["phi_agg2"], marker="s", label="f (quadratic)")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("trend horizon T (business days)")
    ax.set_ylabel("coefficient")
    ax.set_title("Basic model coefficients by trend horizon (paper section 3.4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_coef_by_horizon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(long_horizon["horizon_days"], long_horizon["r2_adj_oos_no_trend"], marker="o", label="EWMA only")
    ax.plot(long_horizon["horizon_days"], long_horizon["r2_adj_oos_with_trend"], marker="s", label="EWMA + trends")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("prediction horizon T (business days)")
    ax.set_ylabel("out-of-sample R^2_adj")
    ax.set_title("Longer-horizon forecasts: improvement from trend factors (section 3.6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_long_horizon_improvement.png", dpi=150)
    plt.close(fig)

    print(basic["table"].to_string(index=False))
    print(long_horizon.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                         help="use the paper's CV/bootstrap repetition counts instead of the reduced run configuration")
    args = parser.parse_args()
    main(full=args.full)
