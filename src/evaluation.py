"""Out-of-sample R-squared, cross-validation and bootstrap (paper section 3.2).

16-fold adjacent cross-validation and resampling-based estimation errors are
not covered by a library geared at this exact panel/QLIKE setup (checked
scikit-learn's cross_validation module, which assumes i.i.d. folds and offers
no QLIKE scoring and no day-level block resampling of a stacked panel), so
both are implemented directly from the paper's description.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import config


def r2_adj(y: np.ndarray, y_hat: np.ndarray, n_params: int) -> float:
    """Adjusted R^2 = 1 - (1-R^2)*(n-1)/(n-k-1), n_params = k (excluding the
    intercept, which the paper's models all carry)."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    assert ss_tot > 0, "degenerate target (zero variance)"
    r2 = 1.0 - ss_res / ss_tot
    n = len(y)
    return 1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1)


def day_folds(dates: pd.Series, n_folds: int) -> list:
    """Split the sorted distinct trading days into ``n_folds`` adjacent,
    (approximately) equal-length blocks (paper section 3.2: 16 adjacent
    576-day periods for N=9216). Returns a list of boolean masks over
    ``dates`` (one mask per fold), aligned to ``dates``'s index."""
    unique_days = np.sort(dates.unique())
    day_fold_id = pd.Series(
        np.minimum((np.arange(len(unique_days)) * n_folds) // len(unique_days), n_folds - 1),
        index=unique_days,
    )
    fold_of_row = dates.map(day_fold_id).to_numpy()
    return [fold_of_row == f for f in range(n_folds)]


def cross_validate(panel: pd.DataFrame, fit_fn: Callable, predict_fn: Callable,
                    n_params: int, n_folds: int = config.CV_FOLDS_RUN,
                    y_col: str = "r2_next") -> dict:
    """Out-of-sample R^2_adj via ``n_folds`` adjacent day-blocks (paper
    section 3.2): train ``fit_fn`` on the 15 (or n_folds-1) complementary
    folds, predict the held-out fold with ``predict_fn``, pool every held-out
    prediction and compute one adjusted R^2 over the pooled out-of-sample
    residuals (paper: "out-of-sample R-squared via 16-fold cross
    validation").

    fit_fn(train_panel) -> fitted object (whatever predict_fn needs)
    predict_fn(fitted, test_panel) -> np.ndarray of predicted y
    """
    masks = day_folds(panel["date"], n_folds)
    oos_y, oos_pred = [], []
    for mask in masks:
        train, test = panel.loc[~mask], panel.loc[mask]
        if len(test) == 0 or len(train) == 0:
            continue
        fitted = fit_fn(train)
        pred = predict_fn(fitted, test)
        oos_y.append(test[y_col].to_numpy())
        oos_pred.append(pred)
    y = np.concatenate(oos_y)
    pred = np.concatenate(oos_pred)
    return {"r2_adj": r2_adj(y, pred, n_params), "n_obs": len(y), "y": y, "pred": pred}


def day_bootstrap_indices(unique_days: np.ndarray, n_samples: int,
                           rng: np.random.Generator) -> list:
    """Bootstrap resamples of trading days with replacement (paper section
    3.2: "draw samples with replacement from the trading days, same days for
    all assets to preserve cross-correlations"). Returns a list of
    ``n_samples`` arrays of resampled days."""
    n = len(unique_days)
    return [rng.choice(unique_days, size=n, replace=True) for _ in range(n_samples)]


def block_bootstrap_indices(unique_days: np.ndarray, n_blocks: int, block_length: int,
                             n_samples: int, rng: np.random.Generator) -> list:
    """Block-bootstrap resamples of contiguous day blocks (paper section 3.2:
    "18 contiguous blocks of 512 days, resample whole blocks to preserve
    serial dependence"). Returns a list of ``n_samples`` day arrays, each the
    concatenation of ``n_blocks`` resampled blocks (with replacement)."""
    n = len(unique_days)
    usable_blocks = min(n_blocks, n // block_length)
    blocks = [unique_days[i * block_length:(i + 1) * block_length] for i in range(usable_blocks)]
    out = []
    for _ in range(n_samples):
        chosen = rng.integers(0, usable_blocks, size=usable_blocks)
        out.append(np.concatenate([blocks[c] for c in chosen]))
    return out


def bootstrap_errors(panel: pd.DataFrame, statistic_fn: Callable, n_samples: int,
                      block: bool = False, n_blocks: int = config.BLOCK_BOOTSTRAP_BLOCKS_RUN,
                      block_length: int = config.BLOCK_LENGTH_DAYS_RUN,
                      seed: int = config.RANDOM_SEED) -> dict:
    """Estimation errors by bootstrapping (paper section 3.2): resample
    trading days (whole-day or whole-block, so every market keeps the same
    resampled days / blocks and cross-correlations are preserved), recompute
    ``statistic_fn(resampled_panel) -> np.ndarray`` on each resample, and
    report half the difference between the 84th and 16th percentile (+/-1
    std-dev equivalent for a Gaussian) of each statistic.

    Returns {"estimates": array (n_samples, n_stats), "error": array
    (n_stats,)} using numpy's default interpolation for the percentiles.
    """
    rng = np.random.default_rng(seed)
    unique_days = np.sort(panel["date"].unique())
    if block:
        samples = block_bootstrap_indices(unique_days, n_blocks, block_length, n_samples, rng)
    else:
        samples = day_bootstrap_indices(unique_days, n_samples, rng)
    # Keys of groupby(...).indices are pandas Timestamps while ``unique_days``
    # (from .unique()) are numpy datetime64, sometimes at a different
    # resolution (ns vs us); normalize both sides through pd.Timestamp so
    # dict lookups below do not silently miss every day.
    day_to_rows = {pd.Timestamp(k): v for k, v in panel.groupby("date").indices.items()}
    estimates = []
    for days in samples:
        rows = np.concatenate([day_to_rows[pd.Timestamp(d)] for d in days
                                if pd.Timestamp(d) in day_to_rows])
        resampled = panel.iloc[rows]
        estimates.append(statistic_fn(resampled))
    estimates = np.asarray(estimates, dtype=float)
    lo, hi = np.percentile(estimates, [16, 84], axis=0)
    return {"estimates": estimates, "error": (hi - lo) / 2.0}
