"""EWMA-based regression models (paper section 3): basic model (6), pure
trend / pure variance variants, Gaussian QMLE (eq. 5), and the enhanced
model (8) with variance-dependent trend terms.

statsmodels.api.OLS is used for the OLS fits (library already covers plain
linear regression; no need for a custom normal-equations implementation).
The QMLE iterative weighted least squares and the enhanced-model regressor
construction are custom (paper equations 5 and 8).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config


def _design_basic(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "sigma2": panel["sigma2"].to_numpy(),
        "phi_agg": panel["phi_agg"].to_numpy(),
        "phi_agg2": panel["phi_agg2"].to_numpy(),
    })


def _design_enhanced(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "sigma2": panel["sigma2"].to_numpy(),
        "phi_sigma": panel["phi_agg_enh"].to_numpy() * (panel["sigma2"].to_numpy() - 1.0),
        "phi_agg2": panel["phi_agg2_enh"].to_numpy(),
    })


def fit_ols(panel: pd.DataFrame, variant: str = "full") -> dict:
    """OLS fit of the next-day variance model (paper eq. 6):
    r_{t+1}^2 = a + d*sigma_t^2 + e*phi_agg + f*phi_agg^2 + eps.

    variant: "full" (all three factors), "trend_only" (drop sigma2),
    "variance_only" (drop phi_agg/phi_agg2).
    """
    design = _design_basic(panel)
    y = panel["r2_next"].to_numpy()
    if variant == "trend_only":
        design = design[["phi_agg", "phi_agg2"]]
    elif variant == "variance_only":
        design = design[["sigma2"]]
    elif variant != "full":
        raise ValueError(f"unknown variant {variant}")
    x = sm.add_constant(design)
    model = sm.OLS(y, x, missing="drop").fit()
    return {"model": model, "params": model.params, "fitted": model.fittedvalues,
            "resid": model.resid, "design": design, "y": y}


def fit_enhanced_ols(panel: pd.DataFrame) -> dict:
    """OLS fit of the enhanced model (paper eq. 8):
    r_{t+1}^2 = a + d*sigma_t^2 + e*phi_agg*(sigma_t^2-1) + f*phi_agg^2 + eps,
    using only trend horizons k=1..6 (paper section 3.5)."""
    design = _design_enhanced(panel)
    y = panel["r2_next"].to_numpy()
    x = sm.add_constant(design)
    model = sm.OLS(y, x, missing="drop").fit()
    return {"model": model, "params": model.params, "fitted": model.fittedvalues,
            "resid": model.resid, "design": design, "y": y}


def predict(params: pd.Series, design: pd.DataFrame) -> np.ndarray:
    x = sm.add_constant(design, has_constant="add")
    x = x[params.index]
    return x.to_numpy() @ params.to_numpy()


def qlike(sigma2_hat: np.ndarray, r2: np.ndarray) -> float:
    """Gaussian quasi-likelihood loss, summed (paper eq. 5):
    QLIKE = sum[log(sigma_hat^2) + r^2/sigma_hat^2]."""
    sigma2_hat = np.clip(sigma2_hat, 1e-6, None)
    return float(np.sum(np.log(sigma2_hat) + r2 / sigma2_hat))


def fit_qmle(panel: pd.DataFrame, variant: str = "full", max_iter: int = 50,
             tol: float = 1e-6) -> dict:
    """Gaussian QMLE of model (6) by iteratively re-weighted least squares
    (paper section 3.3): weights w_t = 1/sigma_hat_{t+1}^2, re-estimated
    after each step until the coefficients converge."""
    design = _design_basic(panel)
    if variant == "trend_only":
        design = design[["phi_agg", "phi_agg2"]]
    elif variant == "variance_only":
        design = design[["sigma2"]]
    elif variant != "full":
        raise ValueError(f"unknown variant {variant}")
    y = panel["r2_next"].to_numpy()
    x = sm.add_constant(design).to_numpy()

    ols = sm.OLS(y, x).fit()
    params = ols.params.copy()
    for _ in range(max_iter):
        sigma2_hat = np.clip(x @ params, 1e-6, None)
        w = 1.0 / sigma2_hat
        wls = sm.WLS(y, x, weights=w).fit()
        new_params = wls.params
        if np.max(np.abs(new_params - params)) < tol:
            params = new_params
            break
        params = new_params
    sigma2_hat = np.clip(x @ params, 1e-6, None)
    params = pd.Series(params, index=sm.add_constant(design).columns)
    return {"params": params, "fitted": sigma2_hat, "y": y, "design": design,
            "qlike_per_obs": qlike(sigma2_hat, y) / len(y)}


def longer_horizon_target(panel: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Cumulative realized variance over the next ``horizon_days`` days,
    sum_{s=1}^{T} r_{t+s}^2 (paper section 3.6), computed per ticker so no
    cumulative sum crosses an asset boundary. ``panel`` must contain a "r"
    (today's normalized return) column and a "ticker" column, sorted by date
    within each ticker. Returns a Series aligned with ``panel.index``, with
    NaN for the last ``horizon_days`` rows of each ticker (look-ahead window
    not available)."""
    r2 = panel["r"].to_numpy() ** 2
    target = np.full(len(panel), np.nan)
    for _, idx in panel.groupby("ticker").groups.items():
        pos = panel.index.get_indexer(idx)
        series = r2[pos]
        n = len(series)
        # sum_{s=1}^{T} r2[t+s] = cumsum(r2)[t+T] - cumsum(r2)[t], 0 beyond range
        cum = np.concatenate(([0.0], np.cumsum(series)))
        result = np.full(n, np.nan)
        valid = n - horizon_days
        if valid > 0:
            result[:valid] = cum[horizon_days:horizon_days + valid] - cum[:valid]
        target[pos] = result
    return pd.Series(target, index=panel.index)
