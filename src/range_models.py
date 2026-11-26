"""Range-based GARCH(1,1) and HAR models with trend factors (paper section on
range-based models, exp_3 methodology steps 4-8).

The squared-range GARCH recursion reuses ``garch_models.garch_recursion``
with the squared range R_t^2 substituted for r_t^2 (paper: "replacing r_t^2
with R_t^2"). The HAR regression is linear, fit with statsmodels.api.OLS (no
custom normal-equations code needed).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config, garch_models


def squared_range(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Un-normalized ((H-L)/(H+L))^2 (paper section 4.3); the per-asset
    normalization factor N (E[R^2]=1) is applied by ``normalized_range``."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    valid = ~(np.isnan(high) | np.isnan(low))
    assert np.all(high[valid] >= low[valid]), "high must be >= low"
    hl = (high - low) / (high + low)
    return hl ** 2


def normalized_range(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """R_t^2 = N * ((H_t-L_t)/(H_t+L_t))^2, N chosen so that E[R^2]=1 over
    the series (paper section 4.3)."""
    raw = squared_range(high, low)
    mean = np.nanmean(raw)
    assert mean > 0, "degenerate (zero) high-low range"
    return raw / mean


def fit_range_garch(panel: pd.DataFrame, with_trend: bool, x0: dict = None,
                     maxiter: int = 200) -> dict:
    """Pooled quasi-ML fit of the range-based GARCH(1,1) recursion using
    R_t^2 (column ``range2``) as the shock, R_{t+1}^2 (``range2_next``) as
    the regression target (paper: range-GARCH baseline/with-trends
    equations). Thin wrapper over ``garch_models.fit_pooled_garch`` on a
    panel whose "r"/"r2_next" columns are aliased to the squared range."""
    aliased = panel.copy()
    aliased["r"] = np.sqrt(aliased["range2"].to_numpy())  # garch_recursion squares r internally
    aliased["r2_next"] = aliased["range2_next"].to_numpy()
    return garch_models.fit_pooled_garch(aliased, with_trend=with_trend, asymmetric=False,
                                          x0=x0, maxiter=maxiter)


def predict_range_garch(fitted: dict, panel: pd.DataFrame) -> np.ndarray:
    aliased = panel.copy()
    aliased["r"] = np.sqrt(aliased["range2"].to_numpy())
    return garch_models.predict_pooled_garch(fitted, aliased)


def _har_design(panel: pd.DataFrame, with_trend: bool) -> pd.DataFrame:
    cols = {
        "daily": panel["range2"].to_numpy(),
        "weekly": panel["range2_w"].to_numpy(),
        "monthly": panel["range2_m"].to_numpy(),
    }
    if with_trend:
        cols["phi_agg"] = panel["phi_agg"].to_numpy()
        cols["phi_agg2"] = panel["phi_agg2"].to_numpy()
    return pd.DataFrame(cols, index=panel.index)


def fit_har(panel: pd.DataFrame, with_trend: bool) -> dict:
    """HAR regression of next-day squared range on daily/weekly/monthly
    squared-range averages (paper eq. (9)), optionally with the simplest
    model's trend factors phi_agg, phi_agg^2 (paper: "HAR model with trends
    uses the simplest model (6) factors")."""
    design = _har_design(panel, with_trend)
    y = panel["range2_next"].to_numpy()
    x = sm.add_constant(design)
    model = sm.OLS(y, x, missing="drop").fit()
    return {"model": model, "params": model.params, "design": design, "y": y,
            "with_trend": with_trend}


def predict_har(fitted: dict, panel: pd.DataFrame) -> np.ndarray:
    design = _har_design(panel, fitted["with_trend"])
    x = sm.add_constant(design, has_constant="add")
    x = x[fitted["params"].index]
    return x.to_numpy() @ fitted["params"].to_numpy()


def har_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add range2 (today), range2_w (trailing 5-day mean), range2_m
    (trailing 22-day mean) and the next-day targets range2_next /
    range2_w_next / range2_m_next, computed per ticker without look-ahead
    (paper eq. (9): beta_w uses (1/5) sum_{i=0}^{4} R_{t-i}^2, i.e. the
    trailing window ending at t, inclusive)."""
    out = []
    for _, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        g["range2_w"] = g["range2"].rolling(config.HAR_WEEKLY_LAG, min_periods=config.HAR_WEEKLY_LAG).mean()
        g["range2_m"] = g["range2"].rolling(config.HAR_MONTHLY_LAG, min_periods=config.HAR_MONTHLY_LAG).mean()
        g["range2_next"] = g["range2"].shift(-1)
        out.append(g)
    panel = pd.concat(out, ignore_index=True)
    return panel.dropna(subset=["range2_w", "range2_m", "range2_next"]).reset_index(drop=True)
