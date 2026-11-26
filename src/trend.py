"""Trend strength and normalized returns (paper section 2.2).

Custom implementation of the paper's equations: no library implements this
specific "smooth weight function" trend strength (checked against `ta`,
`statsmodels` and `vectorbt`, none of which offer a matching trend/momentum
t-statistic construction), so it is implemented directly from the paper.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from src import config


def normalization_factor(horizon_days: int) -> float:
    """M_T such that sum_n w_T(n)^2 = 1 for w_T(n) = M_T n exp(-2n/T).

    Derived (paper section 2.2 gives the constraint, not the closed form)
    from sum_{n=0}^inf n^2 x^n = x(1+x)/(1-x)^3 for x = rho^2, rho = exp(-2/T).
    """
    rho = np.exp(-2.0 / horizon_days)
    x = rho ** 2
    s = x * (1.0 + x) / (1.0 - x) ** 3
    return 1.0 / np.sqrt(s)


def trend_strength(r_hat: np.ndarray, horizon_days: int) -> np.ndarray:
    """phi_T(t) = sum_n w_T(n) r_hat(t-n), computed recursively (paper eq.
    preceding (2)). r_hat is the de-meaned, capped normalized return series.

    The weight w_T(n) = M_T n rho^n is linear in n times a geometric term,
    so phi_T can be obtained from two cascaded AR(1) filters:
        A(t) = r_hat(t) + rho*A(t-1)                 (plain EWMA of r_hat)
        B(t) = rho*(B(t-1) + A(t-1))                  (trend accumulator)
        phi_T(t) = M_T * B(t)
    This recursive form follows directly from w_T(n) = M_T n rho^n and is
    equivalent to summing the infinite series exactly (for a causal, zero
    initial-condition filter).
    """
    rho = np.exp(-2.0 / horizon_days)
    m_t = normalization_factor(horizon_days)
    a = lfilter([1.0], [1.0, -rho], r_hat)
    a_lag = np.concatenate(([0.0], a[:-1]))
    b = lfilter([rho], [1.0, -rho], a_lag)
    return m_t * b


def normalized_returns(close: pd.Series) -> pd.Series:
    """R(t) = r(t)/sigma - mu/sigma (paper section 2.2), r(t)=log-return,
    sigma/mu the long-term (full-sample) standard deviation and mean."""
    log_price = np.log(close.astype(float))
    r = log_price.diff()
    sigma = r.std(ddof=1)
    mu = r.mean()
    assert sigma > 0, "degenerate (constant) price series"
    return (r - mu) / sigma


def capped_demeaned_return(normalized_return: pd.Series) -> np.ndarray:
    """R_hat(t) = R(t) - mu/sigma... already de-meaned in normalized_returns;
    here we only apply the +/-20 sigma cap used for trend-strength inputs
    (paper section 2.2). Returns a numpy array with NaNs replaced by 0
    (used only to initialize the recursive filters before data starts)."""
    r = normalized_return.to_numpy(copy=True)
    r = np.nan_to_num(r, nan=0.0)
    np.clip(r, -config.RETURN_CAP_STD, config.RETURN_CAP_STD, out=r)
    return r


def all_trend_strengths(close: pd.Series) -> pd.DataFrame:
    """Trend strengths phi_{i,T_k}(t) for all 8 horizons k=1..8 (paper
    section 2.2), capped/floored at +/-2.5, aligned to ``close.index``."""
    r_norm = normalized_returns(close)
    r_hat = capped_demeaned_return(r_norm)
    out = {}
    for k in config.TREND_HORIZONS_K:
        t_k = config.TREND_HORIZON_DAYS[k]
        phi = trend_strength(r_hat, t_k)
        out[f"phi_{k}"] = np.clip(phi, -config.TREND_CAP, config.TREND_CAP)
    df = pd.DataFrame(out, index=close.index)
    df["r_norm"] = r_norm.to_numpy()
    return df


def aggregate_trend(df: pd.DataFrame, horizons_k) -> tuple:
    """phi_agg(t) and phi_agg^(2)(t) averaged over the given horizons k
    (paper eq. 3)."""
    cols = [f"phi_{k}" for k in horizons_k]
    phi = df[cols].to_numpy()
    agg = phi.mean(axis=1)
    agg2 = (phi ** 2).mean(axis=1)
    return agg, agg2
