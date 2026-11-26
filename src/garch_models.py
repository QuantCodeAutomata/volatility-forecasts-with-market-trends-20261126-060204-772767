"""GARCH(1,1) / GJR-GARCH with trend factors (paper section on GARCH models,
exp_2 methodology steps 4-7), fit on the pooled, normalized multi-market panel.

The ``arch`` package's ``arch_model`` fits a single series' own-parameter
GARCH and has no hook for extra pooled regressors inside the variance
recursion (checked via its documentation: ``VolatilityProcess`` subclassing
would be needed, which is heavier than writing the five-parameter recursion
directly). The recursion and its quasi-maximum-likelihood fit are implemented
directly from the paper's equations.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config


def garch_recursion(r: np.ndarray, phi_agg: np.ndarray, phi_agg2: np.ndarray,
                     omega: float, alpha: float, beta: float,
                     gamma: float = 0.0, lam: float = 0.0, zeta: float = 0.0,
                     sigma2_init: float = 1.0, floor: float = 1e-6) -> np.ndarray:
    """One market's point-forecast recursion (paper eqs. for GARCH(1,1) and
    GJR-GARCH with trends):

        sigma2_hat[0] = omega + alpha*r[0]^2 + gamma*I(r[0]<0)*r[0]^2
                        + beta*sigma2_init + lam*(sigma2_init-1)*phi_agg[0]
                        + zeta*phi_agg2[0]
        sigma2_hat[t] = omega + (alpha + gamma*I(r[t]<0))*r[t]^2
                        + beta*sigma2_hat[t-1]
                        + lam*(sigma2_hat[t-1]-1)*phi_agg[t] + zeta*phi_agg2[t]

    ``sigma2_hat[t]`` is the forecast of ``r[t+1]^2`` (i.e. aligned with the
    panel's ``r2_next`` column at row t). gamma=0 recovers plain GARCH(1,1);
    gamma != 0 is GJR-GARCH's asymmetric term (I(r_t<0)=1 when r_t<0).
    """
    n = len(r)
    sigma2 = np.empty(n)
    r2 = r ** 2
    neg = (r < 0).astype(float)
    prev = sigma2_init
    for t in range(n):
        value = (omega + (alpha + gamma * neg[t]) * r2[t] + beta * prev
                 + lam * (prev - 1.0) * phi_agg[t] + zeta * phi_agg2[t])
        sigma2[t] = value if value > floor else floor
        prev = sigma2[t]
    return sigma2


def _qlike_sum(sigma2_hat: np.ndarray, r2_next: np.ndarray) -> float:
    sigma2_hat = np.clip(sigma2_hat, 1e-8, None)
    return float(np.sum(np.log(sigma2_hat) + r2_next / sigma2_hat))


def _unpack(x: np.ndarray, names: list) -> dict:
    return dict(zip(names, x))


def fit_pooled_garch(panel: pd.DataFrame, with_trend: bool, asymmetric: bool,
                      x0: Optional[dict] = None, maxiter: int = 200) -> dict:
    """Pooled quasi-maximum-likelihood fit of the GARCH(1,1) / GJR-GARCH
    recursion (with or without trend terms) across every market in
    ``panel`` (paper eqs. for GARCH/GJR-GARCH with trends): one shared
    parameter vector, with the recursion run independently per ticker
    (``panel`` must be sorted by ticker then date) and the QLIKE objective
    summed across all tickers and days.
    """
    names = ["omega", "alpha", "beta"]
    if asymmetric:
        names.append("gamma")
    if with_trend:
        names += ["lam", "zeta"]
    bounds = [config.GARCH_PARAM_BOUNDS[n] for n in names]
    if x0 is None:
        x0 = {"omega": 0.05, "alpha": 0.08, "beta": 0.85, "gamma": 0.05, "lam": -0.05, "zeta": 0.05}
    x0_vec = np.array([x0[n] for n in names])

    groups = [g.sort_values("date") for _, g in panel.groupby("ticker", sort=False)]
    r_groups = [g["r"].to_numpy() for g in groups]
    r2n_groups = [g["r2_next"].to_numpy() for g in groups]
    if with_trend:
        agg_groups = [g["phi_agg_enh"].to_numpy() for g in groups]
        agg2_groups = [g["phi_agg2_enh"].to_numpy() for g in groups]
    else:
        agg_groups = [np.zeros_like(r) for r in r_groups]
        agg2_groups = [np.zeros_like(r) for r in r_groups]

    def objective(x):
        p = _unpack(x, names)
        gamma = p.get("gamma", 0.0)
        lam = p.get("lam", 0.0)
        zeta = p.get("zeta", 0.0)
        total = 0.0
        for r, r2n, agg, agg2 in zip(r_groups, r2n_groups, agg_groups, agg2_groups):
            sigma2_hat = garch_recursion(r, agg, agg2, p["omega"], p["alpha"], p["beta"],
                                          gamma=gamma, lam=lam, zeta=zeta)
            total += _qlike_sum(sigma2_hat, r2n)
        return total

    result = minimize(objective, x0_vec, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
    params = pd.Series(_unpack(result.x, names))
    n_obs = sum(len(r) for r in r_groups)
    return {"params": params, "qlike_per_obs": result.fun / n_obs, "n_obs": n_obs,
            "names": names, "converged": result.success}


def predict_pooled_garch(fitted: dict, panel: pd.DataFrame) -> np.ndarray:
    """Point forecast sigma_hat^2 for every row of ``panel`` using the fitted
    pooled parameters, running the recursion per ticker."""
    params = fitted["params"]
    gamma = params.get("gamma", 0.0)
    lam = params.get("lam", 0.0)
    zeta = params.get("zeta", 0.0)
    with_trend = "lam" in fitted["names"]
    out = pd.Series(index=panel.index, dtype=float)
    for _, g in panel.groupby("ticker", sort=False):
        g = g.sort_values("date")
        agg = g["phi_agg_enh"].to_numpy() if with_trend else np.zeros(len(g))
        agg2 = g["phi_agg2_enh"].to_numpy() if with_trend else np.zeros(len(g))
        sigma2_hat = garch_recursion(g["r"].to_numpy(), agg, agg2,
                                      params["omega"], params["alpha"], params["beta"],
                                      gamma=gamma, lam=lam, zeta=zeta)
        out.loc[g.index] = sigma2_hat
    return out.to_numpy()
