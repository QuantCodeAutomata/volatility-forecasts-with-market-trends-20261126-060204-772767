"""Stochastic volatility models with trend factors: Heston (level variance)
and Leverage SV / LSV (log-variance), exp_4 methodology steps 4-8.

No installed library fits a trend-augmented Heston/LSV pair by the paper's
exact state equations (checked ``arch`` - GARCH-family only, no latent SV
state; ``statsmodels.tsa.statespace`` - generic Kalman filter machinery with
no SV-specific measurement equation), so the quasi-likelihood state-space
filter below is implemented directly from the paper's model and the
log(chi-square) quasi-likelihood approximation of Harvey, Ruiz & Shephard
(1994), using statsmodels' linear Kalman recursion formulas re-derived here
for the paper's nonlinear-in-levels (Heston) and linear-in-logs (LSV) cases.

Both models are latent-state processes, so direct OLS/QMLE (as in
garch_models.py) is not applicable: parameters are fit by (quasi-)maximum
likelihood via a linear(ized) Kalman filter on the one-step prediction
errors, exactly as the paper specifies ("Fit model parameters by maximum
likelihood estimation (not OLS, since SV models have latent variance
processes)", implementation_notes). This is a documented simplification of
the paper's heavier particle-filter/MCMC fitting (results/RESULTS.md,
"Deviations from the paper").
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config


def _recursion(y, phi_agg, phi_agg2, const, ar, lam, zeta, xi, obs_offset, obs_var_fn,
               state0, p0):
    """Linear(ized)-Gaussian Kalman filter quasi-likelihood recursion.

    State: x[t] = const + ar*x[t-1] + lam*phi_agg[t-1] + zeta*phi_agg2[t-1]
    + process noise of variance xi^2 (LSV, log-variance) or xi^2*x[t-1]
    (Heston, level variance, state-dependent diffusion). Observation:
    y[t] = obs_offset + x[t] + noise, noise variance given by
    ``obs_var_fn(x_pred)`` (constant for LSV's log(chi^2_1) approximation;
    state-dependent, moment-matched for Heston's multiplicative r_t^2 noise).

    Returns (total negative log-likelihood, one-step-ahead predictions x_pred).
    """
    n = len(y)
    x_filt = state0
    p_filt = p0
    x_pred_arr = np.empty(n)
    neg_loglik = 0.0
    for t in range(n):
        if t == 0:
            x_pred = state0
            p_pred = p0
        else:
            trend_t = lam * phi_agg[t - 1] + zeta * phi_agg2[t - 1]
            x_pred = const + ar * x_filt + trend_t
            q_var = (xi ** 2) * max(x_filt, 1e-6) if obs_var_fn is HESTON_OBS_VAR else xi ** 2
            p_pred = ar ** 2 * p_filt + q_var
        x_pred_arr[t] = x_pred
        innovation = y[t] - obs_offset - x_pred
        r_var = obs_var_fn(x_pred)
        total_var = max(p_pred + r_var, 1e-8)
        neg_loglik += 0.5 * (np.log(2 * np.pi * total_var) + innovation ** 2 / total_var)
        gain = p_pred / total_var
        x_filt = x_pred + gain * innovation
        p_filt = max((1 - gain) * p_pred, 1e-8)
    return neg_loglik, x_pred_arr


def HESTON_OBS_VAR(x_pred: float) -> float:
    """Moment-matched observation noise for y=r^2=sigma^2*z^2, z~N(0,1):
    Var[sigma^2*z^2 | sigma^2] = 2*sigma^4 (paper's Heston model, level
    variance); plugged in at the predicted state (extended-KF linearization).
    """
    return 2.0 * max(x_pred, 1e-6) ** 2


def lsv_obs_var(_x_pred: float) -> float:
    """Observation noise variance for y=log(r^2)=h+log(z^2): Var[log(z^2)]
    for z~N(0,1), constant (paper's Taylor/LSV log-variance model; Harvey,
    Ruiz & Shephard 1994 quasi-likelihood approximation)."""
    return config.LOG_CHI2_1_VAR


def fit_heston(panel_by_ticker: dict, with_trend: bool, x0: dict = None,
               maxiter: int = 100) -> dict:
    """Pooled quasi-ML fit of the Heston-type level-variance recursion
    (paper): sigma2[t] = sigma2[t-1] + kappa*(theta-sigma2[t-1])
    + lam*phi_agg[t-1] + zeta*phi_agg2[t-1] + xi*sigma[t-1]*eps[t],
    observed through y_t = r_t^2 with moment-matched noise 2*sigma2_pred^2.
    ``panel_by_ticker``: {ticker: DataFrame[r, phi_agg, phi_agg2]} sorted by
    date, already restricted to AGG_HORIZONS_BASIC (paper section 4.2: SV
    models use all 8 horizons, not the enhanced k<=6 subset)."""
    names = ["kappa", "theta", "xi"] + (["lam", "zeta"] if with_trend else [])
    bounds = [config.SV_PARAM_BOUNDS[n] for n in names]
    if x0 is None:
        x0 = {"kappa": 0.05, "theta": 1.0, "xi": 0.2, "lam": 0.01, "zeta": 0.02}
    x0_vec = np.array([x0[n] for n in names])

    def objective(x):
        p = dict(zip(names, x))
        lam = p.get("lam", 0.0)
        zeta = p.get("zeta", 0.0)
        total = 0.0
        for g in panel_by_ticker.values():
            y = g["r"].to_numpy() ** 2
            phi_agg = g["phi_agg"].to_numpy()
            phi_agg2 = g["phi_agg2"].to_numpy()
            const = p["kappa"] * p["theta"]
            ar = 1.0 - p["kappa"]
            nll, _ = _recursion(y, phi_agg, phi_agg2, const, ar, lam, zeta, p["xi"],
                                 obs_offset=0.0, obs_var_fn=HESTON_OBS_VAR,
                                 state0=1.0, p0=1.0)
            total += nll
        return total

    result = minimize(objective, x0_vec, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
    params = pd.Series(dict(zip(names, result.x)))
    n_obs = sum(len(g) for g in panel_by_ticker.values())
    return {"params": params, "neg_loglik": result.fun, "n_obs": n_obs,
            "names": names, "converged": result.success, "kind": "heston"}


def fit_lsv(panel_by_ticker: dict, with_trend: bool, x0: dict = None,
            maxiter: int = 100) -> dict:
    """Pooled quasi-ML fit of the Leverage-SV (log-variance) recursion
    (paper): h[t]-mu = beta*(h[t-1]-mu) + lam*phi_agg[t-1] + zeta*phi_agg2[t-1]
    + xi*eps[t], observed through y_t=log(r_t^2) with the exact
    Harvey-Ruiz-Shephard (1994) Gaussian quasi-likelihood for log(z_t^2)."""
    names = ["mu", "beta", "xi"] + (["lam", "zeta"] if with_trend else [])
    bounds = [config.SV_PARAM_BOUNDS[n] for n in names]
    if x0 is None:
        x0 = {"mu": -0.1, "beta": 0.95, "xi": 0.2, "lam": 0.01, "zeta": 0.02}
    x0_vec = np.array([x0[n] for n in names])

    def objective(x):
        p = dict(zip(names, x))
        lam = p.get("lam", 0.0)
        zeta = p.get("zeta", 0.0)
        total = 0.0
        for g in panel_by_ticker.values():
            r2 = np.clip(g["r"].to_numpy() ** 2, 1e-10, None)
            y = np.log(r2)
            phi_agg = g["phi_agg"].to_numpy()
            phi_agg2 = g["phi_agg2"].to_numpy()
            const = p["mu"] * (1.0 - p["beta"])
            nll, _ = _recursion(y, phi_agg, phi_agg2, const, p["beta"], lam, zeta, p["xi"],
                                 obs_offset=config.LOG_CHI2_1_MEAN, obs_var_fn=lsv_obs_var,
                                 state0=p["mu"], p0=1.0)
            total += nll
        return total

    result = minimize(objective, x0_vec, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
    params = pd.Series(dict(zip(names, result.x)))
    n_obs = sum(len(g) for g in panel_by_ticker.values())
    return {"params": params, "neg_loglik": result.fun, "n_obs": n_obs,
            "names": names, "converged": result.success, "kind": "lsv"}


def leverage_correlation(r: np.ndarray, sigma2_hat: np.ndarray) -> float:
    """Diagnostic leverage correlation rho = corr(z_t, next day's variance
    change), z_t = r_t/sqrt(sigma2_hat_t) (paper's rho = cor(z_t, epsilon_t)).
    The Kalman filters above drop rho from the likelihood for tractability
    (documented simplification, results/RESULTS.md); this reports it
    separately from the filtered state path as a diagnostic only."""
    z = r[:-1] / np.sqrt(np.clip(sigma2_hat[:-1], 1e-8, None))
    d_sigma2 = np.diff(sigma2_hat)
    if np.std(z) == 0 or np.std(d_sigma2) == 0:
        return float("nan")
    return float(np.corrcoef(z, d_sigma2)[0, 1])


def predict_sv(fitted: dict, panel_by_ticker: dict) -> dict:
    """One-step-ahead sigma_hat^2 (paper's evaluation variable) for every
    ticker: for Heston it is the predicted state directly; for LSV it is
    exp(predicted log-variance) (back-transformed point forecast, ignoring
    the small log-normal bias-correction term, a documented simplification).
    """
    p = fitted["params"]
    lam = p.get("lam", 0.0)
    zeta = p.get("zeta", 0.0)
    out = {}
    for ticker, g in panel_by_ticker.items():
        phi_agg = g["phi_agg"].to_numpy()
        phi_agg2 = g["phi_agg2"].to_numpy()
        if fitted["kind"] == "heston":
            y = g["r"].to_numpy() ** 2
            const, ar = p["kappa"] * p["theta"], 1.0 - p["kappa"]
            _, x_pred = _recursion(y, phi_agg, phi_agg2, const, ar, lam, zeta, p["xi"],
                                    0.0, HESTON_OBS_VAR, 1.0, 1.0)
            out[ticker] = np.clip(x_pred, 1e-6, None)
        else:
            r2 = np.clip(g["r"].to_numpy() ** 2, 1e-10, None)
            y = np.log(r2)
            const = p["mu"] * (1.0 - p["beta"])
            _, x_pred = _recursion(y, phi_agg, phi_agg2, const, p["beta"], lam, zeta, p["xi"],
                                    config.LOG_CHI2_1_MEAN, lsv_obs_var, p["mu"], 1.0)
            out[ticker] = np.exp(x_pred)
    return out
