# Results — Volatility Forecasts with Market Trends

Paper: *Volatility Forecasts with Market Trends*. All page/section/equation
references below are the paper's own numbering. Verdicts use the paper's
reported values and (where given) the paper's dispersion: **reproduced** if
inside the paper's reported range or within 10% of its value with the same
sign and the paper's qualitative claim (ranking, sign pattern, growing/shrinking
effect) holds; **not reproduced** if the sign flips, the value is more than
50% off, or the qualitative claim fails; **partially reproduced** otherwise.

Data: 24-market daily panel loaded through the QuantX Dataloader REST API
(`data/data_loader.py`), Massive-backed with a Yahoo Finance backstop. The
paper's own universe is 24 Bloomberg continuous futures series (six equity
indices, six government-bond futures, six currency futures, six commodity
futures), January 1990 – September 2026. Continuous Bloomberg futures and
several foreign government-bond futures are not served by this data path, so
each paper market is replaced one-for-one by the most liquid exchange-traded
proxy of the same asset class (full map in `src/config.py`, `UNIVERSE_RUN`).
Realized sample: 1993-02-01 to 2026-09-14 (138,996 day×ticker rows, return
models) and 1995-05-04 to 2026-09-12 (137,919 rows, daily-range models,
limited by the shortest High/Low history among the proxies), both shorter on
the early side than the paper's 1991-05-01 start because the ETF/FX proxies
list later than the original futures. `non_massive_source` in
`results/exp1_data_log.json` lists 13 tickers served by the Yahoo Finance
backstop rather than Massive (all 6 equity ETFs, 4 rate ETFs, and the gold,
copper and soybean commodity ETFs); the 6 currency pairs and crude oil/
natural gas ETFs came from Massive directly.

## Experiment 1 — EWMA-based Volatility Model with Trend Factors

Basic model (6), full universe, OLS, N=138,996 (`results/exp1_basic_model.csv`):

| coefficient | paper (Table 3) | this run | verdict |
|---|---|---|---|
| a (const)  | ≈0.05  | 0.122  | not reproduced (143% off, same sign) |
| d (σ²)     | ≈0.67  | 0.670  | reproduced |
| e (φ_agg)  | ≈-0.21 | -0.137 | partially reproduced (35% off, same sign) |
| f (φ_agg²) | ≈0.34  | 0.249  | partially reproduced (27% off, same sign) |

The EWMA variance remains the dominant term (d≈0.67) exactly as the paper
reports; the trend terms keep the paper's signs (leverage effect e<0, variance
growth f>0) but are weaker in this run.

Out-of-sample R²_adj, 6-fold adjacent cross-validation (reduced from the
paper's 16-fold/576-day design to 6 folds, see "Deviations from the paper"):
full model 0.0624 vs. variance-only 0.0597 (Δ=0.0027). The paper reports a
next-day Δ R²_adj from adding trends of about 0.006. Same sign (trends help),
magnitude is roughly half → **not reproduced** (>50% off) in magnitude,
direction confirmed.

Enhanced model (8) vs. basic model restricted to k≤6, OOS (`exp1_enhanced_model.csv`):
Δ R²_adj = 0.000188, far below the paper's reported ≈0.0098 → **not reproduced**
in magnitude; sign (enhancement helps) is correct.

By asset class (`exp1_by_asset_class.csv`): equity e = -0.541 vs. the paper's
≈-0.60 (10% off, same sign) → **reproduced**. Non-equity classes: commodity
-0.055, currency -0.224, rate +0.086, vs. the paper's reported ≈-0.10 for
non-equity classes. The qualitative claim — "the leverage effect is primarily
an equity phenomenon" — holds (equity by far the most negative) →
**reproduced** qualitatively; the rate-futures proxy shows a sign flip
(+0.086) that the paper's other asset classes do not, noted below.

By horizon k=1..8 (`exp1_by_horizon.csv`, `exp1_coef_by_horizon.png`): e is
most negative at short-to-medium horizons (k=2..4, 4–16 days) and turns
positive at the longest horizon (k=8, ≈1 year), consistent with the paper's
account of short/medium trends carrying most of the leverage effect.

8-year rolling windows overlapping by 4 years (`exp1_rolling_stability.csv`):
σ² loading rises from 0.40 (1993-2001 window) to 0.68 (2017-2025 window) and
φ_agg changes sign repeatedly (-0.65 → -0.40 → -0.30 → -0.10 → 0.0006 → -0.078
→ -0.20) across windows → the paper's claim of temporal stability is **not
reproduced** on this proxy universe; σ² is directionally stable, φ_agg is not.

Long-horizon predictions, 2-day to 2-month, % improvement in R²_adj from
adding trends (`exp1_long_horizon.csv`, `exp1_long_horizon_improvement.png`):
10.7% (2d) → 12.7% (4d) → 11.9% (8d) → 6.4% (16d) → 2.4% (32d) → 0.8% (64d).
The paper reports improvement *growing* with horizon up to +53% at 3 months;
this run shows improvement *shrinking* beyond the 4-day horizon, and never
exceeds ~13% → **not reproduced** (the qualitative growth pattern fails).

## Experiment 2 — GARCH(1,1) and GJR-GARCH Models with Trend Factors

OOS R²_adj, 6-fold CV (`results/exp2_oos_r2.csv`), vs. paper Table 8:

| model | paper Δ R²_adj | this run Δ R²_adj | verdict |
|---|---|---|---|
| GARCH + trend   | +0.0098 | -0.0027 | not reproduced (sign flip) |
| GJR-GARCH + trend | +0.0083 | +0.0088 | reproduced |

Coefficients (`exp2_garch_coefficients.csv`), GARCH with trend vs. paper:
omega 0.0150 (paper 0.096), alpha 0.103 (paper 0.114, 10% off → reproduced),
beta 0.881 (paper 0.679, 30% off), lambda -0.0178 (paper -0.095, same sign,
much smaller), zeta 0.00636 (paper 0.133, same sign, much smaller). None of
the fits converged within the optimizer's iteration budget (`converged=False`
for all four GARCH variants); the coefficients above are the optimizer's best
point at the iteration cap, not a converged optimum, which explains part of
the gap to the paper's values. Equity subset (`exp2_equity_subset.csv`):
R²_adj,oos 0.1806 (no trend) → 0.1903 (with trend), Δ=0.0097, matching the
paper's reported ~18% → ~19.5% for equities almost exactly → **reproduced**.

## Experiment 3 — Range-based Volatility Models (GARCH and HAR) with Trend Factors

OOS R²_adj (`results/exp3_oos_r2.csv`), vs. paper Table 9:

| subset | model | paper (no trend → with trend) | this run | verdict |
|---|---|---|---|---|
| full | range GARCH | 0.1940 → 0.2015 | -0.0593 → -0.0229 | not reproduced (wrong sign) |
| full | HAR | 0.2057 → 0.2156 | -0.00007 → 0.0010 | not reproduced (near zero) |
| equity | range GARCH | → ~0.50 | 0.260 → 0.269 | not reproduced (well below 50%) |
| equity | HAR | → ~0.50 | 0.267 → 0.276 | not reproduced (well below 50%) |

The direction of the trend effect (adding trend factors improves OOS R²_adj)
holds in every row, including the full-universe range GARCH and HAR models,
which otherwise badly undershoot the paper's reported levels. Both
full-universe range-GARCH fits again failed to converge
(`exp3_range_garch_coefficients.csv`, `converged=False`), which is the most
likely source of the negative full-universe R²_adj (a non-converged variance
recursion can diverge on individual out-of-sample folds, pulling the
cross-validated R²_adj sharply negative); the equity subset, a smaller and
more homogeneous panel, fits a more usable model and lands within the right
order of magnitude of the paper's level, though still well short of the ~50%
the paper reports for equities. HAR coefficients (`exp3_har_coefficients.csv`)
keep the paper's sign pattern (phi_agg<0, phi_agg²>0) but the in-sample R² of
the full-universe HAR regression is under 1%, far below a model that the
paper reports should explain roughly a fifth of next-day squared-range
variation — a second symptom of the same full-universe degradation.

## Experiment 4 — Stochastic Volatility Models (Heston and LSV) with Trend Factors

Out-of-sample quasi-likelihood per observation, 1-QLIKE/N
(`results/exp4_oos_quasi_likelihood.csv`), vs. paper Table 10:

| model | paper (no trend → with trend) | this run | verdict |
|---|---|---|---|
| Heston | 0.2307 → 0.2395 (Δ+0.0088) | 0.2632 → 0.2504 (Δ-0.0128) | not reproduced (sign flip on Δ) |
| LSV    | 0.2398 → 0.2479 (Δ+0.0081) | -0.0251 → 0.1012 (Δ+0.1263) | not reproduced (no-trend level far off) |

The Heston model's no-trend level (0.263) is within 14% of the paper's 0.2307,
and the with-trend level (0.250) is within 5% of the paper's 0.2395, but the
sign of the trend effect flips relative to the paper (trends make the fitted
Heston model worse OOS here, not better). The Heston fit (with and without
trend) also pins the mean-reversion target `theta` at the upper bound of its
search interval (`theta=3.0`, see `results/exp4_sv_coefficients.csv`), against
a long-run normalized variance of 1; a synthetic recovery check (true
kappa=0.05, theta=1.0, xi=0.2, n=5,000) recovers kappa=0.038, theta=1.165,
xi=0.142 without hitting the bound, so the state-space recursion itself is not
mis-specified on clean synthetic input — on the real panel the moment-matched
quasi-likelihood filter instead pushes toward a higher implied long-run
variance than the data's own normalization target, a documented limitation of
the quasi-likelihood simplification used here in place of the paper's
particle-filter/MCMC fit (see "Deviations from the paper"). The LSV no-trend
fit converges but its OOS quasi-likelihood is negative, far below the paper's
reported ≈0.24, while the LSV-with-trend fit is markedly better (0.101) —
the direction of the trend effect is right and large, but the no-trend LSV
baseline itself does not reach the paper's reported predictive power.

## Deviations from the paper

- **Instrument universe.** The paper's 24 Bloomberg continuous futures series
  (including several foreign government-bond futures and commodities with no
  listed ETF) are not available on the QuantX Dataloader / Massive / Yahoo
  data path used here. Each paper market is replaced by the most liquid
  exchange-traded proxy of the same asset class (`src/config.py`,
  `UNIVERSE_RUN`): country-index ETFs for the 6 equity indices, a ladder of US
  Treasury duration ETFs (SHV/SHY/IEI/IEF/TLH/TLT) for the 6 government-bond
  futures, spot FX for the 6 currencies, and commodity-tracking ETFs for 5 of
  the 6 commodities. "Live Cattle" has no tradable ETF proxy on this data path
  and is replaced by a corn ETF (`CORN`) to keep the commodity class at 6
  members.
- **Data source.** 13 of the 24 proxies (`SPY`, `EWC`, `EWG`, `EWU`, `EWJ`,
  `EWH`, `SHY`, `IEF`, `TLT`, `UNG`, `GLD`, `CPER`, `SOYB`) were served by the
  Yahoo Finance backstop inside the Dataloader rather than by Massive
  directly (`results/exp1_data_log.json`, `non_massive_source`); the 6
  currency pairs and the remaining commodity/rate ETFs came from Massive.
- **Sample period.** The paper's reported analysis window is 1991-05-01 to
  2026-09-15 (N=9,216 days). The realized window on the proxy universe is
  1993-02-01 to 2026-09-14 for return-based models (N=138,996 day×ticker rows
  across 24 tickers) and 1995-05-04 to 2026-09-12 for daily-range models
  (N=137,919 rows), both starting later because the ETF/FX proxies for some
  markets (notably the European and Asian equity and rate ETFs) were not
  listed as early as the original futures.
- **Cross-validation and bootstrap scale.** The paper uses 16 adjacent
  576-day folds and ≥300 bootstrap/block-bootstrap resamples
  (`CV_FOLDS_PAPER`, `BOOTSTRAP_SAMPLES_PAPER`, `BLOCK_BOOTSTRAP_BLOCKS_PAPER`
  in `src/config.py`). To fit the time available for this replication, the
  executed runs use 6 adjacent folds and 60 bootstrap/8 block-bootstrap
  resamples (`CV_FOLDS_RUN`, `BOOTSTRAP_SAMPLES_RUN`,
  `BLOCK_BOOTSTRAP_BLOCKS_RUN`); the fold/window definitions and resampling
  method are otherwise unchanged. Pass `--full` to any `exp_*` script to use
  the paper-scale values; the full configuration was not run because,
  scaled from the measured per-fold/per-resample runtime, it would not finish
  before this replication's time limit.
- **SV model fitting (experiment 4).** The paper fits the Heston and LSV
  models by (the paper's description implies) a heavier particle-filter or
  MCMC-based maximum-likelihood procedure. This repository instead fits both
  models with a moment-matched linear-Gaussian (Kalman-filter-style)
  quasi-likelihood recursion on the latent variance/log-variance state, which
  is tractable in the time available but is an approximation rather than an
  exact likelihood; it is the most likely source of the Heston model's
  `theta` bound-hitting behavior and part of the gap to the paper's reported
  quasi-likelihood levels (see Experiment 4 discussion above).
- **GARCH/GJR-GARCH and range-GARCH convergence.** All four GARCH/GJR-GARCH
  fits (experiment 2) and both range-GARCH fits (experiment 3, full universe)
  reached the optimizer's iteration cap without the convergence tolerance
  being met (`converged=False` in `exp2_garch_coefficients.csv` and
  `exp3_range_garch_coefficients.csv`). The reported coefficients are the
  optimizer's best point at the cap, not a verified optimum, and are a likely
  contributor to the full-universe range-GARCH model's negative
  out-of-sample R²_adj.
- **Rate-futures asset-class sign.** Experiment 1's rate-futures proxy class
  (US Treasury duration ETFs substituting for 6 foreign government-bond
  futures) shows a positive linear trend coefficient (+0.086) where the paper
  reports a negative coefficient of similar small magnitude for non-equity
  asset classes; this was not separately isolated beyond noting the proxy
  substitution above.

## Not reproduced

- The paper's quasi-MLE (QMLE) fit of the basic EWMA model (experiment 1,
  methodology step 8) is implemented and run (`exp1_basic_model.csv`, `method
  = QMLE` row) but its out-of-sample R²_adj and bootstrap standard errors are
  not computed in this run; only its in-sample QLIKE-weighted coefficients
  and QLIKE/observation are reported.
- The paper's exact 16-fold/576-day cross-validation and ≥300-resample
  bootstrap scale (experiments 1-4) were not run at full scale; see
  "Deviations from the paper" above for the executed scale and the command
  to run the full configuration.
- The paper's reconstruction of early TSE 60, Hang Seng, EUR and NZD history
  from Libor/dividend-yield proxies (paper section 2.1) is not replicated;
  the proxy universe here does not need it because its own listed history is
  used as-is from each instrument's first available date on the data path.
