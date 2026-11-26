# Volatility Forecasts with Market Trends

Replication of selected experiments from **"Volatility Forecasts with Market
Trends"** (page references below refer to this paper's sections, equations,
tables and figures).

The paper asks whether the current strength of a market's price trend, in
addition to a standard EWMA/GARCH variance estimate, helps forecast next-day
(and longer-horizon) variance. It fits linear and quadratic trend-strength
regression terms on top of five families of variance models — EWMA, GARCH /
GJR-GARCH, range-based GARCH / HAR, and the Heston / Leverage Stochastic
Volatility (LSV) models — across 24 futures markets (six each of equity
indices, interest-rate futures, currencies and commodities).

## Repository layout

```
src/        shared library: config, trend factors, EWMA/variance, GARCH,
            range-based models, SV models, regressions, cross-validation,
            bootstrap, panel assembly (flat modules, no sub-packages)
exp/        one script per experiment (exp_1 .. exp_4)
data/       data_loader.py (QuantX Dataloader market data)
tests/      pytest unit tests for src/ and experiment-level tests
results/    aggregate metrics (.csv/.json) and static .png figures + RESULTS.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Running

```bash
python -m pytest -m "not market_data"   # unit tests (no network)
python -m pytest                        # full suite, needs API keys

python -m exp.exp_1   # EWMA model with trend factors
python -m exp.exp_2   # GARCH(1,1) and GJR-GARCH with trend factors
python -m exp.exp_3   # Range-based GARCH and HAR with trend factors
python -m exp.exp_4   # Heston and Leverage SV with trend factors
```

`reproduce.sh` runs the full sequence above from a clean checkout. Each
experiment script downloads/caches the market data it needs (through
`data/data_loader.py`), fits the paper's models, writes aggregate result
tables and static figures to `results/`, and overwrites any earlier run of
the same experiment.

## Data

Daily OHLCV bars for the 24-market universe are loaded through the QuantX
Dataloader REST API (`data/data_loader.py`), which is backed by Massive
market data with a Yahoo Finance backstop for instruments Massive does not
cover. The paper's own universe is 24 Bloomberg continuous futures series
(six equity indices, six government-bond futures, six currency futures, six
commodity futures) from January 1990 to September 2026; continuous
Bloomberg futures and several foreign government-bond futures are not
served by this data path, so each paper market is replaced by the most
liquid exchange-traded proxy available (see `src/config.py` and "Deviations
from the paper" in `results/RESULTS.md` for the full substitution table and
realized sample dates).

## Paper reference

Title: *Volatility Forecasts with Market Trends*. See `results/RESULTS.md`
for the per-experiment comparison between the paper's reported values and
the values produced by this repository.
