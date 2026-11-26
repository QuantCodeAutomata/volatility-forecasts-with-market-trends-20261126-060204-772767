# Repository guide

This repository reproduces selected experiments from "Volatility Forecasts
with Market Trends": an EWMA variance model, GARCH(1,1)/GJR-GARCH, range-based
GARCH/HAR, and Heston/Leverage-SV stochastic volatility models, each with and
without linear/quadratic market-trend-strength regression terms, fitted on a
24-market daily futures panel.

## Layout

- `src/` — flat library modules (no sub-packages): `config.py` (all paper
  parameters, grouped by topic, each with a unit and source comment),
  `trend.py` (trend-strength factors), `variance.py` (EWMA variance, returns
  normalization), `garch_models.py` (GARCH/GJR-GARCH recursion and fitting),
  `range_models.py` (range-based GARCH and HAR), `sv_models.py` (Heston/LSV
  state-space quasi-likelihood filter), `regression_models.py` (OLS/QMLE
  regressions for the EWMA model), `evaluation.py` (16-fold adjacent
  cross-validation and day/block bootstrap standard errors), `panel.py`
  (assembling the per-(day, ticker) data table from loaded bars).
- `exp/` — one script per experiment, `exp_1.py`..`exp_4.py`, run as
  `python -m exp.exp_N`. Each calls only the `src/` functions above; no
  private data loaders or performance shortcuts live here.
- `data/data_loader.py` — the only place that calls the QuantX Dataloader
  market-data API.
- `tests/` — all pytest tests; `conftest.py` provides a session-scoped,
  cached, reduced real-data fixture for tests marked `market_data`.
- `results/` — aggregate metric tables (`.csv`/`.json`) and static `.png`
  figures written by the experiment scripts, plus `RESULTS.md`.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env            # DATALOADER_API_KEY, MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network
python -m pytest                        # full suite, needs API keys

python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
```

`reproduce.sh` runs the test suite and all four experiments in sequence.

## Conventions

- Every experiment parameter lives in `src/config.py` as an UPPER_CASE
  module-level variable with a comment on its meaning, unit and paper
  section (or "not specified in the paper"). Where a parameter was reduced
  to fit the time available for this replication, both the paper value
  (`..._PAPER`) and the executed value (`..._RUN`) are kept side by side.
- Market data is loaded only through `data/data_loader.py`; no other module
  calls an external data API. Synthetic/test fixtures are generated in
  `tests/conftest.py`.
- Tests marked `@pytest.mark.market_data` use a cached, reduced real-data
  slice and are skipped automatically when `DATALOADER_API_KEY` is absent.
  Run `python -m pytest -m "not market_data"` for a network-free check.
- API keys are read from a local, uncommitted `.env` via `python-dotenv` and
  never appear in code, logs, or `results/`.
- Running an experiment script overwrites its files in `results/`. The
  numbers and verdicts written by hand in `results/RESULTS.md` must be
  re-checked against the freshly written files after any rerun; they are not
  regenerated automatically.
