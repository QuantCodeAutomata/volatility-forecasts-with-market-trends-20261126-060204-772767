#!/usr/bin/env bash
# Runs the test suite and every experiment of
# "Volatility Forecasts with Market Trends" from a clean checkout.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4
