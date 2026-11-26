"""Shared fixtures: synthetic OHLC data and a small, cached real-data slice.

Synthetic data lives only here (tests/), since no experiment of this
replication needs synthetic market data; the experiments all run on the
Dataloader/Massive panel built by data/data_loader.py and src/panel.py.
"""

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: requires DATALOADER_API_KEY/MASSIVE_API_KEY and network access"
    )


@pytest.fixture
def synthetic_close() -> pd.Series:
    """A single simulated GBM-like price path with a mild upward drift and a
    GARCH-like volatility cluster, long enough to warm up the longest (256
    business day) trend-strength filter and leave a reported sample."""
    rng = np.random.default_rng(42)
    n = 2600
    sigma = np.empty(n)
    sigma[0] = 0.01
    shocks = rng.standard_normal(n)
    returns = np.empty(n)
    for t in range(n):
        if t > 0:
            sigma[t] = np.sqrt(0.02 + 0.9 * sigma[t - 1] ** 2 + 0.05 * returns[t - 1] ** 2)
        returns[t] = sigma[t] * shocks[t]
    # Inject a clear upward trend into the second half to test phi's sign.
    returns[1500:] += 0.002
    price = 100.0 * np.exp(np.cumsum(returns))
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(price, index=dates, name="close")


@pytest.fixture
def synthetic_ohlc(synthetic_close) -> pd.DataFrame:
    """OHLC frame built around ``synthetic_close`` with a plausible, bounded
    intraday high/low range for the range-based model tests."""
    rng = np.random.default_rng(7)
    close = synthetic_close
    spread = 0.004 * close
    high = close + spread * rng.uniform(0.2, 1.0, size=len(close))
    low = close - spread * rng.uniform(0.2, 1.0, size=len(close))
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


@pytest.fixture
def synthetic_universe(synthetic_ohlc) -> tuple:
    """A tiny two-ticker universe (one equity-like, one independent series)
    built from ``synthetic_ohlc`` plus an independently-seeded second series,
    for panel-level tests that need more than one ticker."""
    rng = np.random.default_rng(99)
    n = len(synthetic_ohlc)
    shocks = rng.standard_normal(n) * 0.012
    price2 = 50.0 * np.exp(np.cumsum(shocks))
    close2 = pd.Series(price2, index=synthetic_ohlc.index)
    spread2 = 0.004 * close2
    ohlc2 = pd.DataFrame({
        "open": close2.shift(1).fillna(close2.iloc[0]),
        "high": close2 + spread2 * 0.6,
        "low": close2 - spread2 * 0.6,
        "close": close2,
    })
    per_ticker = {"SYN_A": synthetic_ohlc, "SYN_B": ohlc2}
    universe = {"SYN_A": "equity", "SYN_B": "rate"}
    return per_ticker, universe


@pytest.fixture(scope="session")
def market_data_slice():
    """A small real-data slice loaded through data/data_loader.py: two
    tickers, two years, used only by tests marked ``market_data``. The loader
    itself caches the download under /tmp, so repeated test runs do not
    re-download it."""
    if not (os.environ.get("DATALOADER_API_KEY") and os.environ.get("MASSIVE_API_KEY")):
        pytest.skip("DATALOADER_API_KEY/MASSIVE_API_KEY not set")
    from data.data_loader import load_universe

    universe = {"SPY": "equity", "C:EURUSD": "fx"}
    per_ticker, meta = load_universe(universe, "2022-01-01", "2023-12-31", use_cache=True)
    return per_ticker, meta
