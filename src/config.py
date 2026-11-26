"""Parameters for "Volatility Forecasts with Market Trends".

Every module-level constant states its meaning, unit and source. "paper
section X" refers to sections of the paper; "not specified in the paper"
marks values the paper leaves open, for which a reasonable choice was made.
"""

# ---------------------------------------------------------------------------
# Universe (paper section 2.1)
# ---------------------------------------------------------------------------
# The paper uses 24 Bloomberg futures markets (6 equity indices, 6 interest
# rate futures, 6 currencies, 6 commodities) from Jan 1990 to Sep 2026, with
# rolls 5 days before first notice and some pre-1990s reconstructions.
# Bloomberg continuous futures histories and foreign government-bond futures
# are not served by the QuantX Dataloader / Massive data path used here. The
# universe below substitutes the most liquid exchange-traded proxy available
# for each market (country-index ETF for equities, a spread of US Treasury
# duration ETFs for interest-rate futures, spot FX for currency futures,
# commodity-tracking ETFs for commodity futures). This is a deviation from
# the paper's exact instrument list (results/RESULTS.md, "Deviations from the
# paper"); "Live Cattle" has no tradable ETF proxy on the data path and is
# substituted by a Corn ETF to keep the commodity class at 6 members.
UNIVERSE_PAPER_SIZE = 24  # markets used in the paper, paper section 2.1

# ticker -> asset class
UNIVERSE_RUN = {
    # equities: country-index ETF proxy for each futures index
    "SPY": "equity",   # S&P 500
    "EWC": "equity",   # TSE 60 (Canada)
    "EWG": "equity",   # DAX 30 (Germany)
    "EWU": "equity",   # FTSE 100 (UK)
    "EWJ": "equity",   # Nikkei 225 (Japan)
    "EWH": "equity",   # Hang Seng (Hong Kong)
    # interest rates: US Treasury duration ETFs proxy the 6 government-bond futures
    "SHV": "rate",     # proxy for Australia 3y note (shortest available duration)
    "SHY": "rate",     # proxy for UK 10y gilt
    "IEI": "rate",     # proxy for Germany 10y bund
    "IEF": "rate",     # proxy for US 10y note
    "TLH": "rate",     # proxy for Canada 10y bond
    "TLT": "rate",     # proxy for Japan 10y JGB
    # currencies vs USD, spot proxy for the futures
    "C:CADUSD": "fx", "C:EURUSD": "fx", "C:GBPUSD": "fx",
    "C:JPYUSD": "fx", "C:AUDUSD": "fx", "C:NZDUSD": "fx",
    # commodities
    "USO": "commodity",   # Crude Oil
    "UNG": "commodity",   # Natural Gas
    "GLD": "commodity",   # Gold
    "CPER": "commodity",  # Copper
    "SOYB": "commodity",  # Soybeans
    "CORN": "commodity",  # substitutes Live Cattle (no tradable ETF proxy found)
}
UNIVERSE_RUN_SIZE = len(UNIVERSE_RUN)  # instruments actually downloaded and analyzed

SAMPLE_START = "1990-01-01"      # paper section 2.1: data collection start
SAMPLE_END = "2026-09-15"        # paper section 2.1, data cutoff date
ANALYSIS_START = "1991-05-01"    # paper section 2.1: reported-analysis start (N=9216 days); data before this date warms up the recursive trend/variance filters only
IDENTITY_DATE = SAMPLE_END   # Dataloader identity_date, paper gives no explicit snapshot date

# ---------------------------------------------------------------------------
# Trend strength (paper section 2.2, equations preceding (2))
# ---------------------------------------------------------------------------
TREND_HORIZONS_K = list(range(1, 9))          # k = 1..8, paper section 2.2
TREND_HORIZON_DAYS = {k: 2 ** k for k in TREND_HORIZONS_K}  # T_k = 2^k business days
TREND_CAP = 2.5                                # cap/floor of trend strength phi, paper section 2.2
RETURN_CAP_STD = 20.0                          # cap/floor of returns at 20 sigma for trend calc, paper section 2.2
# M_T is derived from sum_n w_T(n)^2 = 1 using the closed form
# sum_{n=0}^inf n^2 rho^(2n) = rho^2(1+rho^2)/(1-rho^2)^3, rho = exp(-2/T).
# (Not given explicitly in the paper; derived here from its normalization
# constraint, paper section 2.2.)

WARMUP_MULTIPLE = 3  # warmup length = WARMUP_MULTIPLE * max(T_k), used only to
# initialize the recursive trend/variance filters before the reported sample
# starts; not specified in the paper.

# ---------------------------------------------------------------------------
# EWMA variance (paper equation 2)
# ---------------------------------------------------------------------------
EWMA_ALPHA = 0.94  # EWMA decay for next-day variance, paper equation 2, k <= 3

# Longer-term prediction horizons (paper section 3.6 / table 7): the paper
# computes T = 2^k for k=1..6 and interpolates linearly to calendar horizons.
# Here the calendar horizons are evaluated directly (business days), which is
# an equivalent but non-interpolated way of obtaining the same horizons
# (interpretation, paper section 3.6).
LONG_HORIZONS_DAYS = {"2-day": 2, "1-week": 5, "2-week": 10,
                       "1-month": 22, "2-month": 44, "3-month": 66}


def ewma_alpha_for_horizon(k: int) -> float:
    """EWMA decay for predicting the 2^k-day variance (paper section 3.6):
    alpha=0.94 for k<=3, else alpha = 1 - 2^-(k+1)."""
    if k <= 3:
        return EWMA_ALPHA
    return 1.0 - 2.0 ** (-(k + 1))


# ---------------------------------------------------------------------------
# Aggregate trend factors (paper equation 3 and section 3.5)
# ---------------------------------------------------------------------------
AGG_HORIZONS_BASIC = list(range(1, 9))   # basic model (6): all k=1..8, paper eq. 3
AGG_HORIZONS_ENHANCED = list(range(1, 7))  # enhanced model (8): k=1..6 (<=3 months), paper section 3.5

# ---------------------------------------------------------------------------
# Cross-validation and bootstrap (paper section 3.2)
# ---------------------------------------------------------------------------
CV_FOLDS_PAPER = 16         # paper section 3.2: 16 adjacent folds of 576 days (9216/16)
BOOTSTRAP_SAMPLES_PAPER = 300  # paper section 3.2 / table 3 caption: >=300 bootstrap samples
BLOCK_BOOTSTRAP_BLOCKS_PAPER = 18   # paper section 3.2: 18 contiguous blocks of 512 days
BLOCK_LENGTH_DAYS_PAPER = 512       # paper section 3.2

# Reduced execution configuration: the paper's instrument-day panel (9216
# days x 24 markets) does not exist here (smaller, shorter-history universe,
# see above), and the paper's CV/bootstrap repetition counts would leave too
# little time for four experiments within the time available for this
# replication. The method (fold/block structure, resampling unit) is
# unchanged; only the repetition counts are reduced. Use --full to request
# the paper's counts.
CV_FOLDS_RUN = 6
BOOTSTRAP_SAMPLES_RUN = 60
BLOCK_BOOTSTRAP_BLOCKS_RUN = 8
BLOCK_LENGTH_DAYS_RUN = 60

# ---------------------------------------------------------------------------
# GARCH / GJR-GARCH (exp_2, methodology steps 4-7)
# ---------------------------------------------------------------------------
GARCH_PARAM_BOUNDS = {
    "omega": (1e-4, 5.0),
    "alpha": (0.0, 0.6),
    "beta": (0.0, 0.999),
    "gamma": (-0.6, 0.6),
    "lam": (-2.0, 2.0),
    "zeta": (-2.0, 2.0),
}  # optimizer bounds; not given in the paper, chosen to keep the recursion stable

# ---------------------------------------------------------------------------
# Range-based models (exp_3, methodology steps 4-8)
# ---------------------------------------------------------------------------
HAR_DAILY_LAG = 1
HAR_WEEKLY_LAG = 5     # paper eq. (9): average of the last 5 days
HAR_MONTHLY_LAG = 22   # paper eq. (9): average of the last 22 days

# ---------------------------------------------------------------------------
# Stochastic volatility models (exp_4, methodology steps 4-7)
# ---------------------------------------------------------------------------
# Fit by an approximate (quasi-)Gaussian Kalman filter likelihood on
# log(r_t^2) / on r_t^2, following the quasi-likelihood approach of Harvey,
# Ruiz & Shephard (1994) for the log-variance (LSV) model, and a
# moment-matched linear-Gaussian filter for the level-based Heston model.
# This is a documented simplification of the paper's heavier
# particle-filter/MCMC-based fitting (paper section 4.2, implementation
# notes); see results/RESULTS.md, "Deviations from the paper".
LOG_CHI2_1_MEAN = -1.2704022563702492   # E[log(z^2)], z ~ N(0,1); digamma(0.5)+log(2)
LOG_CHI2_1_VAR = (3.141592653589793 ** 2) / 2.0  # Var[log(z^2)], z ~ N(0,1)
SV_PARAM_BOUNDS = {
    "kappa": (1e-4, 0.5),
    "theta": (0.2, 3.0),
    "xi": (1e-3, 1.0),
    "beta": (-0.3, 0.3),
    "mu": (-2.0, 0.5),
    "lam": (-1.0, 1.0),
    "zeta": (-1.0, 1.0),
}

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
RANDOM_SEED = 1234  # not specified in the paper; fixed for reproducibility
