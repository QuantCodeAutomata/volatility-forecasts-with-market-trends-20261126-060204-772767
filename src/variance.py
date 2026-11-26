"""EWMA variance (paper equation 2)."""

import numpy as np
from scipy.signal import lfilter

from src import config


def ewma_variance(r: np.ndarray, alpha: float = config.EWMA_ALPHA) -> np.ndarray:
    """sigma_t^2 = alpha*sigma_{t-1}^2 + (1-alpha)*r_t^2 (paper eq. 2).

    r is already normalized so the long-term equilibrium variance is 1; the
    recursive filter is started from 0 and the first WARMUP_MULTIPLE*max(T_k)
    observations are discarded by the caller so the cold start does not bias
    the reported sample (paper section 2.1: "earlier data used only for
    computing initial trend strengths and variances").
    """
    r2 = np.nan_to_num(r.astype(float), nan=1.0) ** 2
    sigma2 = lfilter([1.0 - alpha], [1.0, -alpha], r2)
    return sigma2
