"""
_move_lib.py — shared primitives for the move-probability system.

Everything in the _move_* family imports from here so that "realized vol" and "calibration" mean
exactly one thing across all studies. Volatility estimators, forward-return bucketing,
probabilistic scoring rules and the walk-forward splitter all live here.

CONVENTIONS (deliberate, do not silently change):
  - Volatility is per-DAY standard deviation of LOG returns, never annualized. Annualizing just
    multiplies by sqrt(252) and invites off-by-a-factor bugs when mixing horizons.
  - Bucketing uses SIMPLE returns, because "the stock moves 2%" is a simple-return statement and
    the whole point is to answer the user's question in the user's units.
  - Forward return over h days at row t = close[t+h]/close[t] - 1. Row t is the last OBSERVED bar,
    so every feature used to predict it must be computable from data up to and including t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUCKETS = ("down_big", "down_small", "up_small", "up_big")
BUCKET_LABELS = {"down_big": "跌超阈值", "down_small": "小跌", "up_small": "小涨", "up_big": "涨超阈值"}


# ------------------------------------------------------------------ volatility estimators
# All take price DataFrames/Series and return per-day log-return sigma.

def _logret(close):
    return np.log(close).diff()


def vol_cc(close, n):
    """Close-to-close. The naive baseline: throws away the intraday range."""
    return _logret(close).rolling(n).std(ddof=1)


def vol_ewma(close, lam=0.94):
    """RiskMetrics EWMA. lam=0.94 is the classic daily choice, 0.97 is slower."""
    r = _logret(close)
    return np.sqrt(r.pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def vol_parkinson(high, low, n):
    """Uses the high-low range only. ~5x more efficient than close-close, but blind to
    overnight gaps (it assumes continuous trading), so it UNDERSTATES vol for gappy names."""
    hl = np.log(high / low).pow(2)
    return np.sqrt(hl.rolling(n).mean() / (4 * np.log(2)))


def vol_garman_klass(o, h, l, c, n):
    """Range + open/close. More efficient than Parkinson; still assumes no overnight jump."""
    term = 0.5 * np.log(h / l).pow(2) - (2 * np.log(2) - 1) * np.log(c / o).pow(2)
    return np.sqrt(term.rolling(n).mean().clip(lower=0))


def vol_rogers_satchell(o, h, l, c, n):
    """Drift-independent — does not assume zero mean return, which matters for trending names."""
    term = (np.log(h / c) * np.log(h / o)) + (np.log(l / c) * np.log(l / o))
    return np.sqrt(term.rolling(n).mean().clip(lower=0))


def vol_yang_zhang(o, h, l, c, n, k=None):
    """Yang-Zhang: the only common estimator that handles BOTH overnight jumps and drift.
    Usually the most efficient per bar; the one to beat."""
    ho, lo, co = np.log(h / o), np.log(l / o), np.log(c / o)
    oc = np.log(o / c.shift(1))                     # overnight jump
    sigma_o = oc.rolling(n).var(ddof=1)
    sigma_c = co.rolling(n).var(ddof=1)
    rs = (ho * (ho - co) + lo * (lo - co)).rolling(n).mean()
    if k is None:
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return np.sqrt((sigma_o + k * sigma_c + (1 - k) * rs).clip(lower=0))


def har_features(close, high=None, low=None):
    """Corsi HAR-RV inputs: yesterday's, last week's and last month's realized vol. The modern
    workhorse for vol forecasting -- a linear model on these three usually beats any single
    rolling window, because it captures vol's long-memory decay.
    With daily bars the single-day 'rv_d' term is either |log return| or a one-bar Parkinson."""
    base = _logret(close).abs() if high is None else vol_parkinson(high, low, 1)
    return pd.DataFrame({
        "rv_d": base,
        "rv_w": base.rolling(5).mean(),
        "rv_m": base.rolling(22).mean(),
    })


# ------------------------------------------------------------------ targets
def forward_simple_return(close, h):
    """close[t+h]/close[t] - 1, indexed at t. NaN in the last h rows by construction."""
    return close.shift(-h) / close - 1.0


def realized_vol_forward(close, h):
    """Actual per-day log-return sigma realized over the NEXT h days (r[t+1..t+h]), indexed at t.
    This is what a vol forecast should be scored against.
    rolling(h) at t+h covers exactly r[t+1..t+h]; shift(-h) parks that value back on row t.

    h=1 is a genuinely different object: one return has no dispersion, so rolling(1).std(ddof=1)
    is all-NaN. We fall back to the unbiased single-observation sigma proxy |r| * sqrt(pi/2)
    (E|r| = sigma*sqrt(2/pi) under normality). It is unbiased but very noisy -- log-space noise
    sd ~1.1 -- which is why every h=1 vol R-squared has a hard ceiling around 0.2 that is
    measurement noise in the TARGET, not a failure of the forecast. Do not compare h=1 R-squared
    to h>=5 as if they measured the same thing."""
    r = _logret(close)
    if h == 1:
        return r.abs().shift(-1) * np.sqrt(np.pi / 2)
    return r.rolling(h).std(ddof=1).shift(-h)


def bucketize(fwd_ret: pd.Series | np.ndarray, thr: float) -> pd.Series | np.ndarray:
    """Four mutually exclusive buckets around +/- thr. Ties: exactly -thr counts as down_big,
    exactly 0 counts as down_small, exactly +thr counts as up_big (arbitrary but fixed)."""
    r = fwd_ret
    out = np.where(r <= -thr, "down_big",
          np.where(r <= 0, "down_small",
          np.where(r < thr, "up_small", "up_big")))
    out = np.where(pd.isna(r), None, out)
    return pd.Series(out, index=r.index) if isinstance(r, pd.Series) else out


# ------------------------------------------------------------------ probabilistic scoring
def log_loss(probs: np.ndarray, actual_idx: np.ndarray, eps=1e-12) -> float:
    """Multiclass log loss (lower is better). probs: (n, k) rows summing to 1."""
    p = np.clip(probs[np.arange(len(actual_idx)), actual_idx], eps, 1.0)
    return float(-np.log(p).mean())


def brier_multi(probs: np.ndarray, actual_idx: np.ndarray) -> float:
    """Multiclass Brier score = mean squared error over the one-hot target (lower is better)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual_idx)), actual_idx] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def skill_score(model: float, baseline: float) -> float:
    """Fraction of the baseline's loss removed. >0 means the model beats the baseline;
    0 means it is exactly as good as always predicting the climatological frequencies."""
    return float(1.0 - model / baseline) if baseline else float("nan")


def reliability(pred_p: np.ndarray, hit: np.ndarray, n_bins=10) -> pd.DataFrame:
    """Calibration table for ONE bucket's probability. A well-calibrated forecast has
    observed_freq ~= mean_pred in every bin. Uses equal-count bins so every row has power."""
    df = pd.DataFrame({"p": pred_p, "y": hit.astype(float)}).dropna()
    if len(df) < n_bins * 10:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
    g["gap"] = g.observed - g.mean_pred
    return g


def ece(pred_p: np.ndarray, hit: np.ndarray, n_bins=10) -> float:
    """Expected calibration error: average |observed - predicted| weighted by bin size."""
    r = reliability(pred_p, hit, n_bins)
    if r.empty:
        return float("nan")
    return float((r.n / r.n.sum() * r.gap.abs()).sum())


# ------------------------------------------------------------------ evaluation splitting
def walk_forward_years(index: pd.DatetimeIndex, min_train_years=5):
    """Yield (train_mask, test_mask) per calendar year, expanding window. Fit only on strictly
    prior years -- the repo's standing preference over one static train/test cut."""
    years = sorted(set(index.year))
    for y in years[min_train_years:]:
        yield y, (index.year < y), (index.year == y)


def climatology(actual_idx: np.ndarray, k=4) -> np.ndarray:
    """Unconditional bucket frequencies -- the baseline every model must beat to be worth
    anything. In-sample by construction, which makes it a GENEROUS baseline, not a weak one."""
    counts = np.bincount(actual_idx, minlength=k).astype(float)
    return counts / counts.sum()
