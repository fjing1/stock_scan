"""
move_prob.py — calibrated probability that a symbol moves more than +/-X% over the next N days.

Answers, for any ticker and horizon:  P(up > X%) / P(within +/-X%) / P(down > X%).

ARCHITECTURE: scale-and-shape, not a direction classifier.
    1. sigma_h  -- forecast the h-day return volatility with a log-space HAR regression on
                   range-based realized vol (+ log VIX for index/ETF).
    2. z-shape  -- divide by sigma_h and look up an EMPIRICAL standardized-return distribution.
    3. buckets  -- read the four probabilities straight off that distribution's CDF.

Why this shape and not a classifier: magnitude is forecastable (volatility clusters), direction
essentially is not. This repo has killed six directional edges; pretending otherwise here would
just relabel noise as a probability. So all the skill lives in P(|move| >= X%), and the up/down
split is dominated by two boring structural terms -- drift and skew -- not by a forecast.

DESIGN DECISIONS AND WHY (each one is a research result, not a preference):

  * LOG SPACE EVERYWHERE. z = log(1+r_h)/sigma_h, and the thresholds convert as log(1 +/- X%).
    Dividing a SIMPLE return by a LOG-return sigma injects a spurious positive skew of about
    +0.64*sigma_h. That artifact is large enough to exactly cancel single names' real left skew
    and make them look symmetric. In log space single-name P(z<-2)/P(z>+2) is 1.11 at h=1 rising
    to 1.64 at h=21, with 77-81% of individual names left-skewed.

  * THE EMPIRICAL z TABLE CARRIES THE DRIFT. We deliberately do NOT model mu separately. The
    h=21 failure of a standard normal is a LOCATION defect, not a tail defect -- a normal shifted
    by one scalar (the train median of z) recovers ~107% of the empirical table's gain. Fitting
    the CDF on realized z picks up location, skew and fat tails in one object.

  * PER-HORIZON FITS ABSORB HORIZON SCALING. Volatility does not scale as sqrt(h): the empirical
    exponent is 0.45-0.47 because of negative return autocorrelation, so sqrt(h) overstates the
    63-day index sd by ~20%. We do not model that explicitly -- sigma and the z table are both fit
    per horizon, so any constant per-h multiplicative bias is absorbed by the z table by
    construction. This is why the elaborate AR(1) path-variance machinery is not here: with
    per-h refits it buys 0.06-5.7% of QLIKE, which does not survive the complexity.

  * EARNINGS MULTIPLIER FOR SINGLE NAMES. Without it the model underpredicts P(|move|>2%) on
    windows containing an earnings date by 23.8pp at h=1, and is literally worse than
    climatology there. With it, log loss on those windows drops 10.9%.

  * SUPPORT GATE. The model refuses when log(1+thr)/sigma_h leaves [0.4, 3.0]. Above 3 the
    empirical z has no mass out there (a 5% one-day index move); below 0.4 the band is narrower
    than the noise and the four-bucket question degenerates into a pure direction call.

HONEST LIMITS -- read before trusting a number:
  * Skill is small and it is all in magnitude. Against each ticker's OWN base rate the four-class
    Brier skill is roughly 0.01-0.04, and the magnitude-only skill 0.03-0.18. Against a POOLED
    climatology single-name skill looks ~2x better, but half of that is just knowing that one
    stock is more volatile than another, which is free.
  * The single-name universe used to fit the z tables is survivorship-biased (still-listed names),
    so single-name DOWNSIDE tails are, if anything, too thin.
  * It forecasts a distribution, not a direction. Do not read P(up) > P(down) as a buy signal --
    at h>=5 that inequality is almost entirely the equity drift term and holds nearly always.

Usage:
    ../../vcp_env/bin/python move_prob.py fit              # train and save _move_model.pkl
    ../../vcp_env/bin/python move_prob.py AAPL SPY --thr 2 --horizons 1,5,10,21
    from move_prob import predict; predict("AAPL", horizons=(5,), thr=0.02)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _move_lib as L

MODEL_PATH = Path(__file__).with_name("_move_model.pkl")
HORIZONS = (1, 5, 10, 21)
INDEX_LIKE = {"SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "^GSPC", "^NDX", "^DJI", "^RUT",
              "MAGS", "RSP", "SMH", "XLU", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY",
              "XLB", "XLRE", "XLC", "CIBR", "IHF", "TLT", "GLD", "SLV", "HYG"}
VOL_CLIP = (1e-3, 0.5)        # per-day sigma band; below 1e-3 (1.6% ann) is physically impossible
SUPPORT_LO, SUPPORT_HI = 0.4, 3.0

# Earnings sigma multipliers for single names when the window [t+1, t+h] contains a report.
# Median over 52 symbols with >=5 in-history events; symbol-cluster bootstrap CIs in the study.
EARN_MULT = {1: 1.957, 5: 1.594, 10: 1.355, 21: 1.211}


# ---------------------------------------------------------------- features
def _clip_log(x):
    return np.log(np.clip(x, *VOL_CLIP))


def build_features(close, high=None, low=None, vix=None, volume=None) -> pd.DataFrame:
    """The feature block, all computable from bars up to and including t.

    Clipping into [1e-3, 0.5] BEFORE taking logs is not cosmetic: ~0.5% of one-bar Parkinson
    values are exactly zero (high == low on halted/illiquid bars) and with a 1e-5 floor those
    become ln = -11.5 outliers that cost 0.03-0.04 R-squared.

    Three groups of features, each earned its place by measurement (see _move_fx_screen.py and
    _move_fx_verify_lev.py for the walk-forward deltas):
      HAR   rv_d/rv_w/rv_m/rv_q/ewma97 -- the original scale features.
      LEV   realized semivariance split up/down plus the signed 5-day return. The HAR block is
            SIGN-BLIND (an intraday range and a squared return do not know which way price went),
            so it cannot express the leverage effect. Adding LEV is worth +0.005 BSS2 at index
            h=5 (CI [+0.0017,+0.0077], permutation control -0.0001) and +0.001..+0.002 on single
            names, and it IMPROVES ECE at the same time. Survives dropping 2008/09/20 and
            non-overlapping scoring.
      VOL   dollar volume, volume surprise, Amihud illiquidity. SINGLE NAMES ONLY: worth +0.0022
            at h=1 (CI [+0.0016,+0.0029], 21/21 years) for stocks, but reliably NEGATIVE for
            indices (bootstrap P(delta>0) = 0.00), because index volume is dominated by
            mechanical ETF creation/redemption rather than information arrival.
    Measured and REJECTED, do not re-add: an overnight/intraday variance split and the
    close-to-close / Parkinson ratio (reliably negative), bipower-variation jump separation
    (+0.0002, noise), and the VIX term structure / VVIX (CI spans zero, ECE worse, and
    ^VIX9D/^VIX3M stop at 2026-07-17 in yfinance so they are not live-available anyway)."""
    has_ohlc = high is not None and low is not None
    rv1 = L.vol_parkinson(high, low, 1) if has_ohlc else L._logret(close).abs()
    r = L._logret(close)
    f = pd.DataFrame(index=close.index)
    f["rv_d"] = _clip_log(rv1)
    f["rv_w"] = _clip_log(rv1.rolling(5).mean())
    f["rv_m"] = _clip_log(rv1.rolling(22).mean())
    f["rv_q"] = _clip_log(rv1.rolling(63).mean())
    f["ewma97"] = _clip_log(L.vol_ewma(close, 0.97))
    # LEV
    f["sv_dn"] = _clip_log(np.sqrt((r.where(r < 0, 0.0) ** 2).rolling(22).mean()))
    f["sv_up"] = _clip_log(np.sqrt((r.where(r > 0, 0.0) ** 2).rolling(22).mean()))
    f["r5"] = r.rolling(5).sum()
    if vix is not None:
        f["logvix"] = np.log(np.clip(vix.reindex(close.index).ffill(limit=3) / 100 / np.sqrt(252),
                                     *VOL_CLIP))
    if volume is not None:
        v = pd.to_numeric(volume, errors="coerce").reindex(close.index)
        dv = (v * close).replace(0, np.nan)
        f["dvol"] = np.log(dv.rolling(22).mean().clip(lower=1e3))
        f["vsurp"] = np.log((v / v.rolling(22).mean()).clip(0.05, 20))
        f["amihud"] = np.log((r.abs() / dv).rolling(22).mean().clip(1e-14, 1e-3))
    return f


FEATS_HAR = ["rv_d", "rv_w", "rv_m", "rv_q", "ewma97"]
FEATS_LEV = ["sv_dn", "sv_up", "r5"]
FEATS_VOL = ["dvol", "vsurp", "amihud"]
FEATS_OHLC = FEATS_HAR + FEATS_LEV          # kept as the name other modules import


def feature_cols(group: str, has_vix: bool, has_volume: bool) -> list[str]:
    """Which columns this group actually uses. Volume is single-name only, by measurement."""
    cols = FEATS_HAR + FEATS_LEV
    if group == "index" and has_vix:
        cols = cols + ["logvix"]
    if group == "single" and has_volume:
        cols = cols + FEATS_VOL
    return cols


def _design(f: pd.DataFrame, cols) -> np.ndarray:
    return np.column_stack([np.ones(len(f))] + [f[c].values for c in cols])


# ---------------------------------------------------------------- fitting
def _ols(X, y):
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if keep.sum() < 200:
        return None, 0
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return beta, int(keep.sum())


def build_cache(panel, train_end: str | None = None, verbose=False) -> dict:
    """Build the stacked design matrix / target / forward-log-return arrays once per
    (group, horizon). Every fit downstream is then plain linear algebra on these arrays, which is
    what makes multi-fold kappa estimation cheap instead of an hour of re-deriving features."""
    C, H, Lo = panel["Close"], panel["High"], panel["Low"]
    Vol = panel.get("Volume")
    vix = C["^VIX"] if "^VIX" in C.columns else None
    if train_end:
        cut = pd.Timestamp(train_end)
        C, H, Lo = C[C.index < cut], H[H.index < cut], Lo[Lo.index < cut]
        Vol = Vol[Vol.index < cut] if Vol is not None else None
        vix = vix[vix.index < cut] if vix is not None else None

    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in INDEX_LIKE],
              "single": [s for s in usable if s not in INDEX_LIKE]}
    cache = {}
    for g, syms in groups.items():
        cols = feature_cols(g, vix is not None, Vol is not None)
        for h in HORIZONS:
            X, lr, tg, yr, sid = [], [], [], [], []
            for i, s in enumerate(syms):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                vser = Vol[s].reindex(c.index) if (Vol is not None and s in Vol.columns) else None
                f = build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix, vser)
                if any(x not in f.columns for x in cols):
                    continue
                Xi = _design(f, cols)
                li = np.log(c.shift(-h) / c).values
                ti = _clip_log(L.realized_vol_forward(c, h)).values
                ok = np.isfinite(Xi).all(axis=1) & np.isfinite(li)
                X.append(Xi[ok]); lr.append(li[ok]); tg.append(ti[ok])
                yr.append(c.index.year.values[ok]); sid.append(np.full(int(ok.sum()), i))
            if not X:
                continue
            cache[(g, h)] = dict(X=np.vstack(X), lr=np.concatenate(lr), tgt=np.concatenate(tg),
                                 yr=np.concatenate(yr), sid=np.concatenate(sid),
                                 cols=cols, symbols=syms)
            if verbose:
                print(f"  cached {g:<7} h={h:<3} rows={len(cache[(g, h)]['lr']):>9,}")
    return cache


def _fit_arrays(X, tgt, lr, h, sel):
    """(beta, sorted z) fit on the boolean row selection `sel`."""
    fin = np.isfinite(tgt)
    beta, n = _ols(X[sel & fin], tgt[sel & fin])
    if beta is None:
        return None, None, 0
    sig = np.exp(X[sel] @ beta) * np.sqrt(h)
    z = lr[sel] / sig
    return beta, np.sort(z[np.isfinite(z)]), n


def _calibrate_kappa(z_in, sig_val, lr_val, thrs=(0.01, 0.02, 0.03, 0.05)) -> float:
    """Width multiplier on the reference z distribution, chosen on an INNER fold, never on test.

    Why a WIDTH multiplier and not a sigma multiplier: scaling sigma is a mathematical no-op here,
    because z = return/sigma and the thresholds are divided by the same sigma -- the empirical
    table absorbs any constant scale by construction. Only the table's own width changes anything.

    Why it is needed: the index z table rests on ~6,200 independent dates, so its tails are
    finite-sample thin -- an expanding window that has not yet seen the next crash under-states the
    chance of a big move, which is the dangerous direction for a risk tool. Measured out of sample
    this halves index ECE (h=5: 0.031 -> 0.016) at no cost in sharpness. Single names land at
    kappa ~1.0, i.e. it correctly does nothing where nothing is wrong."""
    grid = np.linspace(0.90, 1.40, 26)
    best, best_err = 1.0, np.inf
    for k in grid:
        zk = z_in * k
        err = 0.0
        for thr in thrs:
            a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
            f_dn = np.searchsorted(zk, a_dn / sig_val, side="right") / len(zk)
            f_up = np.searchsorted(zk, a_up / sig_val, side="right") / len(zk)
            err += abs((f_dn + (1 - f_up)).mean()
                       - float(((lr_val <= a_dn) | (lr_val >= a_up)).mean()))
        if err < best_err:
            best, best_err = float(k), err
    return best


def fit(panel=None, train_end: str | None = None, n_folds=8, verbose=True) -> dict:
    """Fit HAR vol coefficients, the empirical z tables, and the width calibration kappa.

    kappa is the MEDIAN over `n_folds` expanding inner folds (each a held-out calendar year), not
    a single fold. A single fold is regime-dependent: calibrating only on the last two years of
    this sample -- a calm stretch -- returns kappa 0.90-0.98, while the median across 8 folds
    returns 1.05-1.13, matching the walk-forward measurement. Never calibrate width on one fold.

    train_end: ISO date; rows on/after it are excluded so a held-out evaluation is possible."""
    import _move_data as D

    panel = panel or D.load()
    cache = build_cache(panel, train_end=train_end, verbose=verbose)
    model = {"groups": {}, "train_end": train_end, "horizons": list(HORIZONS), "n_folds": n_folds}

    for (g, h), d in sorted(cache.items()):
        X, lr, tgt, yr = d["X"], d["lr"], d["tgt"], d["yr"]
        years = sorted(set(yr))
        kappas = []
        for ty in years[-n_folds:]:
            tr, va = yr < ty, yr == ty
            if tr.sum() < 5000 or va.sum() < 50:
                continue
            b_in, z_in, _ = _fit_arrays(X, tgt, lr, h, tr)
            if b_in is None or len(z_in) < 5000:
                continue
            kappas.append(_calibrate_kappa(z_in, np.exp(X[va] @ b_in) * np.sqrt(h), lr[va]))
        kappa = float(np.median(kappas)) if kappas else 1.0

        beta, z, n = _fit_arrays(X, tgt, lr, h, np.ones(len(lr), bool))
        if beta is None:
            continue
        z = z * kappa
        spec = {"beta": beta, "cols": d["cols"], "n_fit": n, "kappa": kappa,
                "kappa_folds": [round(k, 3) for k in kappas],
                "z": z.astype(np.float32), "n_z": len(z)}

        # No-VIX fallback for the index group. Dropping the logvix COLUMN at predict time while
        # keeping the intercept is not a fallback, it is a bug: the term contributes about
        # 0.47 * log(VIX/100/sqrt(252)) ~ -2.1, so deleting it inflates sigma roughly 8x. The
        # fallback has to be its own fit, with its own z table, or not exist at all.
        opt = [c for c in d["cols"] if c in (["logvix"] + FEATS_VOL)]
        if opt:
            drop = {d["cols"].index(c) + 1 for c in opt}      # +1 for the intercept column
            keep = [j for j in range(X.shape[1]) if j not in drop]
            bn, zn, nn = _fit_arrays(X[:, keep], tgt, lr, h, np.ones(len(lr), bool))
            if bn is not None:
                spec["fallback"] = {"beta": bn, "cols": [c for c in d["cols"] if c not in opt],
                                    "z": (zn * kappa).astype(np.float32), "n_fit": nn}

        model["groups"].setdefault(g, {"symbols": d["symbols"], "per_h": {}})
        model["groups"][g]["per_h"][h] = spec
        if verbose:
            print(f"  {g:<7} h={h:<3} n_fit={n:>9,}  kappa={kappa:.3f} "
                  f"(folds {min(kappas):.2f}-{max(kappas):.2f})  "
                  f"z: med={np.median(z):+.3f} P(z<-2)={np.mean(z < -2):.4f} "
                  f"P(z>2)={np.mean(z > 2):.4f}")

    pd.to_pickle(model, MODEL_PATH)
    if verbose:
        print(f"\nsaved {MODEL_PATH.name}")
    return model


# ---------------------------------------------------------------- prediction
def _cdf(z_sorted: np.ndarray, x: float) -> float:
    """Empirical CDF via binary search. Carries drift, skew and fat tails in one object."""
    return float(np.searchsorted(z_sorted, x, side="right") / len(z_sorted))


def asset_class(symbol: str) -> str:
    return "index" if symbol.upper() in INDEX_LIKE else "single"


def predict_from_bars(symbol, close, high=None, low=None, vix=None, horizons=HORIZONS,
                      thr=0.02, model=None, earnings_in=None, volume=None) -> list[dict]:
    """Core predictor. earnings_in: set of horizons whose window contains an earnings date.
    volume is used for single names only (it measured reliably NEGATIVE for indices)."""
    model = model or pd.read_pickle(MODEL_PATH)
    g = asset_class(symbol)
    grp = model["groups"][g]
    f = build_features(close, high, low, vix, volume if g == "single" else None)
    out = []
    for h in horizons:
        spec = grp["per_h"].get(h)
        if spec is None:
            continue
        cols = spec["cols"]
        if any(c not in f.columns for c in cols):
            fb = spec.get("fallback")
            if fb is None or any(c not in f.columns for c in fb["cols"]):
                out.append({"h": h, "ok": False, "why": "缺少必要特征（无VIX且无降级模型）"})
                continue
            beta, zt, cols = fb["beta"], fb["z"], fb["cols"]
            row = _design(f.iloc[[-1]], cols)
            degraded = True
        else:
            beta, zt, row = spec["beta"], spec["z"], _design(f.iloc[[-1]], cols)
            degraded = False
        if not np.isfinite(row).all():
            out.append({"h": h, "ok": False, "why": "特征不足（历史长度不够或含缺口）"})
            continue

        sig_daily = float(np.exp(row @ beta).item())
        sig_h = sig_daily * np.sqrt(h)
        mult = EARN_MULT.get(h, 1.0) if (earnings_in and h in earnings_in and g == "single") else 1.0
        sig_h *= mult

        a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
        ratio = a_up / sig_h
        z = zt
        F_dn, F_0, F_up = _cdf(z, a_dn / sig_h), _cdf(z, 0.0), _cdf(z, a_up / sig_h)
        p = np.array([F_dn, max(F_0 - F_dn, 0), max(F_up - F_0, 0), max(1 - F_up, 0)])
        p = np.clip(p, 1e-4, None)
        p = p / p.sum()

        # Move SIZES read off the same z table, so they cannot disagree with the probabilities.
        # Never display sigma_h itself as "the expected move": it is a fitted scale, biased ~14%
        # low at every decile (realized/predicted 1.12-1.22). That bias is harmless for the
        # probabilities -- the empirical z table absorbs any constant scale by construction -- but
        # it would be a lie if shown to a user as an expected move.
        qs = np.quantile(z, [0.05, 0.5, 0.95])
        typ = float(np.exp(np.quantile(np.abs(z), 0.5) * sig_h) - 1)

        # Expected (mean) return, E[exp(z*sigma_h)] - 1, over the same z table.
        # WINSORIZED at 0.5%/99.5% first. The raw mean is unusable: the single-name z table
        # contains real microcap jumps out to z ~ +400, and exp(400 * sigma_h) overflows any
        # sensible number -- one observation would set the whole column. z is already sorted,
        # so the cut points are a direct index rather than a quantile scan.
        n_z = len(z)
        lo_i, hi_i = int(0.005 * n_z), int(0.995 * n_z) - 1
        zw = np.clip(z, z[lo_i], z[hi_i])
        exp_ret = float(np.mean(np.exp(zw * sig_h)) - 1.0)

        out.append({
            "h": h, "ok": True, "symbol": symbol, "asset_class": g, "thr": thr,
            "sigma_h": sig_h, "earn_mult": mult,
            "typical_move": typ, "exp_return": exp_ret,
            "q05": float(np.exp(qs[0] * sig_h) - 1),
            "q50": float(np.exp(qs[1] * sig_h) - 1),
            "q95": float(np.exp(qs[2] * sig_h) - 1),
            "support": float(ratio), "in_support": SUPPORT_LO <= ratio <= SUPPORT_HI,
            "degraded": degraded,
            "p_down_big": p[0], "p_down_small": p[1], "p_up_small": p[2], "p_up_big": p[3],
            "p_move": p[0] + p[3], "p_within": p[1] + p[2],
            # P(up by ANY amount) = 1 - F(0). Note this is structurally IDENTICAL for every
            # symbol in the same (asset class, horizon): the evaluation point is 0/sigma_h = 0
            # regardless of sigma, so it cannot vary with the ticker. That is not a bug -- it is
            # this model honestly reporting that it has no directional information. Measured
            # direction skill is 0.0-2.3% over the historical up-share, and NEGATIVE for indices
            # at h >= 10, so any ticker-to-ticker variation here would be noise dressed as signal.
            "p_up_any": p[2] + p[3], "p_down_any": p[0] + p[1],
            "grade": _grade(g, h, thr)[0], "skill": _grade(g, h, thr)[1],
        })
    return out


# Measured out-of-sample grade and magnitude skill per (asset class, horizon, threshold%).
# BSS2 = Brier skill on P(|move| >= thr) against EACH TICKER'S OWN base rate over 21 walk-forward
# test years -- the honest bar, since a user can get the base rate free from the ticker's history.
# Re-measured by _move_validate.py after the LEV+VOL features landed; those raised index h=1/5/10
# by about +0.004 each and single names by about +0.001.
#
# GRADING RULE (two-factor, so a single arbitrary cutoff does not decide it):
#   可用   BSS2 >= 0.05
#   谨慎   0.02 <= BSS2 < 0.05, OR BSS2 < 0.02 while ECE is still good and >=15/21 years positive
#          (a well-calibrated but low-skill answer is worth showing with a caveat, not hiding)
#   不可用 BSS2 < 0.02 AND fewer than 15 of 21 test years positive
#   无法分辨 the empirical z has no mass past the threshold; the support gate refuses first anyway
GRADES = {
    ("index", 1): {1: ("GREEN", .173), 2: ("GREEN", .196), 3: ("GREEN", .214), 5: ("RES", .223)},
    ("index", 5): {1: ("GREEN", .078), 2: ("GREEN", .137), 3: ("GREEN", .160), 5: ("GREEN", .147)},
    ("index", 10): {1: ("GREEN", .051), 2: ("GREEN", .096), 3: ("GREEN", .124), 5: ("GREEN", .136)},
    ("index", 21): {1: ("RED", .006), 2: ("AMBER", .035), 3: ("GREEN", .069), 5: ("GREEN", .119)},
    ("single", 1): {1: ("GREEN", .081), 2: ("GREEN", .115), 3: ("GREEN", .122), 5: ("GREEN", .109)},
    ("single", 5): {1: ("AMBER", .032), 2: ("GREEN", .056), 3: ("GREEN", .075), 5: ("GREEN", .095)},
    ("single", 10): {1: ("AMBER", .021), 2: ("AMBER", .034), 3: ("AMBER", .047), 5: ("GREEN", .068)},
    ("single", 21): {1: ("RED", .013), 2: ("AMBER", .019), 3: ("AMBER", .025), 5: ("AMBER", .041)},
}

# 显示层翻译。内部代码保持英文不变，便于程序化调用和与研究脚本对照。
GRADE_CN = {"GREEN": "可用", "AMBER": "谨慎", "RED": "不可用", "RES": "无法分辨"}


def _grade(g, h, thr):
    """Measured grade + magnitude skill for this cell. Unmeasured (h, thr) inherit the nearest
    measured threshold within the same (group, horizon)."""
    tbl = GRADES.get((g, h))
    if not tbl:
        near_h = min(GRADES, key=lambda k: (k[0] != g, abs(k[1] - h)))
        tbl = GRADES[near_h]
    t = thr * 100
    key = min(tbl, key=lambda k: abs(k - t))
    return tbl[key]


def predict(symbol, horizons=HORIZONS, thr=0.02, period="2y", model=None) -> list[dict]:
    """Live prediction: pull recent bars for `symbol` and forecast. Drops an in-progress bar."""
    import yfinance as yf

    d = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[d["Close"] > 0].dropna(subset=["Close"])
    now_et = pd.Timestamp.now(tz="America/New_York")
    if len(d) and d.index[-1].date() == now_et.date() and now_et.time() < pd.Timestamp("16:00").time():
        d = d.iloc[:-1]
    if len(d) < 80:
        return [{"h": h, "ok": False, "why": "历史不足80根K线"} for h in horizons]

    vix = None
    if asset_class(symbol) == "index":
        v = yf.download("^VIX", period=period, progress=False, auto_adjust=False)
        if isinstance(v.columns, pd.MultiIndex):
            v.columns = v.columns.get_level_values(0)
        vix = pd.to_numeric(v["Close"], errors="coerce").dropna()
    return predict_from_bars(symbol, d["Close"], d["High"], d["Low"], vix,
                             horizons=horizons, thr=thr, model=model,
                             volume=d["Volume"] if "Volume" in d else None)


# ---------------------------------------------------------------- display
def format_rows(rows, wide=True) -> str:
    lines = []
    if wide:
        lines.append(f"  {'期限':<6}{'典型波动':>10}{'涨超阈值':>9}{'区间内':>9}{'跌超阈值':>9}"
                     f"{'幅度超阈值':>12}{'上涨(任意)':>11}{'期望收益':>10}{'偏差情形':>11}{'乐观情形':>11}"
                     f"{'可信度':>8}{'技能分':>8}")
    for r in rows:
        if not r.get("ok"):
            lines.append(f"  h={r['h']:<4} {r.get('why','不可用')}")
            continue
        if not r["in_support"]:
            why = ("阈值远超该标的波动范围，无法分辨" if r["support"] > SUPPORT_HI
                   else "阈值窄于波动噪音，问题退化为纯方向猜测")
            lines.append(f"  {str(r['h'])+'日':<6}{r['typical_move']*100:>9.2f}%   —— {why} ——")
            continue
        star = "*" if r["earn_mult"] > 1 else " "
        lines.append(f"  {str(r['h'])+'日':<6}{r['typical_move']*100:>9.2f}%{r['p_up_big']*100:>8.1f}%"
                     f"{r['p_within']*100:>8.1f}%{r['p_down_big']*100:>8.1f}%"
                     f"{r['p_move']*100:>11.1f}%{star}{r['p_up_any']*100:>10.1f}%"
                     f"{r['exp_return']*100:>+9.2f}%{r['q05']*100:>10.1f}%{r['q95']*100:>10.1f}%"
                     f"{GRADE_CN.get(r['grade'], r['grade']):>8}{r['skill']:>8.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="移动概率预测：未来N日涨/跌超过X%的概率")
    ap.add_argument("symbols", nargs="*", help="代码，如 AAPL SPY")
    ap.add_argument("--thr", type=float, default=2.0, help="阈值百分比，默认2")
    ap.add_argument("--horizons", default="1,5,10,21")
    ap.add_argument("--train-end", default=None, help="fit 模式：只用该日期之前的数据训练")
    args = ap.parse_args()

    if args.symbols and args.symbols[0] == "fit":
        fit(train_end=args.train_end)
        return
    if not args.symbols:
        ap.error("需要至少一个代码，或使用 `move_prob.py fit`")
    if not MODEL_PATH.exists():
        print("模型不存在，先跑：../../vcp_env/bin/python move_prob.py fit", file=sys.stderr)
        sys.exit(1)

    model = pd.read_pickle(MODEL_PATH)
    hs = tuple(int(x) for x in args.horizons.split(","))
    thr = args.thr / 100.0
    for s in args.symbols:
        rows = predict(s, horizons=hs, thr=thr, model=model)
        cls = "指数/ETF" if asset_class(s) == "index" else "个股"
        print(f"\n{s}  ({cls})  阈值 ±{args.thr:g}%")
        print(format_rows(rows))
    print("\n  可信度：可用 / 谨慎（只看「幅度超阈值」一列）/ 不可用（不要看单个数字）/ 无法分辨")
    print("        判定标准为实测技能分：≥.05 可用，.02–.05 谨慎，<.02 不可用")
    print("  技能分 = Brier skill score（无贴切中译，保留原名）：相对「该标的自身历史频率」")
    print("        减少了多少预测误差，来自 21 年滚动前瞻检验")
    print("  偏差情形 / 乐观情形 = 第 5 / 第 95 百分位收益，即「差到什么程度」「好到什么程度」")
    print("  期望收益 = 分布的均值（0.5%/99.5% 截尾）。它等于「漂移 × 波动率」，同样不含方向信息：")
    print("        波动越大的标的这一列必然越高，纯属结构使然，不是说它更值得买。")
    print("        实证上低波动股票的风险调整后收益反而更好，所以不要按这一列排序选股。")
    print("  * = 该窗口内有财报，波动率已按财报乘数放大（仅个股）")
    print("  注意：技能全部集中在「幅度超阈值」一列。涨跌方向的拆分由漂移和偏度决定，不是预测。")
    print("  「上涨(任意)」= P(收益>0)。同一资产类别与期限下，它对所有标的都是同一个数字——")
    print("        因为判定点 0/波动率 = 0，与波动率无关。这正是模型在如实说明：它没有方向信息。")


if __name__ == "__main__":
    main()
