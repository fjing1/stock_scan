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


def build_features(close, high=None, low=None, vix=None) -> pd.DataFrame:
    """The HAR feature block, all computable from bars up to and including t.

    Clipping into [1e-3, 0.5] BEFORE taking logs is not cosmetic: ~0.5% of one-bar Parkinson
    values are exactly zero (high == low on halted/illiquid bars) and with a 1e-5 floor those
    become ln = -11.5 outliers that cost 0.03-0.04 R-squared."""
    has_ohlc = high is not None and low is not None
    rv1 = L.vol_parkinson(high, low, 1) if has_ohlc else L._logret(close).abs()
    f = pd.DataFrame(index=close.index)
    f["rv_d"] = _clip_log(rv1)
    f["rv_w"] = _clip_log(rv1.rolling(5).mean())
    f["rv_m"] = _clip_log(rv1.rolling(22).mean())
    f["rv_q"] = _clip_log(rv1.rolling(63).mean())
    f["ewma97"] = _clip_log(L.vol_ewma(close, 0.97))
    if vix is not None:
        f["logvix"] = np.log(np.clip(vix.reindex(close.index).ffill(limit=3) / 100 / np.sqrt(252),
                                     *VOL_CLIP))
    return f


FEATS_OHLC = ["rv_d", "rv_w", "rv_m", "rv_q", "ewma97"]


def _design(f: pd.DataFrame, cols) -> np.ndarray:
    return np.column_stack([np.ones(len(f))] + [f[c].values for c in cols])


# ---------------------------------------------------------------- fitting
def _ols(X, y):
    keep = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if keep.sum() < 200:
        return None, 0
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return beta, int(keep.sum())


def fit(panel=None, train_end: str | None = None, verbose=True) -> dict:
    """Fit HAR vol coefficients and the empirical z tables, per asset class and horizon.

    train_end: ISO date. Rows on/after it are excluded entirely, so a held-out evaluation is
    possible. None = use everything (what you ship)."""
    import _move_data as D

    panel = panel or D.load()
    C, H, Lo = panel["Close"], panel["High"], panel["Low"]
    vix = C["^VIX"] if "^VIX" in C.columns else None
    if train_end:
        cut = pd.Timestamp(train_end)
        C, H, Lo = C[C.index < cut], H[H.index < cut], Lo[Lo.index < cut]
        vix = vix[vix.index < cut] if vix is not None else None

    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    groups = {"index": [s for s in usable if s in INDEX_LIKE],
              "single": [s for s in usable if s not in INDEX_LIKE]}
    model = {"groups": {}, "train_end": train_end, "horizons": list(HORIZONS)}

    for gname, syms in groups.items():
        cols = FEATS_OHLC + (["logvix"] if (gname == "index" and vix is not None) else [])
        per_h = {}
        for h in HORIZONS:
            Xs, ys = [], []
            for s in syms:
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                f = build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix)
                tgt = _clip_log(L.realized_vol_forward(c, h))
                Xs.append(_design(f, cols))
                ys.append(tgt.values)
            if not Xs:
                continue
            X, y = np.vstack(Xs), np.concatenate(ys)
            beta, n = _ols(X, y)
            if beta is None:
                continue

            # z table, fit with EXACTLY the sigma spec that ships (any constant per-h scale bias
            # in sigma is absorbed here by construction -- that is the point).
            zs = []
            for s in syms:
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                f = build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix)
                sig = np.exp(_design(f, cols) @ beta) * np.sqrt(h)
                lr = np.log(c.shift(-h) / c)          # LOG forward return (see module docstring)
                z = (lr.values / sig)
                zs.append(z[np.isfinite(z)])
            zcat = np.sort(np.concatenate(zs))
            per_h[h] = {"beta": beta, "cols": cols, "n_fit": n,
                        "z": zcat.astype(np.float32), "n_z": len(zcat)}
            if verbose:
                print(f"  {gname:<7} h={h:<3} n_fit={n:>9,}  n_z={len(zcat):>9,}  "
                      f"z: med={np.median(zcat):+.3f} sd={zcat.std():.3f} "
                      f"P(z<-2)={np.mean(zcat < -2):.4f} P(z>2)={np.mean(zcat > 2):.4f}")
        model["groups"][gname] = {"symbols": syms, "per_h": per_h}

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
                      thr=0.02, model=None, earnings_in=None) -> list[dict]:
    """Core predictor. earnings_in: set of horizons whose window contains an earnings date."""
    model = model or pd.read_pickle(MODEL_PATH)
    g = asset_class(symbol)
    grp = model["groups"][g]
    f = build_features(close, high, low, vix)
    out = []
    for h in horizons:
        spec = grp["per_h"].get(h)
        if spec is None:
            continue
        cols = spec["cols"]
        if any(c not in f.columns for c in cols):        # e.g. VIX unavailable for an index
            fallback = [c for c in cols if c in f.columns]
            beta = spec["beta"][:1 + len(fallback)]
            row = _design(f.iloc[[-1]], fallback)
        else:
            beta, row = spec["beta"], _design(f.iloc[[-1]], cols)
        if not np.isfinite(row).all():
            out.append({"h": h, "ok": False, "why": "特征不足（历史长度不够或含缺口）"})
            continue

        sig_daily = float(np.exp(row @ beta))
        sig_h = sig_daily * np.sqrt(h)
        mult = EARN_MULT.get(h, 1.0) if (earnings_in and h in earnings_in and g == "single") else 1.0
        sig_h *= mult

        a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
        ratio = a_up / sig_h
        z = spec["z"]
        F_dn, F_0, F_up = _cdf(z, a_dn / sig_h), _cdf(z, 0.0), _cdf(z, a_up / sig_h)
        p = np.array([F_dn, max(F_0 - F_dn, 0), max(F_up - F_0, 0), max(1 - F_up, 0)])
        p = np.clip(p, 1e-4, None)
        p = p / p.sum()

        out.append({
            "h": h, "ok": True, "symbol": symbol, "asset_class": g, "thr": thr,
            "sigma_daily": sig_daily, "sigma_h": sig_h, "earn_mult": mult,
            "support": float(ratio), "in_support": SUPPORT_LO <= ratio <= SUPPORT_HI,
            "p_down_big": p[0], "p_down_small": p[1], "p_up_small": p[2], "p_up_big": p[3],
            "p_move": p[0] + p[3], "p_within": p[1] + p[2],
            "grade": _grade(g, h, thr),
        })
    return out


def _grade(g, h, thr):
    """Traffic light from the walk-forward study. GREEN = skill positive in 19-21 of 21 test
    years; AMBER = magnitude only; RED = do not show a per-ticker number."""
    t = round(thr * 100, 1)
    if g == "single":
        if h == 21:
            return "RED"          # 39% of tickers score at or below their own base rate
        if h == 10 and t < 2:
            return "RED"
        if h == 10:
            return "AMBER"
        return "GREEN"
    if h == 1 and t >= 5:
        return "RED"              # index cannot resolve a 5% one-day move
    if h >= 10 and t <= 1:
        return "RED"
    if h == 21 and t <= 2:
        return "AMBER"
    return "GREEN"


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
                             horizons=horizons, thr=thr, model=model)


# ---------------------------------------------------------------- display
def format_rows(rows, wide=True) -> str:
    lines = []
    if wide:
        lines.append(f"  {'期限':<6}{'σ_h':>8}{'涨>阈值':>10}{'区间内':>9}{'跌>阈值':>10}"
                     f"{'|移动|>阈值':>13}{'评级':>7}")
    for r in rows:
        if not r.get("ok"):
            lines.append(f"  h={r['h']:<4} {r.get('why','不可用')}")
            continue
        if not r["in_support"]:
            why = "阈值远超波动范围，无法分辨" if r["support"] > SUPPORT_HI else "阈值窄于噪音，退化为方向猜测"
            lines.append(f"  {str(r['h'])+'日':<6}{r['sigma_h']*100:>7.2f}%   —— {why} ——")
            continue
        star = "*" if r["earn_mult"] > 1 else " "
        lines.append(f"  {str(r['h'])+'日':<6}{r['sigma_h']*100:>7.2f}%{r['p_up_big']*100:>9.1f}%"
                     f"{r['p_within']*100:>8.1f}%{r['p_down_big']*100:>9.1f}%"
                     f"{r['p_move']*100:>12.1f}%{star}{r['grade']:>6}")
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
    print("\n  评级 GREEN=可用 / AMBER=只看|移动|一列 / RED=不要看单一数字")
    print("  * = 该窗口内有财报，σ 已按财报乘数放大（仅个股）")
    print("  注意：全部技能都在'|移动|>阈值'这一列。涨跌方向的拆分主要由漂移和偏度决定，不是预测。")


if __name__ == "__main__":
    main()
