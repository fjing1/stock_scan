"""
iv_snapshot.py -- daily per-stock implied-volatility collector (free, keyless, via yfinance).

WHY THIS EXISTS
    move_prob.py forecasts volatility from backward-looking bars only. Option IV is the one
    forward-looking vol input available for free. Measured on an independent 2019-10..2023-07
    per-stock IV history (see _data_probe_iv_effect2.py / _data_probe_iv_effect3.py), adding a
    stock's OWN ATM IV on top of the shipped single-name feature set AND market VIX is worth, in
    out-of-sample forward-log-sigma R^2:
        h=1  +0.0013 [+0.0006,+0.0020]      h=5   +0.0129 [+0.0093,+0.0165]
        h=10 +0.0210 [+0.0136,+0.0278]      h=21  +0.0257 [+0.0115,+0.0418]
    versus market-wide VIX on the same rows, which is worth only +0.0018/+0.0049/+0.0013/+0.0000.
    With the IV strictly lagged one day it is still +0.0095/+0.0184/+0.0261 at h=5/10/21, and it is
    positive in 9-10 of 10 walk-forward quarters. So the feature is real; we just have no history,
    which is why this file accumulates it going forward.

WHAT IT WRITES
    _iv_history.csv, one row per (date, symbol), appended daily and never rewritten. Columns are
    documented in FIELDS below. Re-running the same day is idempotent (existing rows are kept).

EVERY DESIGN DECISION BELOW IS A MEASUREMENT, NOT A PREFERENCE
(_data_probe_iv_chain_anatomy.py, 10 symbols x 8 expiries x both sides, 2026-09-14):

  * ATM BY INTERPOLATION, NOT NEAREST STRIKE. |nearest - interpolated| averages 0.655 vol points
    but that average hides the whole point: on mega-caps the strike grid is 0.13-1.5% of spot wide
    and the choice is irrelevant (AAPL 0.13, SPY 0.22, NVDA 0.19 mean pts). On thin names the grid
    is 4.6-6.5% of spot wide and nearest-strike is off by up to 8.2 vol points (YELP mean 2.67,
    EPAM max 8.19). Interpolating is free, so interpolate.

  * MID OF CALL AND PUT, AND KEEP THE SPREAD AS A QUALITY FLAG. Calls and puts do NOT agree:
    mean(C-P) = +2.24 vol points, mean|C-P| = 4.62, p90 = 9.40, max 17.68. The bias is
    systematically positive (Yahoo's IV solve does not treat carry/dividends consistently, and some
    IVs come off stale last trades rather than the quote). A single side is therefore a coin flip
    on a ~4.6-point error; the mid halves it and cp_spread_pts lets a consumer reject the bad rows.

  * DROP DTE < MIN_DTE (7). The nearest expiry is not merely noisy, it is BIASED LOW, and that is
    the pattern flagged for AAPL (22.9% for the 2026-09-14 expiry vs 27.1% for 2026-09-18). Same-day
    expiry reads, mid of C/P: AAPL 17.6 (0 DTE) vs 25.4 (2 DTE) vs 26.3 (4 DTE); MSFT 18.5/25.5/26.3;
    NVDA 23.9/34.7/36.1; SPY 7.8 (0 DTE) / 10.9 (1) / 14.0 (2) / 15.3 (4). So 0-1 DTE understates by
    7-11 vol points. Mechanism: hours from expiry the option's time value is a few cents, so tick
    discreteness dominates the price the IV is solved from, and the remaining-time denominator is a
    fraction of a day. The 3-7 DTE bucket is also where relative ATM spreads are worst (median
    8.5% of mid, max 200%). Everything >= 7 DTE is stable, so start there.

  * CONSTANT-MATURITY 30-DAY IV BY LINEAR-IN-TOTAL-VARIANCE INTERPOLATION. Raw nearest-expiry ATM
    IV sawtooths with the expiry cycle. Interpolating w*var*T across the two expiries straddling 30
    days is the VIX construction and removes that. This matters: cross-checked against CBOE's own
    single-stock IV30 indices (VXAPL/VXAZN/VXGOG/VXGS/VXIBM, recovered from the Wayback Machine),
    a raw nearest-expiry ATM IV series correlates only 0.32-0.88 with the true constant-maturity
    IV30 with sd(diff) ~3 vol points -- i.e. the sawtooth is a large fraction of the signal. And
    smoothing helps where it should: at h=21 a 5-day-averaged own-IV beat the raw one (+0.0303 vs
    +0.0257 dR^2).

  * LIQUIDITY FILTER: bid > 0 AND (openInterest >= MIN_OI OR volume >= 1). Measured ATM openInterest
    medians collapse with maturity: 1055 (0-2 DTE), 440 (3-7), 215 (8-21), 155 (22-45), 32 (46-120),
    12.5 (120+); ATM volume medians 5961 -> 187 -> 26.5 -> 5 -> 1 -> 2.5; and 15-31% of strikes quote
    a zero bid. A zero-bid strike has no price to invert, so its impliedVolatility is a placeholder,
    not a measurement. Both straddling strikes must survive the filter or the expiry is discarded.

  * SKEW IS NOT WORTH COLLECTING FOR THIS PURPOSE (but it is nearly free, so it is recorded).
    Adding the IV skew and slope on top of ATM IV moved dR^2 by -0.0018..+0.0000 at every horizon.

Usage
    ../../vcp_env/bin/python iv_snapshot.py                  # default universe
    ../../vcp_env/bin/python iv_snapshot.py AAPL MSFT NVDA
    ../../vcp_env/bin/python iv_snapshot.py --universe panel  # every single name in _move_panel.pkl
    from iv_snapshot import collect, iv_term_structure
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

STORE = Path(__file__).with_name("_iv_history.csv")

MIN_DTE = 7           # 0-1 DTE reads 7-11 vol points LOW; see module docstring
MAX_DTE = 400
MIN_OI = 10           # a zero-bid / no-interest strike has no invertible price
TARGET_DAYS = (30, 60, 90)   # constant-maturity points to publish; 30d is the per-stock VIX
MAX_CP_SPREAD = 0.20      # absolute backstop: |callIV - putIV| above 20 vol pts is junk
MAX_CP_SPREAD_REL = 0.30  # RELATIVE gate, and the one that actually binds. Measured on the full
                          # 280-name panel (2026-09-14, n=144 with a usable iv30): the absolute
                          # 20-point rule fires on only 2.8% of rows and misses the real failures,
                          # because a few absolute points is a huge relative error on a low-IV name.
                          # It passed LIN (19.4 pts on iv30=31% -> 63% disagreement), ARR (59%),
                          # DUK (49%) and ATO (47%) as "ok" -- all utilities/REITs. The relative
                          # spread distribution is median 0.134 / p75 0.203 / p90 0.305 / p99 0.609,
                          # so 0.30 flags the worst ~13%: rows where two estimates of the same
                          # quantity disagree by a third of its own level are not data.
EXTRAP_TOL = 5        # days a constant-maturity target may sit outside the listed range unflagged

FIELDS = [
    "date", "symbol", "spot",
    "iv30", "iv60", "iv90",          # constant-maturity, variance-time interpolated, mid of C/P
    "iv30_call", "iv30_put",         # same construction, one side only (for auditing the mid)
    "ts_slope",                      # iv60 - iv30, the term-structure slope
    "iv_near", "dte_near",           # nearest usable expiry, unsmoothed
    "skew25",                        # 25-delta-ish put IV minus ATM IV at the 30d anchor
    "cp_spread_pts",                 # |callIV - putIV| at the 30d anchor, in vol points
    "cp_spread_rel",                 # cp_spread_pts / (100*iv30) -- the gate that actually binds
    "n_exp_used", "atm_oi", "atm_vol", "quality",
]


# ----------------------------------------------------------------- one expiry
def _atm_iv_one_side(df: pd.DataFrame, spot: float) -> tuple[float, float, float, int]:
    """Interpolated ATM IV for one side of one expiry, plus ATM OI/volume and usable-strike count.

    Returns (iv, oi, volume, n_usable). iv is NaN when the expiry cannot be trusted.
    """
    if df is None or df.empty:
        return np.nan, np.nan, np.nan, 0
    d = df.copy()
    for c in ("impliedVolatility", "openInterest", "volume", "bid", "ask", "strike"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["openInterest"] = d["openInterest"].fillna(0)
    d["volume"] = d["volume"].fillna(0)
    # liquidity filter: a zero bid means there is no price to invert
    d = d[(d.bid > 0) & (d.impliedVolatility > 0) & d.strike.notna()
          & ((d.openInterest >= MIN_OI) | (d.volume >= 1))]
    if len(d) < 4:
        return np.nan, np.nan, np.nan, len(d)
    below = d[d.strike <= spot].sort_values("strike")
    above = d[d.strike > spot].sort_values("strike")
    if below.empty or above.empty:
        return np.nan, np.nan, np.nan, len(d)
    lo, hi = below.iloc[-1], above.iloc[0]
    span = hi.strike - lo.strike
    w = 0.0 if span <= 0 else float((spot - lo.strike) / span)
    iv = (1 - w) * float(lo.impliedVolatility) + w * float(hi.impliedVolatility)
    return iv, float(lo.openInterest + hi.openInterest), float(lo.volume + hi.volume), len(d)


def _put_skew(puts: pd.DataFrame, spot: float, atm_iv: float) -> float:
    """IV of the OTM put nearest 90% of spot, minus ATM IV. Cheap skew proxy, no greeks needed."""
    if puts is None or puts.empty or not np.isfinite(atm_iv):
        return np.nan
    d = puts.copy()
    d["impliedVolatility"] = pd.to_numeric(d.impliedVolatility, errors="coerce")
    d["strike"] = pd.to_numeric(d.strike, errors="coerce")
    d["bid"] = pd.to_numeric(d.bid, errors="coerce")
    d = d[(d.bid > 0) & (d.impliedVolatility > 0) & (d.strike < spot)]
    if d.empty:
        return np.nan
    tgt = 0.90 * spot
    row = d.iloc[(d.strike - tgt).abs().argsort()[:1]].iloc[0]
    if abs(row.strike / spot - 0.90) > 0.07:      # no strike anywhere near 90% -> unusable
        return np.nan
    return float(row.impliedVolatility) - atm_iv


# ----------------------------------------------------------------- term structure
def _interp_var_time(pts: list[tuple[float, float]], target: float) -> float:
    """Constant-maturity IV at `target` days, linear in TOTAL VARIANCE (iv^2 * T) -- the VIX rule.

    pts is [(dte, iv), ...] with dte > 0. Interpolates between the two expiries straddling the
    target. Extrapolates flat in IV (not in variance) beyond the ends, which is the conservative
    choice: extrapolating variance linearly off the short end can produce negative variance.
    """
    p = sorted((float(t), float(v)) for t, v in pts if np.isfinite(t) and np.isfinite(v) and t > 0)
    if not p:
        return np.nan
    if len(p) == 1 or target <= p[0][0]:
        return p[0][1]
    if target >= p[-1][0]:
        return p[-1][1]
    for (t1, v1), (t2, v2) in zip(p, p[1:]):
        if t1 <= target <= t2:
            w1, w2 = v1 * v1 * t1, v2 * v2 * t2
            tv = w1 + (w2 - w1) * (target - t1) / (t2 - t1)
            return float(np.sqrt(max(tv, 0.0) / target))
    return np.nan


#: Ladder the chain fetches are spread over. Taking simply the FIRST N expiries is wrong: names
#: with dense weeklies (AAPL/SPY/QQQ/NVDA) have 10 expiries inside 39 days, so iv60 and iv90 would
#: both silently flat-extrapolate off the 39-day point and come out identical, which would make
#: ts_slope a fabricated zero. Picking the expiry nearest each rung keeps the same request budget
#: while actually spanning the curve.
LADDER = (7, 14, 21, 30, 45, 60, 90, 120, 180, 365)


def _pick_expiries(cand: list[tuple[str, int]], budget: int) -> list[tuple[str, int]]:
    """Choose <=budget expiries spread over LADDER so the curve is spanned, not just its front."""
    chosen, used = [], set()
    for rung in LADDER:
        if len(chosen) >= budget:
            break
        rest = [(e, d) for e, d in cand if e not in used]
        if not rest:
            break
        e, d = min(rest, key=lambda x: abs(x[1] - rung))
        used.add(e)
        chosen.append((e, d))
    return sorted(chosen, key=lambda x: x[1])


def iv_term_structure(symbol: str, ticker: yf.Ticker | None = None,
                      max_expiries: int = 10) -> dict:
    """Pull one symbol's chain and return the FIELDS dict. Never raises; sets quality on failure."""
    out = {k: np.nan for k in FIELDS}
    out["symbol"] = symbol
    out["date"] = pd.Timestamp.today().normalize().date().isoformat()
    t = ticker or yf.Ticker(symbol)
    try:
        exps = list(t.options or [])
    except Exception as e:
        out["quality"] = f"no_options:{type(e).__name__}"
        return out
    if not exps:
        out["quality"] = "no_expiries"
        return out
    spot = np.nan
    try:
        spot = float(t.fast_info["last_price"])
    except Exception:
        pass
    if not np.isfinite(spot):
        try:
            spot = float(t.history(period="5d")["Close"].dropna().iloc[-1])
        except Exception:
            out["quality"] = "no_spot"
            return out
    out["spot"] = round(spot, 4)

    today = pd.Timestamp.today().normalize()
    cand = [(e, (pd.Timestamp(e) - today).days) for e in exps]
    cand = [(e, d) for e, d in cand if MIN_DTE <= d <= MAX_DTE]
    cand = _pick_expiries(cand, max_expiries)
    if not cand:
        out["quality"] = "no_expiry_in_dte_window"
        return out

    mids, calls_ts, puts_ts, meta = [], [], [], {}
    for e, dte in cand:
        try:
            ch = t.option_chain(e)
        except Exception:
            continue
        ivc, oic, volc, nc = _atm_iv_one_side(ch.calls, spot)
        ivp, oip, volp, npu = _atm_iv_one_side(ch.puts, spot)
        both = [v for v in (ivc, ivp) if np.isfinite(v)]
        if not both:
            continue
        mid = float(np.mean(both))
        cp = abs(ivc - ivp) if (np.isfinite(ivc) and np.isfinite(ivp)) else np.nan
        mids.append((dte, mid))
        if np.isfinite(ivc):
            calls_ts.append((dte, ivc))
        if np.isfinite(ivp):
            puts_ts.append((dte, ivp))
        meta[dte] = dict(cp=cp, oi=np.nansum([oic, oip]), vol=np.nansum([volc, volp]),
                         puts=ch.puts, atm=mid)
    if not mids:
        out["quality"] = "no_usable_expiry"
        return out

    mids.sort()
    out["iv_near"] = round(mids[0][1], 6)
    out["dte_near"] = int(mids[0][0])
    out["n_exp_used"] = len(mids)
    for tgt in TARGET_DAYS:
        out[f"iv{tgt}"] = round(_interp_var_time(mids, tgt), 6)
    out["iv30_call"] = round(_interp_var_time(calls_ts, 30), 6) if calls_ts else np.nan
    out["iv30_put"] = round(_interp_var_time(puts_ts, 30), 6) if puts_ts else np.nan
    if np.isfinite(out["iv60"]) and np.isfinite(out["iv30"]):
        out["ts_slope"] = round(out["iv60"] - out["iv30"], 6)

    anchor = min(meta, key=lambda d: abs(d - 30))
    m = meta[anchor]
    out["cp_spread_pts"] = round(100 * m["cp"], 3) if np.isfinite(m["cp"]) else np.nan
    out["atm_oi"] = float(m["oi"])
    out["atm_vol"] = float(m["vol"])
    sk = _put_skew(m["puts"], spot, m["atm"])
    out["skew25"] = round(sk, 6) if np.isfinite(sk) else np.nan

    flags = []
    if not np.isfinite(out["iv30"]):
        flags.append("no_iv30")
    rel = (out["cp_spread_pts"] / (100 * out["iv30"])
           if (np.isfinite(out.get("cp_spread_pts", np.nan))
               and np.isfinite(out.get("iv30", np.nan)) and out["iv30"] > 0) else np.nan)
    out["cp_spread_rel"] = rel
    if np.isfinite(out["cp_spread_pts"]) and (
            out["cp_spread_pts"] > 100 * MAX_CP_SPREAD
            or (np.isfinite(rel) and rel > MAX_CP_SPREAD_REL)):
        flags.append("cp_spread_wide")
    if out["n_exp_used"] < 2:
        flags.append("single_expiry_no_interp")
    # Flag every constant-maturity point that had to be EXTRAPOLATED rather than interpolated.
    # EXTRAP_TOL exists because most mid-caps only list the monthly cycle, so with MIN_DTE=7 the
    # front usable expiry is routinely 32 DTE and the 30-day target sits 2 days outside the range.
    # Calling that "extrapolated" is technically true and practically meaningless -- measured on the
    # 278-name panel it fired on 118 of 196 usable rows, almost all of them 1-3 days over. Only a
    # gap wider than EXTRAP_TOL days is a real reach.
    dte_max = max(d for d, _ in mids)
    dte_min = min(d for d, _ in mids)
    for tgt in TARGET_DAYS:
        if tgt > dte_max + EXTRAP_TOL or tgt < dte_min - EXTRAP_TOL:
            flags.append(f"iv{tgt}_extrapolated")
    if np.isfinite(out["atm_oi"]) and out["atm_oi"] < 4 * MIN_OI:
        flags.append("thin_atm")
    out["quality"] = ",".join(flags) if flags else "ok"
    return out


# ----------------------------------------------------------------- store
def collect(symbols, store: Path = STORE, sleep: float = 0.0, verbose: bool = True) -> pd.DataFrame:
    """Snapshot `symbols` and append to the dated store. Idempotent per (date, symbol)."""
    rows, t0 = [], time.time()
    for i, s in enumerate(symbols, 1):
        ts = time.time()
        r = iv_term_structure(s)
        r["_secs"] = round(time.time() - ts, 2)
        rows.append(r)
        if verbose:
            iv = r["iv30"]
            print(f"  [{i:>3}/{len(symbols)}] {s:<6} iv30="
                  f"{(f'{100*iv:6.2f}%' if np.isfinite(iv) else '   n/a')}"
                  f"  near={r['dte_near']}d  nexp={r['n_exp_used']}"
                  f"  cp={r['cp_spread_pts']}  {r['_secs']:>5.2f}s  {r['quality']}")
        if sleep:
            time.sleep(sleep)
    new = pd.DataFrame(rows)[FIELDS + ["_secs"]]
    if store.exists():
        old = pd.read_csv(store)
        both = pd.concat([old, new], ignore_index=True)
        both = both.drop_duplicates(subset=["date", "symbol"], keep="first")
    else:
        both = new
    both.sort_values(["date", "symbol"]).to_csv(store, index=False)
    if verbose:
        ok = int(new.quality.eq("ok").sum())
        got = int(new.iv30.notna().sum())
        print(f"\n{len(new)} symbols in {time.time()-t0:.1f}s "
              f"({(time.time()-t0)/max(len(new),1):.2f}s/symbol) | "
              f"iv30 present {got}/{len(new)} | quality=ok {ok}/{len(new)} | store -> {store}")
    return new


def _panel_universe() -> list[str]:
    import pickle
    import move_prob as MP
    p = pickle.load(open(Path(__file__).with_name("_move_panel.pkl"), "rb"))
    return [s for s in p["Close"].columns if s not in MP.INDEX_LIKE and not s.startswith("^")]


DEFAULT = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ", "JNJ", "KLAC", "INTU", "BMY",
           "EPAM", "PCTY", "QLYS", "YELP", "FORM", "AVAV"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--universe", choices=["default", "panel"], default="default")
    ap.add_argument("--sleep", type=float, default=0.0)
    a = ap.parse_args()
    syms = a.symbols or (_panel_universe() if a.universe == "panel" else DEFAULT)
    print(f"iv_snapshot: {len(syms)} symbols, DTE>={MIN_DTE}, MIN_OI={MIN_OI}, store={STORE.name}")
    collect(syms, sleep=a.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
