"""Exhaustive trend-filter LENGTH sweep: Close > MA(n) for n = 1..200, SMA and EMA.

Holds the validated oversold entry (RSI<40 & %K<20) and the 10-day forward exit FIXED;
varies only the trend-filter MA length, finely, across the whole 1..200 range. This is
~400 configs, so the single best length is almost certainly overfit — the honest output is
the EDGE-vs-LENGTH CURVE: a smooth, broadly elevated REGION is a robust signal; a lone spike
at one length is noise. Reported with train/test stability, a smoothed curve to locate the
stable band, an ASCII plot, and a Deflated-Sharpe haircut for the full search.

CAVEAT: current-names (survivorship) universe; detrended-vs-SPY mean + train/test agreement
are the trustworthy reads. Adjacent lengths are nearly identical signals, so the effective
number of independent trials is far below 400 — read the band, not the argmax.

    python ma_length_curve.py [--names N] [--horizon 10] [--max-n 200]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cycle_patter_for_swing import compute_cycle_stoch  # noqa: E402
from walkforward import deflated_sharpe  # noqa: E402
import yfinance as yf  # noqa: E402

BENCHMARK = "SPY"
SPLIT = pd.Timestamp("2019-01-01")
N_CANDIDATES = 220
MIN_BARS = 2500


def fetch_batch(symbols, period, interval):
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=False, progress=False, group_by="ticker", threads=True)
    out = {}
    for s in symbols:
        try:
            d = raw[s].dropna(how="all").copy()
        except Exception:
            continue
        if len(d):
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def stat(arr):
    a = np.asarray(arr, float); a = a[~np.isnan(a)]
    if len(a) < 20:
        return len(a), float("nan"), float("nan"), float("nan")
    sd = a.std(ddof=1)
    mean = a.mean()
    t = mean / (sd / np.sqrt(len(a))) if sd > 0 else float("nan")
    return len(a), (a > 0).mean() * 100, mean * 100, t


def spark(vals):
    chars = " ▁▂▃▄▅▆▇█"
    v = np.array(vals, float)
    fin = v[~np.isnan(v)]
    if len(fin) == 0:
        return " " * len(v)
    lo, hi = fin.min(), fin.max()
    out = []
    for x in v:
        if np.isnan(x):
            out.append(" ")
        else:
            out.append(chars[int(round((x - lo) / (hi - lo + 1e-12) * (len(chars) - 1)))])
    return "".join(out)


def smooth(v, w=11):
    s = pd.Series(v).rolling(w, center=True, min_periods=3).mean()
    return s.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", type=int, default=N_CANDIDATES)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max-n", type=int, default=200)
    args = ap.parse_args()
    H, MAXN = args.horizon, args.max_n
    lens = list(range(1, MAXN + 1))

    from stock_symbols_1243 import STOCK_SYMBOLS
    cands = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s != BENCHMARK][:args.names]
    print(f"downloading {len(cands)} + {BENCHMARK} (20y daily) ...", flush=True)
    data = {}
    for i in range(0, len(cands), 110):
        data.update(fetch_batch(cands[i:i + 110], "20y", "1d"))
    spy = fetch_batch([BENCHMARK], "20y", "1d").get(BENCHMARK)
    spy_close = spy["Close"].astype(float)
    spy_fwd = spy_close.shift(-H) / spy_close - 1

    keep = {s: d for s, d in data.items() if len(d) >= MIN_BARS and float(d["Close"].iloc[-1]) > 5}
    print(f"universe: {len(keep)} names | entry RSI<40 & %K<20 + Close>MA(n) | exit +{H}d detrended | "
          f"sweeping n=1..{MAXN} (SMA & EMA), cooldown {H}d/name", flush=True)

    # buckets[matype][n] -> {'tr':[...], 'te':[...]}
    buckets = {"SMA": {n: {"tr": [], "te": []} for n in lens},
               "EMA": {n: {"tr": [], "te": []} for n in lens}}
    none_tr, none_te = [], []

    for j, (sym, df) in enumerate(keep.items()):
        if not {"High", "Low", "Close"}.issubset(df.columns):
            continue
        try:
            cs = compute_cycle_stoch(df)
        except Exception:
            continue
        c = df["Close"].astype(float)
        cv = c.values
        oversold = ((cs["rsi"] < 40) & (cs["stoch_k"] < 20)).fillna(False).values
        det = (c.shift(-H) / c - 1 - spy_fwd.reindex(c.index)).values
        dates = c.index
        is_tr = np.asarray(dates < SPLIT)
        # baseline "none"
        last = -10**9
        for i in np.where(oversold)[0]:
            if i - last < H or np.isnan(det[i]):
                continue
            last = i
            (none_tr if is_tr[i] else none_te).append(det[i])
        # length sweep
        for mat in ("SMA", "EMA"):
            for n in lens:
                ma = (c.rolling(n).mean() if mat == "SMA" else c.ewm(span=n, adjust=False).mean()).values
                sig = oversold & (cv > ma)
                last = -10**9
                b = buckets[mat][n]
                for i in np.where(sig)[0]:
                    if i - last < H or np.isnan(det[i]):
                        continue
                    last = i
                    (b["tr"] if is_tr[i] else b["te"]).append(det[i])
        if (j + 1) % 25 == 0:
            print(f"  ...{j+1}/{len(keep)} names", flush=True)

    # aggregate curves
    curve = {}
    for mat in ("SMA", "EMA"):
        rows = {}
        for n in lens:
            ntr, _, mtr, _ = stat(buckets[mat][n]["tr"])
            nte, wte, mte, tte = stat(buckets[mat][n]["te"])
            rows[n] = dict(ntr=ntr, mtr=mtr, nte=nte, wte=wte, mte=mte, tte=tte)
        curve[mat] = rows

    nb, wb, mb, tb = stat(none_te)
    print(f"\nbaseline 'none' (oversold only): OOS mean {mb:+.2f}%/trade  win {wb:.0f}%  n {nb}")

    # sampled table
    sample = [n for n in lens if n in (1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 60, 75, 89,
                                       100, 110, 125, 144, 150, 160, 175, 200) and n <= MAXN]
    print(f"\n{'n':>4} | {'SMA: trMean teMean teWin teN  t':<38} | {'EMA: trMean teMean teWin teN  t':<38}")
    for n in sample:
        s, e = curve["SMA"][n], curve["EMA"][n]
        print(f"{n:>4} | {s['mtr']:>7.2f}{s['mte']:>8.2f}{s['wte']:>6.0f}{s['nte']:>6}{s['tte']:>5.1f}"
              f"            | {e['mtr']:>7.2f}{e['mte']:>8.2f}{e['wte']:>6.0f}{e['nte']:>6}{e['tte']:>5.1f}")

    # ASCII curve of OOS mean vs n (1..MAXN)
    print(f"\nOOS detrended mean%/trade vs MA length (n=1..{MAXN}); each cell = one length:")
    for mat in ("SMA", "EMA"):
        vals = [curve[mat][n]["mte"] for n in lens]
        print(f"  {mat} |{spark(vals)}|")
    print(f"       {'^1':<1}{' '*(48)}{'^50':<1}{' '*(48)}{'^100':<1}{' '*(47)}{'^150':<1}{' '*(45)}{'^200'}")

    # robust band: smoothed OOS mean, require persistence (train>0) and enough trades
    print("\n--- robust band (smoothed OOS mean; only lengths with teN>=300 & trMean>0) ---")
    best = {}
    for mat in ("SMA", "EMA"):
        mte = np.array([curve[mat][n]["mte"] if (curve[mat][n]["nte"] >= 300 and curve[mat][n]["mtr"] > 0)
                        else np.nan for n in lens])
        sm = smooth(mte, 11)
        if np.all(np.isnan(sm)):
            print(f"  {mat}: no length with enough robust trades"); continue
        pk = int(np.nanargmax(sm))
        peak_n = lens[pk]
        thr = 0.9 * np.nanmax(sm)
        band = [lens[i] for i in range(len(lens)) if not np.isnan(sm[i]) and sm[i] >= thr]
        # also raw best by OOS t-stat among robust
        cand = [(n, curve[mat][n]) for n in lens if curve[mat][n]["nte"] >= 300 and curve[mat][n]["mtr"] > 0]
        tbest = max(cand, key=lambda x: (x[1]["tte"] if not np.isnan(x[1]["tte"]) else -9)) if cand else None
        best[mat] = (peak_n, band, tbest)
        print(f"  {mat}: smoothed-peak n≈{peak_n} (smooth mean {np.nanmax(sm):.2f}%); "
              f"robust band n∈[{min(band)}..{max(band)}]; "
              f"best-by-t n={tbest[0]} (teMean {tbest[1]['mte']:+.2f}%, t {tbest[1]['tte']:.1f}, n {tbest[1]['nte']})")

    # DSR haircut for the single best length found (treat full 1..MAXN x2 as the search)
    allcand = []
    for mat in ("SMA", "EMA"):
        for n in lens:
            r = curve[mat][n]
            if r["nte"] >= 300 and r["mtr"] > 0 and not np.isnan(r["tte"]):
                allcand.append((mat, n, r))
    if allcand:
        bm, bn, br = max(allcand, key=lambda x: x[2]["tte"])
        pp = []
        for mat, n, r in allcand:
            a = np.asarray(buckets[mat][n]["te"], float); a = a[~np.isnan(a)]
            if len(a) > 8 and a.std(ddof=1) > 0:
                pp.append(a.mean() / a.std(ddof=1))
        sel = np.asarray(buckets[bm][bn]["te"], float)
        dsr, sr, sr0 = deflated_sharpe(sel, pp, n_trials=2 * MAXN)
        print("\n================ VERDICT ================")
        print(f"  single most-reliable length (OOS t): {bm}({bn})  teMean {br['mte']:+.2f}%/trade, "
              f"t {br['tte']:.1f}, n {br['nte']}")
        print(f"  Deflated Sharpe (haircut for the full 1..{MAXN} x2 search): DSR = {dsr:.2f}  "
              f"({'survives' if dsr > 0.95 else 'a tie within a broad plateau — not a unique magic number'})")
    print("\nReads: look at the CURVE/band, not the argmax. A broad smooth plateau = the edge lives")
    print("in a length REGION (robust); a single tall spike = overfit. Compare the plateau height to")
    print("the 'none' baseline and to the golden-cross (~0.67%/trade) from ma_filter_sweep.py.")


if __name__ == "__main__":
    main()
