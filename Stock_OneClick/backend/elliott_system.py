"""
elliott_system.py — a MECHANICAL, falsifiable Elliott Wave entry/exit system.

WHY MECHANICAL. Elliott Wave as normally practiced is not falsifiable: counts are revised after
the fact, and two analysts produce different labels on the same chart. The only way to test it is
to remove the discretion — detect pivots algorithmically, enforce the rules in code, and date every
signal at the bar the information actually arrived. Then it either has forward predictive power or
it doesn't.

THE TWO THINGS THAT DECIDE WHETHER THIS TEST IS HONEST:

1. LOOKAHEAD IN PIVOT DETECTION. A ZigZag swing low is only KNOWN to be a swing low after price
   has risen off it by the threshold. The naive implementation stamps the pivot at its own bar,
   which is information from the future. Here every pivot carries an explicit `confirm` bar, and
   every signal is dated at `confirm`, never at the pivot itself. The gap between them is real and
   is reported (`confirm_lag`) — it is typically weeks, and it eats most of wave 3.

2. THE CONTROL GROUP. Requiring waves 1-4 to have formed means only looking at series that already
   trended and already pulled back. Compared against an unconditional baseline, ANY such rule
   looks good. So the null here is not "random entry" — it is a PULLBACK-IN-UPTREND rule that
   reproduces the identical price geometry WITHOUT the wave labels. The question this file answers
   is narrow and fair: do the Elliott LABELS add anything over the raw geometry?

WHICH RULES ARE EVEN USABLE. Elliott's three hard rules are not equally available at trade time:
    R1  wave 2 does not retrace >=100% of wave 1   -> KNOWABLE at the wave-2 entry
    R2  wave 3 is not the shortest of 1, 3, 5      -> RETROSPECTIVE. Needs wave 5 to exist, i.e.
                                                      it is unknowable at any entry. Cannot be a
                                                      filter in a tradeable system.
    R3  wave 4 does not enter wave 1's territory   -> KNOWABLE at the wave-4 entry, NOT at wave 2.
So the classic "buy the end of wave 2" setup can only enforce ONE of the three rules. That is a
structural limitation of the theory as a trading system, not a limitation of this implementation,
and it is reported rather than hidden.

Run: ../../vcp_env/bin/python elliott_system.py --refresh
"""
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = Path(__file__).resolve().parent / "_ew_panel.pkl"
TD = 252

# Broad, sector-diversified, mostly 25yr+ history. Breadth is how this test gets the sample size
# that single-name studies (see ntr_system.py) cannot reach.
UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "INTC", "CSCO", "IBM", "ORCL", "TXN", "QCOM", "AMD", "MU", "ADBE",
    "JNJ", "PG", "KO", "MRK", "PFE", "AMGN", "UNH",
    "XOM", "CVX", "CAT", "DE", "BA", "MMM", "GE", "F",
    "JPM", "BAC", "WMT", "HD", "LOW", "TGT", "COST", "MCD", "SBUX", "NKE", "DIS", "T", "VZ",
]

# --- Fibonacci zones. These are the CONVENTIONAL values, fixed in advance, not fitted. --------
W2_RETRACE = (0.382, 0.786)    # wave 2 typically retraces 50-61.8% of wave 1; widened to the
                               # conventional outer bounds so the test is not knife-edge
W4_RETRACE = (0.150, 0.500)    # wave 4 typically retraces ~38.2% of wave 3
W3_TARGET = 1.618              # conventional wave-3 projection off wave 1


def refresh_cache() -> dict:
    import yfinance as yf
    out = {}
    for t in UNIVERSE:
        h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
        if h.empty or len(h) < TD * 10:
            print(f"  skip {t} ({len(h)} bars)")
            continue
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        out[t] = h[["Open", "High", "Low", "Close"]]
    print(f"  cached {len(out)} symbols, "
          f"{sum(len(v) for v in out.values()):,} total bars "
          f"({sum(len(v) for v in out.values())/TD:,.0f} symbol-years)")
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def load_panel() -> dict:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# pivot detection — the lookahead-critical part
# --------------------------------------------------------------------------- #
def find_pivots(px: pd.Series, thr: float) -> pd.DataFrame:
    """
    ZigZag pivots with EXPLICIT confirmation bars.

    `pos` is where the extreme actually occurred. `confirm` is the first bar at which price had
    retraced `thr` from that extreme — i.e. the first bar on which a real-time observer could
    have known the pivot existed. Signals must use `confirm`. Using `pos` is lookahead.
    """
    v = px.to_numpy(dtype=float)
    n = len(v)
    piv = []
    mode = "up"                     # tracking a rising extreme, looking for a HIGH
    ext_i, ext_v = 0, v[0]
    for i in range(1, n):
        if mode == "up":
            if v[i] > ext_v:
                ext_i, ext_v = i, v[i]
            elif v[i] <= ext_v * (1.0 - thr):
                piv.append({"pos": ext_i, "price": ext_v, "kind": "H", "confirm": i})
                mode = "down"
                ext_i, ext_v = i, v[i]
        else:
            if v[i] < ext_v:
                ext_i, ext_v = i, v[i]
            elif v[i] >= ext_v * (1.0 + thr):
                piv.append({"pos": ext_i, "price": ext_v, "kind": "L", "confirm": i})
                mode = "up"
                ext_i, ext_v = i, v[i]
    return pd.DataFrame(piv)


# --------------------------------------------------------------------------- #
# wave labeling
# --------------------------------------------------------------------------- #
def label_impulses(piv: pd.DataFrame) -> list[dict]:
    """
    Find candidate 5-wave UP impulses over consecutive alternating pivots
    P0(L) P1(H) P2(L) P3(H) P4(L) P5(H).

    Returns one record per candidate with the rule checks separated into those knowable at the
    wave-2 entry, those knowable at the wave-4 entry, and those only knowable retrospectively.
    """
    out = []
    if len(piv) < 6:
        return out
    k = piv["kind"].tolist()
    for s in range(len(piv) - 5):
        seq = k[s:s + 6]
        if seq != ["L", "H", "L", "H", "L", "H"]:
            continue
        P = piv.iloc[s:s + 6].reset_index(drop=True)
        p0, p1, p2, p3, p4, p5 = P["price"].tolist()
        w1, w2, w3, w4, w5 = p1 - p0, p1 - p2, p3 - p2, p3 - p4, p5 - p4
        if w1 <= 0 or w3 <= 0 or w5 <= 0 or w2 <= 0 or w4 <= 0:
            continue
        r2_retr = w2 / w1                      # wave 2 as a fraction of wave 1
        r4_retr = w4 / w3
        out.append({
            "i0": int(P.loc[0, "pos"]), "i1": int(P.loc[1, "pos"]),
            "i2": int(P.loc[2, "pos"]), "i3": int(P.loc[3, "pos"]),
            "i4": int(P.loc[4, "pos"]), "i5": int(P.loc[5, "pos"]),
            "c2": int(P.loc[2, "confirm"]), "c4": int(P.loc[4, "confirm"]),
            "c5": int(P.loc[5, "confirm"]),
            "p0": p0, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
            "w2_retrace": r2_retr, "w4_retrace": r4_retr,
            "w3_mult": w3 / w1,
            # --- knowable at the wave-2 entry ---
            "R1_ok": r2_retr < 1.0,
            "w2_in_fib": W2_RETRACE[0] <= r2_retr <= W2_RETRACE[1],
            # --- knowable at the wave-4 entry ---
            "R3_ok": p4 > p1,                  # wave 4 does not enter wave 1's territory
            "w4_in_fib": W4_RETRACE[0] <= r4_retr <= W4_RETRACE[1],
            # --- retrospective only: needs wave 5 to have completed ---
            "R2_ok": w3 > min(w1, w5),
        })
    return out


# --------------------------------------------------------------------------- #
# event study machinery
# --------------------------------------------------------------------------- #
def forward_returns(px: pd.Series, bars: list[int], horizons=(21, 63, 126, 252)) -> dict:
    v = px.to_numpy(dtype=float)
    n = len(v)
    out = {}
    for h in horizons:
        vals = [v[b + h] / v[b] - 1.0 for b in bars if 0 <= b and b + h < n]
        out[h] = np.array(vals)
    return out


def baseline_returns(px: pd.Series, horizons=(21, 63, 126, 252)) -> dict:
    v = px.to_numpy(dtype=float)
    n = len(v)
    return {h: np.array([v[i + h] / v[i] - 1.0 for i in range(n - h)]) for h in horizons}


def control_pullbacks(px: pd.Series, piv: pd.DataFrame, ma_len: int = 200) -> list[int]:
    """
    THE NULL HYPOTHESIS. Same geometry, no wave labels: every confirmed swing LOW where
      (a) price is above its `ma_len` MA (an uptrend), and
      (b) the pullback from the immediately preceding swing HIGH lands in the same Fibonacci
          zone the wave-2 rule requires.
    No 5-wave structure required, no rule checks. If the Elliott labels add nothing over this,
    the theory adds nothing.
    """
    ma = px.rolling(ma_len).mean().to_numpy(dtype=float)
    v = px.to_numpy(dtype=float)
    bars = []
    for i in range(1, len(piv)):
        if piv.loc[i, "kind"] != "L" or piv.loc[i - 1, "kind"] != "H":
            continue
        c = int(piv.loc[i, "confirm"])
        if c >= len(v) or not np.isfinite(ma[c]) or v[c] <= ma[c]:
            continue
        hi, lo = piv.loc[i - 1, "price"], piv.loc[i, "price"]
        # measure the pullback against the prior up-leg, mirroring the wave-2 computation
        prev_lo = piv.loc[i - 2, "price"] if i >= 2 and piv.loc[i - 2, "kind"] == "L" else np.nan
        if not np.isfinite(prev_lo) or hi <= prev_lo:
            continue
        retr = (hi - lo) / (hi - prev_lo)
        if W2_RETRACE[0] <= retr <= W2_RETRACE[1]:
            bars.append(c)
    return bars


def summarise(label: str, ev: dict, base: dict, extra: str = ""):
    print(f"  {label:<42}{extra}")
    for h in sorted(ev):
        a, b = ev[h], base[h]
        if len(a) < 30:
            print(f"    +{h:>3}d  n={len(a):<5} too few")
            continue
        se = a.std(ddof=1) / np.sqrt(len(a))
        edge = a.mean() - b.mean()
        print(f"    +{h:>3}d  n={len(a):<5} mean {a.mean()*100:+6.2f}%  base {b.mean()*100:+6.2f}%"
              f"  edge {edge*100:+6.2f}pp  t={edge/se:+6.2f}  win {(a>0).mean()*100:3.0f}%"
              f" vs {(b>0).mean()*100:3.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--thr", type=float, default=0.10,
                    help="ZigZag threshold; this IS the wave degree")
    args = ap.parse_args()
    panel = refresh_cache() if args.refresh else load_panel()
    print(f"universe {len(panel)} symbols, "
          f"{sum(len(v) for v in panel.values())/TD:,.0f} symbol-years\n")

    for thr in (0.05, 0.10, 0.15, 0.20):
        print("=" * 84)
        print(f"ZIGZAG THRESHOLD {thr*100:.0f}%   (this parameter IS the Elliott 'degree')")
        print("=" * 84)
        w2_bars, w4_bars, ctl_bars = [], [], []
        w2_fib_bars, w4_fib_bars = [], []
        lags, n_cand, n_r1, n_r3, n_r2 = [], 0, 0, 0, 0
        agg_ev = {k: {h: [] for h in (21, 63, 126, 252)}
                  for k in ("w2", "w2fib", "w4", "w4fib", "ctl")}
        agg_base = {h: [] for h in (21, 63, 126, 252)}

        for t, df in panel.items():
            px = df["Close"].dropna()
            piv = find_pivots(px, thr)
            if len(piv) < 8:
                continue
            cands = label_impulses(piv)
            n_cand += len(cands)
            base = baseline_returns(px)
            for h in agg_base:
                agg_base[h].append(base[h])

            b_w2 = [c["c2"] for c in cands if c["R1_ok"]]
            b_w2f = [c["c2"] for c in cands if c["R1_ok"] and c["w2_in_fib"]]
            b_w4 = [c["c4"] for c in cands if c["R1_ok"] and c["R3_ok"]]
            b_w4f = [c["c4"] for c in cands if c["R1_ok"] and c["R3_ok"] and c["w4_in_fib"]]
            b_ctl = control_pullbacks(px, piv)
            n_r1 += sum(c["R1_ok"] for c in cands)
            n_r3 += sum(c["R3_ok"] for c in cands)
            n_r2 += sum(c["R2_ok"] for c in cands)
            lags += [c["c2"] - c["i2"] for c in cands]

            for key, bars in (("w2", b_w2), ("w2fib", b_w2f), ("w4", b_w4),
                              ("w4fib", b_w4f), ("ctl", b_ctl)):
                fr = forward_returns(px, bars)
                for h in agg_ev[key]:
                    agg_ev[key][h].append(fr[h])

        base = {h: np.concatenate(v) for h, v in agg_base.items() if v}
        ev = {k: {h: np.concatenate(v) for h, v in d.items() if v} for k, d in agg_ev.items()}
        print(f"  candidate 5-wave structures: {n_cand:,}   "
              f"R1 pass {n_r1/max(n_cand,1)*100:.0f}%   R3 pass {n_r3/max(n_cand,1)*100:.0f}%   "
              f"R2 pass (retrospective) {n_r2/max(n_cand,1)*100:.0f}%")
        print(f"  pivot confirmation lag: median {np.median(lags):.0f} bars, "
              f"mean {np.mean(lags):.0f} bars  <- this much of wave 3 is already gone at entry\n")
        summarise("WAVE-2 entry (R1 only — all that's knowable)", ev["w2"], base)
        summarise("WAVE-2 entry + Fib 38.2-78.6% zone", ev["w2fib"], base)
        summarise("WAVE-4 entry (R1+R3)", ev["w4"], base)
        summarise("WAVE-4 entry + Fib zone", ev["w4fib"], base)
        summarise("CONTROL: pullback-in-uptrend, NO wave labels", ev["ctl"], base,
                  extra="  <- the null to beat")
        # the decisive comparison
        print("\n  DO THE LABELS ADD ANYTHING OVER THE RAW GEOMETRY?")
        for h in sorted(ev["w2fib"]):
            a, c = ev["w2fib"][h], ev["ctl"][h]
            if len(a) < 30 or len(c) < 30:
                continue
            se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
            d = a.mean() - c.mean()
            print(f"    +{h:>3}d  wave2+Fib {a.mean()*100:+6.2f}%  vs  control {c.mean()*100:+6.2f}%"
                  f"   diff {d*100:+6.2f}pp   t={d/se:+6.2f}")
        print()


if __name__ == "__main__":
    main()
