"""_move_spec_display.py -- render the exact display for the spec's worked example."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import _move_data as D
import _move_lib as L

VOL_CLIP = (1e-3, 0.5)
HORIZONS = (1, 5, 10, 21)
INDEX_SYMS = {"SPY", "QQQ", "IWM", "DIA", "^GSPC"}


def clip_log(x):
    return np.log(np.clip(x, *VOL_CLIP))


art = pd.read_pickle("_move_spec_ship2.pkl")
p = D.load()
O, H, Lo, C = p["Open"].copy(), p["High"].copy(), p["Low"].copy(), p["Close"].copy()
for df in (O, H, Lo, C):
    df.mask(df <= 0, inplace=True)
vix = p["Close"]["^VIX"]
cov = C.notna().sum()
usable = [s for s in C.columns if s != "^VIX" and cov[s] >= 500]
SG = [s for s in usable if s not in INDEX_SYMS]

# rebuild quintile tables + per-quintile display quantiles
QT, EDGES, DISP = {}, {}, {}
for hz in HORIZONS:
    a = art[("single", "MAIN", hz)]
    zs, sd = [], []
    for s in SG:
        c = C[s].dropna()
        if len(c) < 400:
            continue
        o, hh, ll = O[s].reindex(c.index), H[s].reindex(c.index), Lo[s].reindex(c.index)
        r1 = L.vol_parkinson(hh, ll, 1)
        X = np.column_stack([np.ones(len(c)), clip_log(r1), clip_log(r1.rolling(5).mean()),
                             clip_log(r1.rolling(22).mean()), clip_log(r1.rolling(63).mean()),
                             clip_log(L.vol_yang_zhang(o, hh, ll, c, 21)),
                             clip_log(L.vol_ewma(c, 0.97))])
        lr = np.log(c.shift(-hz) / c).values
        tg = clip_log(L.realized_vol_forward(c, hz)).values
        ok = np.isfinite(X).all(axis=1) & np.isfinite(lr) & np.isfinite(tg)
        s_d = np.exp(X[ok] @ a["beta"])
        zs.append(lr[ok] / (s_d * np.sqrt(hz))); sd.append(s_d)
    z = np.concatenate(zs); s_d = np.concatenate(sd)
    ann = s_d * np.sqrt(252)
    e = np.percentile(ann, [20, 40, 60, 80]); EDGES[hz] = e
    q = np.digitize(ann, e)
    QT[hz] = [np.sort(z[q == k]) for k in range(5)]
    DISP[hz] = [(np.percentile(QT[hz][k], 10), np.percentile(QT[hz][k], 90),
                 (np.percentile(QT[hz][k], 75) - np.percentile(QT[hz][k], 25)) / 1.349)
                for k in range(5)]

print("=== SINGLE-NAME per-quintile display constants (q10, q90, iqs) ===")
for hz in HORIZONS:
    print(f"h={hz:<3} edges_ann=" + " ".join(f"{x:.4f}" for x in EDGES[hz]))
    for k in range(5):
        q10, q90, iqs = DISP[hz][k]
        print(f"     Q{k+1}: q10={q10:+.4f} q90={q90:+.4f} iqs={iqs:.4f} n={len(QT[hz][k]):,}")

IDISP = {}
for hz in HORIZONS:
    z = art[("index", "MAIN", hz)]["z"]
    IDISP[hz] = (np.percentile(z, 10), np.percentile(z, 90),
                 (np.percentile(z, 75) - np.percentile(z, 25)) / 1.349)
print("\n=== INDEX display constants (q10, q90, iqs) ===")
for hz in HORIZONS:
    print(f"h={hz:<3} q10={IDISP[hz][0]:+.4f} q90={IDISP[hz][1]:+.4f} iqs={IDISP[hz][2]:.4f}")

EM = {1: 1.82, 5: 1.48, 10: 1.33, 21: 1.21}
ecache = json.load(open("../../gold_pine_script/pead_earnings_cache.json"))


def predict(sym, thr=0.02, force_earn=None):
    c = C[sym].dropna()
    g = "index" if sym in INDEX_SYMS else "single"
    o, hh, ll = O[sym].reindex(c.index), H[sym].reindex(c.index), Lo[sym].reindex(c.index)
    r1 = L.vol_parkinson(hh, ll, 1)
    if g == "index":
        X = np.column_stack([np.ones(len(c)), clip_log(r1), clip_log(r1.rolling(5).mean()),
                             clip_log(r1.rolling(22).mean()),
                             clip_log(L.vol_yang_zhang(o, hh, ll, c, 21)),
                             clip_log(vix.reindex(c.index).ffill(limit=3) / 100 / np.sqrt(252))])
    else:
        X = np.column_stack([np.ones(len(c)), clip_log(r1), clip_log(r1.rolling(5).mean()),
                             clip_log(r1.rolling(22).mean()), clip_log(r1.rolling(63).mean()),
                             clip_log(L.vol_yang_zhang(o, hh, ll, c, 21)),
                             clip_log(L.vol_ewma(c, 0.97))])
    row = X[-1]
    out = []
    for hz in HORIZONS:
        a = art[(g, "MAIN", hz)]
        sd = float(np.exp(row @ a["beta"]))
        if g == "single":
            k = int(np.digitize([sd * np.sqrt(252)], EDGES[hz])[0])
            zs = QT[hz][k]; q10, q90, iqs = DISP[hz][k]; lab = f"Q{k+1}"
        else:
            zs = a["z"]; q10, q90, iqs = IDISP[hz]; lab = "-"
        m = EM[hz] if (force_earn and hz >= force_earn) else 1.0
        sh = sd * np.sqrt(hz) * m
        n = len(zs)
        Fd = np.searchsorted(zs, np.log(1 - thr) / sh, side="right") / n
        F0 = np.searchsorted(zs, 0.0, side="right") / n
        Fu = np.searchsorted(zs, np.log(1 + thr) / sh, side="right") / n
        pr = np.clip(np.array([Fd, F0 - Fd, Fu - F0, 1 - Fu]), 1e-4, None)
        pr = pr / pr.sum()
        lo = np.exp(q10 * sh) - 1
        hi = np.exp(q90 * sh) - 1
        out.append(dict(h=hz, sd=sd, sh=sh, q=lab, p=pr, lo=lo, hi=hi, iqs=iqs, em=m,
                        typ=sh * iqs))
    return g, c, out


print("\n\n=== FULL PER-TICKER READOUT (real numbers, 2026-09-10 close) ===")
for sym, fe in [("SPY", None), ("AAPL", None), ("AAPL", 5), ("NVDA", None), ("JPM", None)]:
    g, c, rows = predict(sym, 0.02, fe)
    tag = "  [earnings forced in window for illustration]" if fe else ""
    print(f"\n{sym}  {g}  close {c.iloc[-1]:.2f}  {pd.Timestamp(c.index[-1]).date()}{tag}")
    print(f"  {'h':>3} {'vol-bkt':>7} {'typ move':>9} {'80% band':>18} "
          f"{'P(dn>2%)':>9} {'P(|r|<2%)':>10} {'P(up>2%)':>9} {'P(|move|>2%)':>13} {'up:dn':>7}")
    for r in rows:
        pr = r["p"]
        ratio = pr[3] / pr[0] if pr[0] > 0 else np.inf
        print(f"  {r['h']:>3} {r['q']:>7} {r['typ']*100:>8.2f}% "
              f"{f'{r[chr(108)+chr(111)]*100:+.1f}% .. {r[chr(104)+chr(105)]*100:+.1f}%':>18} "
              f"{pr[0]*100:>8.1f}% {(pr[1]+pr[2])*100:>9.1f}% {pr[3]*100:>8.1f}% "
              f"{(pr[0]+pr[3])*100:>12.1f}% {ratio:>7.2f}")

print("\n\n=== COMPACT SCAN LINE (h=5, thr=2%) ===")
print(f"{'sym':<6}{'σ5':>7}{'dn':>6}{'mid':>6}{'up':>6}{'|mv|':>7}{'up:dn':>7}  flags")
for sym in ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "XOM", "JPM", "BLNK"]:
    if sym not in C.columns:
        continue
    g, c, rows = predict(sym, 0.02)
    r = [x for x in rows if x["h"] == 5][0]
    pr = r["p"]
    print(f"{sym:<6}{r['typ']*100:>6.1f}%{pr[0]*100:>5.0f}%{(pr[1]+pr[2])*100:>5.0f}%"
          f"{pr[3]*100:>5.0f}%{(pr[0]+pr[3])*100:>6.0f}%{pr[3]/pr[0]:>7.2f}  {r['q']}")
