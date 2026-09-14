"""
_move_iv_bss2.py — the gate the data-source investigation left open: does per-stock ATM implied
volatility improve the move-probability model on BSS2, the metric this system is actually scored on?

WHY THIS EXISTS. The free-data probe measured own-IV at +0.021 / +0.026 OOS R-squared on LOG
REALIZED VOL at h=10/21, on top of the full shipped feature set including market VIX. That is a
volatility-forecasting result, not a probability-calibration result, and in this repo the two have
repeatedly diverged: the macro-calendar study found FOMC windows raise realized log-vol by +0.183
and still bought only +0.0011 BSS2, and the earlier vol-estimator round found the highest-R-squared
model was the more conditionally biased one. So R-squared is a screen; BSS2 decides.

TRAP THIS SCRIPT IS BUILT TO AVOID (it already produced one confident false negative upstream):
comparing an IV-augmented model fit on the 938-day IV window against a baseline fit on the full
25-year panel makes IV look strictly worse. EVERY model here is restricted to the identical common
sample before anything is fit. Row counts are printed so the restriction is visible.

DATA. HuggingFace gauss314/options-IV-SP500 -> _data_probe_hf_iv_sp500.csv
  3.16M rows, 3,893 symbols, 938 trading days, 2019-10-14 -> 2023-07-28. Column ATM_IV is in
  annualised vol POINTS (28.09 = 28.09%). It is a nearest-expiry-style figure, not a true IV30:
  against CBOE's own VXAPL it runs bias -2.32 vol points, sd(diff) 3.06. That level offset is
  harmless here because every model is refit on this data, but it is why the live yfinance
  collector (iv_snapshot.py, a properly interpolated constant-maturity IV30) CANNOT be spliced
  onto this history without a bridge study.

FOLDS. The IV window is only ~3.8 years, so yearly expanding folds give at most 2-3 test years.
This script uses expanding QUARTER folds with a minimum 4-quarter train, which yields ~11 OOS
quarters -- thin, and reported as thin.

Run: ../../vcp_env/bin/python _move_iv_bss2.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M

T0 = time.time()
RNG = np.random.default_rng(23)
IV_CSV = "_data_probe_hf_iv_sp500.csv"
HS = (1, 5, 10, 21)
THRS = (0.02, 0.05)
MIN_TRAIN_Q = 4
SHRINK = 40.0


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def load_iv() -> pd.DataFrame:
    iv = pd.read_csv(IV_CSV, usecols=["symbol", "date", "ATM_IV"])
    iv["date"] = pd.to_datetime(iv["date"])
    iv = iv[(iv.ATM_IV > 1) & (iv.ATM_IV < 400)]          # vol points; strip degenerate rows
    log(f"IV panel: {len(iv):,} rows, {iv.symbol.nunique():,} symbols, "
        f"{iv.date.nunique():,} dates, {iv.date.min().date()} -> {iv.date.max().date()}")
    return iv


def build(panel, iv):
    """Single names only (IV is a single-name feature; index IV is already covered by VIX).
    Stacks the shipped baseline design plus an iv block, on the IV window only."""
    C, H, Lo, V = panel["Close"], panel["High"], panel["Low"], panel.get("Volume")
    vix = C["^VIX"] if "^VIX" in C.columns else None
    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    syms = [s for s in usable if s not in M.INDEX_LIKE]
    ivw = iv.pivot_table(index="date", columns="symbol", values="ATM_IV", aggfunc="last")
    have = [s for s in syms if s in ivw.columns]
    log(f"panel single names {len(syms)}, of which {len(have)} appear in the IV panel")

    out = {}
    for h in HS:
        Xb, Xi, lr, tg, yr, q, sid = [], [], [], [], [], [], []
        for i, s in enumerate(have):
            c = C[s].dropna()
            if len(c) < 400:
                continue
            vser = V[s].reindex(c.index) if (V is not None and s in V.columns) else None
            f = M.build_features(c, H[s].reindex(c.index), Lo[s].reindex(c.index), vix, vser)
            cols = M.feature_cols("single", vix is not None, vser is not None)
            if any(x not in f.columns for x in cols):
                continue
            ivs = ivw[s].reindex(c.index)
            # Lag by one day: the IV print is same-day, and a one-day lag makes the feature
            # unambiguously available at decision time. The probe measured the gain surviving
            # this lag (+0.0095/+0.0184/+0.0261 at h=5/10/21 vs +0.0129/+0.0210/+0.0257 unlagged).
            ivl = np.log(ivs.shift(1) / 100.0 / np.sqrt(252))
            tgt = M._clip_log(L.realized_vol_forward(c, h))
            lrf = np.log(c.shift(-h) / c)
            base = M._design(f, cols)
            ok = (np.isfinite(base).all(1) & np.isfinite(lrf.values) & np.isfinite(tgt.values)
                  & np.isfinite(ivl.values))
            if ok.sum() < 50:
                continue
            Xb.append(base[ok]); Xi.append(ivl.values[ok, None])
            lr.append(lrf.values[ok]); tg.append(tgt.values[ok])
            yr.append(c.index.year.values[ok])
            q.append((c.index.year.values[ok] * 4 + (c.index.quarter.values[ok] - 1))[:])
            sid.append(np.full(int(ok.sum()), i))
        if not lr:
            continue
        out[h] = dict(base=np.vstack(Xb), iv=np.vstack(Xi), lr=np.concatenate(lr),
                      tgt=np.concatenate(tg), yr=np.concatenate(yr), q=np.concatenate(q),
                      sid=np.concatenate(sid))
        log(f"h={h:<3} common-sample rows={len(out[h]['lr']):>8,} "
            f"symbols={len(set(out[h]['sid']))} quarters={len(set(out[h]['q']))}")
    return out


def walk(base, extra, lr, tgt, q, sid, h, thr):
    ones = np.ones((len(lr), 1))
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    big = ((lr <= a_dn) | (lr >= a_up)).astype(float)
    nsym = int(sid.max()) + 1
    P, PS, BIG, QQ = [], [], [], []
    qs = sorted(set(q))
    for tq in qs[MIN_TRAIN_Q:]:
        tr, te = q < tq, q == tq
        if tr.sum() < 3000 or te.sum() < 200:
            continue
        Xtr = np.hstack([ones[tr], base[tr]] + ([extra[tr]] if extra is not None else []))
        Xte = np.hstack([ones[te], base[te]] + ([extra[te]] if extra is not None else []))
        b, _n = M._ols(Xtr, tgt[tr])      # M._ols returns (beta, n_used), not beta
        if b is None:
            continue
        s_tr, s_te = np.exp(Xtr @ b) * np.sqrt(h), np.exp(Xte @ b) * np.sqrt(h)
        z = np.sort(lr[tr] / s_tr)
        f_dn = np.searchsorted(z, a_dn / s_te, "right") / len(z)
        f_up = np.searchsorted(z, a_up / s_te, "right") / len(z)
        P.append(np.clip(f_dn + (1 - f_up), 1e-4, 1 - 1e-4))
        cnt = np.bincount(sid[tr], weights=big[tr], minlength=nsym)
        tot = np.bincount(sid[tr], minlength=nsym).astype(float)
        PS.append(((cnt + SHRINK * big[tr].mean()) / (tot + SHRINK))[sid[te]])
        BIG.append(big[te]); QQ.append(np.full(int(te.sum()), tq))
    if not P:
        return None
    return np.concatenate(P), np.concatenate(BIG), np.concatenate(PS), np.concatenate(QQ)


def bss2(p, bg, ps):
    return 1 - np.nanmean((p - bg) ** 2) / np.nanmean((ps - bg) ** 2)


def main():
    panel = D.load()
    cache = build(panel, load_iv())
    rows = []
    for h, d in sorted(cache.items()):
        for thr in THRS:
            r0 = walk(d["base"], None, d["lr"], d["tgt"], d["q"], d["sid"], h, thr)
            r1 = walk(d["base"], d["iv"], d["lr"], d["tgt"], d["q"], d["sid"], h, thr)
            rp = walk(d["base"], d["iv"][RNG.permutation(len(d["iv"]))], d["lr"], d["tgt"],
                      d["q"], d["sid"], h, thr)
            if not (r0 and r1):
                continue
            b0, b1 = bss2(*r0[:3]), bss2(*r1[:3])
            bp = bss2(*rp[:3]) if rp else np.nan
            # quarter-block bootstrap: the coarsest honest unit, absorbs overlap and the
            # cross-sectional correlation inside a quarter
            qs = np.unique(r1[3]); idx = {u: np.flatnonzero(r1[3] == u) for u in qs}
            bt = np.empty(400)
            for i in range(400):
                m = np.concatenate([idx[u] for u in RNG.choice(qs, len(qs), replace=True)])
                bt[i] = bss2(r1[0][m], r1[1][m], r1[2][m]) - bss2(r0[0][m], r0[1][m], r0[2][m])
            per_q = [bss2(r1[0][idx[u]], r1[1][idx[u]], r1[2][idx[u]])
                     > bss2(r0[0][idx[u]], r0[1][idx[u]], r0[2][idx[u]]) for u in qs]
            rows.append(dict(h=h, thr=thr, n=len(r1[0]), q_oos=len(qs), base=b0, d=b1 - b0,
                             d_perm=bp - b0, lo=np.percentile(bt, 2.5), hi=np.percentile(bt, 97.5),
                             p_gt0=(bt > 0).mean(), q_better=f"{sum(per_q)}/{len(per_q)}",
                             d_ece=L.ece(r1[0], r1[1].astype(bool)) - L.ece(r0[0], r0[1].astype(bool))))
            log(f"h={h:<3} thr={thr*100:.0f}% n={len(r1[0]):>7,} base={b0:+.4f} "
                f"d={b1-b0:+.4f} CI[{rows[-1]['lo']:+.4f},{rows[-1]['hi']:+.4f}] "
                f"P>0={rows[-1]['p_gt0']:.2f} perm={bp-b0:+.4f} q+{rows[-1]['q_better']} "
                f"dECE={rows[-1]['d_ece']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_pickle("_move_iv_bss2.pkl")
    print("\n" + "=" * 112)
    print("PER-STOCK ATM IV -- delta BSS2 on P(|move| >= thr), single names, common sample only")
    print("perm = IV column shuffled across rows; must be ~0 for the gain to be information")
    print("=" * 112)
    if not df.empty:
        print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print("\nsaved _move_iv_bss2.pkl")


if __name__ == "__main__":
    main()
