"""
_move_fx_screen.py — one consolidated screening pass over every candidate factor family that the
failed research workflow left unfinished.

WHY THIS EXISTS: the six-agent workflow stalled because a single walk-forward over 1.1M
single-name rows with per-symbol refits took 11.5 hours per cell (measured: the model-form study
logged 41,356s cumulative). This script tests the same candidate factors but is engineered to
finish in minutes: all candidates are plain added OLS columns, the per-symbol climatology is a
bincount instead of a per-symbol loop, and the single-name universe is strided down.

WHAT IS ALREADY SETTLED and therefore NOT retested here (from the salvaged agent output):
  * MODEL FORM: comprehensively tested and dead. delta BSS2 at thr=2% for squared terms,
    per-symbol fixed effects (2 shrinkages), spline in log-sigma, logistic, multinomial logit,
    Platt and isotonic recalibration are ALL within +/-0.002, most negative. Best was per-symbol
    fixed effects at +0.0004..+0.0006 on single names. Nothing there.
  * MACRO CALENDAR at index h=1: FOMC-in-window raises realized log vol by +0.183 (the largest
    calendar effect measured) yet buys only +0.0011 BSS2 with a CI spanning zero; CPI-in-window
    is significantly WORSE (-0.0038, p=.005); day-of-week is worse. Retested here only at other
    horizons, because h=1/2% for an index sits so far in the tail that a 20% vol bump barely
    moves the probability -- other horizons are where it could still pay.

CANDIDATE BLOCKS (each added to the shipped baseline, measured on identical folds):
  LEV   leverage / sign asymmetry -- realized semivariance up vs down, signed recent return
  GAP   overnight vs intraday variance split, plus the close-to-close / Parkinson ratio
  JUMP  bipower variation and the jump component (RV - BV)
  VOL   dollar volume, volume surprise, Amihud illiquidity
  VIXC  the vol-derivatives complex (index only): VIX9D, term slope, VVIX, VXN
  FOMC  FOMC-in-window dummy, using the fetched date list
  NOISE control block -- same parameter count, deliberately uninformative. Any real block must
        beat this, otherwise the "gain" is just added degrees of freedom.

Run: ../../vcp_env/bin/python _move_fx_screen.py [--stride 3] [--out _move_fx_screen.pkl]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L
import move_prob as M

T0 = time.time()
HS = (1, 5, 21)
THRS = (0.02, 0.05)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
CLIP = (1e-3, 0.5)
RNG = np.random.default_rng(7)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def cl(x):
    return np.log(np.clip(x, *CLIP))


def build(stride: int):
    """Per (group, h): baseline design matrix + each candidate block, stacked over symbols."""
    p = D.load()
    C, H, Lo, V = p["Close"], p["High"], p["Low"], p.get("Volume")
    vix_all = C["^VIX"] if "^VIX" in C.columns else None
    vd = pd.read_pickle("_move_fx_voldrv.pkl") if Path("_move_fx_voldrv.pkl").exists() else None
    macro = (json.load(open("_move_fx_macro_dates.json"))
             if Path("_move_fx_macro_dates.json").exists() else None)
    fomc = pd.to_datetime(macro["fomc"]) if macro else pd.DatetimeIndex([])

    usable = [s for s in C.columns if s != "^VIX" and C[s].notna().sum() >= 500]
    idx_syms = [s for s in usable if s in M.INDEX_LIKE]
    sgl_syms = [s for s in usable if s not in M.INDEX_LIKE][::stride]
    log(f"universe: {len(idx_syms)} index, {len(sgl_syms)} single (stride {stride})")

    out = {}
    for gname, syms in (("index", idx_syms), ("single", sgl_syms)):
        for h in HS:
            acc = {k: [] for k in ("base", "LEV", "GAP", "JUMP", "VOL", "VIXC", "FOMC", "NOISE")}
            lr_a, tg_a, yr_a, sid_a = [], [], [], []
            for i, s in enumerate(syms):
                c = C[s].dropna()
                if len(c) < 400:
                    continue
                hi, lo = H[s].reindex(c.index), Lo[s].reindex(c.index)
                op = p["Open"][s].reindex(c.index)
                r = np.log(c).diff()

                # ---- baseline block (exactly the shipped features)
                rv1 = L.vol_parkinson(hi, lo, 1)
                base = [cl(rv1), cl(rv1.rolling(5).mean()), cl(rv1.rolling(22).mean()),
                        cl(rv1.rolling(63).mean()), cl(L.vol_ewma(c, 0.97))]
                if gname == "index" and vix_all is not None:
                    base.append(np.log(np.clip(vix_all.reindex(c.index).ffill(limit=3)
                                               / 100 / np.sqrt(252), *CLIP)))

                # ---- LEV: realized semivariance + signed return. The baseline is sign-blind.
                dn = r.where(r < 0, 0.0) ** 2
                up = r.where(r > 0, 0.0) ** 2
                lev = [cl(np.sqrt(dn.rolling(22).mean())), cl(np.sqrt(up.rolling(22).mean())),
                       r.rolling(5).sum().fillna(0.0)]

                # ---- GAP: overnight vs intraday variance, and the gap-regime ratio
                on = np.log(op / c.shift(1)) ** 2
                intra = np.log(c / op) ** 2
                ratio = L.vol_cc(c, 21) / L.vol_parkinson(hi, lo, 21)
                gap = [cl(np.sqrt(on.rolling(22).mean())), cl(np.sqrt(intra.rolling(22).mean())),
                       np.log(np.clip(ratio, 0.3, 4.0))]

                # ---- JUMP: bipower variation separates the continuous part from jumps
                bv = (r.abs() * r.abs().shift(1) * np.pi / 2).rolling(22).mean()
                rv = (r ** 2).rolling(22).mean()
                jmp = (rv - bv).clip(lower=0)
                jumpb = [cl(np.sqrt(bv)), np.log1p(jmp * 1e4)]

                # ---- VOL: volume level, surprise, Amihud illiquidity
                if V is not None and s in V.columns:
                    vv = pd.to_numeric(V[s], errors="coerce").reindex(c.index)
                    dv = (vv * c).replace(0, np.nan)
                    volb = [np.log(dv.rolling(22).mean().clip(lower=1e3)),
                            np.log((vv / vv.rolling(22).mean()).clip(0.05, 20).fillna(1.0)),
                            np.log((r.abs() / dv).rolling(22).mean().clip(1e-14, 1e-3))]
                else:
                    volb = [pd.Series(np.nan, index=c.index)] * 3

                # ---- VIXC: the vol-derivatives complex, index group only
                if gname == "index" and vd is not None:
                    pick = "^VIX9D" if h <= 5 else "^VIX3M"
                    a = vd[pick].reindex(c.index)
                    b = vd["^VVIX"].reindex(c.index)
                    slope = (vd["^VIX"] / vd["^VIX3M"]).reindex(c.index)
                    vixb = [np.log(np.clip(a / 100 / np.sqrt(252), *CLIP)),
                            np.log(np.clip(b / 100, 0.1, 5.0)),
                            np.log(slope.clip(0.4, 2.5))]
                else:
                    vixb = [pd.Series(np.nan, index=c.index)] * 3

                # ---- FOMC: does an FOMC decision fall inside [t+1, t+h]?
                fl = pd.Series(0.0, index=c.index)
                if len(fomc):
                    pos = c.index.searchsorted(fomc)
                    hit = np.zeros(len(c))
                    for q in pos[(pos > 0) & (pos < len(c))]:
                        hit[max(0, q - h):q] = 1.0
                    fl = pd.Series(hit, index=c.index)

                nz = pd.Series(RNG.standard_normal(len(c)), index=c.index)

                tgt = cl(L.realized_vol_forward(c, h))
                lrf = np.log(c.shift(-h) / c)
                blocks = {"base": base, "LEV": lev, "GAP": gap, "JUMP": jumpb,
                          "VOL": volb, "VIXC": vixb, "FOMC": [fl], "NOISE": [nz]}
                fin = np.isfinite(lrf.values)
                for k, cols in blocks.items():
                    acc[k].append(np.column_stack([x.values for x in cols])[fin])
                lr_a.append(lrf.values[fin]); tg_a.append(tgt.values[fin])
                yr_a.append(c.index.year.values[fin]); sid_a.append(np.full(int(fin.sum()), i))

            if not lr_a:
                continue
            out[(gname, h)] = {k: np.vstack(v) for k, v in acc.items()}
            out[(gname, h)].update(lr=np.concatenate(lr_a), tgt=np.concatenate(tg_a),
                                   yr=np.concatenate(yr_a), sid=np.concatenate(sid_a))
            log(f"built {gname:<7} h={h:<3} rows={len(out[(gname,h)]['lr']):>9,}")
    return out


def ols(X, y):
    k = np.isfinite(X).all(1) & np.isfinite(y)
    if k.sum() < 300:
        return None
    b, *_ = np.linalg.lstsq(X[k], y[k], rcond=None)
    return b


def probs(z_sorted, sig, thr):
    a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
    n = len(z_sorted)
    f_dn = np.searchsorted(z_sorted, a_dn / sig, "right") / n
    f_up = np.searchsorted(z_sorted, a_up / sig, "right") / n
    return np.clip(f_dn + (1 - f_up), 1e-4, 1 - 1e-4)


def run(cache, out_path):
    rows = []
    for (g, h), d in sorted(cache.items()):
        lr, tgt, yr, sid = d["lr"], d["tgt"], d["yr"], d["sid"]
        ones = np.ones((len(lr), 1))
        years = sorted(set(yr))
        tests = [k for k in ("LEV", "GAP", "JUMP", "VOL", "VIXC", "FOMC", "NOISE")
                 if np.isfinite(d[k]).all(1).mean() > 0.5]
        for thr in THRS:
            a_dn, a_up = np.log(1 - thr), np.log(1 + thr)
            big = ((lr <= a_dn) | (lr >= a_up)).astype(float)
            P = {k: [] for k in ["BASE"] + tests}
            PS, BIG, YY = [], [], []
            for ty in years[MIN_TRAIN_YEARS:]:
                tr, te = yr < ty, yr == ty
                if tr.sum() < 4000 or te.sum() < 40:
                    continue
                Xb_tr, Xb_te = np.hstack([ones[tr], d["base"][tr]]), np.hstack([ones[te], d["base"][te]])
                for k in ["BASE"] + tests:
                    Xtr = Xb_tr if k == "BASE" else np.hstack([Xb_tr, d[k][tr]])
                    Xte = Xb_te if k == "BASE" else np.hstack([Xb_te, d[k][te]])
                    b = ols(Xtr, tgt[tr])
                    if b is None:
                        P[k].append(np.full(int(te.sum()), np.nan)); continue
                    s_tr = np.exp(np.nan_to_num(Xtr, nan=0.0) @ b) * np.sqrt(h)
                    s_te = np.exp(np.nan_to_num(Xte, nan=0.0) @ b) * np.sqrt(h)
                    zz = np.sort((lr[tr] / s_tr)[np.isfinite(lr[tr] / s_tr)])
                    P[k].append(probs(zz, s_te, thr))
                # per-symbol climatology via bincount (not a per-symbol loop)
                nsym = int(sid.max()) + 1
                cnt = np.bincount(sid[tr], weights=big[tr], minlength=nsym)
                tot = np.bincount(sid[tr], minlength=nsym).astype(float)
                pooled = big[tr].mean()
                ps_sym = (cnt + SHRINK * pooled) / (tot + SHRINK)
                PS.append(ps_sym[sid[te]]); BIG.append(big[te]); YY.append(np.full(int(te.sum()), ty))
            if not PS:
                continue
            ps, bg, yy = np.concatenate(PS), np.concatenate(BIG), np.concatenate(YY)
            bs = lambda q: np.nanmean((q - bg) ** 2)
            base_p = np.concatenate(P["BASE"])
            b_base = 1 - bs(base_p) / bs(ps)
            rec = {"group": g, "h": h, "thr": thr, "n": len(bg), "bss2_base": b_base,
                   "ece_base": L.ece(base_p, bg.astype(bool))}
            for k in tests:
                q = np.concatenate(P[k])
                rec[f"d_{k}"] = (1 - bs(q) / bs(ps)) - b_base
                rec[f"ece_{k}"] = L.ece(q, bg.astype(bool)) - rec["ece_base"]
                yrs = [(np.nanmean((q[yy == u] - bg[yy == u]) ** 2)
                        < np.nanmean((base_p[yy == u] - bg[yy == u]) ** 2)) for u in np.unique(yy)]
                rec[f"yrs_{k}"] = f"{int(np.sum(yrs))}/{len(yrs)}"
            rows.append(rec)
            log(f"{g:<7} h={h:<3} thr={thr*100:.0f}%  base={b_base:+.4f}  " +
                "  ".join(f"{k}{rec[f'd_{k}']:+.4f}({rec[f'yrs_{k}']})" for k in tests))
    df = pd.DataFrame(rows)
    df.to_pickle(out_path)

    print("\n" + "=" * 118)
    print("DELTA BSS2 vs shipped baseline  (NOISE is the same-parameter-count control: real blocks must beat it)")
    print("=" * 118)
    for thr in THRS:
        sub = df[df.thr == thr]
        if sub.empty:
            continue
        cols = [c[2:] for c in sub.columns if c.startswith("d_")]
        print(f"\n--- thr={thr*100:.0f}% ---")
        print(f"{'grp':<8}{'h':>3}{'n':>10}{'BASE':>9}" + "".join(f"{c:>11}" for c in cols))
        for _, r in sub.iterrows():
            print(f"{r.group:<8}{r.h:>3}{r.n:>10,}{r.bss2_base:>9.4f}"
                  + "".join(f"{r.get('d_'+c, np.nan):>+11.4f}" for c in cols))
    print(f"\nsaved {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", default="_move_fx_screen.pkl")
    a = ap.parse_args()
    run(build(a.stride), a.out)
