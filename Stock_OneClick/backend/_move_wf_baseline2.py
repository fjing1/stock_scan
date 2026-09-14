"""
_move_wf_baseline2.py — supplement to _move_wf_baseline.py.

Answers the questions the main table raised:
  A. what are the `invalid value in log` rows, and do outlier z's move anything?
  B. SCALE vs DIRECTION: is P(|r|>=thr) calibrated even when P(up_big)/P(down_big) is not?
     (crisis years split the mass one-sidedly; the main table conflates the two failures)
  C. probability-floor sensitivity: is the log loss at thr>=3%, h=1 an artifact of FLOOR=1e-4?
  D. EWMA overshoot diagnostic: sd(z) and realized/forecast vol ratio by sigma decile.
  E. RECENT-WINDOW headline: OOS scores restricted to test years 2021-2026, which is the
     honest answer to "what should the shipped system score on NEW data".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_lib as L
import _move_wf_baseline as W

HORIZONS = W.HORIZONS
THRESHOLDS = W.THRESHOLDS


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100, flush=True)


def main():
    close, idx_syms, singles = W.load_panel()

    # ------------------------------------------------------- A. bad-price diagnosis
    hdr("A — non-positive / broken price rows (source of the numpy log warning)")
    bad = (close <= 0)
    print(f"close<=0 cells: {int(bad.to_numpy().sum())}  in symbols: "
          f"{list(bad.columns[bad.any()])}")
    lr = np.log(close.where(close > 0)).diff()
    ext = lr.abs() > 0.9
    n_ext = int(ext.to_numpy().sum())
    print(f"|daily log return| > 0.9 (i.e. >2.5x or <0.4x in a day): {n_ext} cells")
    if n_ext:
        w = np.argwhere(ext.to_numpy())
        for r, c in w[:15]:
            print(f"   {close.index[r].date()} {close.columns[c]:<6} "
                  f"logret={lr.iat[r, c]:+.3f}  px {close.iat[r-1, c]:.4f} -> {close.iat[r, c]:.4f}")
    print("NOTE: these fat outliers sit in the z sample but the model reads the z distribution "
          "through an empirical CDF (searchsorted), so a handful of extreme z's shift no quantile "
          "materially. They DO blow up the moment-based skew/kurtosis in Table 6.")

    groups = {"INDEX": close[idx_syms], "SINGLES": close[singles]}
    res = {}
    for g, cg in groups.items():
        print(f"\nre-running walk-forward for {g} ...", flush=True)
        res[g], _ = W.run_group(g, cg, verbose=False)

    # ------------------------------------------------------- B. scale vs direction
    hdr("B1 — SCALE calibration: P(|r|>=thr) = P(up_big)+P(down_big), all OOS")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>11}{'pred_move':>11}{'obs_move':>10}{'gap':>8}"
          f"{'ECE_move':>10}{'Brier2':>9}{'Brier2clim':>12}{'BSS2':>8}")
    for g in groups:
        for h in HORIZONS:
            for thr in THRESHOLDS:
                rm, rb = res[g][(h, thr, "EMP")], res[g][(h, thr, "CLIM")]
                p, y = rm["p"].astype(float), rm["y"].astype(int)
                pm = p[:, 0] + p[:, 3]
                pbm = rb["p"].astype(float)[:, 0] + rb["p"].astype(float)[:, 3]
                hit = (y == 0) | (y == 3)
                b2 = float(((pm - hit) ** 2).mean())
                b2c = float(((pbm - hit) ** 2).mean())
                print(f"{g:<8}{h:>3}{thr:>6.0%}{len(y):>11,}{pm.mean():>11.3f}{hit.mean():>10.3f}"
                      f"{hit.mean()-pm.mean():>8.3f}{L.ece(pm, hit):>10.4f}"
                      f"{b2:>9.4f}{b2c:>12.4f}{L.skill_score(b2, b2c):>8.4f}", flush=True)

    hdr("B2 — SCALE (|r|>=2%) vs DIRECTION split, crisis and calm years")
    print(f"{'grp':<8}{'year':>6}{'h':>3}{'n':>9}"
          f"{'pred|r|':>9}{'obs|r|':>8}{'gap_scale':>10}"
          f"{'p_up/p_tot':>11}{'obs_up/tot':>11}{'gap_dir':>9}   diagnosis")
    for g in groups:
        for y in (2008, 2020, 2022, 2011, 2017, 2013):
            for h in (1, 5, 21):
                rm = res[g][(h, 0.02, "EMP")]
                sel = np.isin(rm["pos"], np.flatnonzero(close.index.year == y))
                if sel.sum() == 0:
                    continue
                p, yy = rm["p"][sel].astype(float), rm["y"][sel].astype(int)
                pm = (p[:, 0] + p[:, 3]).mean()
                hit = ((yy == 0) | (yy == 3))
                om = hit.mean()
                sh_p = p[:, 3].sum() / (p[:, 0] + p[:, 3]).sum()
                sh_o = (yy == 3).sum() / max(1, hit.sum())
                gs, gd = om - pm, sh_o - sh_p
                diag = ("scale OK" if abs(gs) < 0.03 else
                        ("scale UNDER" if gs > 0 else "scale OVER")) + ", " + \
                       ("dir OK" if abs(gd) < 0.05 else
                        ("dir too BEARISH" if gd > 0 else "dir too BULLISH"))
                print(f"{g:<8}{y:>6}{h:>3}{sel.sum():>9,}{pm:>9.3f}{om:>8.3f}{gs:>10.3f}"
                      f"{sh_p:>11.3f}{sh_o:>11.3f}{gd:>9.3f}   {diag}", flush=True)

    hdr("B3 — reliability of P(|r|>=2%) (scale only), h=5 and h=21")
    for g in groups:
        for h in (5, 21):
            rm = res[g][(h, 0.02, "EMP")]
            p, yy = rm["p"].astype(float), rm["y"].astype(int)
            pm = p[:, 0] + p[:, 3]
            hit = (yy == 0) | (yy == 3)
            print(f"\n--- {g} h={h} P(|r|>=2%)  ECE={L.ece(pm, hit):.4f}  n={len(yy):,} ---")
            print(L.reliability(pm, hit, 10).to_string(float_format=lambda v: f"{v:9.4f}"),
                  flush=True)

    # ------------------------------------------------------- C. floor sensitivity
    hdr("C — probability-floor sensitivity of LOG LOSS (Brier is floor-insensitive)")
    print("re-floors the STORED probabilities: p_alt = clip(p, f)/sum. Because FLOOR=1e-4 was "
          "already applied, this can only relax upward from 1e-4; to test tighter floors we "
          "recompute the raw empirical CDF cells from rawmin where they were the binding cell.")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'flr%':>7}{'LL@1e-4':>10}{'LL@1e-3':>10}"
          f"{'LL@1e-2':>10}{'clim@1e-4':>11}{'skill@1e-3':>12}")
    for g in groups:
        for h in HORIZONS:
            for thr in THRESHOLDS:
                rm, rb = res[g][(h, thr, "EMP")], res[g][(h, thr, "CLIM")]
                y = rm["y"].astype(int)
                lls = []
                for f in (1e-4, 1e-3, 1e-2):
                    q = np.clip(rm["p"].astype(float), f, None)
                    q /= q.sum(axis=1, keepdims=True)
                    lls.append(L.log_loss(q, y))
                ll0 = L.log_loss(rb["p"].astype(float), y)
                q = np.clip(rm["p"].astype(float), 1e-3, None); q /= q.sum(axis=1, keepdims=True)
                q0 = np.clip(rb["p"].astype(float), 1e-3, None); q0 /= q0.sum(axis=1, keepdims=True)
                print(f"{g:<8}{h:>3}{thr:>6.0%}{100*(rm['rawmin']<W.FLOOR).mean():>7.2f}"
                      f"{lls[0]:>10.4f}{lls[1]:>10.4f}{lls[2]:>10.4f}{ll0:>11.4f}"
                      f"{L.skill_score(L.log_loss(q, y), L.log_loss(q0, y)):>12.4f}", flush=True)

    # ------------------------------------------------------- D. EWMA overshoot
    hdr("D — EWMA sigma bias by forecast decile: realized/forecast vol ratio (IN-SAMPLE descriptive)")
    print(f"{'grp':<8}{'h':>3}{'dec':>4}{'n':>9}{'sig_fcst':>10}{'sig_real':>10}"
          f"{'ratio':>8}{'sd(z)':>8}{'mean(z)':>9}")
    for g, cg in groups.items():
        for h in (1, 5, 21):
            sig1 = L.vol_ewma(cg, W.LAM)
            if h == 1:
                # rolling(1).std(ddof=1) is NaN by construction; use the single-|r| unbiased
                # estimator sigma_hat = |r|*sqrt(pi/2)  (E|r| = sigma*sqrt(2/pi))
                real = np.log(cg.where(cg > 0)).diff().abs().shift(-1) * math.sqrt(math.pi / 2)
            else:
                real = L.realized_vol_forward(cg, h)
            A = W.flatten(cg, h)
            z = A["fwd_l"] / A["sig_h"]
            f_ = sig1.to_numpy()
            r_ = real.to_numpy()
            ok = np.zeros(f_.shape, bool)
            # rebuild the same mask flatten() used, so arrays line up
            m = np.zeros(f_.shape, bool)
            m[A["pos"], A["sym"]] = True
            fv, rv = f_[m], r_[m]
            good = np.isfinite(fv) & np.isfinite(rv)
            d = pd.qcut(pd.Series(fv[good]).rank(method="first"), 10, labels=False)
            df = pd.DataFrame({"f": fv[good], "r": rv[good], "z": z[good], "d": d})
            for dd, gg in df.groupby("d"):
                print(f"{g:<8}{h:>3}{int(dd):>4}{len(gg):>9,}{gg.f.mean():>10.4f}"
                      f"{gg.r.mean():>10.4f}{gg.r.mean()/gg.f.mean():>8.3f}"
                      f"{gg.z.std():>8.3f}{gg.z.mean():>9.3f}", flush=True)
            print()

    # ------------------------------------------------------- E. recent window headline
    hdr("E — RECENT WINDOW OOS (test years 2021-2026 only): the shipping expectation")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>10}{'n_dates':>8}{'BSS':>8}{'lo95':>8}{'hi95':>8}"
          f"{'LLskill':>9}{'ECE_up':>8}{'ECE_dn':>8}{'ECE_move':>9}{'BSS2_move':>10}")
    recent = np.flatnonzero(close.index.year >= 2021)
    for g in groups:
        for h in HORIZONS:
            for thr in THRESHOLDS:
                rm, rb = res[g][(h, thr, "EMP")], res[g][(h, thr, "CLIM")]
                sel = np.isin(rm["pos"], recent)
                sm = {k: v[sel] for k, v in rm.items()}
                sb = {k: v[sel] for k, v in rb.items()}
                p, y = sm["p"].astype(float), sm["y"].astype(int)
                pb = sb["p"].astype(float)
                bss = L.skill_score(L.brier_multi(p, y), L.brier_multi(pb, y))
                mu, lo, hi, nb = W.block_bootstrap_bss(sm, sb, h, nboot=300)
                iu, idn = 3, 0
                pm, pbm = p[:, 0] + p[:, 3], pb[:, 0] + pb[:, 3]
                hit = (y == 0) | (y == 3)
                b2 = float(((pm - hit) ** 2).mean()); b2c = float(((pbm - hit) ** 2).mean())
                print(f"{g:<8}{h:>3}{thr:>6.0%}{len(y):>10,}"
                      f"{len(np.unique(sm['pos'])):>8,}{bss:>8.4f}{lo:>8.4f}{hi:>8.4f}"
                      f"{L.skill_score(L.log_loss(p, y), L.log_loss(pb, y)):>9.4f}"
                      f"{L.ece(p[:, iu], y == iu):>8.4f}{L.ece(p[:, idn], y == idn):>8.4f}"
                      f"{L.ece(pm, hit):>9.4f}{L.skill_score(b2, b2c):>10.4f}", flush=True)

    hdr("E2 — per-index-symbol OOS BSS at thr=2% (the 5 'indices' are near-duplicates; "
        "SPY/^GSPC/DIA share ~1 effective series)")
    print(f"{'sym':<8}{'h':>3}{'n':>8}{'BSS':>9}{'LLskill':>9}{'ECE_up':>8}{'ECE_dn':>8}")
    for s in idx_syms:
        r, _ = W.run_group(s, close[[s]], verbose=False)
        for h in HORIZONS:
            rm, rb = r[(h, 0.02, "EMP")], r[(h, 0.02, "CLIM")]
            p, y = rm["p"].astype(float), rm["y"].astype(int)
            pb = rb["p"].astype(float)
            print(f"{s:<8}{h:>3}{len(y):>8,}"
                  f"{L.skill_score(L.brier_multi(p, y), L.brier_multi(pb, y)):>9.4f}"
                  f"{L.skill_score(L.log_loss(p, y), L.log_loss(pb, y)):>9.4f}"
                  f"{L.ece(p[:, 3], y == 3):>8.4f}{L.ece(p[:, 0], y == 0):>8.4f}", flush=True)


if __name__ == "__main__":
    main()
