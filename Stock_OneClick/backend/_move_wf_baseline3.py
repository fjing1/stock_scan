"""
_move_wf_baseline3.py — third pass: does the pooled-fit model work for an INDIVIDUAL ticker?

The pooled numbers answer "is the harness skilful on average". A user asks about ONE ticker, so
the shipping question is the DISTRIBUTION of per-symbol out-of-sample scores, not the pool mean.
Everything here is sliced out of the same walk-forward run as _move_wf_baseline.py (pooled z-fit,
train years only), just grouped by symbol at scoring time -- so no extra fitting, no lookahead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _move_lib as L
import _move_wf_baseline as W


def hdr(s):
    print("\n" + "=" * 100)
    print(s)
    print("=" * 100, flush=True)


def sym_metrics(rm, rb, sel):
    p, y = rm["p"][sel].astype(float), rm["y"][sel].astype(int)
    pb = rb["p"][sel].astype(float)
    pm, pbm = p[:, 0] + p[:, 3], pb[:, 0] + pb[:, 3]
    hit = (y == 0) | (y == 3)
    b2, b2c = float(((pm - hit) ** 2).mean()), float(((pbm - hit) ** 2).mean())
    return dict(
        n=int(sel.sum()),
        bss=L.skill_score(L.brier_multi(p, y), L.brier_multi(pb, y)),
        lls=L.skill_score(L.log_loss(p, y), L.log_loss(pb, y)),
        bss2=L.skill_score(b2, b2c),
        ece_up=L.ece(p[:, 3], y == 3), ece_dn=L.ece(p[:, 0], y == 0),
        ece_move=L.ece(pm, hit),
        gap_move=float(hit.mean() - pm.mean()),
    )


def main():
    close, idx_syms, singles = W.load_panel()
    groups = {"INDEX": close[idx_syms], "SINGLES": close[singles]}
    res = {}
    for g, cg in groups.items():
        print(f"running {g} ...", flush=True)
        res[g], _ = W.run_group(g, cg, verbose=False)

    hdr("F1 — per-INDEX-SYMBOL OOS scores from the POOLED walk-forward fit, thr=2%, all 21 test yrs")
    print(f"{'sym':<8}{'h':>3}{'n':>7}{'BSS4':>8}{'LLskill':>9}{'BSS2_move':>10}"
          f"{'ECE_up':>8}{'ECE_dn':>8}{'ECE_move':>9}{'gap_move':>9}")
    for h in W.HORIZONS:
        rm, rb = res["INDEX"][(h, 0.02, "EMP")], res["INDEX"][(h, 0.02, "CLIM")]
        for si, s in enumerate(idx_syms):
            m = sym_metrics(rm, rb, rm["sym"] == si)
            print(f"{s:<8}{h:>3}{m['n']:>7,}{m['bss']:>8.4f}{m['lls']:>9.4f}{m['bss2']:>10.4f}"
                  f"{m['ece_up']:>8.4f}{m['ece_dn']:>8.4f}{m['ece_move']:>9.4f}"
                  f"{m['gap_move']:>9.4f}", flush=True)
        print()

    hdr("F2 — DISTRIBUTION of per-symbol OOS scores across the 231 single names (thr=2%)")
    print("This is the shipping-risk number: how often is the model WORSE than climatology "
          "for the particular ticker a user typed?")
    print(f"{'h':>3}{'thr':>6}{'metric':<12}{'n_syms':>8}{'p05':>9}{'p25':>9}{'median':>9}"
          f"{'p75':>9}{'p95':>9}{'mean':>9}{'frac<0':>9}")
    dist_rows = []
    for h in W.HORIZONS:
        for thr in (0.02, 0.05):
            rm, rb = res["SINGLES"][(h, thr, "EMP")], res["SINGLES"][(h, thr, "CLIM")]
            recs = []
            for si in range(len(singles)):
                sel = rm["sym"] == si
                if sel.sum() < 250:
                    continue
                recs.append({"sym": singles[si], **sym_metrics(rm, rb, sel)})
            df = pd.DataFrame(recs)
            for met in ("bss", "lls", "bss2", "ece_up", "ece_move"):
                v = df[met].dropna()
                print(f"{h:>3}{thr:>6.0%}{met:<12}{len(v):>8}"
                      f"{v.quantile(.05):>9.4f}{v.quantile(.25):>9.4f}{v.median():>9.4f}"
                      f"{v.quantile(.75):>9.4f}{v.quantile(.95):>9.4f}{v.mean():>9.4f}"
                      f"{(v < 0).mean():>9.3f}", flush=True)
            print()
            if thr == 0.02:
                d = df.copy(); d["h"] = h
                dist_rows.append(d)
    pd.concat(dist_rows).to_csv(Path(__file__).with_name("_move_wf_baseline_persymbol.csv"),
                               index=False)

    hdr("F3 — worst and best single names by BSS4 at h=5, thr=2% (n>=250 OOS rows)")
    d = [x for x in dist_rows if x["h"].iloc[0] == 5][0].sort_values("bss")
    cols = ["sym", "n", "bss", "lls", "bss2", "ece_up", "ece_move", "gap_move"]
    print("WORST 12:\n" + d[cols].head(12).to_string(index=False,
          float_format=lambda v: f"{v:9.4f}"))
    print("\nBEST 12:\n" + d[cols].tail(12).to_string(index=False,
          float_format=lambda v: f"{v:9.4f}"))

    hdr("F4 — is the surviving skill just the LOW-VOL / HIGH-VOL cross-section? "
        "per-symbol BSS4 (h=5, thr=2%) vs that symbol's mean sigma")
    d2 = [x for x in dist_rows if x["h"].iloc[0] == 5][0].copy()
    sig = L.vol_ewma(close[singles], W.LAM).mean()
    d2["sigma"] = d2["sym"].map(sig)
    d2["q"] = pd.qcut(d2["sigma"].rank(method="first"), 5, labels=False)
    print(f"{'sigma_q':>8}{'n_syms':>8}{'mean_sigma':>12}{'BSS4':>9}{'BSS2':>9}"
          f"{'ECE_up':>9}{'gap_move':>10}")
    for q, gg in d2.groupby("q"):
        print(f"{int(q):>8}{len(gg):>8}{gg.sigma.mean():>12.4f}{gg.bss.mean():>9.4f}"
              f"{gg.bss2.mean():>9.4f}{gg.ece_up.mean():>9.4f}{gg.gap_move.mean():>10.4f}",
              flush=True)


if __name__ == "__main__":
    main()
