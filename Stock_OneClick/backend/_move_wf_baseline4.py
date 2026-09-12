"""
_move_wf_baseline4.py — the decisive table: skill against the PER-SYMBOL climatology.

_move_wf_baseline.py scores against a POOLED climatology. For a single-ticker product that is a
soft baseline: it does not know that BLNK is 4x more volatile than KO, so part of the model's
"skill" is only that cross-sectional fact, which a user could get for free from the ticker's own
history. CLIMSYM (that symbol's own train-year bucket frequencies, shrunk by 40 pseudo-counts
toward the pooled ones) removes that freebie. Everything the model earns above CLIMSYM is
genuine conditional information -- vol timing plus distribution shape.
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


def sc(rm, rb, sel=None):
    p = rm["p"].astype(float) if sel is None else rm["p"][sel].astype(float)
    y = rm["y"].astype(int) if sel is None else rm["y"][sel].astype(int)
    pb = rb["p"].astype(float) if sel is None else rb["p"][sel].astype(float)
    pm, pbm = p[:, 0] + p[:, 3], pb[:, 0] + pb[:, 3]
    hit = (y == 0) | (y == 3)
    b2, b2c = float(((pm - hit) ** 2).mean()), float(((pbm - hit) ** 2).mean())
    return dict(n=len(y),
                bss=L.skill_score(L.brier_multi(p, y), L.brier_multi(pb, y)),
                lls=L.skill_score(L.log_loss(p, y), L.log_loss(pb, y)),
                bss2=L.skill_score(b2, b2c),
                ece_up=L.ece(p[:, 3], y == 3), ece_dn=L.ece(p[:, 0], y == 0),
                ece_move=L.ece(pm, hit))


def main():
    close, idx_syms, singles = W.load_panel()
    groups = {"INDEX": close[idx_syms], "SINGLES": close[singles]}
    res = {}
    for g, cg in groups.items():
        print(f"running {g} ...", flush=True)
        res[g], _ = W.run_group(g, cg, verbose=False)

    hdr("G1 — POOLED OOS skill vs the PER-SYMBOL climatology (CLIMSYM), all 21 test years")
    print("BSS4/LLskill/BSS2 are now measured against 'quote this ticker's own base rate'.")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>10}"
          f"{'BSS4vsPOOL':>11}{'BSS4vsSYM':>10}{'LLsk_vsSYM':>11}{'BSS2_vsSYM':>11}"
          f"{'CLIMSYMvsPOOL':>14}")
    rows = []
    for g in groups:
        for h in W.HORIZONS:
            for thr in W.THRESHOLDS:
                rm = res[g][(h, thr, "EMP")]
                rp, rs = res[g][(h, thr, "CLIM")], res[g][(h, thr, "CLIMSYM")]
                a, b = sc(rm, rp), sc(rm, rs)
                c = sc(rs, rp)
                rows.append(dict(grp=g, h=h, thr=thr, n=a["n"], bss_pool=a["bss"],
                                 bss_sym=b["bss"], lls_sym=b["lls"], bss2_sym=b["bss2"],
                                 climsym_gain=c["bss"]))
                print(f"{g:<8}{h:>3}{thr:>6.0%}{a['n']:>10,}"
                      f"{a['bss']:>11.4f}{b['bss']:>10.4f}{b['lls']:>11.4f}{b['bss2']:>11.4f}"
                      f"{c['bss']:>14.4f}", flush=True)
    pd.DataFrame(rows).to_csv(Path(__file__).with_name("_move_wf_baseline_table_g1.csv"),
                              index=False)

    hdr("G2 — full model ladder vs CLIMSYM (BSS4). CONSTSYM should now collapse to ~0, "
        "because a frozen per-symbol sigma carries almost the same information as CLIMSYM.")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}" + "".join(f"{m:>11}" for m in W.MODELS) +
          f"{'vol_gain':>10}{'shape_gain':>11}{'loc/scale':>11}")
    for g in groups:
        for h in W.HORIZONS:
            for thr in W.THRESHOLDS:
                rs = res[g][(h, thr, "CLIMSYM")]
                v = {m: sc(res[g][(h, thr, m)], rs)["bss"] for m in W.MODELS}
                print(f"{g:<8}{h:>3}{thr:>6.0%}" + "".join(f"{v[m]:>11.4f}" for m in W.MODELS) +
                      f"{v['EMP']-v['CONSTSYM']:>10.4f}"
                      f"{v['EMP']-v['NORMFIT']:>11.4f}{v['NORMFIT']-v['NORMSTD']:>11.4f}",
                      flush=True)

    hdr("G3 — per-symbol distribution of BSS4 vs CLIMSYM across 231 singles (the shipping risk)")
    print(f"{'h':>3}{'thr':>6}{'metric':<10}{'n_syms':>8}{'p05':>9}{'p25':>9}{'median':>9}"
          f"{'p75':>9}{'p95':>9}{'mean':>9}{'frac<=0':>9}")
    keep = {}
    for h in W.HORIZONS:
        for thr in (0.02, 0.05):
            rm, rs = res["SINGLES"][(h, thr, "EMP")], res["SINGLES"][(h, thr, "CLIMSYM")]
            recs = []
            for si in range(len(singles)):
                sel = rm["sym"] == si
                if sel.sum() < 250:
                    continue
                recs.append({"sym": singles[si], **sc(rm, rs, sel)})
            df = pd.DataFrame(recs)
            keep[(h, thr)] = df
            for met in ("bss", "lls", "bss2", "ece_up", "ece_move"):
                v = df[met].dropna()
                print(f"{h:>3}{thr:>6.0%}{met:<10}{len(v):>8}"
                      f"{v.quantile(.05):>9.4f}{v.quantile(.25):>9.4f}{v.median():>9.4f}"
                      f"{v.quantile(.75):>9.4f}{v.quantile(.95):>9.4f}{v.mean():>9.4f}"
                      f"{(v <= 0).mean():>9.3f}", flush=True)
            print()

    hdr("G4 — vs CLIMSYM: does skill still line up with the symbol's vol quintile? (h=5, 2%)")
    d = keep[(5, 0.02)].copy()
    sig = L.vol_ewma(close[singles], W.LAM).mean()
    d["sigma"] = d["sym"].map(sig)
    d["q"] = pd.qcut(d["sigma"].rank(method="first"), 5, labels=False)
    print(f"{'sigma_q':>8}{'n_syms':>8}{'mean_sigma':>12}{'BSS4':>9}{'BSS2':>9}{'frac<=0':>9}")
    for q, gg in d.groupby("q"):
        print(f"{int(q):>8}{len(gg):>8}{gg.sigma.mean():>12.4f}{gg.bss.mean():>9.4f}"
              f"{gg.bss2.mean():>9.4f}{(gg.bss <= 0).mean():>9.3f}", flush=True)

    hdr("G5 — RECENT WINDOW (2021-2026) vs CLIMSYM with date-block bootstrap CI, thr=2% & 5%")
    print(f"{'grp':<8}{'h':>3}{'thr':>6}{'n':>10}{'n_dates':>8}{'BSS4':>8}{'lo95':>8}{'hi95':>8}"
          f"{'BSS2':>8}{'lo95_2':>8}{'hi95_2':>8}{'ECE_up':>8}{'ECE_move':>9}")
    recent = np.flatnonzero(close.index.year >= 2021)
    for g in groups:
        for h in W.HORIZONS:
            for thr in (0.02, 0.05):
                rm, rs = res[g][(h, thr, "EMP")], res[g][(h, thr, "CLIMSYM")]
                sel = np.isin(rm["pos"], recent)
                sm = {k: v[sel] for k, v in rm.items()}
                ss = {k: v[sel] for k, v in rs.items()}
                m = sc(sm, ss)
                _, lo, hi, _ = W.block_bootstrap_bss(sm, ss, h, nboot=300)
                # 2-class bootstrap: reuse the machinery on collapsed 2-column probs
                sm2 = dict(sm); ss2 = dict(ss)
                for d_ in (sm2, ss2):
                    q = d_["p"].astype(float)
                    d_["p"] = np.column_stack([q[:, 0] + q[:, 3], q[:, 1] + q[:, 2]]).astype(np.float32)
                sm2["y"] = np.where((sm["y"] == 0) | (sm["y"] == 3), 0, 1).astype(np.int8)
                ss2["y"] = sm2["y"]
                _, lo2, hi2, _ = W.block_bootstrap_bss(sm2, ss2, h, nboot=300)
                print(f"{g:<8}{h:>3}{thr:>6.0%}{m['n']:>10,}{len(np.unique(sm['pos'])):>8,}"
                      f"{m['bss']:>8.4f}{lo:>8.4f}{hi:>8.4f}"
                      f"{m['bss2']:>8.4f}{lo2:>8.4f}{hi2:>8.4f}"
                      f"{m['ece_up']:>8.4f}{m['ece_move']:>9.4f}", flush=True)

    hdr("G6 — by-year BSS4 vs CLIMSYM (thr=2%): how many years is the model actually useful?")
    yrs = sorted(set(close.index.year))[W.MIN_TRAIN_YEARS:]
    yearpos = {y: np.flatnonzero(close.index.year == y) for y in yrs}
    for g in groups:
        print(f"\n--- {g} ---")
        print(f"{'year':<6}" + "".join(f"{f'h{h}':>9}" for h in W.HORIZONS))
        neg = {h: 0 for h in W.HORIZONS}
        for y in yrs:
            line = f"{y:<6}"
            for h in W.HORIZONS:
                rm, rs = res[g][(h, 0.02, "EMP")], res[g][(h, 0.02, "CLIMSYM")]
                sel = np.isin(rm["pos"], yearpos[y])
                b = sc(rm, rs, sel)["bss"]
                neg[h] += b <= 0
                line += f"{b:>9.4f}"
            print(line)
        print("neg_yrs" + "".join(f"{neg[h]:>9d}" for h in W.HORIZONS) + f"   of {len(yrs)}")


if __name__ == "__main__":
    main()
