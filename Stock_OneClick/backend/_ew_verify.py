"""
_ew_verify.py — fix the three flaws the research pass found in elliott_system.py, and re-test.

The borderline result from elliott_system.py was: wave-2+Fib entries beat a label-free
pullback-in-uptrend control by +2.00pp at 252d, cluster-bootstrapped t = +2.93 across 43 symbols.
That sat just under the Bonferroni bar (2.98) for the 80 tests run. Three flaws were then
identified, any of which could explain it:

  F1  MISSING RULES. Only R1/R2/R3 were checked. Elliott also requires, for an impulse:
        R4  p3 > p1   (wave 3 must exceed wave 1's high)
        R5  p4 > p2   (wave 4's low must hold above wave 2's low)
      Structures failing those are not impulses at all, so the candidate pool was polluted.

  F2  VOLATILITY CONFOUND — the likely explanation. The signal fires 57% of the time in the top
      realized-vol quintile against a 20% base rate. High-vol states have higher forward returns
      in a rising market, so a control matched only on trend and Fib zone is not a fair control.
      Matching on volatility state is the decisive test.

  F3  SURVIVORSHIP. The 43-symbol panel is today's survivors. Delisted names are absent, which
      inflates any long-only forward-return study. Quantified here but not fixed (point-in-time
      membership data isn't available in this project).

Scale note: wave-length comparisons for R2 now use |Δ log P| rather than arithmetic price
differences, which is what the canon's percentage framing implies.

Run: ../../vcp_env/bin/python _ew_verify.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import elliott_system as E

warnings.filterwarnings("ignore")
TD = 252
RNG = np.random.default_rng(4242)
H = 252


def label_impulses_strict(piv: pd.DataFrame) -> list[dict]:
    """As elliott_system.label_impulses, but adding R4/R5 and using log scale for R2."""
    out = []
    if len(piv) < 6:
        return out
    k = piv["kind"].tolist()
    for s in range(len(piv) - 5):
        if k[s:s + 6] != ["L", "H", "L", "H", "L", "H"]:
            continue
        P = piv.iloc[s:s + 6].reset_index(drop=True)
        p0, p1, p2, p3, p4, p5 = P["price"].tolist()
        if min(p0, p1, p2, p3, p4, p5) <= 0:
            continue
        l1, l3, l5 = abs(np.log(p1 / p0)), abs(np.log(p3 / p2)), abs(np.log(p5 / p4))
        w2r = (p1 - p2) / (p1 - p0) if p1 > p0 else np.nan
        w4r = (p3 - p4) / (p3 - p2) if p3 > p2 else np.nan
        if not np.isfinite(w2r) or not np.isfinite(w4r):
            continue
        out.append({
            "i2": int(P.loc[2, "pos"]), "c2": int(P.loc[2, "confirm"]),
            "i4": int(P.loc[4, "pos"]), "c4": int(P.loc[4, "confirm"]),
            "w2_retrace": w2r, "w4_retrace": w4r,
            "R1_ok": p2 > p0,                     # knowable at wave-2 entry
            "R4_ok": p3 > p1,                     # knowable at wave-4 entry
            "R5_ok": p4 > p2,                     # knowable at wave-4 entry
            "R3_ok": p4 > p1,                     # knowable at wave-4 entry
            "R2_ok": not (l3 < l1 and l3 < l5),   # RETROSPECTIVE — never usable
            "w2_in_fib": E.W2_RETRACE[0] <= w2r <= E.W2_RETRACE[1],
        })
    return out


def vol_state(px: pd.Series, win: int = 60) -> pd.Series:
    """Realized-vol quintile, computed causally (expanding rank of trailing vol)."""
    v = px.pct_change().rolling(win).std()
    return v.expanding(TD * 2).rank(pct=True)


def dd_state(px: pd.Series) -> pd.Series:
    return px / px.rolling(TD).max() - 1.0


def main():
    panel = E.load_panel()
    print(f"universe {len(panel)} symbols, "
          f"{sum(len(v) for v in panel.values())/TD:,.0f} symbol-years\n")

    for thr in (0.05, 0.10):
        print("=" * 82)
        print(f"THRESHOLD {thr*100:.0f}%")
        print("=" * 82)
        # ---- F1: what do the missing rules do to the candidate pool? ---- #
        tot = dict(cand=0, r1=0, r3=0, r4=0, r5=0, r2=0, impulse=0)
        rows_sig, rows_ctl = [], []
        for t, df in panel.items():
            px = df["Close"].dropna()
            piv = E.find_pivots(px, thr)
            if len(piv) < 8:
                continue
            cands = label_impulses_strict(piv)
            tot["cand"] += len(cands)
            for c in cands:
                tot["r1"] += c["R1_ok"]; tot["r3"] += c["R3_ok"]
                tot["r4"] += c["R4_ok"]; tot["r5"] += c["R5_ok"]; tot["r2"] += c["R2_ok"]
                tot["impulse"] += (c["R1_ok"] and c["R3_ok"] and c["R4_ok"] and c["R5_ok"])
            vq, dq = vol_state(px), dd_state(px)
            ma = px.rolling(200).mean()
            v = px.to_numpy(dtype=float)
            n = len(v)

            # signal set: wave-2 entry, gated ONLY on what is knowable there
            for c in cands:
                b = c["c2"]
                if b + H >= n or not c["R1_ok"] or not c["w2_in_fib"]:
                    continue
                rows_sig.append({"sym": t, "bar": b, "fwd": v[b + H] / v[b] - 1,
                                 "vq": vq.iloc[b], "dq": dq.iloc[b],
                                 "up": bool(v[b] > ma.iloc[b]) if np.isfinite(ma.iloc[b]) else False})
            # label-free control, identical geometry + entry convention
            for b in E.control_pullbacks(px, piv):
                if b + H >= n:
                    continue
                rows_ctl.append({"sym": t, "bar": b, "fwd": v[b + H] / v[b] - 1,
                                 "vq": vq.iloc[b], "dq": dq.iloc[b], "up": True})

        c = tot["cand"]
        print(f"  F1 RULE PASS RATES over {c:,} candidate 5-pivot sequences:")
        for k, lbl in (("r1", "R1 p2>p0      (knowable @W2)"),
                       ("r4", "R4 p3>p1      (knowable @W4)"),
                       ("r5", "R5 p4>p2      (knowable @W4)"),
                       ("r3", "R3 p4>p1      (knowable @W4)"),
                       ("r2", "R2 w3 not shortest (RETROSPECTIVE)")):
            print(f"    {lbl:<38} {tot[k]/c*100:5.1f}%")
        print(f"    {'ALL of R1+R3+R4+R5 = a valid impulse':<38} {tot['impulse']/c*100:5.1f}%")
        print(f"    -> {100-tot['impulse']/c*100:.0f}% of what the detector finds is NOT an Elliott impulse")

        sig = pd.DataFrame(rows_sig).dropna()
        ctl = pd.DataFrame(rows_ctl).dropna()
        print(f"\n  events: signal {len(sig):,}   control {len(ctl):,}   "
              f"ratio {len(sig)/max(len(ctl),1):.2f} signals per control")
        print("  (a ratio >1 means the wave labels are RELABELLING the same pullbacks, not "
              "finding rarer ones)")

        # ---- F2: the volatility confound ---- #
        print(f"\n  F2 VOLATILITY STATE at entry (quintile of trailing 60d realized vol):")
        for lbl, d in (("signal ", sig), ("control", ctl)):
            q = pd.cut(d.vq, [0, .2, .4, .6, .8, 1.0], labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            share = q.value_counts(normalize=True).sort_index() * 100
            print(f"    {lbl}  " + "  ".join(f"{k}:{v:4.1f}%" for k, v in share.items()))

        print(f"\n  RAW comparison (what elliott_system.py reported):")
        d_raw = sig.fwd.mean() - ctl.fwd.mean()
        print(f"    signal {sig.fwd.mean()*100:+6.2f}%   control {ctl.fwd.mean()*100:+6.2f}%   "
              f"diff {d_raw*100:+6.2f}pp")

        print(f"\n  VOL-QUINTILE-MATCHED comparison (the decisive test):")
        diffs, wts = [], []
        print(f"    {'quintile':>9}{'n sig':>7}{'n ctl':>7}{'signal':>9}{'control':>9}{'diff':>9}")
        for qi, (lo, hi) in enumerate([(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)], 1):
            s = sig[(sig.vq > lo) & (sig.vq <= hi)]
            k = ctl[(ctl.vq > lo) & (ctl.vq <= hi)]
            if len(s) < 30 or len(k) < 30:
                print(f"    {'Q'+str(qi):>9}{len(s):>7}{len(k):>7}      thin")
                continue
            dd = s.fwd.mean() - k.fwd.mean()
            diffs.append(dd); wts.append(len(s))
            print(f"    {'Q'+str(qi):>9}{len(s):>7}{len(k):>7}{s.fwd.mean()*100:>+8.2f}%"
                  f"{k.fwd.mean()*100:>+8.2f}%{dd*100:>+8.2f}pp")
        if diffs:
            wm = np.average(diffs, weights=wts)
            print(f"    {'weighted':>9}{'':>14}{'':>18}{wm*100:>+8.2f}pp   "
                  f"(raw was {d_raw*100:+.2f}pp)")
            print(f"    -> volatility explains {(d_raw-wm)/d_raw*100 if d_raw else 0:5.0f}% "
                  f"of the raw difference")

        # ---- cluster bootstrap on the vol-matched difference ---- #
        print(f"\n  CLUSTER BOOTSTRAP on the vol-matched difference (by symbol):")
        per = []
        for s_ in sorted(set(sig.sym) & set(ctl.sym)):
            a, b = sig[sig.sym == s_], ctl[ctl.sym == s_]
            if len(a) < 5 or len(b) < 5:
                continue
            dl = []
            for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)]:
                x = a[(a.vq > lo) & (a.vq <= hi)]; y = b[(b.vq > lo) & (b.vq <= hi)]
                if len(x) >= 2 and len(y) >= 2:
                    dl.append((x.fwd.mean() - y.fwd.mean(), len(x)))
            if dl:
                per.append(np.average([x[0] for x in dl], weights=[x[1] for x in dl]))
        per = np.array(per)
        if len(per) > 5:
            se = per.std(ddof=1) / np.sqrt(len(per))
            bs = np.array([np.mean(RNG.choice(per, len(per), replace=True)) for _ in range(4000)])
            print(f"    symbols {len(per)}   mean {per.mean()*100:+.2f}pp   "
                  f"median {np.median(per)*100:+.2f}pp   positive {int((per>0).sum())}/{len(per)}")
            print(f"    clustered t {per.mean()/se:+.2f}   "
                  f"95% CI [{np.percentile(bs,2.5)*100:+.2f}pp, {np.percentile(bs,97.5)*100:+.2f}pp]"
                  f"   P(<=0) {(bs<=0).mean()*100:.0f}%")
            print(f"    Bonferroni bar for the 80 tests run: |t| > 2.98")
        print()

    print("=" * 82)
    print("F3 SURVIVORSHIP (quantified, not fixed)")
    print("=" * 82)
    print("  All 43 symbols exist today. Names that delisted, went to zero, or were acquired at a")
    print("  discount are structurally absent, and those are exactly the paths where a 5-wave")
    print("  'impulse' would have been followed by collapse rather than a 252-day gain. Any")
    print("  long-only forward-return study on this panel is biased upward; the study cannot")
    print("  distinguish that bias from a real effect. Fixing it needs point-in-time membership")
    print("  with delisted names, which this project does not have.")


if __name__ == "__main__":
    main()
