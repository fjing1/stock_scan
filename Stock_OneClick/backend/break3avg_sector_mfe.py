#!/usr/bin/env python3
"""Historical max-up-potential (MFE) of break SMA 8/22 signals, by SECTOR.

For every break-8/22 signal in the broad universe (full history), measure the Maximum
Favorable Excursion (highest high reached after entry, as % of entry close) and the
Maximum Adverse Excursion (worst dip), over several forward windows. Aggregate by
sector to answer: when a sector breaks out, how much upside typically follows, and
how often does it reach +10% / +20%?

    ../../vcp_env/bin/python break3avg_sector_mfe.py
    ../../vcp_env/bin/python break3avg_sector_mfe.py --window 60

Output per sector: n signals, median/75th/90th MFE (max up) at H bars, P(reach +10/+20%),
median MAE (downside endured). Also a strong-RS vs all split overall.
Caveat: close-to-close highs/lows, no costs; survivorship (delisted names absent) inflates
absolute MFE — trust the CROSS-SECTIONAL (sector-vs-sector) ranking most.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402
from break3avg_scan import break3avg_signal, load_all_market, _rs_tier  # noqa: E402
from break3avg_sector import SECTORS, ASSIGN_ORDER, batch_dl  # noqa: E402

WIN = 60          # max forward window for MFE/MAE + target-hit
TARGETS = [0.05, 0.10, 0.20]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=WIN, help="forward window in bars for MFE (default 60)")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()
    H = args.window

    root = BACKEND_DIR.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import stock_symbols_1243 as u

    attr_to_label = {attr: label for label, attr in SECTORS}
    sector_of = {}
    for attr in ASSIGN_ORDER:
        for s in getattr(u, attr, []):
            sector_of.setdefault(s, attr_to_label[attr])
    all_syms = list(sector_of.keys())
    print(f"Universe: {len(all_syms)} names across {len(SECTORS)} sectors. Downloading full history (batched)...")
    frames = batch_dl([scan.to_yfinance_symbol(s) for s in all_syms], args.period)

    try:
        spy = scan.download_daily("SPY", period="max")["Close"].astype(float)
        spy_ret = spy / spy.shift(252) - 1
    except Exception:
        spy_ret = None

    recs = []
    for sym, df in frames.items():
        sec = sector_of.get(sym)
        if sec is None or len(df) < 300:
            continue
        c = df["Close"].astype(float).values
        hi = df["High"].astype(float).values
        lo = df["Low"].astype(float).values
        dates = df.index
        e = break3avg_signal(df, 8, 22, "SMA", "SMA").values
        for i in np.where(e)[0]:
            if i + 1 >= len(c):
                continue
            end = min(i + 1 + H, len(c))
            mfe = hi[i + 1:end].max() / c[i] - 1
            mae = lo[i + 1:end].min() / c[i] - 1
            rec = {"sector": sec, "mfe": mfe, "mae": mae}
            for t in TARGETS:
                rec[f"hit{int(t*100)}"] = 1 if hi[i + 1:end].max() >= c[i] * (1 + t) else 0
            # RS tier
            if spy_ret is not None and i >= 252:
                s12 = c[i] / c[i - 252] - 1
                sp = spy_ret.asof(dates[i])
                rec["tier"] = _rs_tier(s12 - sp) if np.isfinite(sp) else "n/a"
            else:
                rec["tier"] = "n/a"
            recs.append(rec)
    d = pd.DataFrame(recs)
    print(f"got {frames.__len__()} names with data, {len(d)} break-8/22 signals (H={H} bars)\n")
    if d.empty:
        return 0

    def block(sub):
        return (len(sub), sub["mfe"].median(), sub["mfe"].quantile(.75), sub["mfe"].quantile(.90),
                sub[f"hit10"].mean(), sub[f"hit20"].mean(), sub["mae"].median())

    print(f"=== max UP potential (MFE) of break-8/22 by sector, within {H} bars ===")
    print(f"  {'sector':<18}{'n':>6}{'MFE med':>9}{'MFE 75th':>10}{'MFE 90th':>10}"
          f"{'P(+10%)':>9}{'P(+20%)':>9}{'MAE med':>9}")
    rows = []
    for label, _ in SECTORS:
        sub = d[d["sector"] == label]
        if len(sub) < 30:
            continue
        rows.append((label,) + block(sub))
    for r in sorted(rows, key=lambda x: x[2], reverse=True):   # sort by median MFE
        label, n, med, q75, q90, p10, p20, mae = r
        print(f"  {label:<18}{n:>6}{med:>+9.1%}{q75:>+10.1%}{q90:>+10.1%}"
              f"{p10:>9.0%}{p20:>9.0%}{mae:>+9.1%}")

    allrow = ("ALL",) + block(d)
    print(f"  {'-'*72}")
    label, n, med, q75, q90, p10, p20, mae = allrow
    print(f"  {'ALL sectors':<18}{n:>6}{med:>+9.1%}{q75:>+10.1%}{q90:>+10.1%}{p10:>9.0%}{p20:>9.0%}{mae:>+9.1%}")

    # strong vs weak RS (max-up-potential lift from conviction)
    print(f"\n=== max up potential by RS tier (all sectors, within {H} bars) ===")
    print(f"  {'tier':<10}{'n':>7}{'MFE med':>9}{'MFE 75th':>10}{'P(+10%)':>9}{'P(+20%)':>9}{'MAE med':>9}")
    for tier in ["strong", "mid", "weak"]:
        sub = d[d["tier"] == tier]
        if len(sub) < 30:
            continue
        n, med, q75, q90, p10, p20, mae = block(sub)
        print(f"  {tier:<10}{n:>7}{med:>+9.1%}{q75:>+10.1%}{p10:>9.0%}{p20:>9.0%}{mae:>+9.1%}")

    print(f"\nRead: 'MFE med' = the typical peak a break reaches within {H} bars (your realistic")
    print("target ceiling); '90th' = best-case runners. P(+10/20%) = odds of reaching that target.")
    print("MAE med = the dip you typically endure first (informs the stop). Sector ranking is the")
    print("robust part; absolute MFE is inflated by survivorship (failed names' breaks are absent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
