#!/usr/bin/env python3
"""Sector breakout breadth: which SECTOR is currently breaking up (break SMA 8/22)?

For each GICS-ish sector (lists in stock_symbols_1243.py), count how many names have
a break SMA 8/22 signal within the last --within bars, as a % of names with data
(= breakout breadth). Rank sectors by breadth so you can see where the breakouts are
concentrated. Also flags strong-RS (hi-conv) share and lists the strong-RS names.

    ../../vcp_env/bin/python break3avg_sector.py                # within 5 bars
    ../../vcp_env/bin/python break3avg_sector.py --within 1     # today only
    ../../vcp_env/bin/python break3avg_sector.py --list-strong  # print strong-RS names per sector

Uses the same batched downloader + break3avg_signal + 12m-RS tier as break3avg_scan.
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
from break3avg_scan import break3avg_signal, _rs_tier  # noqa: E402

SECTORS = [
    ("Technology", "TECH_STOCKS"),
    ("Healthcare", "HEALTHCARE_STOCKS"),
    ("Financials", "FINANCIAL_STOCKS"),
    ("Consumer Disc.", "CONSUMER_DISCRETIONARY"),
    ("Consumer Staples", "CONSUMER_STAPLES"),
    ("Energy", "ENERGY_STOCKS"),
    ("Materials/Indust.", "MATERIALS_INDUSTRIALS"),
    ("Real Estate/REIT", "REAL_ESTATE_REITS"),
    ("Utilities", "UTILITIES"),
    ("Comm. Services", "COMMUNICATION_SERVICES"),
]

# Representative SPDR sector ETF per bucket (Materials/Indust. -> both XLI & XLB)
SECTOR_ETF = {
    "Technology": ["XLK"], "Healthcare": ["XLV"], "Financials": ["XLF"],
    "Consumer Disc.": ["XLY"], "Consumer Staples": ["XLP"], "Energy": ["XLE"],
    "Materials/Indust.": ["XLI", "XLB"], "Real Estate/REIT": ["XLRE"],
    "Utilities": ["XLU"], "Comm. Services": ["XLC"],
}

# Source lists overlap (e.g. ENERGY_STOCKS is polluted with ~52 utility names). Assign
# each ticker to its MOST-SPECIFIC sector first so Utilities/Real-Estate claim their own
# names before the broader/contaminated buckets (Energy) can grab them.
ASSIGN_ORDER = ["UTILITIES", "REAL_ESTATE_REITS", "ENERGY_STOCKS", "FINANCIAL_STOCKS",
                "CONSUMER_STAPLES", "MATERIALS_INDUSTRIALS", "HEALTHCARE_STOCKS",
                "CONSUMER_DISCRETIONARY", "COMMUNICATION_SERVICES", "TECH_STOCKS"]


def batch_dl(symbols, period):
    import yfinance as yf
    out = {}
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]
        try:
            raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                              progress=False, group_by="ticker", threads=True)
        except Exception:
            continue
        for s in chunk:
            try:
                d = raw[s].dropna(how="all")
            except Exception:
                continue
            if len(d) >= 60:
                out[s] = d
        print(f"  ...{min(i+100,len(symbols))}/{len(symbols)} downloaded, {len(out)} with data", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--within", type=int, default=5, help="signal within N latest bars (default 5)")
    ap.add_argument("--period", default="2y")
    ap.add_argument("--list-strong", action="store_true", help="print strong-RS names per sector")
    args = ap.parse_args()

    root = BACKEND_DIR.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import stock_symbols_1243 as u

    sector_of, all_syms = {}, []
    attr_to_label = {attr: label for label, attr in SECTORS}
    # assign most-specific sector first (fixes utilities stolen by the polluted Energy list)
    for attr in ASSIGN_ORDER:
        for s in getattr(u, attr, []):
            sector_of.setdefault(s, attr_to_label[attr])
    for _, attr in SECTORS:
        all_syms += list(getattr(u, attr, []))
    all_syms = list(dict.fromkeys(all_syms))
    print(f"Sectors: {len(SECTORS)}   unique names: {len(all_syms)}   downloading (batched)...")

    frames = batch_dl([scan.to_yfinance_symbol(s) for s in all_syms], args.period)

    # sector ETFs (breakout at the ETF level, alongside constituent breadth)
    etf_syms = sorted({e for lst in SECTOR_ETF.values() for e in lst})
    etf_frames = batch_dl(etf_syms, args.period)

    # SPY 12m for RS
    try:
        spy = scan.download_daily("SPY", period=args.period)["Close"].astype(float)
        spy_12m = float(spy.iloc[-1] / spy.iloc[-1 - 252] - 1) if len(spy) > 252 else float(spy.iloc[-1] / spy.iloc[0] - 1)
    except Exception:
        spy_12m = np.nan
    print(f"SPY trailing 12m (RS baseline): {spy_12m:+.1%}\n")

    def etf_status(label: str) -> str:
        """Break-8/22 status of the sector's ETF(s) within the window (e.g. 'XLE✓' or 'XLI·XLB✓')."""
        marks = []
        for etf in SECTOR_ETF.get(label, []):
            df = etf_frames.get(etf)
            if df is None or len(df) < 60:
                marks.append(f"{etf}?")
                continue
            e = break3avg_signal(df, 8, 22, "SMA", "SMA")
            marks.append(f"{etf}✓" if e.iloc[-args.within:].any() else etf)
        return " ".join(marks)

    # per-sector tallies
    agg = {label: {"data": 0, "hits": 0, "strong": 0, "names": []} for label, _ in SECTORS}
    for sym, df in frames.items():
        sec = sector_of.get(sym)
        if sec is None:
            continue
        agg[sec]["data"] += 1
        e = break3avg_signal(df, 8, 22, "SMA", "SMA")
        if not e.iloc[-args.within:].any():
            continue
        agg[sec]["hits"] += 1
        c = df["Close"].astype(float)
        lb = 252
        s12 = (c.iloc[-1] / c.iloc[-1 - lb] - 1) if len(c) > lb else (c.iloc[-1] / c.iloc[0] - 1)
        tier = _rs_tier(s12 - spy_12m) if np.isfinite(spy_12m) else "n/a"
        if tier == "strong":
            agg[sec]["strong"] += 1
        agg[sec]["names"].append((sym, tier, s12 - spy_12m))

    rows = []
    for label, _ in SECTORS:
        a = agg[label]
        breadth = a["hits"] / a["data"] if a["data"] else 0.0
        strong_breadth = a["strong"] / a["data"] if a["data"] else 0.0
        rows.append((label, a["data"], a["hits"], breadth, a["strong"], strong_breadth))
    rows.sort(key=lambda r: r[3], reverse=True)

    print(f"=== break SMA 8/22 breakout BREADTH by sector (within {args.within} bars) ===")
    print(f"  {'sector':<18}{'names':>6}{'hits':>6}{'breadth%':>10}{'strong':>8}{'ETF break':>14}")
    for label, n, hits, br, st, stbr in rows:
        bar = "#" * int(br * 40)
        print(f"  {label:<18}{n:>6}{hits:>6}{br:>9.0%}{st:>8}{etf_status(label):>14}  {bar}")

    top = rows[0]
    print(f"\n  -> hottest sector: {top[0]}  ({top[3]:.0%} of names breaking out, {top[4]} strong-RS)")
    etf_hits = [lbl for lbl, *_ in rows if "✓" in etf_status(lbl)]
    print(f"  -> sector ETFs breaking out: {', '.join(etf_hits) if etf_hits else 'none'}")

    if args.list_strong:
        print("\n=== strong-RS breakout names per sector (hi-conv) ===")
        for label, _ in SECTORS:
            strong = sorted([x for x in agg[label]["names"] if x[1] == "strong"], key=lambda x: -x[2])
            if strong:
                print(f"  {label}: " + ", ".join(f"{s}(+{rs:.0%})" for s, _, rs in strong[:12]))
    print("\nBreadth = share of the sector's names with a fresh break 8/22. High breadth = the")
    print("sector is broadly breaking out (rotation IN); strong% = the hi-conv (strong-RS) share.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
