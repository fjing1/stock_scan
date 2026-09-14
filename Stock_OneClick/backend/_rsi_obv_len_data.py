#!/usr/bin/env python3
"""Builds the cached daily OHLCV panel for the OBV-RSI length study.

Deliberately BROAD (a few hundred names sampled across the full ~1243 universe, plus the
index and the survivor basket) because [[exit-strategy-edge-is-survivorship]]: edges that
look real on the 40-name survivor basket evaporate on a wide universe. Keeping all three
groups lets the study report them side by side.

    ../../vcp_env/bin/python _rsi_obv_len_data.py            # build/refresh the cache
    ../../vcp_env/bin/python _rsi_obv_len_data.py --n 600    # wider sample
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402
from rsi_ma_sweep import BASKET, INDEX  # noqa: E402

PANEL = BACKEND_DIR / "_rsi_obv_len_panel.pkl"
MIN_BARS = 1000          # ~4y minimum so per-year walk-forward has something to chew on
COLS = ["Open", "High", "Low", "Close", "Volume"]


def sample_universe(n: int) -> tuple[list[str], dict[str, str]]:
    """Return (symbols, group_map). Groups: index / basket / broad."""
    root = BACKEND_DIR.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import stock_symbols_1243 as u  # noqa: E402

    broad = [s for s in u.STOCK_SYMBOLS if s not in set(BASKET) | set(INDEX)]
    step = max(1, len(broad) // n)
    broad = broad[::step][:n]
    group = {}
    for s in INDEX:
        group[s] = "index"
    for s in BASKET:
        group.setdefault(s, "basket")
    for s in broad:
        group.setdefault(s, "broad")
    return list(group), group


def build(symbols: list[str], period: str, chunk: int = 80) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    pairs = [(s, scan.to_yfinance_symbol(s)) for s in symbols]
    for i in range(0, len(pairs), chunk):
        batch = pairs[i:i + chunk]
        try:
            data = yf.download([y for _, y in batch], period=period, group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as exc:
            print(f"  batch {i} failed: {exc}")
            continue
        for orig, y in batch:
            try:
                d = data[y][COLS].dropna()
            except Exception:
                continue
            if len(d) >= MIN_BARS and (d["Volume"] > 0).mean() > 0.9:
                out[orig] = d.astype(float)
        print(f"  ...{min(i + chunk, len(pairs))}/{len(pairs)} fetched, {len(out)} usable")
    return out


def load() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    with open(PANEL, "rb") as fh:
        blob = pickle.load(fh)
    return blob["data"], blob["group"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=400, help="how many broad-universe names to sample")
    ap.add_argument("--period", default="max")
    args = ap.parse_args()

    symbols, group = sample_universe(args.n)
    print(f"building panel for {len(symbols)} names (period={args.period})...")
    data = build(symbols, args.period)
    if not data:
        print("no data")
        return 1
    group = {s: g for s, g in group.items() if s in data}
    idx = pd.DatetimeIndex(sorted(set().union(*[d.index for d in data.values()])))
    with open(PANEL, "wb") as fh:
        pickle.dump({"data": data, "group": group}, fh)
    counts = pd.Series(list(group.values())).value_counts().to_dict()
    print(f"saved {len(data)} names {counts} covering {idx.min().date()}..{idx.max().date()} -> {PANEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
