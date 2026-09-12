"""
_move_data.py — cached OHLC panel for the move-probability system (`move_prob.py`).

Build once, pickle, reuse — same pattern as _vix_data.py / _btc_data.py, so every study in the
_move_* family reads identical bars instead of each re-downloading and silently disagreeing.

Why OHLC and not just closes: the best volatility forecasters use the intraday range. A
close-to-close std throws away the high and the low, and Yang-Zhang / Garman-Klass are several
times more statistically efficient per bar. We need Open/High/Low/Close to compute them.

Universe: indices (SPY QQQ IWM DIA + ^GSPC ^VIX) plus a deterministic stratified sample of single
names drawn across all 10 sector buckets in ../../stock_symbols_1243.py.

SURVIVORSHIP WARNING (matters, read before trusting any number): stock_symbols_1243.py is a list
of names that are still listed TODAY. Every company that blew up and delisted is missing. For a
*calibration* study this is less fatal than for an alpha study — we are predicting the SIZE of a
move, not trying to earn a return — but it still biases the tails: survivors had fewer -40%
gap-downs than the true historical cross-section. So single-name tail probabilities from this
panel are, if anything, TOO LOW. The index series (SPY/QQQ/^GSPC) have no such bias and are the
honest benchmark.

Partial-bar guard: the last row is dropped if the US cash session has not closed. Unclosed bars
have silently corrupted several prior analyses in this repo.

Run to (re)build:  ../../vcp_env/bin/python _move_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

PANEL = Path(__file__).with_name("_move_panel.pkl")
FIELDS = ("Open", "High", "Low", "Close")
INDICES = ["SPY", "QQQ", "IWM", "DIA", "^GSPC", "^VIX"]
N_PER_SECTOR = 30          # deterministic stride sample per sector bucket
PERIOD = "25y"


def universe() -> list[str]:
    """Indices + a deterministic (stride, not random) sample from each sector bucket.
    Stride sampling keeps this reproducible without Math.random-style nondeterminism."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import stock_symbols_1243 as S

    buckets = ["TECH_STOCKS", "HEALTHCARE_STOCKS", "FINANCIAL_STOCKS", "CONSUMER_DISCRETIONARY",
               "CONSUMER_STAPLES", "ENERGY_STOCKS", "MATERIALS_INDUSTRIALS", "UTILITIES",
               "REAL_ESTATE_REITS", "COMMUNICATION_SERVICES"]
    picked = []
    for b in buckets:
        names = [s for s in getattr(S, b, []) if "." not in s]     # BRK.A etc. break yfinance
        if not names:
            continue
        stride = max(1, len(names) // N_PER_SECTOR)
        picked += names[::stride][:N_PER_SECTOR]
    # dict.fromkeys preserves order while de-duplicating (names appear in several buckets)
    return list(dict.fromkeys(INDICES + picked))


def build() -> dict:
    syms = universe()
    print(f"downloading {len(syms)} symbols, period={PERIOD} ...")
    out = {f: [] for f in FIELDS}
    CHUNK = 40
    for i in range(0, len(syms), CHUNK):
        part = syms[i:i + CHUNK]
        d = yf.download(part, period=PERIOD, progress=False, auto_adjust=True,
                        group_by="column", threads=True)
        if d is None or d.empty:
            print(f"  chunk {i // CHUNK}: empty, skipped")
            continue
        for f in FIELDS:
            if f not in d.columns.get_level_values(0):
                continue
            sub = d[f]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(part[0])
            out[f].append(sub)
        print(f"  chunk {i // CHUNK + 1}/{(len(syms) - 1) // CHUNK + 1}: {len(part)} symbols")

    panel = {}
    for f in FIELDS:
        if not out[f]:
            continue
        df = pd.concat(out[f], axis=1)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        panel[f] = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Non-positive price guard. yfinance back-adjustment can emit NEGATIVE adjusted closes
    # (CBIO had 2,189 of them here). np.log() quietly NaNs those so the vol estimators skip them,
    # but a simple return close[t+h]/close[t]-1 across a sign flip returns finite garbage -- we
    # measured -1.82, i.e. a "-182% move" -- which then lands in the down_big bucket and poisons
    # both the climatology and the left tail. Blank them at the source instead.
    bad = 0
    for f in FIELDS:
        if f in panel:
            mask = panel[f] <= 0
            bad += int(mask.sum().sum())
            panel[f] = panel[f].mask(mask)
    if bad:
        print(f"⚠️  blanked {bad:,} non-positive price cells (yfinance back-adjustment artifacts)")

    now_et = pd.Timestamp.now(tz="America/New_York")
    ref = panel["Close"]
    if ref.index[-1].date() == now_et.date() and now_et.time() < pd.Timestamp("16:00").time():
        print(f"⚠️  dropping in-progress bar {ref.index[-1].date()} (US session still open)")
        panel = {f: v.iloc[:-1] for f, v in panel.items()}

    pd.to_pickle(panel, PANEL)
    return panel


def load(rebuild: bool = False) -> dict:
    if rebuild or not PANEL.exists():
        return build()
    return pd.read_pickle(PANEL)


if __name__ == "__main__":
    p = build()
    c = p["Close"]
    print(f"\npanel {PANEL.name}: {c.shape[1]} symbols x {len(c)} rows  "
          f"{c.index[0].date()} → {c.index[-1].date()}")
    cov = c.notna().sum().sort_values()
    print(f"coverage: min {cov.iloc[0]} ({cov.index[0]}), median {int(cov.median())}, "
          f"max {cov.iloc[-1]} ({cov.index[-1]})")
    print(f"fields: {sorted(p)}")
    print(f"total non-null closes: {int(c.notna().sum().sum()):,}")
