#!/usr/bin/env python3
"""Persistent intraday bar store — beat the yfinance 60-day sub-hourly cap by
accumulating bars locally.

yfinance only serves ~60 days of 15m (and shorter) bars. Fetch-only-live scans are
therefore permanently stuck at a rolling 60-day window. This module keeps a local
CSV per symbol/interval and, on each run, fetches ONLY recent bars and appends the
newly-finalized ones — so history compounds past 60 days for as long as you keep
running it, and each run is cheap (pull ~5 days, not 60).

Design:
  * store at  <repo>/Stock_OneClick/bars/<interval>/<SYMBOL>.csv  (tz-aware ET index)
  * first run per symbol seeds the full --seed-period (default 60d); later runs pull
    only --recent (default 7d) and merge
  * overlapping timestamps: the FRESH fetch wins (intraday bars get revised as they
    finalize, so the last partial bar is corrected next run)
  * split continuity: if the fresh overlap is a uniform re-scale of stored bars
    (yfinance back-adjusts splits across the whole series), stored OHLC is rescaled
    onto the new basis before merge so old+new stay continuous
  * gap flag: if the store's newest bar is older than the fetchable window, there is
    an unfillable hole (system was off >~60d) — it's reported, not silently stitched

Reuses scan_stocks' symbol mapping + yf normalization. Run with the project venv:

    ../../vcp_env/bin/python intraday_store.py                 # 15m, universe from input
    ../../vcp_env/bin/python intraday_store.py --interval 5m
    ../../vcp_env/bin/python intraday_store.py --symbols NVDA,AMD --recent 5d
    ../../vcp_env/bin/python intraday_store.py --seed            # force full re-seed

Accessor for downstream signal code:  get_history("NVDA", "15m")  -> full local df.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402  — symbol mapping + yf normalization + input loader

try:
    import yfinance as yf
except Exception:  # pragma: no cover - yfinance always present in this env
    yf = None

BASE_DIR = BACKEND_DIR.parent
STORE_DIR = BASE_DIR / "bars"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
# yfinance sub-hourly history caps (days) — used to detect unfillable gaps
FETCH_CAP_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 60}


def _safe_name(symbol: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(symbol).strip().upper())


def store_path(symbol: str, interval: str) -> Path:
    return STORE_DIR / interval / f"{_safe_name(symbol)}.csv"


def load_bars(symbol: str, interval: str) -> pd.DataFrame:
    """Load the accumulated local bars (tz-aware ET index 'ts'); empty df if none."""
    p = store_path(symbol, interval)
    if not p.exists():
        return pd.DataFrame(columns=OHLCV)
    df = pd.read_csv(p)
    if "ts" not in df.columns or df.empty:
        return pd.DataFrame(columns=OHLCV)
    idx = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
    df = df.drop(columns=["ts"]).set_index(idx)
    df.index.name = "ts"
    for c in OHLCV:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[[c for c in OHLCV if c in df.columns]].sort_index()


# public accessor for downstream signal code
def get_history(symbol: str, interval: str = "15m") -> pd.DataFrame:
    return load_bars(symbol, interval)


def save_bars(symbol: str, interval: str, df: pd.DataFrame) -> None:
    p = store_path(symbol, interval)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # persist the index as a UTC ISO string so tz round-trips cleanly across machines
    ts = pd.to_datetime(out.index, utc=True)
    out.insert(0, "ts", ts.strftime("%Y-%m-%dT%H:%M:%SZ"))
    out.to_csv(p, index=False)


def _fetch_intraday(symbol: str, interval: str, period: str) -> pd.DataFrame | None:
    if yf is None:
        return None
    yf_symbol = scan.to_yfinance_symbol(symbol)
    df = yf.Ticker(yf_symbol).history(period=period, interval=interval, auto_adjust=False)
    df = scan.normalize_yf_df(df)
    if df is None or df.empty:
        return None
    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    df.index.name = "ts"
    cols = [c for c in OHLCV if c in df.columns]
    return df[cols].dropna(how="all").sort_index()


def _split_factor(old: pd.DataFrame, new: pd.DataFrame, min_overlap: int = 8,
                  rel_std_tol: float = 0.01, min_dev: float = 0.01) -> float | None:
    """If the fresh fetch is a UNIFORM rescale of the stored overlap (a split that
    yfinance back-adjusted across the series), return the factor r such that
    new ≈ old * r. Else None. Dividend gaps / real moves are non-uniform → None."""
    common = old.index.intersection(new.index)
    if len(common) < min_overlap:
        return None
    o = pd.to_numeric(old.loc[common, "Close"], errors="coerce")
    n = pd.to_numeric(new.loc[common, "Close"], errors="coerce")
    mask = (o > 0) & (n > 0)
    if int(mask.sum()) < min_overlap:
        return None
    ratio = (n[mask] / o[mask]).to_numpy()
    mean_r = float(np.mean(ratio))
    if mean_r <= 0:
        return None
    rel_std = float(np.std(ratio) / mean_r)
    if rel_std <= rel_std_tol and abs(mean_r - 1.0) > min_dev:
        return mean_r
    return None


def merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Pure merge (no network). Fresh bars win on overlap; stored bars are rescaled
    onto the new basis if a split is detected. Returns (merged, note)."""
    old = old if old is not None else pd.DataFrame(columns=OHLCV)
    new = new if new is not None else pd.DataFrame(columns=OHLCV)
    if new.empty:
        return old.sort_index(), "no-new"
    if old.empty:
        return new.sort_index(), "seeded"

    note = "appended"
    r = _split_factor(old, new)
    if r is not None:
        old = old.copy()
        for c in ("Open", "High", "Low", "Close"):
            if c in old.columns:
                old[c] = old[c] * r
        if "Volume" in old.columns:
            old["Volume"] = old["Volume"] / r  # split scales volume inversely
        note = f"split-adjusted x{r:.4f}"

    combined = pd.concat([old, new])
    # keep the LAST occurrence per timestamp -> fresh fetch overrides stored/partial bars
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined, note


def update_symbol(symbol: str, interval: str, recent: str, seed_period: str) -> dict:
    old = load_bars(symbol, interval)
    period = seed_period if old.empty else recent
    new = _fetch_intraday(symbol, interval, period)
    if new is None or new.empty:
        return {"symbol": symbol, "status": "no-data", "bars": len(old), "added": 0, "note": ""}

    gap_note = ""
    if not old.empty:
        cap = FETCH_CAP_DAYS.get(interval, 60)
        newest_old = old.index.max()
        oldest_new = new.index.min()
        # if there's daylight between the store's newest bar and the fetch window start,
        # and the fetch didn't reach back to touch the store, we have an unfillable hole
        if oldest_new > newest_old + pd.Timedelta(minutes=1) and old.index.intersection(new.index).empty:
            gap_note = f"GAP: store ends {newest_old:%Y-%m-%d %H:%M}, fetch starts {oldest_new:%Y-%m-%d %H:%M} (>{cap}d off?)"

    before = len(old)
    merged, note = merge_bars(old, new)
    save_bars(symbol, interval, merged)
    added = len(merged) - before
    full_note = "; ".join(x for x in [note, gap_note] if x)
    return {"symbol": symbol, "status": "ok", "bars": len(merged),
            "added": max(added, 0), "note": full_note,
            "range": f"{merged.index.min():%Y-%m-%d}..{merged.index.max():%Y-%m-%d %H:%M}"}


def update_all(symbols: list[str], interval: str, recent: str, seed_period: str,
               workers: int = 6) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    syms = [s for s in dict.fromkeys(str(x).strip().upper() for x in symbols) if s]
    results: list[dict] = []
    workers = max(1, min(workers, len(syms) or 1))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(update_symbol, s, interval, recent, seed_period): s for s in syms}
        for fut in as_completed(futs):
            done += 1
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"symbol": futs[fut], "status": f"error:{type(e).__name__}",
                                "bars": 0, "added": 0, "note": str(e)[:80]})
            if done % 25 == 0 or done == len(syms):
                print(f"  progress {done}/{len(syms)}", flush=True)
    order = {s: i for i, s in enumerate(syms)}
    results.sort(key=lambda r: order.get(r["symbol"], 1e9))
    return results


def _universe() -> list[str]:
    try:
        df_in, _ = scan.load_input_and_meta(scan.INPUT_FILE)
        return df_in["symbol"].astype(str).str.strip().str.upper().tolist()
    except Exception:
        return list(getattr(scan, "A_POOL_SYMBOLS", []))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", default="15m", help="15m / 5m / 30m / 60m ...")
    ap.add_argument("--recent", default="7d", help="incremental fetch window on subsequent runs")
    ap.add_argument("--seed-period", default="60d", help="first-run backfill window (yfinance cap)")
    ap.add_argument("--symbols", default="", help="comma list; overrides the input-file universe")
    ap.add_argument("--seed", action="store_true", help="force a full re-seed (fetch seed-period for all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N symbols (testing)")
    args = ap.parse_args()

    syms = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
            if args.symbols else _universe())
    if args.limit:
        syms = syms[:args.limit]
    if not syms:
        print("No symbols. Provide --symbols or populate stock_input_template.xlsx.")
        return 0

    recent = args.seed_period if args.seed else args.recent
    print(f"Intraday store: {len(syms)} symbols  interval={args.interval}  "
          f"{'RE-SEED ' + args.seed_period if args.seed else 'recent=' + recent}  -> {STORE_DIR/args.interval}")
    results = update_all(syms, args.interval, recent, args.seed_period, workers=args.workers)

    ok = [r for r in results if r["status"] == "ok"]
    added = sum(r["added"] for r in ok)
    nodata = [r["symbol"] for r in results if r["status"] == "no-data"]
    errs = [r for r in results if r["status"].startswith("error")]
    gaps = [r for r in ok if "GAP" in r.get("note", "")]
    splits = [r for r in ok if "split" in r.get("note", "")]
    print(f"\n✅ updated {len(ok)}/{len(results)} symbols, +{added} new bars total")
    if splits:
        print(f"↺ split-adjusted continuity applied: {', '.join(r['symbol'] for r in splits)}")
    if gaps:
        print(f"⚠ history gaps (system off >~cap): {', '.join(r['symbol'] for r in gaps)}")
    if nodata:
        print(f"∅ no data: {', '.join(nodata[:20])}{' ...' if len(nodata) > 20 else ''}")
    if errs:
        print(f"✗ errors: {', '.join(r['symbol'] for r in errs[:20])}")
    if ok:
        sample = ok[0]
        print(f"e.g. {sample['symbol']}: {sample['bars']} bars  {sample.get('range','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
