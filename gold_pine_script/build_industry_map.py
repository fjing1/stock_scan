"""One-time (resumable) sweep: cache each universe ticker's sector + industry.

Runs ~1,200 yfinance .info calls, so it's slow — but RESUMABLE: it checkpoints
every 50 names and skips already-cached tickers on re-run. Output:
tradingview_scripts/industry_map.json with:
    by_ticker   : {SYM: {sector, industry}}
    by_industry : {industry: [SYM, ...]}   <- comprehensive peer lists

sector_confirm.py (and the scanner) read this for industry peer breadth across
EVERY industry in the universe, instead of curated lists / a weak fallback.

    python build_industry_map.py          # run / resume the sweep
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "industry_map.json")


def load():
    if os.path.exists(PATH):
        try:
            return json.load(open(PATH))
        except Exception:
            pass
    return {"by_ticker": {}, "by_industry": {}, "generated_utc": None, "n": 0}


def save(data):
    by_ind = {}
    for t, meta in data["by_ticker"].items():
        ind = meta.get("industry")
        if ind:
            by_ind.setdefault(ind, []).append(t)
    data["by_industry"] = {k: sorted(v) for k, v in sorted(by_ind.items())}
    data["n"] = len(data["by_ticker"])
    data["generated_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = PATH + ".tmp"
    json.dump(data, open(tmp, "w"), indent=1)
    os.replace(tmp, PATH)


def main():
    from stock_symbols_1243 import STOCK_SYMBOLS
    data = load()
    done = set(data["by_ticker"].keys())
    todo = [s for s in dict.fromkeys(STOCK_SYMBOLS) if s not in done]
    print(f"industry map: {len(done)} cached, {len(todo)} to fetch", flush=True)
    for i, s in enumerate(todo):
        try:
            info = yf.Ticker(s).info
            data["by_ticker"][s] = {"sector": info.get("sector"),
                                    "industry": info.get("industry")}
        except Exception:
            data["by_ticker"][s] = {"sector": None, "industry": None}
        if (i + 1) % 50 == 0:
            save(data)
            print(f"  ...{i+1}/{len(todo)} fetched ({len(data['by_ticker'])} total)", flush=True)
    save(data)
    inds = data["by_industry"]
    classified = sum(1 for m in data["by_ticker"].values() if m.get("industry"))
    print(f"DONE: {data['n']} tickers ({classified} classified), {len(inds)} industries.")
    print("top industries by count:")
    for ind, ts in sorted(inds.items(), key=lambda kv: -len(kv[1]))[:15]:
        print(f"  {len(ts):>3}  {ind}")


if __name__ == "__main__":
    main()
