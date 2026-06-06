"""
拉取美股市值 > 指定阈值（默认 100 亿美元）的个股列表，排除 ETF。

数据源：yfinance（Yahoo Finance）。流程：
1) 下载全市场代码表（纳斯达克/纽交所/美交所）；
2) 逐个用 yfinance 拉取公司基本信息；
3) 按市值和 quoteType 过滤，输出 Excel（默认 .xlsx），也可指定 .csv。

用法：
    python3 stock_list_10B.py --out stock_list_10B.xlsx --min-mcap 10000000000 --workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd
import requests
import yfinance as yf


NASDAQ_URL = "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
OTHER_URL = "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"


@dataclass
class TickerInfo:
    symbol: str
    name: str
    exchange: str
    market_cap: float


def fetch_symbol_lists() -> Set[str]:
    """拉取 NASDAQ / NYSE / AMEX 全部代码集合（不含ETF标记处理）。"""
    symbols: Set[str] = set()
    for url in (NASDAQ_URL, OTHER_URL):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        # 跳过头尾说明行
        for line in lines:
            if "|" not in line or line.startswith("File Creation Time"):
                continue
            parts = line.split("|")
            sym = parts[0].strip().upper()
            if not sym or sym == "SYMBOL":
                continue
            # 对 NASDAQ/OTHER，两张表最后一行会是 "Symbol|..." 或者空；已过滤
            symbols.add(sym)
    return symbols


def get_market_cap(info: dict) -> Optional[float]:
    for k in ("marketCap", "market_cap"):
        if k in info and info[k] is not None:
            return float(info[k])
    # fast_info 兜底
    return None


def fetch_one(sym: str, min_mcap: float) -> Optional[TickerInfo]:
    """从 yfinance 拉取单票信息，按市值和 ETF 过滤。"""
    try:
        tk = yf.Ticker(sym)
        info = tk.get_info()
    except Exception:
        return None

    quote_type = (info.get("quoteType") or info.get("quote_type") or "").lower()
    if quote_type == "etf":
        return None

    mcap = get_market_cap(info)
    if mcap is None or mcap < min_mcap:
        return None

    name = info.get("shortName") or info.get("longName") or ""
    exch = info.get("exchange") or info.get("fullExchangeName") or ""
    return TickerInfo(symbol=sym, name=name, exchange=exch, market_cap=mcap)


def collect(min_mcap: float, workers: int) -> List[TickerInfo]:
    symbols = sorted(fetch_symbol_lists())
    results: List[TickerInfo] = []

    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_sym = {ex.submit(fetch_one, sym, min_mcap): sym for sym in symbols}
        for fut in futures.as_completed(fut_to_sym):
            ti = fut.result()
            if ti is not None:
                results.append(ti)
    return results


def save_file(rows: Iterable[TickerInfo], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"symbol": r.symbol, "name": r.name, "exchange": r.exchange, "market_cap": int(r.market_cap)}
        for r in sorted(rows, key=lambda x: x.market_cap, reverse=True)
    ]
    df = pd.DataFrame(data)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        # 默认写 Excel
        df.to_excel(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="拉取市值超阈值的美股（非 ETF）列表")
    parser.add_argument("--min-mcap", type=float, default=10_000_000_000, help="市值下限（美元）")
    parser.add_argument("--out", type=Path, default=Path("stock_list_10B.xlsx"), help="输出文件路径（.xlsx 或 .csv）")
    parser.add_argument("--workers", type=int, default=16, help="并发线程数")
    args = parser.parse_args()

    t0 = time.time()
    rows = collect(args.min_mcap, args.workers)
    save_file(rows, args.out)
    print(f"✅ 完成：{len(rows)} 只，耗时 {time.time()-t0:.1f}s，已保存 {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
