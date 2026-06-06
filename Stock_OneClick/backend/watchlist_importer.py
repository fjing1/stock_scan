from __future__ import annotations

import json
import re
import shutil
import ssl
from datetime import datetime
from html import unescape
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


META_COLUMNS = [
    "symbol",
    "name",
    "exchange",
    "sector",
    "industry",
    "market_cap",
    "group",
    "note",
    "enable",
]

EXCHANGE_ALIASES = {
    "NASDAQ": "NASDAQ",
    "NYSE": "NYSE",
    "AMEX": "AMEX",
    "ARCA": "AMEX",
    "SP": "SP",
    "TVC": "TVC",
    "CBOE": "CBOE",
    "NYMEX": "NYMEX",
    "COMEX": "COMEX",
    "PEPPERSTONE": "PEPPERSTONE",
}


def _normalize_exchange(exchange: str) -> str:
    exchange = str(exchange or "").strip().upper()
    return EXCHANGE_ALIASES.get(exchange, exchange)


def _split_tv_symbol(token: str) -> tuple[str, str]:
    token = str(token or "").strip().strip('"').strip("'")
    exchange = ""
    symbol = token.upper()
    if ":" in symbol:
        exchange, symbol = symbol.split(":", 1)
        exchange = _normalize_exchange(exchange)
    return symbol.strip().upper(), exchange


def _parse_group_marker(line: str) -> str | None:
    marker = str(line or "").strip()
    if not marker.startswith("###"):
        return None
    group = marker.lstrip("#").strip()
    return group or None


def parse_tradingview_symbols(symbols: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    current_group = "00 市场环境"

    for item in symbols:
        token = str(item or "").strip()
        if not token:
            continue
        group = _parse_group_marker(token)
        if group:
            current_group = group
            continue

        symbol, exchange = _split_tv_symbol(token)
        if not symbol or symbol in seen:
            continue
        if not re.search(r"[A-Z]", symbol):
            continue
        seen.add(symbol)
        rows.append({"symbol": symbol, "exchange": exchange, "group": current_group})
    return rows


def parse_tradingview_watchlist(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    current_group = "98 TradingView导入"
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            group = _parse_group_marker(line)
            if group:
                current_group = group
            continue
        if line.lower() in {"symbol", "symbols", "ticker", "tickers"}:
            continue
        if line.startswith("[") and line.endswith("]"):
            continue
        group = _parse_group_marker(line)
        if group:
            current_group = group
            continue

        parts = re.split(r"[\s,;]+", line)
        for token in parts:
            token = token.strip().strip('"').strip("'")
            if not token or token.startswith("#"):
                continue
            if token.lower() in {"symbol", "description", "exchange"}:
                continue
            if not re.search(r"[A-Za-z]", token):
                continue

            symbol, exchange = _split_tv_symbol(token)
            if not symbol or symbol.startswith("###"):
                continue
            if symbol in seen:
                continue
            seen.add(symbol)
            rows.append({"symbol": symbol, "exchange": exchange, "group": current_group})
    return rows


def _write_grouped_watchlist_snapshot(parsed: list[dict[str, str]], out_path: Path) -> None:
    lines: list[str] = []
    current_group = None
    for row in parsed:
        group = row.get("group", "") or "98 TradingView导入"
        if group != current_group:
            if lines:
                lines.append("")
            lines.append(f"[{group}]")
            current_group = group
        prefix = row.get("exchange", "")
        symbol = row["symbol"]
        lines.append(f"{prefix}:{symbol}" if prefix else symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_excel_tables(excel_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到 Excel 扫描池：{excel_path}")

    xls = pd.ExcelFile(excel_path)
    try:
        df_input = pd.read_excel(xls, sheet_name="Sheet1_Input")
    except ValueError:
        df_input = pd.DataFrame(columns=["symbol"])
    try:
        df_meta = pd.read_excel(xls, sheet_name="Sheet2_Classified")
    except ValueError:
        df_meta = pd.DataFrame(columns=META_COLUMNS)

    if "symbol" not in df_input.columns:
        df_input = pd.DataFrame(columns=["symbol"])
    df_input["symbol"] = df_input["symbol"].astype(str).str.strip().str.upper()
    df_input = df_input[df_input["symbol"] != ""]

    for col in META_COLUMNS:
        if col not in df_meta.columns:
            df_meta[col] = pd.NA
    df_meta["symbol"] = df_meta["symbol"].astype(str).str.strip().str.upper()
    return df_input, df_meta


def _backup_excel(excel_path: Path, history_dir: Path, label: str) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = history_dir / f"stock_input_template_before_{label}_{stamp}.xlsx"
    shutil.copy2(excel_path, backup_path)
    return backup_path


def _write_excel_tables(excel_path: Path, df_input: pd.DataFrame, df_meta: pd.DataFrame) -> None:
    df_input = df_input.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    df_meta = df_meta[META_COLUMNS].drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    df_meta["enable"] = pd.to_numeric(df_meta["enable"], errors="coerce").fillna(1).astype(int)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_input[["symbol"]].to_excel(writer, sheet_name="Sheet1_Input", index=False)
        df_meta[META_COLUMNS].to_excel(writer, sheet_name="Sheet2_Classified", index=False)


def import_watchlist_file(path: Path, excel_path: Path, history_dir: Path) -> dict[str, object]:
    path = Path(path)
    excel_path = Path(excel_path)
    history_dir = Path(history_dir)
    if not path.exists():
        raise FileNotFoundError(f"找不到 watchlist 文件：{path}")

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    parsed = parse_tradingview_watchlist(text)
    if not parsed:
        raise ValueError("没有从文件里解析到 TradingView symbol。请确认导出的是 watchlist .txt。")

    df_input, df_meta = _read_excel_tables(excel_path)
    existing = set(df_input["symbol"].dropna().astype(str).str.upper())
    existing_meta = set(df_meta["symbol"].dropna().astype(str).str.upper())
    added_symbols = []
    new_meta_rows = []

    for item in parsed:
        symbol = item["symbol"]
        exchange = item.get("exchange", "")
        if symbol not in existing:
            new_input_row = pd.DataFrame([{"symbol": symbol}])
            if df_input.empty:
                df_input = new_input_row
            else:
                df_input = pd.concat([df_input, new_input_row], ignore_index=True)
            existing.add(symbol)
            added_symbols.append(symbol)
        if symbol not in existing_meta:
            new_meta_rows.append(
                {
                    "symbol": symbol,
                    "name": "",
                    "exchange": exchange,
                    "sector": "",
                    "industry": "",
                    "market_cap": pd.NA,
                    "group": item.get("group") or "98 TradingView导入",
                    "note": f"Imported from {path.name}",
                    "enable": 1,
                }
            )
            existing_meta.add(symbol)
        elif exchange and "exchange" in df_meta.columns:
            mask = df_meta["symbol"] == symbol
            empty_exchange = df_meta.loc[mask, "exchange"].isna() | df_meta.loc[mask, "exchange"].astype(str).str.strip().eq("")
            df_meta.loc[mask & empty_exchange, "exchange"] = exchange

    if new_meta_rows:
        if df_meta.empty:
            df_meta = pd.DataFrame(new_meta_rows)
        else:
            df_meta = pd.DataFrame(df_meta.to_dict("records") + new_meta_rows)

    backup_path = _backup_excel(excel_path, history_dir, "tv_import")
    _write_excel_tables(excel_path, df_input, df_meta)

    return {
        "parsed_count": len(parsed),
        "added_count": len(added_symbols),
        "added_symbols": added_symbols,
        "backup_path": backup_path,
        "excel_path": excel_path,
    }


def fetch_tradingview_watchlist(url: str) -> dict[str, object]:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            html = response.read().decode("utf-8", errors="replace")
    except URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        with urlopen(request, timeout=30, context=ssl._create_unverified_context()) as response:
            html = response.read().decode("utf-8", errors="replace")

    scripts = re.findall(
        r'<script[^>]+type="application/prs\.init-data\+json"[^>]*>(.*?)</script>',
        html,
        flags=re.S,
    )
    for raw in scripts:
        data = json.loads(unescape(raw))
        watchlist = data.get("sharedWatchlist", {}).get("list")
        if watchlist and isinstance(watchlist.get("symbols"), list):
            return watchlist
    raise ValueError("没有在 TradingView 页面里找到 sharedWatchlist.list.symbols。")


def sync_watchlist_url(url: str, excel_path: Path, history_dir: Path, export_dir: Path | None = None) -> dict[str, object]:
    excel_path = Path(excel_path)
    history_dir = Path(history_dir)
    watchlist = fetch_tradingview_watchlist(url)
    parsed = parse_tradingview_symbols(watchlist["symbols"])
    if not parsed:
        raise ValueError("TradingView 链接里没有解析到可用 symbol。")

    df_input_old, df_meta_old = _read_excel_tables(excel_path)
    old_meta = {
        str(row["symbol"]).strip().upper(): row
        for row in df_meta_old[META_COLUMNS].to_dict("records")
        if str(row.get("symbol", "")).strip()
    }

    new_input = pd.DataFrame({"symbol": [row["symbol"] for row in parsed]})
    new_meta_rows = []
    added_symbols = []
    for row in parsed:
        symbol = row["symbol"]
        old = dict(old_meta.get(symbol, {}))
        if not old:
            added_symbols.append(symbol)
        merged = {
            "symbol": symbol,
            "name": old.get("name", ""),
            "exchange": row.get("exchange") or old.get("exchange", ""),
            "sector": old.get("sector", ""),
            "industry": old.get("industry", ""),
            "market_cap": old.get("market_cap", pd.NA),
            "group": row.get("group") or old.get("group", "98 TradingView导入"),
            "note": old.get("note", ""),
            "enable": old.get("enable", 1),
        }
        new_meta_rows.append(merged)

    new_meta = pd.DataFrame(new_meta_rows, columns=META_COLUMNS)
    backup_path = _backup_excel(excel_path, history_dir, "tv_url_sync")
    _write_excel_tables(excel_path, new_input, new_meta)

    snapshot_path = None
    if export_dir is not None:
        export_dir = Path(export_dir)
        snapshot_path = export_dir / "tv_watchlist_from_url_latest.txt"
        _write_grouped_watchlist_snapshot(parsed, snapshot_path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _write_grouped_watchlist_snapshot(parsed, history_dir / f"tv_watchlist_from_url_{stamp}.txt")

    return {
        "watchlist_id": watchlist.get("id"),
        "watchlist_name": watchlist.get("name"),
        "modified": watchlist.get("modified"),
        "parsed_count": len(parsed),
        "added_count": len(added_symbols),
        "added_symbols": added_symbols,
        "backup_path": backup_path,
        "excel_path": excel_path,
        "snapshot_path": snapshot_path,
    }
