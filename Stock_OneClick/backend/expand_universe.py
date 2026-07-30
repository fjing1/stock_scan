#!/usr/bin/env python3
"""Expand the scan universe with curated THEMATIC names, added as enable=0
(research tier) so the live scan (enable=1) is unchanged until you flip them on.

Why enable=0 direct-to-Sheet2: scan_stocks.enrich_meta_with_yfinance() adds any
brand-new Sheet1 symbol with enable=1 (goes live), but never touches the enable of
a row already in Sheet2. So we write new names into BOTH sheets with enable=0 and
the enricher will respect that.

Safety: every new ticker is validated against yfinance (must return recent daily
bars) before it's written — kills typos/hallucinated tickers. The existing 162
rows are left byte-for-byte intact (enable, group, note preserved). The workbook is
backed up (timestamped) to history/ before writing.

    ../../vcp_env/bin/python expand_universe.py            # validate + preview only (themes)
    ../../vcp_env/bin/python expand_universe.py --write     # actually write the workbook (themes)
    ../../vcp_env/bin/python expand_universe.py --write --enable 0   # (default) research tier

Bulk / rules-based path (e.g. top-N by market cap from stock_list_10B.py output):
    ../../vcp_env/bin/python expand_universe.py --from-csv ../reports/us_mcap_top.csv --top-n 1000
    ../../vcp_env/bin/python expand_universe.py --from-csv ../reports/us_mcap_top.csv --top-n 1000 --write
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import scan_stocks as scan  # noqa: E402

BASE_DIR = BACKEND_DIR.parent
INPUT_FILE = BASE_DIR / "stock_input_template.xlsx"
HISTORY_DIR = BASE_DIR / "history"
META_COLS = ["symbol", "name", "exchange", "sector", "industry", "market_cap", "group", "note", "enable"]

# Curated thematic extensions — liquid, well-known US large/mid-caps grouped to
# extend the existing scheme (existing groups run 00..21). Dedup vs current list
# happens at merge, so overlaps here are harmless.
THEMES: dict[str, list[str]] = {
    "22 生物科技 / 制药": ["LLY", "MRK", "PFE", "ABBV", "BMY", "AMGN", "GILD", "VRTX", "REGN",
                       "MRNA", "BIIB", "ZTS", "ALNY", "NBIX", "ARGX"],
    "23 金融科技 / 支付": ["V", "MA", "PYPL", "FI", "FIS", "GPN", "COIN", "SOFI", "AFRM",
                       "HOOD", "NU", "TOST", "BILL"],
    "24 国防 / 航空航天": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "AXON", "TDG",
                       "HWM", "LDOS", "KTOS"],
    "25 银行 / 金融": ["JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "USB", "PNC", "TFC",
                    "BLK", "AXP", "BX", "KKR", "APO"],
    "26 消费 / 零售": ["WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD", "TJX",
                    "LULU", "CMG", "PG", "KO", "PEP", "MELI"],
    "27 医疗设备 / 服务": ["ISRG", "MDT", "SYK", "BSX", "ABT", "TMO", "DHR", "CVS", "HCA",
                       "ELV", "MCK", "IDXX"],
    "28 工业 / 机械": ["CAT", "DE", "HON", "GE", "MMM", "EMR", "PH", "ITW", "UNP", "CSX",
                    "GEV", "PWR", "ETN"],
    "29 软件 / 云": ["MSFT", "ORCL", "CRM", "ADBE", "NOW", "SNOW", "NET", "CRWD", "ZS",
                  "MDB", "TEAM", "WDAY", "INTU", "SNPS", "DASH", "ABNB"],
    "30 EV / 汽车": ["TSLA", "RIVN", "LCID", "GM", "F"],
    "31 半导体扩展": ["AMD", "INTC", "MU", "QCOM", "TXN", "LRCX", "KLAC", "ON", "ADI",
                  "NXPI", "MCHP", "TSM", "ASML", "ARM", "AMAT", "TER"],
    "32 中概股": ["BABA", "PDD", "JD", "NIO", "LI", "XPEV", "BIDU"],
    "33 能源扩展": ["XOM", "PSX", "MPC", "VLO", "WMB", "KMI", "LNG", "DVN", "FANG", "TRGP"],
    "34 材料 / 金属": ["FCX", "NEM", "STLD", "DOW", "LIN", "APD", "SHW", "ALB", "CRS"],
    "35 公用事业扩展": ["D", "AEP", "EXC", "XEL", "PEG", "ED", "VST", "CEG"],
}


def _validate(sym: str) -> dict | None:
    """Return {'ok': True, ...basic info} if yfinance serves recent daily bars."""
    try:
        d = scan.download_daily(sym, period="5d")
    except Exception:
        d = None
    if d is None or d.empty:
        return None
    info = {}
    try:
        info = scan._fetch_yf_info(sym)[1] or {}
    except Exception:
        info = {}
    return {"symbol": sym, "name": info.get("name", ""), "exchange": info.get("exchange", ""),
            "sector": info.get("sector", ""), "industry": info.get("industry", ""),
            "market_cap": info.get("market_cap", float("nan"))}


def _write_workbook(df_in, df_meta, valid, group_of, enable):
    """Append validated new names (enable flag) to both sheets, after a backup.
    valid: {sym: {name,exchange,sector,industry,market_cap}}; group_of: {sym: group}."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = HISTORY_DIR / f"stock_input_template_backup_{ts}.xlsx"
    shutil.copy2(INPUT_FILE, backup)

    new_meta_rows = [{
        "symbol": s, "name": v.get("name", ""), "exchange": v.get("exchange", ""),
        "sector": v.get("sector", ""), "industry": v.get("industry", ""),
        "market_cap": v.get("market_cap", float("nan")), "group": group_of.get(s, v.get("sector", "")),
        "note": "", "enable": enable,
    } for s, v in valid.items()]
    for c in META_COLS:
        if c not in df_meta.columns:
            df_meta[c] = pd.NA
    df_meta_out = pd.concat([df_meta[META_COLS], pd.DataFrame(new_meta_rows)], ignore_index=True)
    df_in_out = pd.concat([df_in, pd.DataFrame({"symbol": list(valid.keys())})], ignore_index=True)
    df_in_out["symbol"] = df_in_out["symbol"].astype(str).str.strip().str.upper()
    df_in_out = df_in_out.drop_duplicates("symbol").reset_index(drop=True)

    with pd.ExcelWriter(INPUT_FILE, engine="openpyxl") as w:
        df_in_out.to_excel(w, sheet_name="Sheet1_Input", index=False)
        df_meta_out.to_excel(w, sheet_name="Sheet2_Classified", index=False)
    print(f"\n💾 wrote {INPUT_FILE}  (backup: {backup.name})")
    print(f"   Sheet1_Input: {len(df_in_out)}   Sheet2_Classified: {len(df_meta_out)}")
    print(f"   new names are enable={enable}. Flip to 1 in Sheet2 to send any live.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="write the workbook (default: preview only)")
    ap.add_argument("--enable", type=int, default=0, help="enable flag for new names (default 0 = research)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--from-csv", default="", help="bulk source CSV (symbol,name,market_cap,...) e.g. from stock_list_10B.py")
    ap.add_argument("--top-n", type=int, default=1000, help="with --from-csv: keep the top-N by market cap")
    args = ap.parse_args()

    if not INPUT_FILE.exists():
        print(f"Missing {INPUT_FILE}"); return 1
    xls = pd.ExcelFile(INPUT_FILE)
    df_in = pd.read_excel(xls, "Sheet1_Input")
    df_meta = pd.read_excel(xls, "Sheet2_Classified") if "Sheet2_Classified" in xls.sheet_names else pd.DataFrame(columns=META_COLS)
    have = set(df_in["symbol"].astype(str).str.strip().str.upper()) | \
           set(df_meta.get("symbol", pd.Series(dtype=str)).astype(str).str.strip().str.upper())
    print(f"current universe: {len(have)} symbols")

    valid: dict[str, dict] = {}
    group_of: dict[str, str] = {}
    dropped: list[str] = []

    if args.from_csv:
        # ---- bulk rules-based path: top-N by market cap from a scan CSV ----
        src = Path(args.from_csv)
        if not src.is_absolute():
            src = (BACKEND_DIR / src).resolve()
        if not src.exists():
            print(f"CSV not found: {src}"); return 1
        cdf = pd.read_csv(src)
        cdf["symbol"] = cdf["symbol"].astype(str).str.strip().str.upper()
        cdf["market_cap"] = pd.to_numeric(cdf["market_cap"], errors="coerce")
        cdf = cdf.dropna(subset=["symbol", "market_cap"]).sort_values("market_cap", ascending=False)
        top = cdf.head(args.top_n)
        print(f"CSV {src.name}: {len(cdf)} rows -> top {args.top_n} by mcap")
        for _, r in top.iterrows():
            s = r["symbol"]
            if not s or s in have or s in valid:
                continue
            sector = str(r.get("sector", "") or "")
            valid[s] = {"name": str(r.get("name", "") or ""), "exchange": str(r.get("exchange", "") or ""),
                        "sector": sector, "industry": str(r.get("industry", "") or ""),
                        "market_cap": float(r["market_cap"])}
            group_of[s] = f"90 研究池-{sector}" if sector else "90 研究池"
    else:
        # ---- curated thematic path (yfinance-validated) ----
        cand: dict[str, str] = {}
        for group, syms in THEMES.items():
            for s in syms:
                s = s.strip().upper()
                if s and s not in have and s not in cand:
                    cand[s] = group
        print(f"themed candidates (new): {len(cand)}  -> validating against yfinance ...")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_validate, s): s for s in cand}
            done = 0
            for fut in as_completed(futs):
                done += 1
                s = futs[fut]
                info = fut.result()
                if info:
                    valid[s] = info
                    group_of[s] = cand[s]
                else:
                    dropped.append(s)
                if done % 25 == 0 or done == len(cand):
                    print(f"  validated {done}/{len(cand)}", flush=True)

    print(f"\n✅ valid new names: {len(valid)}   ✗ dropped: {len(dropped)}")
    if dropped:
        print("   dropped:", ", ".join(sorted(dropped)))
    total = len(have) + len(valid)
    print(f"resulting universe: {len(have)} existing + {len(valid)} new (enable={args.enable}) = {total}")
    by_group: dict[str, int] = {}
    for s in valid:
        by_group[group_of.get(s, "?")] = by_group.get(group_of.get(s, "?"), 0) + 1
    for g in sorted(by_group):
        print(f"   {g}: {by_group[g]}")

    if not args.write:
        print("\n(preview only — re-run with --write to apply)")
        return 0
    if not valid:
        print("nothing to write."); return 0

    _write_workbook(df_in, df_meta, valid, group_of, args.enable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
