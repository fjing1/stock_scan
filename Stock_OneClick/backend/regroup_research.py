#!/usr/bin/env python3
"""Regroup the bulk research names (enable=0, currently in coarse "90 研究池-<sector>"
buckets) into meaningful thematic groups, reusing the existing scheme (22-35) and
adding new numbered groups where needed.

Deterministic industry->group mapping (first match wins on the yfinance/NASDAQ
`industry` label; sector fallback; explicit catch-all that is REPORTED, never
silent). Live names (enable=1) and the already-curated thematic names (groups 00-35)
are left untouched. Backs up the workbook before writing.

    ../../vcp_env/bin/python regroup_research.py            # preview only
    ../../vcp_env/bin/python regroup_research.py --write     # apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
INPUT_FILE = BASE_DIR / "stock_input_template.xlsx"
HISTORY_DIR = BASE_DIR / "history"
META_COLS = ["symbol", "name", "exchange", "sector", "industry", "market_cap", "group", "note", "enable"]

# New groups introduced by this pass (extend the 00-35 scheme):
#   36 房地产/REIT  37 保险  38 电信/媒体  39 商业服务  40 消费必需品  41 消费服务/休闲  42 运输/物流

# Ordered (first substring match wins). Keys matched lowercase against `industry`.
INDUSTRY_MAP = [
    # --- finance ---
    ("reit", "36 房地产 / REIT"),
    ("real estate investment", "36 房地产 / REIT"),
    ("real estate", "36 房地产 / REIT"),
    ("insurance", "37 保险"), ("insurer", "37 保险"),
    ("bank", "25 银行 / 金融"),
    ("investment banker", "25 银行 / 金融"), ("broker", "25 银行 / 金融"),
    ("investment manager", "25 银行 / 金融"), ("investment trust", "25 银行 / 金融"),
    ("finance: consumer", "23 金融科技 / 支付"), ("finance companies", "25 银行 / 金融"),
    ("savings inst", "25 银行 / 金融"),
    # --- tech / software ---
    ("semiconductor", "31 半导体扩展"), ("electronic components", "31 半导体扩展"),
    ("software", "29 软件 / 云"), ("edp services", "29 软件 / 云"),
    ("programming data", "29 软件 / 云"), ("computer", "29 软件 / 云"),
    ("internet", "29 软件 / 云"),
    # --- health ---
    ("biotechnology", "22 生物科技 / 制药"), ("pharmaceutical", "22 生物科技 / 制药"),
    ("pharma", "22 生物科技 / 制药"),
    ("medical", "27 医疗设备 / 服务"), ("dental", "27 医疗设备 / 服务"),
    ("health", "27 医疗设备 / 服务"), ("hospital", "27 医疗设备 / 服务"),
    # --- defense / aerospace ---
    ("aerospace", "24 国防 / 航空航天"), ("military", "24 国防 / 航空航天"),
    ("defense", "24 国防 / 航空航天"),
    # --- transport ---
    ("air freight", "42 运输 / 物流"), ("trucking", "42 运输 / 物流"),
    ("marine transport", "42 运输 / 物流"), ("railroad", "42 运输 / 物流"),
    ("airline", "42 运输 / 物流"), ("transportation", "42 运输 / 物流"),
    ("freight", "42 运输 / 物流"),
    # --- telecom / media ---
    ("telecommunications", "38 电信 / 媒体"), ("broadcasting", "38 电信 / 媒体"),
    ("cable", "38 电信 / 媒体"), ("television", "38 电信 / 媒体"),
    ("radio", "38 电信 / 媒体"), ("media", "38 电信 / 媒体"),
    ("wireless", "38 电信 / 媒体"), ("telephone", "38 电信 / 媒体"),
    # --- energy / utilities / materials ---
    ("natural gas", "33 能源扩展"), ("oil", "33 能源扩展"), ("coal", "33 能源扩展"),
    ("electric utilities", "35 公用事业扩展"), ("power generation", "35 公用事业扩展"),
    ("utilities", "35 公用事业扩展"), ("water supply", "35 公用事业扩展"),
    ("precious metals", "34 材料 / 金属"), ("steel", "34 材料 / 金属"),
    ("iron ore", "34 材料 / 金属"), ("mining", "34 材料 / 金属"),
    ("aluminum", "34 材料 / 金属"), ("metal", "34 材料 / 金属"),
    ("chemical", "34 材料 / 金属"),
    # --- industrials ---
    ("machinery", "28 工业 / 机械"), ("metal fabrication", "28 工业 / 机械"),
    ("electrical product", "28 工业 / 机械"), ("industrial", "28 工业 / 机械"),
    ("containers/packaging", "28 工业 / 机械"), ("environmental", "28 工业 / 机械"),
    ("homebuilding", "28 工业 / 机械"), ("construction", "28 工业 / 机械"),
    ("building", "28 工业 / 机械"), ("engineering", "28 工业 / 机械"),
    # --- consumer ---
    ("beverage", "40 消费必需品"), ("packaged food", "40 消费必需品"),
    ("food", "40 消费必需品"), ("cosmetic", "40 消费必需品"),
    ("package goods", "40 消费必需品"), ("meat", "40 消费必需品"),
    ("poultry", "40 消费必需品"), ("agricult", "40 消费必需品"),
    ("tobacco", "40 消费必需品"), ("farming", "40 消费必需品"),
    ("restaurant", "41 消费服务 / 休闲"), ("hotel", "41 消费服务 / 休闲"),
    ("resort", "41 消费服务 / 休闲"), ("amusement", "41 消费服务 / 休闲"),
    ("recreation", "41 消费服务 / 休闲"), ("leisure", "41 消费服务 / 休闲"),
    ("gaming", "41 消费服务 / 休闲"), ("casino", "41 消费服务 / 休闲"),
    ("auto parts", "30 EV / 汽车"), ("motor vehicle", "30 EV / 汽车"),
    ("automotive", "30 EV / 汽车"), ("auto dealers", "30 EV / 汽车"),
    ("retail", "26 消费 / 零售"), ("apparel", "26 消费 / 零售"),
    ("department", "26 消费 / 零售"), ("shoe", "26 消费 / 零售"),
    ("consumer electronics", "26 消费 / 零售"), ("consumer specialties", "26 消费 / 零售"),
    # --- business services ---
    ("business services", "39 商业服务"), ("diversified commercial", "39 商业服务"),
    ("commercial services", "39 商业服务"), ("advertising", "39 商业服务"),
    ("consulting", "39 商业服务"), ("staffing", "39 商业服务"),
    ("professional services", "39 商业服务"),
]

# sector-level fallback (matched lowercase against `sector`)
SECTOR_MAP = [
    ("finance", "25 银行 / 金融"), ("technology", "29 软件 / 云"),
    ("health", "22 生物科技 / 制药"), ("industrials", "28 工业 / 机械"),
    ("utilities", "35 公用事业扩展"), ("energy", "33 能源扩展"),
    ("real estate", "36 房地产 / REIT"), ("consumer discretionary", "26 消费 / 零售"),
    ("consumer staples", "40 消费必需品"), ("telecommunications", "38 电信 / 媒体"),
    ("basic materials", "34 材料 / 金属"),
]
CATCHALL = "43 研究池-其他"


def classify(industry: str, sector: str) -> tuple[str, str]:
    ind = str(industry or "").strip().lower()
    sec = str(sector or "").strip().lower()
    for kw, grp in INDUSTRY_MAP:
        if kw in ind:
            return grp, "industry"
    for kw, grp in SECTOR_MAP:
        if kw in sec:
            return grp, "sector"
    return CATCHALL, "catchall"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    xls = pd.ExcelFile(INPUT_FILE)
    df_in = pd.read_excel(xls, "Sheet1_Input")
    m = pd.read_excel(xls, "Sheet2_Classified")
    for c in META_COLS:
        if c not in m.columns:
            m[c] = pd.NA
    m["enable"] = pd.to_numeric(m["enable"], errors="coerce").fillna(1).astype(int)

    # target = enable=0 rows currently in the coarse bulk buckets
    mask = (m["enable"] == 0) & m["group"].astype(str).str.startswith("90 研究池")
    n = int(mask.sum())
    print(f"regrouping {n} bulk research names (live + curated 22-35 untouched)")

    new_groups, how = [], []
    for _, r in m[mask].iterrows():
        g, src = classify(r.get("industry"), r.get("sector"))
        new_groups.append(g); how.append(src)
    m.loc[mask, "group"] = new_groups

    res = m[mask].copy()
    res["_how"] = how
    print("\n=== resulting group distribution ===")
    print(res["group"].value_counts().sort_index().to_string())
    print(f"\nmapped via industry: {how.count('industry')} | via sector: {how.count('sector')} | catch-all: {how.count('catchall')}")
    if how.count("catchall"):
        ca = res[res["_how"] == "catchall"]
        print("\n⚠ catch-all (unmapped — review these industries):")
        print(ca["industry"].astype(str).value_counts().to_string())

    newly = sorted(set(new_groups) & {"36 房地产 / REIT", "37 保险", "38 电信 / 媒体",
                                      "39 商业服务", "40 消费必需品", "41 消费服务 / 休闲",
                                      "42 运输 / 物流", CATCHALL})
    print(f"\nNEW groups introduced: {', '.join(newly)}")

    if not args.write:
        print("\n(preview only — re-run with --write to apply)")
        return 0

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = HISTORY_DIR / f"stock_input_template_backup_{ts}.xlsx"
    shutil.copy2(INPUT_FILE, backup)
    with pd.ExcelWriter(INPUT_FILE, engine="openpyxl") as w:
        df_in.to_excel(w, sheet_name="Sheet1_Input", index=False)
        m[META_COLS].to_excel(w, sheet_name="Sheet2_Classified", index=False)
    print(f"\n💾 wrote {INPUT_FILE}  (backup: {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
