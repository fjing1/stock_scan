"""
_fund_data.py — point-in-time fundamental panel from SEC XBRL. Free, keyless, no vendor.

Same build-once/pickle/reuse pattern as _move_data.py and _vix_data.py.

WHY THIS EXISTS AT ALL: every fundamental study in this repo has been blocked by not having
fundamentals. SEC's XBRL API closes that for free, and — the part that matters — it stamps every
fact with the date it was FILED, not just the period it covers. That is what makes a fundamental
backtest honest. Most fundamental datasets give you the restated final number and silently let you
trade on information that did not exist yet; here we can align on `filed` and never do that.

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

THREE THINGS THAT WILL SILENTLY CORRUPT THIS IF NOT HANDLED, all verified against real filings:

  1. TAG MIGRATION. Companies change which us-gaap tag they report a concept under. Apple's
     `Revenues` series has 11 facts and STOPS IN 2018 -- it moved to
     `RevenueFromContractWithCustomerExcludingAssessedTax`. A single-tag lookup silently returns a
     truncated series and every growth rate computed from it is garbage. So every concept is a
     PRIORITY LADDER, and we record which tag actually supplied each row.

  2. FOREIGN PRIVATE ISSUERS. Arm Holdings (CIK 1973239) files 20-F/6-K, not 10-K/10-Q, and its
     fiscal year ends 31 March so `fy`/`fp` do not mean what they mean for a US filer. Any code
     that filters on form == "10-K" drops these companies entirely.

  3. DUPLICATE FACTS. The same (concept, period) appears in multiple filings as it gets restated,
     and annual filings also carry quarterly comparatives. We keep ALL versions -- that is the
     point of point-in-time -- and let the caller ask for "as known on date D".

RATE LIMIT AND THE USER-AGENT TRAP: SEC asks for <= 10 requests/second and a real User-Agent.
It is stricter than "any non-empty string" -- it wants something that looks like a genuine contact
address. Verified today: UA "stock_scan-research fundamentals@localhost" returns 403 on every
endpoint, while "stock_scan research contact@example.com" returns 200 on the identical request.
Set FUND_USER_AGENT to your own real email.

Run:  ../../vcp_env/bin/python _fund_data.py --symbols AAPL NVDA ARM
      ../../vcp_env/bin/python _fund_data.py --watchlist        # today's buy watchlist
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_fund_cache"
PANEL = HERE / "_fund_panel.pkl"
TICKER_MAP = CACHE / "company_tickers.json"

UA = os.environ.get("FUND_USER_AGENT", "stock_scan research contact@example.com")
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
SEC_SLEEP = 0.12          # ~8 req/s, under SEC's 10/s ceiling

# Annual and quarterly report forms across US filers AND foreign private issuers.
ANNUAL = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F"}
INTERIM = {"10-Q", "10-Q/A", "6-K"}

# Concept -> priority ladder of us-gaap (or dei) tags. First tag that has data for a given period
# wins, and the winning tag is recorded so tag switches are auditable rather than invisible.
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "revenue_related_party": ["RevenueFromRelatedParties"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfServices"],
    "gross_profit": ["GrossProfit"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
    "short_term_inv": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                       "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "debt_total": ["DebtLongtermAndShorttermCombinedAmount", "LongTermDebt",
                   "LongTermDebtNoncurrent"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "contract_asset": ["ContractWithCustomerAssetNetCurrent", "ContractWithCustomerAssetNet"],
    "deferred_revenue": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
    "rpo": ["RevenueRemainingPerformanceObligation"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding",
                       "WeightedAverageNumberOfSharesOutstandingBasic"],
    "eps_diluted": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
}
UNITLESS = {"shares_diluted", "eps_diluted"}     # shares / USD-per-share, not USD


def _get(url: str, tries: int = 3):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
        except Exception as exc:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        time.sleep(1.5 * (i + 1))       # 403/429/5xx -> back off
    return None


def ticker_to_cik(refresh: bool = False) -> dict[str, str]:
    """SEC's own ticker->CIK map. 10,422 registrants in one file."""
    CACHE.mkdir(exist_ok=True)
    if TICKER_MAP.exists() and not refresh:
        raw = json.loads(TICKER_MAP.read_text())
    else:
        r = _get("https://www.sec.gov/files/company_tickers.json")
        if r is None:
            raise RuntimeError(
                "SEC 拒绝了请求（通常是 403）。最常见原因是 User-Agent 不像真实联系方式——"
                f"当前是 {UA!r}。设一个真实邮箱：export FUND_USER_AGENT='你的名字 you@domain.com'")
        raw = r.json()
        TICKER_MAP.write_text(json.dumps(raw))
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}


def fetch_companyfacts(cik: str, refresh: bool = False) -> dict | None:
    """Raw companyfacts JSON, gzipped to disk. ~3.8MB raw for AAPL, so the cache is gitignored;
    the extracted panel is what gets tracked."""
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"CIK{cik}.json.gz"
    if p.exists() and not refresh:
        with gzip.open(p, "rt") as f:
            return json.load(f)
    r = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    time.sleep(SEC_SLEEP)
    if r is None:
        return None
    with gzip.open(p, "wt") as f:
        f.write(r.text)
    return r.json()


def extract(symbol: str, cf: dict) -> pd.DataFrame:
    """Flatten companyfacts into one long frame, keeping EVERY reported version of every fact.

    Columns: symbol, concept, tag, start, end, val, form, filed, accn, fy, fp, annual
    `filed` is the point-in-time key: a row is only knowable on or after that date.
    """
    rows = []
    facts = cf.get("facts", {})
    for concept, ladder in CONCEPTS.items():
        for tag in ladder:
            for ns in ("us-gaap", "ifrs-full", "dei"):
                node = facts.get(ns, {}).get(tag)
                if not node:
                    continue
                for unit, arr in node.get("units", {}).items():
                    if concept not in UNITLESS and unit != "USD":
                        continue
                    for f in arr:
                        if f.get("val") is None or not f.get("end") or not f.get("filed"):
                            continue
                        form = f.get("form", "")
                        rows.append({
                            "symbol": symbol, "concept": concept, "tag": tag, "ns": ns,
                            "unit": unit, "start": f.get("start"), "end": f["end"],
                            "val": float(f["val"]), "form": form, "filed": f["filed"],
                            "accn": f.get("accn"), "fy": f.get("fy"), "fp": f.get("fp"),
                            "annual": form in ANNUAL,
                        })
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    # duration facts (income/cash-flow) carry start; instant facts (balance sheet) do not
    d["days"] = (d["end"] - d["start"]).dt.days
    d["kind"] = d["start"].isna().map({True: "instant", False: "duration"})
    return d.drop_duplicates(subset=["symbol", "concept", "tag", "start", "end", "accn"])


def build(symbols: list[str], refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    cikmap = ticker_to_cik(refresh=refresh)
    out, missing = [], []
    for i, s in enumerate(symbols, 1):
        cik = cikmap.get(s.upper())
        if cik is None:
            missing.append((s, "no CIK (ETF, ADR without filings, or delisted)"))
            continue
        cf = fetch_companyfacts(cik, refresh=refresh)
        if cf is None:
            missing.append((s, "no companyfacts"))
            continue
        d = extract(s.upper(), cf)
        if d.empty:
            missing.append((s, "no mapped concepts"))
            continue
        out.append(d)
        if verbose:
            ann = d[d.annual]
            print(f"  [{i:>3}/{len(symbols)}] {s:<6} facts={len(d):>6} concepts={d.concept.nunique():>2} "
                  f"forms={sorted(d.form.unique())[:3]} 最早{d.end.min().date()} 最新{d.end.max().date()}",
                  flush=True)
    if not out:
        raise RuntimeError("nothing fetched")
    fresh = pd.concat(out, ignore_index=True)

    # MERGE, never replace. Overwriting silently destroyed ARM/AAPL/NVDA/QCOM the first time a
    # watchlist fetch ran after a per-symbol fetch, which would permanently break auto-resolution
    # in fund_forward.py for any symbol absent from the most recent fetch. A long-lived ledger
    # needs a panel that only ever grows.
    prior = pd.DataFrame()
    if PANEL.exists():
        try:
            prior = pd.read_pickle(PANEL).get("panel", pd.DataFrame())
        except Exception:
            prior = pd.DataFrame()
    if not prior.empty:
        keep = prior[~prior.symbol.isin(set(fresh.symbol.unique()))]   # refetched symbols replaced
        panel = pd.concat([keep, fresh], ignore_index=True)
    else:
        panel = fresh
    panel = panel.drop_duplicates(subset=["symbol", "concept", "tag", "start", "end", "accn"])
    pd.to_pickle({"panel": panel, "missing": missing, "built": pd.Timestamp.now(tz="UTC")}, PANEL)
    if verbose:
        added = panel.symbol.nunique() - (prior.symbol.nunique() if not prior.empty else 0)
        print(f"\n面板 {PANEL.name}: {len(panel):,} 条 facts, {panel.symbol.nunique()} 标的"
              f"（本次新增/刷新 {fresh.symbol.nunique()}，净增 {added}）, "
              f"{panel.concept.nunique()} 概念")
        if missing:
            print(f"未取到 {len(missing)} 个: " + ", ".join(f"{s}({w})" for s, w in missing[:8])
                  + (" ..." if len(missing) > 8 else ""))
    return panel


def load() -> pd.DataFrame:
    if not PANEL.exists():
        raise FileNotFoundError(f"{PANEL.name} 不存在，先跑 _fund_data.py")
    return pd.read_pickle(PANEL)["panel"]


# ------------------------------------------------------------------ point-in-time access
def as_of(panel: pd.DataFrame, when, concept: str | None = None,
          annual_only: bool = False) -> pd.DataFrame:
    """Everything that was PUBLICLY KNOWN on `when`. This is the whole reason for this module:
    filter on `filed <= when`, never on `end <= when`."""
    d = panel[panel.filed <= pd.Timestamp(when)]
    if concept:
        d = d[d.concept == concept]
    if annual_only:
        d = d[d.annual]
    return d


def latest(panel: pd.DataFrame, symbol: str, concept: str, when=None,
           kind: str | None = None, min_days: int | None = None,
           max_days: int | None = None) -> pd.Series | None:
    """The most recently ENDED period for `concept`, using only filings available at `when`.
    Ties on `end` are broken by the later `filed` (the restatement that was current then).

    min_days/max_days select the period length: quarterly duration facts are ~90 days, annual
    ~365. Without them a mix of Q and FY facts comes back and every ratio silently mixes scales."""
    d = panel[(panel.symbol == symbol.upper()) & (panel.concept == concept)]
    if when is not None:
        d = d[d.filed <= pd.Timestamp(when)]
    if kind:
        d = d[d.kind == kind]
    if min_days is not None:
        d = d[d.days >= min_days]
    if max_days is not None:
        d = d[d.days <= max_days]
    if d.empty:
        return None
    return d.sort_values(["end", "filed"]).iloc[-1]


def series(panel: pd.DataFrame, symbol: str, concept: str, *, annual: bool,
           when=None) -> pd.DataFrame:
    """Time series of one concept for one symbol, one row per period end, point-in-time safe.
    annual=True -> ~365-day durations (or instants from annual filings); False -> ~90-day."""
    d = panel[(panel.symbol == symbol.upper()) & (panel.concept == concept)]
    if when is not None:
        d = d[d.filed <= pd.Timestamp(when)]
    if d.empty:
        return pd.DataFrame()
    if (d.kind == "duration").any():
        d = d[d.kind == "duration"]
        d = d[(d.days >= 300) & (d.days <= 400)] if annual else d[(d.days >= 60) & (d.days <= 120)]
    else:
        d = d[d.annual] if annual else d
    if d.empty:
        return pd.DataFrame()
    # one row per period end: keep the latest-filed version known at `when`
    d = d.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")
    return d[["end", "val", "tag", "form", "filed", "accn", "days"]].reset_index(drop=True)


def watchlist_symbols(limit: int | None = None) -> list[str]:
    """Symbols from the most recent TV buy-signal file, stripped of exchange prefixes."""
    base = HERE.parent / "tv_buy_signals"
    files = sorted(base.glob("tv_buy_today_2*.txt"))
    if not files:
        return []
    syms = []
    for line in files[-1].read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            syms.append(line.split(":")[-1].strip().upper())
    syms = list(dict.fromkeys(syms))
    return syms[:limit] if limit else syms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--watchlist", action="store_true", help="用最近一份买入信号清单")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重新抓取")
    a = ap.parse_args()
    syms = a.symbols or (watchlist_symbols() if a.watchlist else ["AAPL", "NVDA", "ARM"])
    print(f"User-Agent: {UA}")
    print(f"抓取 {len(syms)} 个标的的 SEC XBRL ...")
    build(syms, refresh=a.refresh)
