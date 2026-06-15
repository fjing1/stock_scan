"""Sector / industry trend confirmation for a stock signal (top-down overlay).

A bottom-up dip-in-uptrend buy is higher-conviction when the whole industry is
healthy. Given a stock, this:
  1. identifies its sector + granular industry (yfinance .info),
  2. maps it to a sector SPDR ETF and (when known) a thematic industry ETF,
  3. scores those ETFs' trend health,
  4. finds industry peers (same granular industry within the repo sector list) and
     computes breadth (% above MA200, median momentum, how many are themselves dips),
  5. returns a SECTOR-CONFIRMATION verdict to overlay on the stock's own signal.

    python sector_confirm.py NTR
    python sector_confirm.py CF --peers 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yfinance as yf  # noqa: E402

# granular industry -> curated liquid peer set (primary source for breadth)
INDUSTRY_PEERS = {
    "Agricultural Inputs": ["CF", "NTR", "MOS", "IPI", "CTVA", "FMC", "SMG", "ICL", "BG"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "INTC", "MU", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "ADI", "MCHP"],
    "Oil & Gas E&P": ["EOG", "COP", "DVN", "FANG", "OXY", "APA", "CTRA", "HES", "MRO"],
    "Gold": ["NEM", "GOLD", "AEM", "KGC", "AU", "FNV", "WPM"],
    "Biotechnology": ["AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA", "INCY"],
    "Banks - Regional": ["USB", "PNC", "TFC", "FITB", "MTB", "HBAN", "RF", "KEY", "CFG"],
}
# yfinance broad sector -> S&P SPDR sector ETF
SECTOR_SPDR = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Basic Materials": "XLB", "Real Estate": "XLRE",
    "Utilities": "XLU", "Communication Services": "XLC",
}
# granular industry -> thematic ETF (curated; closest liquid proxy)
INDUSTRY_ETF = {
    "Agricultural Inputs": "MOO", "Semiconductors": "SMH",
    "Oil & Gas E&P": "XOP", "Oil & Gas Equipment & Services": "OIH",
    "Gold": "GDX", "Biotechnology": "XBI",
    "Banks - Regional": "KRE", "Banks - Diversified": "KBE",
    "Residential Construction": "XHB", "Airlines": "JETS",
    "Software - Application": "IGV", "Software - Infrastructure": "IGV",
    "Internet Retail": "XRT", "Aerospace & Defense": "ITA",
    "Steel": "SLX", "Copper": "COPX", "Solar": "TAN",
}
# yfinance sector -> repo sector list name (in stock_symbols_1243)
SECTOR_REPO = {
    "Technology": "TECH_STOCKS", "Healthcare": "HEALTHCARE_STOCKS",
    "Financial Services": "FINANCIAL_STOCKS", "Consumer Cyclical": "CONSUMER_DISCRETIONARY",
    "Consumer Defensive": "CONSUMER_STAPLES", "Energy": "ENERGY_STOCKS",
    "Industrials": "MATERIALS_INDUSTRIALS", "Basic Materials": "MATERIALS_INDUSTRIALS",
    "Real Estate": "REAL_ESTATE_REITS", "Utilities": "UTILITIES",
    "Communication Services": "COMMUNICATION_SERVICES",
}


def fetch(symbols, period="2y"):
    syms = symbols if isinstance(symbols, list) else [symbols]
    raw = yf.download(syms, period=period, interval="1d", auto_adjust=False,
                      progress=False, group_by="ticker", threads=True)
    out = {}
    for s in syms:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = set(raw.columns.get_level_values(0))
                if s in lvl0:                      # (ticker, field) layout
                    d = raw[s].copy()
                else:                              # single ticker -> (field, ticker)
                    d = raw.copy()
                    d.columns = d.columns.get_level_values(0)
            else:
                d = raw.copy()
            d = d.dropna(how="all")
        except Exception:
            continue
        if len(d) and "Close" in d.columns:
            if getattr(d.index, "tz", None) is not None:
                d.index = d.index.tz_localize(None)
            out[s] = d
    return out


def trend_health(df):
    """0-5 trend score + components for a price series."""
    if df is None or len(df) < 220:
        return None
    c = df["Close"].astype(float)
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200).mean()
    px = float(c.iloc[-1])
    above200 = px > ma200.iloc[-1]
    ma_align = ma50.iloc[-1] > ma200.iloc[-1]
    ma_rising = ma50.iloc[-1] > ma50.iloc[-11]
    mom3 = px / float(c.iloc[-63]) - 1
    mom6 = px / float(c.iloc[-126]) - 1
    score = sum([above200, ma_align, ma_rising, mom3 > 0, mom6 > 0])
    return dict(px=px, pct200=(px / ma200.iloc[-1] - 1) * 100, above200=above200,
                ma_align=ma_align, ma_rising=ma_rising, mom3=mom3 * 100,
                mom6=mom6 * 100, score=score)


def is_dip(df):
    """dip-in-uptrend (close>MA200 & RSI14<40 & StochK<20) on the last bar."""
    try:
        from cycle_patter_for_swing import compute_cycle_stoch
        cs = compute_cycle_stoch(df)
        c = df["Close"].astype(float)
        return bool((c.iloc[-1] > c.rolling(200).mean().iloc[-1])
                    and cs["rsi"].iloc[-1] < 40 and cs["stoch_k"].iloc[-1] < 20)
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--peers", type=int, default=35, help="max sector members to scan for industry peers")
    args = ap.parse_args(argv)
    sym = args.symbol.upper()

    info = yf.Ticker(sym).info
    sector, industry = info.get("sector"), info.get("industry")
    print(f"=== Sector confirmation for {sym} ===")
    print(f"sector: {sector}   industry: {industry}")

    spdr = SECTOR_SPDR.get(sector)
    them = INDUSTRY_ETF.get(industry)
    etfs = [e for e in [them, spdr] if e]
    print(f"sector ETF: {spdr}   thematic ETF: {them or '(none mapped)'}")

    # --- ETF trend health ---
    print("\n-- ETF trend health (0-5; >=4 = healthy uptrend) --")
    print(f"  {'ETF':<5}{'px':>9}{'%>MA200':>9}{'MA50>200':>9}{'MA50↑':>7}{'3m%':>7}{'6m%':>7}{'score':>7}")
    etf_scores = []
    for e in etfs:
        d = fetch(e).get(e)
        h = trend_health(d)
        if h:
            etf_scores.append(h["score"])
            print(f"  {e:<5}{h['px']:>9.2f}{h['pct200']:>+9.1f}{str(h['ma_align']):>9}"
                  f"{str(h['ma_rising']):>7}{h['mom3']:>+7.1f}{h['mom6']:>+7.1f}{h['score']:>5}/5")

    # --- industry peers: curated set if available, else scan repo sector list ---
    peers = []
    if industry in INDUSTRY_PEERS:
        peers = [p for p in INDUSTRY_PEERS[industry] if p != sym]
    else:
        repo = SECTOR_REPO.get(sector)
        if repo:
            import stock_symbols_1243 as ss
            members = [s for s in getattr(ss, repo, []) if s != sym][: args.peers]
            print(f"\nscanning up to {len(members)} {sector} members for industry == '{industry}' ...", flush=True)
            for m in members:
                try:
                    if yf.Ticker(m).info.get("industry") == industry:
                        peers.append(m)
                except Exception:
                    continue
    peers = [sym] + peers
    pdata = fetch(peers)

    print(f"\n-- industry peers in '{industry}' ({len([p for p in peers if p in pdata])} found) --")
    print(f"  {'sym':<6}{'px':>9}{'%>MA200':>9}{'3m%':>7}{'trend/5':>8}{'dip?':>6}")
    rows = []
    for p in peers:
        d = pdata.get(p)
        h = trend_health(d)
        if not h:
            continue
        dip = is_dip(d)
        rows.append((p, h, dip))
        print(f"  {p:<6}{h['px']:>9.2f}{h['pct200']:>+9.1f}{h['mom3']:>+7.1f}{h['score']:>6}/5"
              f"{('  YES' if dip else '   -'):>6}")

    # --- breadth + verdict ---
    if rows:
        above = np.mean([1 if r[1]["above200"] else 0 for r in rows]) * 100
        med3 = np.median([r[1]["mom3"] for r in rows])
        med_score = np.median([r[1]["score"] for r in rows])
        dips = [r[0] for r in rows if r[2]]
        etf_avg = np.mean(etf_scores) if etf_scores else None
        n = len(rows)
        print(f"\n-- INDUSTRY BREADTH ({n} peers) --")
        print(f"  peers above MA200: {above:.0f}%   median 3m mom: {med3:+.1f}%   median trend: {med_score:.0f}/5")
        print(f"  sector/thematic ETF avg trend: {etf_avg:.1f}/5" if etf_avg is not None else "  (no ETF)")
        print(f"  peers also flashing a dip-in-uptrend: {', '.join(dips) if dips else 'none'}")

        if n < 4:
            print("\n  >>> SECTOR CONFIRMATION: INSUFFICIENT peers — lean on ETF trend above.")
        else:
            structural = above >= 60 and med_score >= 3 and (etf_avg is None or etf_avg >= 3)
            momentum_up = med3 > 0
            etf_weak = etf_avg is not None and etf_avg < 3
            if structural and momentum_up:
                v = "STRONG — industry in a healthy, RISING trend (confirms a dip-buy)"
            elif structural and not momentum_up:
                v = "CONSTRUCTIVE — long-term uptrend intact but industry is PULLING BACK now (classic dip backdrop; confirm the bottom is in)"
            elif (above < 40 or med_score <= 2) or etf_weak:
                v = "WEAK — industry trend does NOT confirm (dip-buys here risk catching a falling sector)"
            else:
                v = "MIXED — partial confirmation only"
            print(f"\n  >>> SECTOR CONFIRMATION for {sym}: {v}")


if __name__ == "__main__":
    main()
