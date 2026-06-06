#!/usr/bin/env python3
"""Regime-gate calculator: turn each scan date into a target exposure.

From the labeled dataset (build_dataset.py), for every date computes the two
same-day, look-ahead-free gate inputs — the market-context state and
SELL_share = #SELL signals / (#BUY + #SELL) — and the resulting target long
gross per the strategy in docs/STRATEGY_PROPOSAL.md. Writes/updates
../reports/gate_log.csv (one row per date) and prints the latest gate.

The gate can only CUT exposure as risk rises; its worst case is foregone upside.
No network. Run:
    ../../vcp_env/bin/python build_dataset.py
    ../../vcp_env/bin/python gate_calc.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DS = BASE_DIR / "reports" / "strategy_dataset.csv"
DEFAULT_LOG = BASE_DIR / "reports" / "gate_log.csv"

STRONG = {"强势/中性看涨", "强势", "中性看涨"}


def gate(state: str, sell_share: float):
    """Return (target_gross_pct, hedge, action). Order = most-restrictive first.
    The action text names the DRIVER (market state vs SELL_share breadth) so a
    blank-state, breadth-driven de-risk is never mistaken for a regime call."""
    s = str(state or "").strip()
    ss = float(sell_share) if pd.notna(sell_share) else 0.0
    if s in {"看跌/避险", "风险升高"}:
        return 0, "cash; optional small SPY/QQQ put", f"RISK-OFF (state={s}): 0% long, 100% cash"
    if ss >= 0.60:
        return 0, "cash; optional small SPY/QQQ put", f"RISK-OFF (breadth SELL_share={ss:.2f}>=0.60): 0% long, 100% cash"
    if s == "谨慎看涨":
        return 0, "start index hedge, rest cash", "CAUTION (state=谨慎看涨): no NEW longs, hold <=15%, begin hedge"
    if ss >= 0.40:
        return 0, "start index hedge, rest cash", f"CAUTION (breadth SELL_share={ss:.2f}>=0.40): no NEW longs, begin hedge"
    if s in STRONG and ss < 0.30:
        return 40, "rest cash", "RISK-ON (state strong, breadth ok): up to 40% long, equal-weight"
    if s in STRONG and ss < 0.40:
        return 20, "rest cash", f"RISK-ON soft (state strong, SELL_share={ss:.2f}): half size ~20% (interpolated)"
    return 0, "cash", "NO SIGNAL (blank/legacy state, breadth not elevated): 0% long"


def compute(ds_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ds_path)
    rows = []
    for dt, g in df.groupby("date"):
        state = ""
        st = g["state"].dropna().astype(str)
        st = st[st.str.strip() != ""]
        if len(st):
            state = st.mode().iloc[0]
        buy_n = int((g["side"] == "BUY").sum())
        sell_n = int((g["side"] == "SELL").sum())
        tot = buy_n + sell_n
        sell_share = round(sell_n / tot, 3) if tot else np.nan
        tg, hedge, action = gate(state, sell_share)
        rows.append({"date": dt, "state": state or "(blank)", "buy_n": buy_n, "sell_n": sell_n,
                     "sell_share": sell_share, "target_gross_pct": tg, "hedge": hedge, "action": action})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", default=str(DEFAULT_DS))
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    args = ap.parse_args()
    ds = Path(args.ds)
    if not ds.exists():
        print(f"Dataset not found: {ds}\nRun build_dataset.py first."); return 1

    gates = compute(ds)
    log = Path(args.log)
    # merge with any existing log (new dates overwrite old rows for the same date)
    if log.exists():
        try:
            old = pd.read_csv(log)
            keep = old[~old["date"].isin(gates["date"])]
            gates = pd.concat([keep, gates], ignore_index=True)
        except Exception:
            pass
    gates = gates.sort_values("date").reset_index(drop=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    gates.to_csv(log, index=False)

    print(f"WROTE {log}  ({len(gates)} dates)\n")
    show = gates[["date", "state", "buy_n", "sell_n", "sell_share", "target_gross_pct", "action"]]
    print(show.to_string(index=False))
    latest = gates.iloc[-1]
    print(f"\n>>> LATEST GATE ({latest['date']}): {latest['state']} | "
          f"SELL_share={latest['sell_share']} | TARGET LONG GROSS = {latest['target_gross_pct']}% | "
          f"hedge: {latest['hedge']}")
    print(f">>> ACTION: {latest['action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
