"""Is ARM's weakness 'idiosyncratic deterioration' or 'mean reversion after an idiosyncratic
melt-up'? Test the relative-strength claim across every window, not just 3 months.
"""
import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.width", 200)
tk = ["ARM", "SMH", "NVDA", "QCOM", "AVGO", "SPY", "QQQ"]
d = yf.download(tk, start="2023-09-14", auto_adjust=False, progress=False)["Close"]
d.index = pd.to_datetime(d.index).tz_localize(None)
d = d.dropna(how="all")
d.to_pickle("_arm_rel_px.pkl")
print(f"rows {len(d)}  {d.index[0].date()} .. {d.index[-1].date()}")
print(d.tail(3).round(2).to_string())

print("\n" + "=" * 100)
print("ARM TOTAL RETURN vs PEERS ACROSS WINDOWS (close-to-close, ending 2026-09-14)")
print("=" * 100)
wins = {"1w (5d)": 5, "1m (21d)": 21, "3m (63d)": 63, "6m (126d)": 126,
        "12m (252d)": 252, "since IPO": len(d) - 1}
rows = []
for lab, n in wins.items():
    if n >= len(d):
        n = len(d) - 1
    r = {}
    for c in d.columns:
        s = d[c].dropna()
        if len(s) <= n:
            continue
        r[c] = 100 * (s.iloc[-1] / s.iloc[-1 - n] - 1)
    r["ARM-SMH"] = r.get("ARM", np.nan) - r.get("SMH", np.nan)
    r["ARM-NVDA"] = r.get("ARM", np.nan) - r.get("NVDA", np.nan)
    r["ARM-SPY"] = r.get("ARM", np.nan) - r.get("SPY", np.nan)
    rows.append(pd.Series(r, name=lab))
tbl = pd.DataFrame(rows)
print(tbl.round(1).to_string())

print("\n" + "=" * 100)
print("*** THE WINDOW IS DOING THE WORK ***")
print("=" * 100)
a, s_ = d["ARM"], d["SMH"]
print(f"  3-month ARM vs SMH: {tbl.loc['3m (63d)', 'ARM-SMH']:+.1f}pt  <- the brief's key fact")
print(f"  6-month ARM vs SMH: {tbl.loc['6m (126d)', 'ARM-SMH']:+.1f}pt")
print(f"  12-month ARM vs SMH: {tbl.loc['12m (252d)', 'ARM-SMH']:+.1f}pt")
print("\n  Where the 3-month window starts matters completely:")
for start in ["2026-03-31", "2026-05-06", "2026-06-01", "2026-06-18", "2026-07-29", "2026-08-18"]:
    ai = a.index.searchsorted(pd.Timestamp(start))
    ra = 100 * (a.iloc[-1] / a.iloc[ai] - 1)
    rs = 100 * (s_.iloc[-1] / s_.iloc[ai] - 1)
    note = {"2026-03-31": "FY2026 year-end", "2026-05-06": "Q4 FY26 earnings / AGI CPU $1bn",
            "2026-06-18": "ALL-TIME HIGH close 439.46", "2026-07-29": "Q1 FY27 earnings",
            "2026-08-18": "the day ARM opened at exactly 257.00"}.get(start, "")
    print(f"    from {start} ({a.index[ai].date()}, ARM {a.iloc[ai]:7.2f}): ARM {ra:+7.1f}%  SMH {rs:+6.1f}%  "
          f"ARM-SMH {ra - rs:+7.1f}pt   {note}")

print("\n  The 3-month lookback begins INSIDE a vertical melt-up. ARM ran 151.28 (2026-03-31) ->")
n_days = (a.index.searchsorted(pd.Timestamp('2026-06-18')) - a.index.searchsorted(pd.Timestamp('2026-03-31')))
print(f"  439.46 (2026-06-18), +{100 * (439.46 / 151.28 - 1):.0f}% in {n_days} trading sessions, while SMH did "
      f"{100 * (s_.loc['2026-06-18'] / s_.loc['2026-03-31'] - 1):+.0f}%.")
print("  Measuring 'relative weakness' from that peak measures the melt-up, not deterioration.")
print(f"  Over 12 months ARM is {tbl.loc['12m (252d)', 'ARM-SMH']:+.0f}pt vs SMH and "
      f"{tbl.loc['12m (252d)', 'ARM-SPY']:+.0f}pt vs SPY.")

print("\n" + "=" * 100)
print("RETRACEMENT ARITHMETIC")
print("=" * 100)
lo, hi, now = 151.28, 439.46, 239.01
print(f"  swing low {lo} (2026-03-31) -> high {hi} (2026-06-18) -> now {now}")
print(f"  retracement of the up-leg: {100 * (hi - now) / (hi - lo):.1f}%")
for f in (0.5, 0.618, 0.786, 1.0):
    lvl = hi - f * (hi - lo)
    print(f"    {f:.3f} retrace = {lvl:7.2f}  ({100 * (lvl / now - 1):+6.1f}% from here, "
          f"{100 * (lvl / 257 - 1):+6.1f}% vs cost 257)")
print(f"\n  Cost 257.00 = {100 * (hi - 257) / (hi - lo):.1f}% retracement -> the entry was made")
print("  roughly two-thirds of the way down a completed melt-up, not at a base.")

print("\n" + "=" * 100)
print("BETA / IDIOSYNCRATIC DECOMPOSITION vs SMH (daily, closed bars, last 252d)")
print("=" * 100)
ret = np.log(d[["ARM", "SMH", "NVDA", "SPY"]]).diff().dropna().iloc[:-1]  # drop partial bar
for win, lab in ((252, "last 252d"), (63, "last 63d")):
    r2 = ret.tail(win)
    x, y = r2["SMH"].values, r2["ARM"].values
    beta = np.cov(x, y)[0, 1] / np.var(x)
    alpha = y.mean() - beta * x.mean()
    resid = y - (alpha + beta * x)
    r_sq = 1 - resid.var() / y.var()
    print(f"  {lab}: beta to SMH {beta:.2f}  R^2 {r_sq:.2f}  "
          f"idiosyncratic vol {100 * resid.std() * np.sqrt(252):.0f}% annualised  "
          f"(total ARM vol {100 * y.std() * np.sqrt(252):.0f}%)")
print("\n  R^2 that low means SMH explains only a minority of ARM's variance. ARM is mostly its own")
print("  risk. That cuts BOTH ways: the melt-up was not the sector either.")
