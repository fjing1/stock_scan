"""ARM price path vs the dated primary-source event calendar."""
import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.width", 200)
t = yf.Ticker("ARM")
px = t.history(period="max", auto_adjust=False)
px.index = px.index.tz_localize(None)
# drop the in-progress bar per repo convention
print("last bars:")
print(px[["Open", "High", "Low", "Close", "Volume"]].tail(4))
px.to_pickle("_arm_px.pkl")

c = px["Close"]
print(f"\nrows {len(c)}  first {c.index[0].date()} last {c.index[-1].date()}  last close {c.iloc[-1]:.2f}")

# EVENT DATES from SEC filings (6-K furnish dates = after-close earnings)
events = {
    "2023-11-08": "Q2 FY24 earnings (first as public co)",
    "2024-02-07": "Q3 FY24 earnings",
    "2024-05-08": "Q4 FY24 earnings",
    "2024-07-31": "Q1 FY25 earnings",
    "2024-11-06": "Q2 FY25 earnings",
    "2025-02-05": "Q3 FY25 earnings",
    "2025-05-07": "Q4 FY25 earnings",
    "2025-07-30": "Q1 FY26 earnings",
    "2025-11-05": "Q2 FY26 earnings",
    "2026-02-04": "Q3 FY26 earnings",
    "2026-05-06": "Q4 FY26 earnings",
    "2026-07-29": "Q1 FY27 earnings",
}
print("\n=== reaction to each earnings 6-K (day-of -> next 1/5/10 sessions) ===")
for d, lab in events.items():
    ts = pd.Timestamp(d)
    idx = c.index.searchsorted(ts)
    if idx >= len(c):
        continue
    base = c.iloc[idx]
    row = [f"{lab:38s} {c.index[idx].date()} close {base:8.2f}"]
    for h in (1, 5, 10, 21):
        j = idx + h
        if j < len(c):
            row.append(f"+{h}d {100 * (c.iloc[j] / base - 1):+7.2f}%")
    print("  ".join(row))

print("\n=== monthly closes, last 18m ===")
m = c.resample("ME").last()
print(m.tail(18).round(2).to_string())

print("\n=== the 3-month drawdown: daily since 2026-06-01 ===")
seg = px.loc["2026-06-01":, ["Open", "High", "Low", "Close", "Volume"]].copy()
seg["chg%"] = seg["Close"].pct_change() * 100
big = seg[seg["chg%"].abs() > 4.5]
print("days with |move| > 4.5%:")
print(big.round(2).to_string())

print(f"\nsegment high {seg['Close'].max():.2f} on {seg['Close'].idxmax().date()}")
print(f"segment low  {seg['Close'].min():.2f} on {seg['Close'].idxmin().date()}")

print("\n=== all-time drawdown context ===")
print(f"ATH close {c.max():.2f} on {c.idxmax().date()}")
print(f"current {c.iloc[-1]:.2f}, {100 * (c.iloc[-1] / c.max() - 1):+.1f}% from ATH close")
