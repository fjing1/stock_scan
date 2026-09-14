"""FALSIFICATION TEST of the user's stated mechanism:
"Apple just released new iPhones, so from a fundamentals view ARM should be long-term bullish."

Apple iPhone announcement/launch dates are Apple press-release facts (apple.com newsroom /
Apple 8-K-free product announcements); they are public, dated, non-controversial.
If the mechanism is real, ARM should show a positive drift after iPhone launches.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 210)
d = pd.read_pickle("_arm_rel_px.pkl")
arm, smh, aapl_missing = d["ARM"], d["SMH"], None

# Apple's September iPhone announcement events (announcement date, ship date)
# ARM IPO'd 2023-09-14, so 2023 is the first testable year.
EV = [
    ("2023-09-12", "iPhone 15 announced"),
    ("2023-09-22", "iPhone 15 on sale"),
    ("2024-09-09", "iPhone 16 announced"),
    ("2024-09-20", "iPhone 16 on sale"),
    ("2025-09-09", "iPhone 17 announced"),
    ("2025-09-19", "iPhone 17 on sale"),
    ("2026-09-09", "iPhone (2026 cycle) announced -- the user's trigger"),
]
print("=" * 108)
print("ARM AROUND APPLE iPHONE LAUNCHES -- does the user's mechanism show up in the tape?")
print("=" * 108)
print(f"  {'event':>48} {'date':>12} {'ARM close':>10} " + " ".join(f"{'+' + str(h) + 'd':>8}" for h in (1, 5, 10, 21, 63)))
rows = []
for ds, lab in EV:
    ts = pd.Timestamp(ds)
    i = arm.index.searchsorted(ts)
    if i >= len(arm):
        continue
    base = arm.iloc[i]
    cells, rec = [], {"event": lab, "date": arm.index[i].date()}
    for h in (1, 5, 10, 21, 63):
        j = i + h
        if j < len(arm):
            v = 100 * (arm.iloc[j] / base - 1)
            vs = 100 * (smh.iloc[j] / smh.iloc[i] - 1)
            cells.append(f"{v:+7.1f}%")
            rec[f"ARM+{h}"] = v
            rec[f"rel+{h}"] = v - vs
        else:
            cells.append("     n/a")
    rows.append(rec)
    print(f"  {lab:>48} {str(arm.index[i].date()):>12} {base:>10.2f} " + " ".join(cells))

R = pd.DataFrame(rows)
print("\n  ABSOLUTE ARM return after the event (%):")
for h in (1, 5, 10, 21, 63):
    col = f"ARM+{h}"
    v = R[col].dropna()
    print(f"    +{h:>2}d: mean {v.mean():+6.1f}%  median {v.median():+6.1f}%  "
          f"positive {int((v > 0).sum())}/{len(v)}")
print("\n  ARM MINUS SMH after the event (pt) -- isolating the Apple-specific claim:")
for h in (1, 5, 10, 21, 63):
    col = f"rel+{h}"
    v = R[col].dropna()
    print(f"    +{h:>2}d: mean {v.mean():+6.1f}pt  median {v.median():+6.1f}pt  "
          f"positive {int((v > 0).sum())}/{len(v)}")

print("\n" + "=" * 108)
print("THE MOST DIRECT TEST: the user's own trigger")
print("=" * 108)
i0 = arm.index.searchsorted(pd.Timestamp("2026-09-09"))
print(f"  Apple's 2026 iPhone announcement was 2026-09-09. ARM closed {arm.iloc[i0]:.2f} that day.")
for k in range(i0, len(arm)):
    print(f"    {arm.index[k].date()}  ARM {arm.iloc[k]:7.2f} ({100 * (arm.iloc[k] / arm.iloc[i0] - 1):+6.2f}% vs 09-09)   "
          f"SMH {smh.iloc[k]:7.2f} ({100 * (smh.iloc[k] / smh.iloc[i0] - 1):+6.2f}%)")
print(f"\n  In the {len(arm) - i0 - 1} sessions since the iPhone announcement ARM is "
      f"{100 * (arm.iloc[-1] / arm.iloc[i0] - 1):+.1f}% and SMH is {100 * (smh.iloc[-1] / smh.iloc[i0] - 1):+.1f}%.")
print("  The mechanism has now been live for a week and has produced the opposite sign.")
print("  (One week proves nothing on its own -- which is exactly the point: it also proves nothing")
print("   in the other direction, and it is the only evidence the 'new iPhone' argument has.)")

print("\n" + "=" * 108)
print("WHY THE MECHANISM CANNOT WORK ARITHMETICALLY  [from the 20-F, not from the tape]")
print("=" * 108)
print("  Even a perfect iPhone cycle reaches ARM through a very narrow pipe:")
print("   1. ARM royalty revenue FY2026 = $2,613m of $4,920m total (20-F Note 4).")
print("   2. Mobile applications processors = 43% of royalty = $1,124m = 22.8% of TOTAL revenue")
print("      (20-F FY2026 risk factor + business section).")
print("   3. That $1,124m is ALL mobile-AP vendors: Apple, Qualcomm, MediaTek, Samsung, Google,")
print("      Unisoc and others. Apple is a fraction of a fraction.")
print("   4. ARM's royalty is accrued 'in the quarter in which the customer SHIPS their products'")
print("      (Q1 FYE27 letter, revenue-recognition note) -- so the relevant variable is Apple's")
print("      unit SHIPMENTS, not the excitement of a launch event, and ARM books it a quarter later")
print("      ('Referenced figures are based on the most recent royalty report data that relates to")
print("      the prior quarter').")
print("   5. ARM's own filing: '>99% market share in mobile applications processors for many years'")
print("      and 'our substantial existing market share may limit opportunities for future growth.'")
print("      ARM already gets paid on essentially every smartphone. There is no share to win.")
print("  => Apple's iPhone cycle changes ARM's UNIT count, in a segment that is 22.8% of revenue,")
print("     where ARM already has 99%+ share. It cannot be the driver of a re-rating.")
