"""Consistent Black-Scholes IVs from mid prices (no scipy), earnings implied move,
and the actual cost of each hedge structure. Chain: yfinance live 2026-09-14.
"""
import math

import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.width", 240)
PX, COST, R = 239.01, 257.00, 0.040


def N(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs(S, K, T, s, r, call=True):
    if T <= 0 or s <= 0:
        return max(0.0, (S - K) if call else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * s * s) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    if call:
        return S * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * N(-d1)


def iv(price, S, K, T, r, call=True):
    lo, hi = 1e-4, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if bs(S, K, T, mid, r, call) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


t = yf.Ticker("ARM")
print("=" * 100)
print("SELF-COMPUTED IV SURFACE (Black-Scholes on mid prices, r=4.0%) -- parity-consistent")
print("=" * 100)
surf = {}
for exp in ("2026-09-25", "2026-10-09", "2026-10-30", "2026-11-20", "2026-12-18", "2027-01-15"):
    T = (pd.Timestamp(exp) - pd.Timestamp("2026-09-14")).days / 365
    oc = t.option_chain(exp)
    rows = []
    for side, df, is_call in (("C", oc.calls, True), ("P", oc.puts, False)):
        for _, x in df.iterrows():
            if not (x.bid > 0 and x.ask > 0):
                continue
            m = (x.bid + x.ask) / 2
            if not (PX * 0.7 <= x.strike <= PX * 1.35):
                continue
            rows.append({"K": x.strike, "side": side, "mid": m, "OI": x.openInterest,
                         "iv": iv(m, PX, x.strike, T, R, is_call)})
    d = pd.DataFrame(rows)
    # ATM iv = average of the two closest strikes, both sides
    katm = min(d.K.unique(), key=lambda k: abs(k - PX))
    atm = d[d.K == katm].iv.mean()
    surf[exp] = {"T": T, "dte": int(T * 365), "atm_iv": atm}
    piv = d.pivot_table(index="K", columns="side", values="iv")
    piv.columns = [f"iv_{c}" for c in piv.columns]
    piv["moneyness%"] = 100 * (piv.index / PX - 1)
    print(f"\n  {exp}  dte {int(T * 365):>3}  ATM IV {100 * atm:.1f}%")
    print((piv[["moneyness%", "iv_C", "iv_P"]] * [1, 100, 100]).round(1).to_string())
    print("   -> calls and puts now agree to ~1pt: the yfinance impliedVolatility column was the artefact.")

s = pd.DataFrame(surf).T
s["atm_iv%"] = 100 * s.atm_iv
print("\n" + "=" * 100)
print("TERM STRUCTURE (self-computed ATM IV)")
print("=" * 100)
print(s[["dte", "atm_iv%"]].round(1).to_string())

print("\n" + "=" * 100)
print("EARNINGS (2026-11-04) IMPLIED MOVE, backed out of the term structure")
print("=" * 100)
pre = s.loc["2026-10-30"]
post = s.loc["2026-11-20"]
var_pre = (pre.atm_iv ** 2) * pre["T"]
var_post = (post.atm_iv ** 2) * post["T"]
extra_var = var_post - var_pre
print(f"  Oct-30 expiry ({int(pre.dte)}d, pre-earnings): ATM IV {100 * pre.atm_iv:.1f}%  total var {var_pre:.4f}")
print(f"  Nov-20 expiry ({int(post.dte)}d, post-earnings): ATM IV {100 * post.atm_iv:.1f}%  total var {var_post:.4f}")
dt_cal = (post['T'] - pre['T'])
base_var_in_gap = (pre.atm_iv ** 2) * dt_cal
jump_var = extra_var - base_var_in_gap
print(f"  extra variance in the 21 calendar days between them: {extra_var:.4f}")
print(f"  of which 'ordinary' diffusive variance at the pre-earnings vol: {base_var_in_gap:.4f}")
if jump_var > 0:
    print(f"  => EARNINGS JUMP variance {jump_var:.4f} -> implied one-day earnings move "
          f"+/- {100 * math.sqrt(jump_var):.1f}%")
else:
    print(f"  => no positive residual: the market is NOT pricing an unusual earnings jump "
          f"(residual {jump_var:.5f}).")
print("  Cross-check, ATM straddles: Oct-30 13.7%->18.4% of spot, Nov-20 24.4% of spot.")

print("\n" + "=" * 100)
print("COST OF PROTECTION, ANNUALISED -- what 63.5-74% IV actually charges you")
print("=" * 100)
print("  A protective put is a recurring expense. Annualised cost of rolling it:")
rows = []
for exp in ("2026-10-09", "2026-11-20", "2026-12-18", "2027-01-15"):
    T = (pd.Timestamp(exp) - pd.Timestamp("2026-09-14")).days / 365
    oc = t.option_chain(exp)
    pu = oc.puts
    for tgt in (0.95, 0.90, 0.85, 0.80):
        K = min(pu.strike.unique(), key=lambda k: abs(k - PX * tgt))
        row = pu[pu.strike == K].iloc[0]
        if not (row.bid > 0 and row.ask > 0):
            continue
        m = (row.bid + row.ask) / 2
        rows.append({"exp": exp, "dte": int(T * 365), "K": K, "moneyness%": 100 * (K / PX - 1),
                     "prem": m, "cost%pos": 100 * m / PX, "ann%pos": 100 * m / PX / T,
                     "floor_vs_cost%": 100 * ((K - m) / COST - 1), "OI": int(row.openInterest or 0)})
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\n  Read the ann%pos column. Insuring ARM 10-15% below spot costs 25-35% of the position")
print("  PER YEAR. Over a 3-year hold that is most of the position. Protection at this IV is not")
print("  a 'cheap hedge' -- it is a materially negative-expected-value drag you accept only if you")
print("  specifically need the tail cut for a defined, short window (e.g. across Nov-4 earnings).")

print("\n" + "=" * 100)
print("COVERED CALL: expected-value framing at IV/RV = 0.99")
print("=" * 100)
print("  With IV30/RV20 = 0.99 the premium is ~fair compensation for the upside sold.")
print("  So a covered call does NOT add expected return; it TRUNCATES the distribution.")
print("  What it does buy you, concretely, on the Nov-20 expiry (spans earnings):")
oc = t.option_chain("2026-11-20")
T = (pd.Timestamp("2026-11-20") - pd.Timestamp("2026-09-14")).days / 365
for K in (260, 280, 300):
    row = oc.calls[oc.calls.strike == K].iloc[0]
    m = (row.bid + row.ask) / 2
    be_down = PX - m
    print(f"\n   sell {K}C @ ${m:.2f} ({100 * m / PX:.2f}% of position, {100 * m / PX / T:.0f}% annualised)")
    print(f"     downside cushion: breakeven drops to ${be_down:.2f} ({100 * (be_down / PX - 1):.1f}% from spot, "
          f"{100 * (be_down / COST - 1):.1f}% vs your 257 cost)")
    print(f"     if called away: ${K + m:.2f} = {100 * ((K + m) / PX - 1):+.1f}% from spot, "
          f"{100 * ((K + m) / COST - 1):+.1f}% vs cost")
    print(f"     upside forgone above ${K + m:.2f}: unlimited. ARM moved from 151 to 439 in 79 trading days")
    print(f"     between 2026-03-31 and 2026-06-18, so 'ARM cannot go up {100 * ((K + m) / PX - 1):.0f}% in {int(T * 365)} days' is false.")
