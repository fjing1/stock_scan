"""ARM: the arithmetic of each option the holder actually faces.

Price/vol/chain: yfinance live, 2026-09-14.
Position: cost 257.00, price 239.01.
Repo conventions: ATR22 5x trailing stop; partial in-progress bar dropped before stats.
"""
import numpy as np
import pandas as pd
import yfinance as yf

pd.set_option("display.width", 240)
rng = np.random.default_rng(7)

PX = 239.01
COST = 257.00
ATR22 = 14.95        # given, measured by the repo
STOP = 204.70        # repo 5xATR22 trailing stop, given

px = pd.read_pickle("_arm_px.pkl")
c = px["Close"].copy()
# repo convention: drop the in-progress bar. 2026-09-14 volume was non-round -> partial.
c_closed = c.iloc[:-1]
r = np.log(c_closed).diff().dropna()
print("=" * 104)
print("MOVE MODEL -- empirical block bootstrap on ARM's own closed daily bars")
print("=" * 104)
print(f"  closed bars used: {len(r)}  ({r.index[0].date()} .. {r.index[-1].date()})")
print(f"  daily sd {r.std():.4f} -> annualised {100 * r.std() * np.sqrt(252):.1f}%")
print(f"  last-60d daily sd {r.tail(60).std():.4f} -> annualised {100 * r.tail(60).std() * np.sqrt(252):.1f}%")
print(f"  ATR22 given = {ATR22:.2f} = {100 * ATR22 / PX:.1f}% of price")

# stationary block bootstrap, blocks of 5, using the trailing 2y of returns (regime-relevant)
pool = r.tail(504).values
NSIM, HMAX = 200_000, 21
paths = np.empty((NSIM, HMAX))
B = 5
nb = int(np.ceil(HMAX / B))
for k in range(nb):
    starts = rng.integers(0, len(pool) - B, NSIM)
    blk = np.stack([pool[s:s + B] for s in starts])
    lo, hi = k * B, min((k + 1) * B, HMAX)
    paths[:, lo:hi] = blk[:, : hi - lo]
# demean to zero drift (we are NOT forecasting direction)
paths = paths - pool.mean()
cum = np.cumsum(paths, axis=1)
S = PX * np.exp(cum)

print("\n  bootstrap horizons (zero-drift):")
hdr = f"    {'h':>3} {'sd%':>7} {'P<257':>7} {'p05':>9} {'p25':>9} {'p50':>9} {'p75':>9} {'p95':>9} {'E|move|%':>9}"
print(hdr)
for h in (5, 10, 21):
    s = S[:, h - 1]
    q = np.percentile(s, [5, 25, 50, 75, 95])
    print(f"    {h:>3} {100 * cum[:, h - 1].std():>7.1f} {100 * (s < COST).mean():>6.1f}% "
          f"{q[0]:>9.2f} {q[1]:>9.2f} {q[2]:>9.2f} {q[3]:>9.2f} {q[4]:>9.2f} "
          f"{100 * np.abs(s / PX - 1).mean():>9.1f}")
print("\n  For reference, the numbers already measured by the repo's calibrated model:")
print("    P(still below 257) = 76.7% / 67.6% / 59.2% at 5 / 10 / 21 days")
print("    typical 21-day move 14.1%; P(|move|>10% in 21d) = 60.9%; 5th pct at 21d = -28.8%")
print("  My independent bootstrap is in the same family; use the repo model's headline figures,")
print("  and use the bootstrap only for the TOUCH probabilities below (which it can compute and")
print("  a terminal-distribution model cannot).")

print("\n" + "=" * 104)
print("1. THE SUNK-COST FRAME")
print("=" * 104)
print(f"  Cost {COST:.2f} is 92.3rd pctile of ARM's entire post-IPO closing history. It is a fact about")
print(f"  the past. The only live question is whether ARM at {PX:.2f} beats the alternatives.")
print(f"  'Wait for breakeven' requires +{100 * (COST / PX - 1):.1f}% from here.")
print(f"  Repo model P(still below {COST:.0f}): 76.7% @5d, 67.6% @10d, 59.2% @21d.")
print(f"  Complement -- P(back above cost): 23.3% @5d, 32.4% @10d, 40.8% @21d.")
print(f"  So the single most likely outcome of waiting one month is: still underwater, with a")
print(f"  {100 * np.abs(S[:, 20] / PX - 1).mean():.1f}% typical absolute move in between.")
print(f"  Symmetry check: the same {100 * (COST / PX - 1):.1f}% is available from ANY stock. The 257 anchor gives")
print("  ARM no advantage in earning it.")

print("\n" + "=" * 104)
print("2. POSITION SIZING ARITHMETIC")
print("=" * 104)
p05_21 = np.percentile(S[:, 20], 5)
adverse = -28.8  # repo model's 5th pct at 21d, %
print(f"  ATR22 = {ATR22:.2f} = {100 * ATR22 / PX:.1f}% of price. Typical 21-day move 14.1%. "
      f"P(|move|>10% in 21d) = 60.9%.")
print(f"  Adverse case used: repo model 5th percentile at 21 days = {adverse:.1f}%  "
      f"(my bootstrap 5th pct: {100 * (p05_21 / PX - 1):.1f}%)")
print(f"\n  {'port weight':>12} {'$100k port':>12} {'$500k port':>12} {'$1m port':>12} {'% of PORTFOLIO at 5th pct':>28}")
for w in (0.02, 0.05, 0.10, 0.20):
    dollars = [w * p * adverse / 100 for p in (100_000, 500_000, 1_000_000)]
    print(f"  {w:>11.0%} {dollars[0]:>12,.0f} {dollars[1]:>12,.0f} {dollars[2]:>12,.0f} "
          f"{w * adverse:>27.2f}%")
print("\n  Read that last column: at a 5% weight a 1-in-20 bad month costs 1.44% of the portfolio.")
print("  At 20% it costs 5.76%. Same stock, same volatility -- 4x the pain.")
print("  Also: a 20% weight in a 63.5%-IV name contributes roughly 0.20 x 63.5% = 12.7% of")
print("  standalone vol to the portfolio before any diversification credit.")
print("\n  Weight at which the 21d 5th-pct loss equals a chosen portfolio pain budget:")
for budget in (1.0, 2.0, 3.0, 5.0):
    print(f"    tolerate {budget:.0f}% portfolio drawdown at the 21d 5th pct -> max ARM weight "
          f"{100 * budget / abs(adverse):.1f}%")
print("\n  DIAGNOSTIC QUESTION: if the position were 2% instead of its current size, would the")
print("  -7% still feel like it needs a decision? If no, the problem is SIZE, not ARM.")

print("\n" + "=" * 104)
print("4. THE STOP ARITHMETIC  (repo 5xATR22 trailing stop = 204.70)")
print("=" * 104)
print(f"  Stop {STOP:.2f} = {100 * (STOP / PX - 1):.1f}% from {PX:.2f} and {100 * (STOP / COST - 1):.1f}% from cost {COST:.2f}")
print(f"  NOTE / discrepancy: the brief states the stop is -16.8% from here. 204.70/239.01-1 = "
      f"{100 * (STOP / PX - 1):.1f}%.")
print(f"  -16.8% implies a reference price of {STOP / (1 - 0.168):.2f}, i.e. the figure was computed before")
print(f"  today's -9.7% bar. The -20.4% vs cost checks out exactly ({100 * (STOP / COST - 1):.2f}%).")
print(f"  Using the live 239.01, the stop is {100 * (STOP / PX - 1):.1f}% away = {(PX - STOP) / ATR22:.2f} x ATR22.")
touch21 = (S[:, :21].min(axis=1) <= STOP).mean()
touch10 = (S[:, :10].min(axis=1) <= STOP).mean()
touch5 = (S[:, :5].min(axis=1) <= STOP).mean()
closeblw21 = (S[:, 20] <= STOP).mean()
print(f"\n  bootstrap P(TOUCH {STOP:.2f} intraday-equivalent, on closes):  5d {100 * touch5:.1f}%   "
      f"10d {100 * touch10:.1f}%   21d {100 * touch21:.1f}%")
print(f"  bootstrap P(CLOSE below {STOP:.2f} at day 21):                  {100 * closeblw21:.1f}%")
print(f"  => the touch probability is {touch21 / closeblw21:.2f}x the close-below probability. A stop is hit by the")
print("     PATH, not the endpoint -- that gap is the whole reason stops get run on high-vol names.")
print("  Using true intraday lows would raise the touch number further; these are close-to-close paths,")
print("  so treat the figures as a FLOOR on the true touch probability.")
print(f"\n  {'port weight':>12} {'loss @stop vs price':>22} {'loss @stop vs cost':>22} {'$ on $500k port':>18}")
for w in (0.02, 0.05, 0.10, 0.20):
    lp = w * 100 * (STOP / PX - 1)
    lc = w * 100 * (STOP / COST - 1)
    print(f"  {w:>11.0%} {lp:>21.2f}% {lc:>21.2f}% {w * 500_000 * (STOP / COST - 1):>18,.0f}")
print(f"\n  The trap: a {100 * touch21:.0f}%-of-the-time touch over 21 days with ZERO assumed drift. On top of that")
print("  these are CLOSE-to-close paths; ARM's average daily high-low range is")
print(f"  {100 * ((px['High'] - px['Low']) / px['Close']).tail(60).mean():.1f}% of price over the last 60 bars, so the true intraday touch rate is")
print("  materially higher. A mechanical 5xATR stop on a 74%-vol name gets run on path noise.")
print("  Either widen it, size down so you do not need it, or accept getting stopped out often.")

# ---------- OPTIONS ----------
print("\n" + "=" * 104)
print("3. THE OPTIONS ARITHMETIC -- LIVE CHAIN, yfinance, 2026-09-14")
print("=" * 104)
t = yf.Ticker("ARM")
exps = t.options
print(f"  expiries available: {exps[:14]}")
IV30, IV60, RV20 = 63.5, 68.6, 63.9
print(f"  IV30 {IV30}%  IV60 {IV60}%  RV20 {RV20}%  ->  IV30/RV20 = {IV30 / RV20:.2f}")
print("  IV/RV = 0.99 means options are priced at FAIR value vs recent realised vol.")
print("  You are NOT being paid a premium to sell vol, and you are NOT getting a discount to buy it.")
print("  You are paying/receiving roughly the statistically correct price. So the option choice is")
print("  about RESHAPING the payoff, not about capturing an edge.")


def chain(exp):
    o = t.option_chain(exp)
    return o.calls, o.puts


targets = []
today = pd.Timestamp("2026-09-14")
for e in exps:
    dte = (pd.Timestamp(e) - today).days
    if dte in range(25, 40) or dte in range(55, 80) or dte in range(80, 130):
        targets.append((e, dte))
seen = set()
picks = []
for e, d in targets:
    band = "30d" if d < 45 else ("60d" if d < 80 else "90d+")
    if band in seen:
        continue
    seen.add(band)
    picks.append((e, d, band))
print(f"\n  expiries chosen: {[(e, d, b) for e, d, b in picks]}")

for exp, dte, band in picks:
    calls, puts = chain(exp)
    for df in (calls, puts):
        df["mid"] = np.where(
            (df.bid > 0) & (df.ask > 0), (df.bid + df.ask) / 2,
            df.lastPrice)
    print("\n" + "-" * 104)
    print(f"  EXPIRY {exp}  ({dte} days)   [{band}]")
    print("-" * 104)

    print("  (a) COVERED CALL -- sell upside, collect premium")
    print(f"    {'strike':>8} {'%OTM':>7} {'bid':>7} {'ask':>7} {'mid':>7} {'prem%px':>8} {'ann%':>7} "
          f"{'OI':>7} {'IV%':>6} {'called-away tot ret from 239':>28} {'from cost 257':>15}")
    cc = calls[(calls.strike >= PX * 0.98) & (calls.strike <= PX * 1.45)].sort_values("strike")
    for _, row in cc.iterrows():
        k, mid = row.strike, row["mid"]
        if not mid or mid <= 0:
            continue
        prem = 100 * mid / PX
        ann = prem * 365 / dte
        tot_px = 100 * ((k + mid) / PX - 1)
        tot_cost = 100 * ((k + mid) / COST - 1)
        print(f"    {k:>8.0f} {100 * (k / PX - 1):>6.1f}% {row.bid:>7.2f} {row.ask:>7.2f} {mid:>7.2f} "
              f"{prem:>7.2f}% {ann:>6.1f}% {int(row.openInterest or 0):>7} {100 * (row.impliedVolatility or 0):>5.0f} "
              f"{tot_px:>27.2f}% {tot_cost:>14.2f}%")

    print("\n  (b) PROTECTIVE PUT -- pay to cap the left tail")
    print(f"    {'strike':>8} {'%OTM':>7} {'bid':>7} {'ask':>7} {'mid':>7} {'cost%px':>8} {'ann%':>7} "
          f"{'OI':>7} {'IV%':>6} {'floor(px-prem)':>15} {'worst vs cost':>14}")
    pp = puts[(puts.strike >= PX * 0.66) & (puts.strike <= PX * 1.02)].sort_values("strike", ascending=False)
    for _, row in pp.iterrows():
        k, mid = row.strike, row["mid"]
        if not mid or mid <= 0:
            continue
        cost_pct = 100 * mid / PX
        floor = k - mid
        print(f"    {k:>8.0f} {100 * (k / PX - 1):>6.1f}% {row.bid:>7.2f} {row.ask:>7.2f} {mid:>7.2f} "
              f"{cost_pct:>7.2f}% {cost_pct * 365 / dte:>6.1f}% {int(row.openInterest or 0):>7} "
              f"{100 * (row.impliedVolatility or 0):>5.0f} {floor:>15.2f} {100 * (floor / COST - 1):>13.2f}%")

    print("\n  (c) COLLAR -- put financed by call. net debit(+)/credit(-) as % of price")
    pk_list = [k for k in sorted(puts.strike.unique()) if PX * 0.80 <= k <= PX * 0.96]
    ck_list = [k for k in sorted(calls.strike.unique()) if PX * 1.04 <= k <= PX * 1.35]
    pmid = {r.strike: r["mid"] for _, r in puts.iterrows() if r["mid"] and r["mid"] > 0}
    cmid = {r.strike: r["mid"] for _, r in calls.iterrows() if r["mid"] and r["mid"] > 0}
    print(f"    {'put K':>7} {'call K':>7} {'put$':>7} {'call$':>7} {'net$':>8} {'net%px':>8} "
          f"{'floor':>8} {'cap':>8} {'max loss vs cost':>17} {'max gain vs cost':>17}")
    for pk in pk_list:
        if pk not in pmid:
            continue
        for ck in ck_list:
            if ck not in cmid:
                continue
            net = pmid[pk] - cmid[ck]
            if abs(net) > PX * 0.035:
                continue
            floor, cap = pk - net, ck - net
            print(f"    {pk:>7.0f} {ck:>7.0f} {pmid[pk]:>7.2f} {cmid[ck]:>7.2f} {net:>8.2f} "
                  f"{100 * net / PX:>7.2f}% {floor:>8.2f} {cap:>8.2f} "
                  f"{100 * (floor / COST - 1):>16.2f}% {100 * (cap / COST - 1):>16.2f}%")
