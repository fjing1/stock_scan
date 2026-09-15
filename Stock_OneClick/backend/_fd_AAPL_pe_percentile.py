"""_fd_AAPL_pe_percentile.py -- AAPL's trailing multiple percentile, done correctly.

THREE BUGS FOUND AND FIXED IN GETTING HERE, each of which produced a plausible but wrong answer:

  1. Summing four quarterly rows for TTM. Apple's 10-K carries no ~90-day Q4 duration fact, so the
     last four quarterly rows span 15 months with Q4 missing. fund_metrics.py guards revenue with a
     250-300 day span test (so revenue_ttm silently falls back to the 3-quarter-stale fiscal year)
     but applies NO guard to EPS, so its P/E of 39.5x divides by a gapped 8.44 instead of 8.71.

  2. Split contamination. yfinance's unadjusted Close is split-adjusted; XBRL EPS is as-filed.
     Apple split 7:1 (2014-06-09) and 4:1 (2020-08-31), so as-filed EPS from before a split is 28x
     or 4x too large. Mixing bases inside one TTM subtraction produced NEGATIVE trailing P/E and a
     nonsense 2021 median of -62x. Fixed by rescaling each fact by the cumulative split factor that
     applied AFTER its filing date -- not by discarding data.

  3. Restatement lookahead. Taking the LAST-filed revision of each period (to dodge bug 2) dates
     every figure by a republication up to a year later, so 2021 prices were being divided by 2020
     earnings, inflating the 2021 median P/E to 52x. Point-in-time requires the FIRST-filed
     revision, which is what this script uses.

The percentile is therefore built from: first-filed facts only (no lookahead), rescaled for splits,
TTM constructed as FY + latest YTD - prior-year YTD (immune to the missing Q4).
"""
import numpy as np
import pandas as pd
import yfinance as yf

import _fund_data as F

pd.set_option("display.width", 220)
SYM, PRICE = "AAPL", 333.08
EPS_TTM, REV_TTM, SH_NOW = 8.71, 466_823e6, 14_594_180_000
ONE_OFF_TAX, ONE_OFF_Q = 10_175e6, pd.Timestamp("2024-09-28")
panel = F.load()

# Apple split history. A fact FILED before a split date reports per-share figures on the pre-split
# basis, so it must be divided by every split factor that came after it.
SPLITS = [(pd.Timestamp("2014-06-09"), 7.0), (pd.Timestamp("2020-08-31"), 4.0)]


def split_factor(filed):
    f = 1.0
    for d, r in SPLITS:
        if filed < d:
            f *= r
    return f


def dur(concept, per_share=False):
    d = panel[(panel.symbol == SYM) & (panel.concept == concept)].copy()
    d = d[d["start"].notna()].copy()
    d["days"] = (d["end"] - d["start"]).dt.days
    d["val"] = d["val"].astype(float)
    if per_share:
        d["val"] = d["val"] / d["filed"].map(split_factor)
    elif concept == "shares_diluted":
        d["val"] = d["val"] * d["filed"].map(split_factor)
    # POINT IN TIME: first filing that disclosed each period
    return d.sort_values(["filed"]).drop_duplicates(["start", "end"], keep="first") \
            .sort_values(["end", "days"])


def ttm_timeline(concept, per_share=False):
    d = dur(concept, per_share)
    ann = d[(d.days >= 340) & (d.days <= 400)].sort_values("end")
    ev = [(r.filed, r.val, r.start, r.end, "FY") for _, r in ann.iterrows()]
    for _, r in d[(d.days >= 60) & (d.days <= 320)].iterrows():
        pf = ann[ann.end < r.start]
        if pf.empty:
            continue
        fy = pf.iloc[-1]
        pool = d[(d.start.between(fy.start - pd.Timedelta(days=7), fy.start + pd.Timedelta(days=7))) &
                 (d.days.between(r.days - 12, r.days + 12))]
        if pool.empty:
            continue
        ev.append((r.filed, fy.val + r.val - pool.val.iloc[-1],
                   r.end - pd.Timedelta(days=364), r.end, f"{r.days}d YTD"))
    e = pd.DataFrame(ev, columns=["filed", "ttm", "win_start", "period_end", "how"])
    return e.sort_values(["filed", "period_end"]).drop_duplicates("filed", keep="last").set_index("filed")


eps_t = ttm_timeline("eps_diluted", per_share=True)
rev_t = ttm_timeline("revenue")
shq = dur("shares_diluted")
shq = shq[(shq.days >= 60) & (shq.days <= 120)]
sh_t = shq.sort_values(["filed", "end"]).drop_duplicates("filed", keep="last").set_index("filed")["val"]

print("=" * 128)
print("POINT-IN-TIME TTM EPS TIMELINE (split-rescaled, first-filed only) -- spot-check")
print("=" * 128)
chk = eps_t.copy()
chk["price_that_day"] = np.nan
print(chk[["ttm", "period_end", "how"]].iloc[::4].tail(14).to_string(float_format=lambda x: f"{x:,.2f}"))
print(f"\n  latest: TTM EPS {eps_t['ttm'].iloc[-1]:.2f} for period ending "
      f"{eps_t['period_end'].iloc[-1].date()}, filed {eps_t.index[-1].date()}  "
      f"(independent derivation: {EPS_TTM})")

eps_t["has_oneoff"] = (eps_t.win_start <= ONE_OFF_Q) & (eps_t.period_end >= ONE_OFF_Q)
eps_t["sh"] = sh_t.reindex(eps_t.index, method="ffill")
eps_t["ttm_n"] = np.where(eps_t.has_oneoff, eps_t.ttm + ONE_OFF_TAX / eps_t.sh, eps_t.ttm)

raw = yf.download(SYM, start="2010-01-01", progress=False, auto_adjust=False)
close = raw["Close"][SYM] if isinstance(raw["Close"], pd.DataFrame) else raw["Close"]
close.index = close.index.tz_localize(None)

h = pd.DataFrame({"close": close})
for c, s in [("eps", eps_t["ttm"]), ("eps_n", eps_t["ttm_n"]), ("rev", rev_t["ttm"]), ("sh", sh_t)]:
    h[c] = s.reindex(h.index, method="ffill")
h = h.dropna()
h["pe"], h["pe_n"] = h.close / h.eps, h.close / h.eps_n
h["ps"] = h.close / (h.rev / h.sh)
bad = (h.pe <= 0) | (h.pe > 150)
print(f"\n  rows with implausible P/E after the split fix: {bad.sum()} of {len(h)}")
h = h[~bad]

now = {"pe": PRICE / EPS_TTM, "pe_n": PRICE / EPS_TTM, "ps": PRICE / (REV_TTM / SH_NOW)}
for win, lbl in [("2010-01-01", "full history 2010->today"), ("2016-09-14", "last 10 years"),
                 ("2021-09-14", "last 5 years"), ("2023-09-14", "last 3 years")]:
    w = h[h.index >= win]
    print("\n" + "=" * 128)
    print(f"{lbl}   n={len(w)}   {w.index[0].date()} -> {w.index[-1].date()}")
    print("=" * 128)
    for k, name in [("pe", "trailing GAAP P/E as reported"),
                    ("pe_n", "trailing GAAP P/E, one-off tax added back"),
                    ("ps", "price / sales per share (tax-immune)")]:
        s = w[k]
        print(f"  {name:42s} today {now[k]:6.2f} | med {s.median():6.2f} p25 {s.quantile(.25):6.2f} "
              f"p75 {s.quantile(.75):6.2f} p90 {s.quantile(.90):6.2f} max {s.max():6.2f} "
              f"| PCTILE {(s < now[k]).mean()*100:5.1f}  vs med {now[k]/s.median()-1:+6.1%}")

print("\n" + "=" * 128)
print("MEDIAN BY CALENDAR YEAR (sanity: these should track the well-known re-rating story)")
print("=" * 128)
print(h.groupby(h.index.year)[["pe", "pe_n", "ps", "close"]].median()
      .to_string(float_format=lambda x: f"{x:,.1f}"))
h.to_pickle("_fd_AAPL_pe_history.pkl")
print("\nsaved _fd_AAPL_pe_history.pkl")
