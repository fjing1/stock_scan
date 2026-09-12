#!/usr/bin/env python3
"""nvda_system.py — is NVDA a buy-and-hold, or can a swing/timing rule beat holding it?

Built on the aapl_system.py methodology, because NVDA is the same trap in a more extreme form:
it is the single best-performing large-cap of the era, so ANY rule fitted to that chart will
look good, and the only honest question is whether it beats simply holding. Guards:

  1. The benchmark is BUY-AND-HOLD NVDA — a brutal bar. A rule that halves the drawdown while
     giving up 10%/yr of CAGR is a failure, not a "risk-managed improvement".
  2. GENERALISATION: the winning rule is re-run unchanged on a semis/large-cap-tech peer basket
     that deliberately includes the era's LOSERS (INTC, MU, QCOM, CSCO, IBM). If it only works
     on NVDA it is fitted to one price path.
  3. ERA SPLITS: NVDA has been at least four different companies (gaming -> crypto -> pandemic
     -> AI datacentre). A rule must survive era-by-era, not just full-sample.
  4. AFTER-TAX: this is decisive here and usually omitted. Buy-and-hold defers one LTCG bill to
     the end; a swing rule realises gains every trade, mostly at SHORT-term rates. On a name
     compounding this fast, that gap dwarfs most timing alpha (cf. the BTC trend finding).
  5. Deflated Sharpe over every configuration tried, plus an annual-block bootstrap of the
     Sharpe difference vs buy-and-hold.

    ../../vcp_env/bin/python nvda_system.py --refresh
    ../../vcp_env/bin/python nvda_system.py --start 2015-01-01
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from rsi_obv_length_study import deflated_sharpe  # noqa: E402
from rsi_obv_slope_scan import resolve_partial_cutoff, trim_partial  # noqa: E402

CACHE = BACKEND_DIR / "_nvda_panel.pkl"
TD = 252
COST = 0.0005            # 5bps per turn (NVDA is among the most liquid names on the tape)
RISK_FREE = 0.0182
LTCG = 0.20              # long-term capital gains (held > 1 year)
STCG = 0.35              # short-term = ordinary income; a high-bracket US assumption

TARGET = "NVDA"
# Control group: the era's semis/tech LOSERS are in here on purpose.
PEERS = ["AMD", "AVGO", "MU", "QCOM", "TXN", "INTC", "MRVL", "AMAT", "LRCX", "KLAC",
         "CSCO", "IBM", "ORCL", "MSFT", "AAPL"]
CONTEXT = ["SPY", "QQQ", "SMH"]

# NVDA has been several different businesses; a rule should survive each.
ERAS = [("1999-2008 gaming/GPU", "1999-01-01", "2008-12-31"),
        ("2009-2015 post-GFC", "2009-01-01", "2015-12-31"),
        ("2016-2018 crypto boom/bust", "2016-01-01", "2018-12-31"),
        ("2019-2022 pandemic + bust", "2019-01-01", "2022-12-31"),
        ("2023-     AI datacentre", "2023-01-01", "2099-12-31")]


# ------------------------------------------------------------------ data
def refresh_cache() -> dict:
    import yfinance as yf
    out = {}
    for t in [TARGET] + PEERS + CONTEXT:
        h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
        if h.empty:
            print(f"  WARN no data {t}")
            continue
        if getattr(h.index, "tz", None) is not None:
            h.index = h.index.tz_localize(None)
        out[t] = h[["Open", "High", "Low", "Close", "Volume"]]
        print(f"  {t:<8} {h.index[0].date()} -> {h.index[-1].date()}  {len(h)} bars")
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def load_panel() -> dict:
    if not CACHE.exists():
        return refresh_cache()
    with open(CACHE, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ indicators
def rsi(x: pd.Series, n: int) -> pd.Series:
    d = x.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(df: pd.DataFrame, n: int = 22) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def banded_state(px: pd.Series, n: int, band: float = 0.02) -> pd.Series:
    """Hysteresis-banded trend: flips only when price clears the MA by +/-band. The band is
    what stops a 200MA rule from whipsawing itself to death on a 60%-vol stock."""
    ma = px.rolling(n).mean()
    st = pd.Series(np.nan, index=px.index)
    st[px > ma * (1 + band)] = 1.0
    st[px < ma * (1 - band)] = 0.0
    return st.ffill().fillna(1.0)


# ------------------------------------------------------------------ strategies -> position (0/1) on signal day
def s_hold(df):
    return pd.Series(1.0, index=df.index)


def s_ma(df, n, band=0.02):
    return banded_state(df["Close"], n, band)


def s_ensemble(df, lengths=(50, 100, 150, 200), band=0.02):
    votes = sum(banded_state(df["Close"], n, band) for n in lengths)
    return (votes >= len(lengths) / 2).astype(float)


def s_rsi_mr(df, lo=5, n=2, max_hold=10, trend=True):
    """Connors-style dip buy: RSI(n) < lo, only above the 200MA if trend=True.
    Exit on the first up-close or after max_hold bars."""
    c = df["Close"].values
    r = rsi(df["Close"], n).values
    s200 = df["Close"].rolling(200).mean().values
    pos = np.zeros(len(c))
    i = 200
    while i < len(c) - 1:
        ok = r[i] < lo and (not trend or (np.isfinite(s200[i]) and c[i] > s200[i]))
        if ok:
            j, held = i + 1, 0
            pos[i] = 1.0
            while j < len(c):
                held += 1
                pos[j] = 1.0
                if c[j] > c[j - 1] or held >= max_hold:
                    break
                j += 1
            i = j + 1
        else:
            i += 1
    return pd.Series(pos, index=df.index)


def s_combo(df, band=0.02):
    """Repo strategy #24: hold outright while above the 200MA; below it, only take
    mean-reversion dip bounces. The rule that beat buy-hold on SPY/QQQ."""
    trend = banded_state(df["Close"], 200, band)
    mr = s_rsi_mr(df, lo=5, n=2, trend=False)
    return ((trend > 0) | (mr > 0)).astype(float)


def s_atr_trail(df, mult=5.0, n=22, re_ma=50):
    """Hold, but exit on a mult x ATR trailing stop from the running high; re-enter when
    close reclaims the re_ma SMA. This is the repo's existing exit applied to a hold."""
    c = df["Close"].values
    a = atr(df, n).values
    ma = df["Close"].rolling(re_ma).mean().values
    pos = np.zeros(len(c))
    in_pos, peak = False, np.nan
    for i in range(1, len(c)):
        if not in_pos:
            if np.isfinite(ma[i]) and c[i] > ma[i]:
                in_pos, peak = True, c[i]
        else:
            peak = max(peak, c[i])
            if np.isfinite(a[i]) and c[i] < peak - mult * a[i]:
                in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    return pd.Series(pos, index=df.index)


def s_break3avg(df, fast=8, slow=22, exit_ma=22):
    """The repo's break3avg entry (the one signal with a documented real edge), held until
    close drops below the slow SMA."""
    o, h, l, c = (df[k].astype(float) for k in ("Open", "High", "Low", "Close"))
    p = (o + h + l + c) / 4
    f, s, s3 = p.rolling(fast).mean(), p.rolling(slow).mean(), p.rolling(3).mean()
    entry = ((c > f) & (c.shift(1) <= f.shift(1)) & (f > f.shift(1))
             & (c > s) & (c.shift(1) <= s.shift(1)) & (s3 > s))
    ex = c < c.rolling(exit_ma).mean()
    pos = pd.Series(np.where(entry, 1.0, np.where(ex, 0.0, np.nan)), index=df.index)
    return pos.ffill().fillna(0.0)


STRATEGIES = [
    ("BUY & HOLD", s_hold),
    ("MA50 banded", lambda d: s_ma(d, 50)),
    ("MA100 banded", lambda d: s_ma(d, 100)),
    ("MA150 banded", lambda d: s_ma(d, 150)),
    ("MA200 banded", lambda d: s_ma(d, 200)),
    ("MA250 banded", lambda d: s_ma(d, 250)),
    ("SMA ensemble vote", s_ensemble),
    ("COMBO #24 (200MA + MR)", s_combo),
    ("RSI(2)<5 MR, above 200MA", lambda d: s_rsi_mr(d, trend=True)),
    ("RSI(2)<5 MR, anywhere", lambda d: s_rsi_mr(d, trend=False)),
    ("ATR trail 5x, re-entry MA50", lambda d: s_atr_trail(d, 5.0)),
    ("ATR trail 3x, re-entry MA50", lambda d: s_atr_trail(d, 3.0)),
    ("break3avg 8/22", s_break3avg),
]


# ------------------------------------------------------------------ evaluation
def evaluate(df: pd.DataFrame, sig: pd.Series) -> tuple[pd.Series, dict]:
    """Daily net returns and headline metrics. Signal acts NEXT bar; cash earns the risk-free."""
    ret = df["Close"].pct_change().fillna(0.0)
    rf_d = (1 + RISK_FREE) ** (1 / TD) - 1
    pos = sig.shift(1).reindex(ret.index).fillna(0.0).clip(0, 1)
    turn = pos.diff().abs().fillna(0.0)
    net = pos * ret + (1 - pos) * rf_d - turn * COST
    eq = (1 + net).cumprod()
    yrs = len(net) / TD
    vol = net.std(ddof=1) * math.sqrt(TD)
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    return net, {"cagr": cagr * 100, "vol": vol * 100,
                 "maxdd": ((eq / eq.cummax()) - 1).min() * 100,
                 "sharpe": (cagr - RISK_FREE) / vol if vol > 0 else np.nan,
                 "inmkt": pos.mean() * 100,
                 "flips": int((pos.diff().abs() > 0).sum()),
                 "total": eq.iloc[-1]}


def after_tax_terminal(df: pd.DataFrame, sig: pd.Series) -> tuple[float, int, float]:
    """Terminal wealth after capital-gains tax, modelled trade by trade.

    Each completed round trip realises a gain taxed at STCG if held < 1 year, LTCG otherwise;
    losses accumulate and offset later gains. Buy-and-hold naturally pays one LTCG bill at the
    end on the whole move, which is the entire point of the comparison. Returns
    (after-tax terminal multiple, number of trades, share of gains taxed short-term)."""
    px = df["Close"]
    pos = sig.shift(1).reindex(px.index).fillna(0.0).clip(0, 1).values
    p = px.values
    dates = px.index
    rf_d = (1 + RISK_FREE) ** (1 / TD) - 1

    wealth, carry_loss = 1.0, 0.0
    st_gain = lt_gain = 0.0
    trades = 0
    i = 0
    while i < len(p):
        if pos[i] <= 0:
            wealth *= (1 + rf_d)
            i += 1
            continue
        # pos[i]==1 means the bar-i return p[i]/p[i-1] is EARNED, so the position was taken
        # at the close of bar i-1. Entry and exit prices must bracket exactly the earned bars,
        # otherwise the trade P&L silently disagrees with evaluate()'s daily compounding.
        first = i
        while i < len(p) and pos[i] > 0:
            i += 1
        last = i - 1                                   # last bar whose return was earned
        entry_px = p[first - 1] if first > 0 else p[0]
        gross = p[last] / entry_px
        still_open = last == len(p) - 1
        gross *= (1 - COST) ** (1 if still_open else 2)
        gain = wealth * (gross - 1)
        trades += 1
        held_days = (dates[last] - dates[max(first - 1, 0)]).days
        if gain > 0:
            offset = min(gain, carry_loss)
            carry_loss -= offset
            taxable = gain - offset
            rate = LTCG if held_days >= 365 else STCG
            if held_days >= 365:
                lt_gain += taxable
            else:
                st_gain += taxable
            wealth += gain - taxable * rate
        else:
            carry_loss += -gain
            wealth += gain
    tot = st_gain + lt_gain
    return wealth, trades, (st_gain / tot * 100 if tot > 0 else 0.0)


def block_bootstrap_sharpe_diff(net_a: pd.Series, net_b: pd.Series, n_boot=2000, seed=7):
    """Annual-block bootstrap of Sharpe(b) - Sharpe(a). Blocks by calendar year preserve the
    within-year autocorrelation that makes naive daily resampling far too optimistic."""
    rng = np.random.default_rng(seed)
    idx = net_a.index
    years = sorted({d.year for d in idx})
    pos_by_year = {y: np.where(idx.year == y)[0] for y in years}
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([pos_by_year[y] for y in rng.choice(years, len(years), replace=True)])
        a, b = net_a.values[sel], net_b.values[sel]
        sa, sb = a.std(ddof=1), b.std(ddof=1)
        if sa > 0 and sb > 0:
            out.append((b.mean() / sb - a.mean() / sa) * math.sqrt(TD))
    return np.array(out)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download the price panel")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--include-partial", action="store_true")
    args = ap.parse_args()

    panel = refresh_cache() if args.refresh else load_panel()
    cutoff = resolve_partial_cutoff(args.include_partial)

    def frame(t):
        d = panel[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
        d = trim_partial(d, cutoff)
        return d[d.index >= args.start]

    nv = frame(TARGET)
    print(f"\n{TARGET}: {nv.index[0].date()} .. {nv.index[-1].date()}  ({len(nv)} bars, "
          f"{len(nv)/TD:.1f}y)   costs {COST*1e4:.0f}bps/turn")

    # ---------------- baseline character
    bh_net, bh_m = evaluate(nv, s_hold(nv))
    c = nv["Close"]
    dd = (c / c.cummax() - 1)
    print(f"\nBUY & HOLD: {bh_m['total']:,.0f}x   CAGR {bh_m['cagr']:.1f}%   vol {bh_m['vol']:.0f}%   "
          f"maxDD {bh_m['maxdd']:.0f}%   Sharpe {bh_m['sharpe']:.2f}")
    print("  the path you have to survive — worst drawdowns:")
    for lo, hi in [(-1.0, -0.75), (-0.75, -0.55), (-0.55, -0.35)]:
        n_days = int(((dd <= hi) & (dd > lo)).sum())
        if n_days:
            print(f"    {abs(hi)*100:.0f}%+ below the high on {n_days:,} days "
                  f"({n_days/len(dd)*100:.1f}% of the time)")

    # ---------------- strategy comparison
    print(f"\n{'='*104}\nPRE-TAX: every strategy vs buy-and-hold (full sample)\n{'='*104}")
    print(f"  {'strategy':<30}{'mult':>12}{'CAGR':>8}{'vol':>7}{'maxDD':>8}{'Sharpe':>8}"
          f"{'in mkt':>8}{'trades':>8}{'vs B&H CAGR':>13}")
    results = {}
    for name, fn in STRATEGIES:
        sig = fn(nv)
        net, m = evaluate(nv, sig)
        results[name] = (sig, net, m)
        print(f"  {name:<30}{m['total']:>11,.0f}x{m['cagr']:>8.1f}{m['vol']:>7.0f}{m['maxdd']:>8.0f}"
              f"{m['sharpe']:>8.2f}{m['inmkt']:>8.0f}{m['flips']:>8}"
              f"{m['cagr']-bh_m['cagr']:>+13.1f}")

    beat_cagr = [n for n, (_, _, m) in results.items() if n != "BUY & HOLD" and m["cagr"] > bh_m["cagr"]]
    beat_sh = [n for n, (_, _, m) in results.items() if n != "BUY & HOLD" and m["sharpe"] > bh_m["sharpe"]]
    print(f"\n  beat buy-hold on CAGR : {beat_cagr or 'NONE'}")
    print(f"  beat buy-hold on Sharpe: {beat_sh or 'NONE'}")

    # ---------------- era stability
    print(f"\n{'='*104}\nERA STABILITY — CAGR by regime (NVDA has been four different companies)\n{'='*104}")
    hdr = f"  {'strategy':<30}" + "".join([f"{e[0].split()[0]:>13}" for e in ERAS])
    print(hdr)
    for name, (sig, _, _) in results.items():
        cells = []
        for _, a, b in ERAS:
            sub = nv[(nv.index >= a) & (nv.index <= b)]
            if len(sub) < 120:
                cells.append(f"{'—':>13}")
                continue
            _, m = evaluate(sub, sig.reindex(sub.index))
            cells.append(f"{m['cagr']:>12.0f}%")
        print(f"  {name:<30}" + "".join(cells))

    # ---------------- after tax
    print(f"\n{'='*104}\nAFTER-TAX (LTCG {LTCG:.0%} / STCG {STCG:.0%}, trade-by-trade, losses carried)\n{'='*104}")
    print(f"  {'strategy':<30}{'pre-tax':>12}{'after-tax':>12}{'tax drag':>11}"
          f"{'trades':>8}{'% gains ST':>12}{'after-tax CAGR':>16}")
    yrs = len(nv) / TD
    tax_rows = []
    for name, (sig, _, m) in results.items():
        w, n_tr, st_share = after_tax_terminal(nv, sig)
        cagr_at = w ** (1 / yrs) - 1
        drag = (m["cagr"] / 100) - cagr_at
        tax_rows.append((name, m["total"], w, cagr_at))
        print(f"  {name:<30}{m['total']:>11,.0f}x{w:>11,.0f}x{drag*100:>10.1f}%"
              f"{n_tr:>8}{st_share:>11.0f}%{cagr_at*100:>15.1f}%")
    best_at = max(tax_rows, key=lambda r: r[3])
    print(f"\n  -> highest AFTER-TAX CAGR: {best_at[0]}  ({best_at[3]*100:.1f}%/yr, {best_at[2]:,.0f}x)")

    # ---------------- best challenger vs hold: bootstrap + DSR
    challengers = {n: v for n, v in results.items() if n != "BUY & HOLD"}
    best_name = max(challengers, key=lambda n: challengers[n][2]["sharpe"])
    best_sig, best_net, best_m = challengers[best_name]
    print(f"\n{'='*104}\nIS THE BEST CHALLENGER REAL?  ({best_name})\n{'='*104}")
    boot = block_bootstrap_sharpe_diff(bh_net, best_net)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  annual-block bootstrap of Sharpe({best_name}) - Sharpe(buy-hold), 2000 draws:")
    print(f"    mean {boot.mean():+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"P(beats hold) {(boot > 0).mean()*100:.0f}%")
    sharpes = np.array([m["sharpe"] for _, _, m in results.values() if np.isfinite(m["sharpe"])])
    dsr = deflated_sharpe(best_m["sharpe"], len(best_net), len(sharpes),
                          float(np.var(sharpes, ddof=1)), float(best_net.skew()),
                          float(best_net.kurtosis() + 3.0))
    print(f"  Deflated Sharpe over the {len(sharpes)} configurations tried: {dsr:.3f} "
          f"({'survives' if dsr > 0.95 else 'does NOT survive'} the multiple-testing haircut)")

    # ---------------- generalisation to peers
    print(f"\n{'='*104}\nGENERALISATION — the same rule, unchanged, on peers (losers included)\n{'='*104}")
    print(f"  {'name':<8}{'B&H CAGR':>11}{'rule CAGR':>11}{'delta':>9}{'B&H Sh':>9}{'rule Sh':>9}{'better?':>9}")
    wins = tot = 0
    for t in PEERS:
        if t not in panel:
            continue
        d = frame(t)
        if len(d) < 5 * TD:
            continue
        _, mh = evaluate(d, s_hold(d))
        rule_fn = dict(STRATEGIES)[best_name]
        _, mr_ = evaluate(d, rule_fn(d))
        tot += 1
        better = mr_["cagr"] > mh["cagr"]
        wins += int(better)
        print(f"  {t:<8}{mh['cagr']:>10.1f}%{mr_['cagr']:>10.1f}%{mr_['cagr']-mh['cagr']:>+9.1f}"
              f"{mh['sharpe']:>9.2f}{mr_['sharpe']:>9.2f}{('Y' if better else 'n'):>9}")
    print(f"  -> '{best_name}' beat buy-hold CAGR on {wins}/{tot} peers")

    # ---------------- where NVDA stands right now
    print(f"\n{'='*104}\nCURRENT STATE — what each rule says on the last closed bar "
          f"({nv.index[-1].date()})\n{'='*104}")
    last = nv["Close"].iloc[-1]
    print(f"\n  close {last:,.2f}   {(last/nv['Close'].cummax().iloc[-1]-1)*100:+.1f}% from all-time high"
          f"   ATR22 {atr(nv).iloc[-1]:,.2f} ({atr(nv).iloc[-1]/last*100:.1f}% of price)")
    for n in (50, 100, 200, 250):
        ma_n = nv["Close"].rolling(n).mean().iloc[-1]
        print(f"    MA{n:<4} {ma_n:>10,.2f}   price is {(last/ma_n-1)*100:+6.1f}% vs it")
    print(f"\n  {'strategy':<30}{'position now':>14}")
    for name, (sig, _, _) in results.items():
        print(f"  {name:<30}{('LONG' if sig.iloc[-1] > 0 else 'cash'):>14}")

    print(f"\n{'='*104}")
    print("Caveats: close-to-close, long-only + cash at the risk-free, 5bps/turn, no slippage on")
    print("gaps, no options/leverage. NVDA is chosen WITH HINDSIGHT as a winner — that alone makes")
    print("buy-and-hold hard to beat and makes any 'edge' found here suspect. The tax model assumes")
    print("a taxable account at top US rates; in an IRA/401k the after-tax column collapses to the")
    print("pre-tax one and timing rules look relatively better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
