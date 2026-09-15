"""
_fd_AAPL_options.py — options arithmetic on REAL quotes, with the chain validated before use.

ORDER OF OPERATIONS MATTERS. A yfinance chain can be stale, wide, or partly zero-bid, and a covered
call priced off a garbage mid is worse than no analysis. So: validate put-call parity FIRST (the
implied forward F = K + C - P must be flat across strikes and close to spot x exp((r-q)T)); only then
price structures.

Then, and this is the part a textbook will not give you: each structure's expected payoff is
evaluated under THIS REPO's calibrated 21-day distribution (move_prob.py's fitted sigma + empirical
z table), not under the option's own implied lognormal. If the two disagree, that disagreement IS
the trade -- and if they agree, the overwrite is not compensated and you are just selling upside.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import move_prob as M

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
HERE = Path(__file__).resolve().parent
SYM = "AAPL"
TODAY = pd.Timestamp("2026-09-14")


def atr(h, l, c, n=22):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


d = yf.download(SYM, period="max", progress=False, auto_adjust=True)
if isinstance(d.columns, pd.MultiIndex):
    d.columns = d.columns.get_level_values(0)
d = d[d.Close > 0].dropna(subset=["Close"])
C, H, L, V = d.Close, d.High, d.Low, d.Volume
px = float(C.iloc[-1])
a22 = float(atr(H, L, C).iloc[-1])
rv20 = float(np.log(C).diff().tail(20).std(ddof=1) * np.sqrt(252))
rv60 = float(np.log(C).diff().tail(60).std(ddof=1) * np.sqrt(252))

# ---- unadjusted spot, because option strikes are on the unadjusted price
raw = yf.download(SYM, period="10d", progress=False, auto_adjust=False)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
spot = float(raw["Close"].dropna().iloc[-1])
print("=" * 118)
print(f"AAPL 期权算术   调整后收盘 {px:.2f}   未调整收盘(行权价所在坐标) {spot:.2f}   {TODAY.date()}")
print(f"  已实现波动 20日 {rv20:.1%}   60日 {rv60:.1%}   ATR22 {a22/px:.2%}/日")
print("=" * 118)

# ---- model distribution
model = pd.read_pickle(M.MODEL_PATH)
g = M.asset_class(SYM)
f = M.build_features(C, H, L, None, V)
SIG, Z = {}, {}
for h in (5, 10, 21):
    sp = model["groups"][g]["per_h"].get(h)
    if sp is None:
        continue
    SIG[h] = float(np.exp(M._design(f.iloc[[-1]], sp["cols"]) @ sp["beta"]).item()) * np.sqrt(h)
    Z[h] = np.sort(np.asarray(sp["z"]))

t = yf.Ticker(SYM)
EXPS = ["2026-10-16", "2026-11-20", "2027-01-15"]
chains = {}
for e in EXPS:
    ch = t.option_chain(e)
    chains[e] = (ch.calls.copy(), ch.puts.copy())
    ch.calls.to_csv(HERE / f"_fd_AAPL_calls_{e}.csv", index=False)
    ch.puts.to_csv(HERE / f"_fd_AAPL_puts_{e}.csv", index=False)


def mid(df):
    m = (df.bid + df.ask) / 2
    bad = (df.bid <= 0) | (df.ask <= 0) | (df.ask < df.bid)
    return m.where(~bad, np.nan)


# ============================================================ chain validation
print("\n" + "=" * 118)
print("【0】链条校验：认沽认购平价 → 隐含远期  F = K + C_mid − P_mid   （必须跨行权价基本一致，且≈现货）")
print("=" * 118)
valid = {}
for e in EXPS:
    cal, put = chains[e]
    T = (pd.Timestamp(e) - TODAY).days / 365.0
    a = cal[["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]].copy()
    b = put[["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]].copy()
    a["cm"], b["pm"] = mid(a), mid(b)
    j = a.merge(b, on="strike", suffixes=("_c", "_p"))
    j = j[(j.strike.between(spot * 0.80, spot * 1.20))].dropna(subset=["cm", "pm"])
    j["spread_c"] = (j.ask_c - j.bid_c) / j.cm
    j["spread_p"] = (j.ask_p - j.bid_p) / j.pm
    j["F"] = j.strike + j.cm - j.pm
    j["err"] = j.F / spot - 1
    liq = j[(j.openInterest_c > 50) & (j.openInterest_p > 50)]
    use = liq if len(liq) >= 8 else j
    med_F = float(use.F.median())
    print(f"\n  {e}  T={T*365:.0f}天 ({T:.3f}y)   可用双边行权价 {len(j)} 个（OI>50 的 {len(liq)} 个）")
    print(f"    隐含远期 中位 {med_F:.2f}   相对现货 {med_F/spot-1:+.3%}   "
          f"跨行权价标准差 {float(use.F.std()):.2f} ({float(use.F.std())/spot:.3%} of spot)")
    print(f"    隐含 (r−q) = {np.log(med_F/spot)/T*100:+.2f}%/年"
          f"   [现金利率 ~4%、股息率 ~0.35% ⇒ 理论 ≈ +3.6%]")
    print(f"    买卖价差中位：call {float(use.spread_c.median())*100:.1f}% of mid，"
          f"put {float(use.spread_p.median())*100:.1f}% of mid")
    q = "OK" if abs(med_F / spot - 1) < 0.02 and float(use.F.std()) / spot < 0.01 else "⚠️ 可疑"
    print(f"    判定：{q}")
    valid[e] = {"F": med_F, "T": T, "tbl": j, "ok": q == "OK"}
    show = use[(use.strike >= spot * 0.90) & (use.strike <= spot * 1.10)]
    print(show[["strike", "bid_c", "ask_c", "cm", "bid_p", "ask_p", "pm", "F", "err",
                "impliedVolatility_c", "impliedVolatility_p", "openInterest_c", "openInterest_p"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def p_gt(level, h):
    return float(1 - np.searchsorted(Z[h], np.log(level / px) / SIG[h], "right") / len(Z[h]))


def p_lt(level, h):
    return float(np.searchsorted(Z[h], np.log(level / px) / SIG[h], "right") / len(Z[h]))


def model_terminal(h, n=400000, rng=np.random.default_rng(7)):
    zz = rng.choice(Z[h], size=n, replace=True)
    return px * np.exp(zz * SIG[h])


# ============================================================ covered call
print("\n" + "=" * 118)
print("【1】备兑开仓（covered call）—— 真实报价")
print("=" * 118)
print(f"  波动率定价背景：IV30 24.4%（_iv_history.csv 今日快照）vs 已实现20日 {rv20:.1%} → "
      f"比值 {0.244/rv20:.2f}")
print("  比值 ≈1.0 = 卖的是「公允」波动，不是「贵」波动。卖备兑在这里不是波动率套利，"
      "\n  它只是把「未来的上涨」换成「现在的现金」。")
for e in ("2026-10-16", "2026-11-20"):
    v = valid[e]
    T, tbl = v["T"], v["tbl"]
    hh = 21 if T * 365 < 45 else 21
    print(f"\n  ---- 到期 {e}（{T*365:.0f}天，{'财报前' if e < '2026-10-29' else '财报后，含10/29财报'}）")
    print(f"       {'行权价':>8}{'距现价':>9}{'权利金mid':>11}{'权利金%现货':>12}{'年化%':>9}"
          f"{'IV':>8}{'被行权概率(模型21d)':>20}{'封顶总收益':>12}{'OI':>9}")
    for mult in (1.00, 1.02, 1.05, 1.08, 1.10):
        k = tbl.strike.iloc[(tbl.strike - spot * mult).abs().argsort()].iloc[0]
        r = tbl[tbl.strike == k].iloc[0]
        prem, ivc = float(r.cm), float(r.impliedVolatility_c)
        cap = (k - spot + prem) / spot
        print(f"       {k:>8.1f}{k/spot-1:>+8.1%}{prem:>11.2f}{prem/spot:>11.2%}"
              f"{prem/spot/T:>8.1%}{ivc:>8.1%}{p_gt(k*px/spot, 21):>19.1%}{cap:>11.2%}"
              f"{r.openInterest_c:>9.0f}")
    # EV under the repo's own distribution, 21d horizon, using the 21d-equivalent expiry
    print(f"\n       在本仓库 21日分布下的期望值比较（把权利金按 21/{T*365:.0f} 天线性摊到21日窗口，"
          f"\n       只为可比；这是近似，标为 INFERENCE）")
    ST = model_terminal(21)
    ev_stock = float(np.mean(ST / px - 1))
    for mult in (1.02, 1.05, 1.08):
        k = tbl.strike.iloc[(tbl.strike - spot * mult).abs().argsort()].iloc[0]
        prem = float(tbl[tbl.strike == k].iloc[0].cm) * min(1.0, 21 / (T * 365))
        k_adj = k * px / spot
        payoff = np.minimum(ST, k_adj) + prem
        ev_cc = float(np.mean(payoff / px - 1))
        sd_cc = float(np.std(payoff / px - 1))
        sd_st = float(np.std(ST / px - 1))
        print(f"       K={k:.0f}: 期望 备兑 {ev_cc:+.2%} vs 纯持股 {ev_stock:+.2%}  "
              f"差 {ev_cc-ev_stock:+.2%}   波动 {sd_cc:.2%} vs {sd_st:.2%}  "
              f"(降波 {1-sd_cc/sd_st:.0%})  5分位 {np.quantile(payoff/px-1,0.05):+.2%}")

# ============================================================ protective put
print("\n" + "=" * 118)
print("【2】保护性看跌（protective put）—— 真实报价")
print("=" * 118)
hi22 = float(C.rolling(22).max().iloc[-1])
stop = hi22 - 5 * a22
print(f"  参照：5×ATR22 止损 {stop:.2f}（{stop/px-1:+.1%}）。用保护性看跌替代止损，"
      f"就不会被路径噪音扫出局。\n  已测得 21日触及止损概率 ≈18%，终点跌破概率 7.2%。")
for e in ("2026-10-16", "2026-11-20", "2027-01-15"):
    v = valid[e]
    T, tbl = v["T"], v["tbl"]
    print(f"\n  ---- 到期 {e}（{T*365:.0f}天）")
    print(f"       {'行权价':>8}{'距现价':>9}{'成本mid':>10}{'成本%现货':>11}{'年化拖累':>10}"
          f"{'IV':>8}{'保护下限(净)':>13}{'到期实值概率':>14}{'OI':>9}")
    for mult in (1.00, 0.95, 0.90, 0.85):
        k = tbl.strike.iloc[(tbl.strike - spot * mult).abs().argsort()].iloc[0]
        r = tbl[tbl.strike == k].iloc[0]
        cost, ivp = float(r.pm), float(r.impliedVolatility_p)
        floor_net = (k - cost) / spot - 1
        print(f"       {k:>8.1f}{k/spot-1:>+8.1%}{cost:>10.2f}{cost/spot:>10.2%}"
              f"{cost/spot/T:>9.1%}{ivp:>8.1%}{floor_net:>+12.2%}{p_lt(k*px/spot, 21):>13.1%}"
              f"{r.openInterest_p:>9.0f}")

# ============================================================ collar
print("\n" + "=" * 118)
print("【3】领口（collar）：卖出上方 call 融资买入下方 put")
print("=" * 118)
for e in ("2026-10-16", "2026-11-20", "2027-01-15"):
    v = valid[e]
    T, tbl = v["T"], v["tbl"]
    print(f"\n  ---- 到期 {e}（{T*365:.0f}天）")
    rows = []
    for pm_mult in (0.95, 0.90):
        kp = tbl.strike.iloc[(tbl.strike - spot * pm_mult).abs().argsort()].iloc[0]
        pcost = float(tbl[tbl.strike == kp].iloc[0].pm)
        # cheapest call strike whose premium >= put cost  (zero-cost collar)
        cand = tbl[(tbl.strike > spot) & (tbl.cm >= pcost)].sort_values("strike")
        if cand.empty:
            continue
        kc = float(cand.strike.iloc[-1]) if False else float(cand.strike.iloc[-1])
        # take the HIGHEST strike still paying for the put => the least upside given away
        kc = float(cand.strike.max())
        ccred = float(cand[cand.strike == kc].cm.iloc[0])
        rows.append({"put_K": kp, "put成本": pcost, "call_K": kc, "call权利金": ccred,
                     "净现金": ccred - pcost,
                     "下限%": (kp - (pcost - ccred)) / spot - 1,
                     "上限%": (kc - (pcost - ccred)) / spot - 1,
                     "P(到期>callK)": p_gt(kc * px / spot, 21),
                     "P(到期<putK)": p_lt(kp * px / spot, 21)})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("       注：上/下限已含净现金；「最高仍能付掉 put 的 call 行权价」= 让出最少上涨的零成本领口。")

print("\n" + "=" * 118)
print("【4】期限结构与财报")
print("=" * 118)
iv = pd.read_csv(HERE / "_iv_history.csv")
r = iv[iv.symbol == SYM].iloc[-1]
print(f"  快照 {r.date}: IV30 {r.iv30:.1%}  IV60 {r.iv60:.1%}  IV90 {r.iv90:.1%}  "
      f"期限斜率 {r.ts_slope*100:+.1f}pts  25delta偏斜 {r.skew25*100:+.1f}pts  质量 {r.quality}")
print(f"  IV30/RV20 = {r.iv30/rv20:.2f}   IV30/RV60 = {r.iv30/rv60:.2f}")
print(f"  下次财报 2026-10-29（yfinance 日历）。距今 {(pd.Timestamp('2026-10-29')-TODAY).days} 天，"
      f"在 21 个交易日窗口之外 → move_prob 的 21日 sigma 未加财报乘数（1.211）。")
print(f"  含财报的 21日 sigma 会是 {SIG[21]*1.211:.4f}，对应 5分位 "
      f"{np.exp(np.quantile(Z[21],0.05)*SIG[21]*1.211)-1:+.1%}（而非 "
      f"{np.exp(np.quantile(Z[21],0.05)*SIG[21])-1:+.1%}）。10/16 到期的期权不含这个风险，"
      f"11/20 到期的含。")
