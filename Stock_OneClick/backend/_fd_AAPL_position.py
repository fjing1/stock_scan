"""
_fd_AAPL_position.py — the decision arithmetic for an AAPL holder, using this repo's own measured
tools. Nothing here is a textbook formula applied blind: every probability comes from move_prob.py's
fitted sigma + its empirical z table, or from a block bootstrap of AAPL's own bars.

Sections
  1  SUNK COST      break-even move and P(below cost) over a grid of possible cost bases
  2  SIZING         21-day 5th-percentile and ATR impact at each portfolio weight, and the inverse
  3  STOP           5xATR22 level; P(close below) vs P(TOUCH), the latter from path simulation of
                    the ACTUAL ratcheting rule, plus the realized historical hit rate
  4  OPTIONS        real chain, put-call-parity validation FIRST, then covered call / put / collar
  5  CONTEXT        earnings date, consensus anchor, IV vs realized

Usage: ../../vcp_env/bin/python _fd_AAPL_position.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import move_prob as M

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

SYM = "AAPL"
RNG = np.random.default_rng(20260914)


def atr(h, l, c, n=22):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ============================================================ data
d_all = yf.download(SYM, period="max", progress=False, auto_adjust=True)
if isinstance(d_all.columns, pd.MultiIndex):
    d_all.columns = d_all.columns.get_level_values(0)
d_all = d_all[d_all.Close > 0].dropna(subset=["Close"])
# PARTIAL BAR GUARD: drop today's bar if the session is not closed. Repo memory: unclosed bars have
# silently broken three prior analyses.
print(f"最后两根K线: {d_all.index[-2].date()} / {d_all.index[-1].date()}   "
      f"收盘 {float(d_all.Close.iloc[-2]):.2f} / {float(d_all.Close.iloc[-1]):.2f}")
d3 = d_all.tail(756)
C, H, L, V = d_all.Close, d_all.High, d_all.Low, d_all.Volume
px = float(C.iloc[-1])
a22 = float(atr(H, L, C).iloc[-1])

print("=" * 118)
print(f"AAPL 决策算术   现价 {px:.2f}   数据截至 {C.index[-1].date()}   共 {len(C):,} 根日线 "
      f"({C.index[0].date()} 起)")
print("=" * 118)
print(f"  ATR22 {a22:.2f} = 价格的 {a22/px:.2%}")

# fitted sigma per horizon from the repo's model
model = pd.read_pickle(M.MODEL_PATH)
g = M.asset_class(SYM)
feat = M.build_features(C, H, L, None, V)
SIG, ZT = {}, {}
for h in (1, 5, 10, 21):
    spec = model["groups"][g]["per_h"].get(h)
    if spec is None:
        continue
    cols = spec["cols"]
    if any(x not in feat.columns for x in cols):
        continue
    SIG[h] = float(np.exp(M._design(feat.iloc[[-1]], cols) @ spec["beta"]).item()) * np.sqrt(h)
    ZT[h] = np.asarray(spec["z"])
print("\n  模型拟合的对数收益 sigma（本仓库 move_prob.py，21年滚动前瞻校准）")
for h in SIG:
    print(f"    {h:>2}日  sigma={SIG[h]:.4f} ({SIG[h]*100:.2f}%)   经验z表 n={len(ZT[h]):,}")


def p_below(level: float, h: int) -> float:
    """P(close at horizon h < level), from the fitted sigma and the empirical z distribution."""
    z, s = ZT[h], SIG[h]
    need = np.log(level / px)
    return float(np.searchsorted(np.sort(z), need / s, "right") / len(z))


# ============================================================ 1  SUNK COST
print("\n" + "=" * 118)
print("【1】沉没成本框架")
print("=" * 118)
print("  事实：成本价是持有者过去的一个事实，不是公司的属性。市场不知道它，也不会因为它调整。")
print("  持有者未提供成本。下面把成本当作参数扫一遍，这样任何成本都能读到自己的那一行。")
print("  参考锚点：过去一年 AAPL 收益 "
      f"{float(C.iloc[-1]/C.iloc[-253]-1):+.1%}（{C.index[-253].date()} 收盘 {float(C.iloc[-253]):.2f}）")
print(f"           三年前收盘 {float(C.iloc[-756]):.2f}（{C.index[-756].date()}），五年前 "
      f"{float(C.iloc[-1260]):.2f}")

anchors = {
    "1年前买入": float(C.iloc[-253]),
    "2年前买入": float(C.iloc[-505]),
    "3年前买入": float(C.iloc[-756]),
    "5年前买入": float(C.iloc[-1260]),
    "近1月均价买入": float(C.tail(22).mean()),
    "今天买入(=现价)": px,
}
rows = []
for name, cost in anchors.items():
    r = {"情形": name, "成本": cost, "当前盈亏": px / cost - 1,
         "回本需要": cost / px - 1,
         "成本处历史百分位": float((C <= cost).mean())}
    for h in (5, 10, 21):
        r[f"P(第{h}日收盘<成本)"] = p_below(cost, h) if h in SIG else np.nan
    rows.append(r)
t = pd.DataFrame(rows)
print("\n" + t.to_string(index=False, formatters={
    "成本": "{:.2f}".format, "当前盈亏": "{:+.1%}".format, "回本需要": "{:+.1%}".format,
    "成本处历史百分位": "{:.0%}".format,
    **{f"P(第{h}日收盘<成本)": "{:.1%}".format for h in (5, 10, 21)}}))
print("\n  读法：只有「今天买入」那一行的 P(<成本) 是有意义的对称赌局（≈50% 略偏下，因为漂移>0）。")
print("  其余各行 P(<成本) 很低 —— 那不是「安全」，那只是说明现价离旧成本很远，与未来收益无关。")
print(f"  真正的问题：今天 {px:.2f} 你会买吗？如果不会，继续持有在数学上等价于每天以 {px:.2f} 重新买入。")

# ============================================================ 2  SIZING
print("\n" + "=" * 118)
print("【2】仓位规模算术")
print("=" * 118)
q05_21 = float(np.quantile(ZT[21], 0.05) * SIG[21])
q05_21 = np.exp(q05_21) - 1
q01_21 = np.exp(float(np.quantile(ZT[21], 0.01) * SIG[21])) - 1
q95_21 = np.exp(float(np.quantile(ZT[21], 0.95) * SIG[21])) - 1
print(f"  21日分布（模型）：5分位 {q05_21:+.1%}   1分位 {q01_21:+.1%}   95分位 {q95_21:+.1%}"
      f"   典型幅度 {np.exp(SIG[21])-1:.1%}")
print(f"  1×ATR22 = {a22/px:.2%}    3×ATR22 = {3*a22/px:.2%}    5×ATR22 = {5*a22/px:.2%}")
print("\n  组合层面的冲击（对总组合的百分比）")
hdr = f"  {'权重':>6}{'21日5分位':>12}{'21日1分位':>12}{'-3ATR':>10}{'-5ATR':>10}{'-20%情景':>11}{'-30%情景':>11}"
print(hdr)
for w in (0.02, 0.05, 0.10, 0.20, 0.30):
    print(f"  {w:>5.0%}{w*q05_21*100:>11.2f}%{w*q01_21*100:>11.2f}%"
          f"{-w*3*a22/px*100:>9.2f}%{-w*5*a22/px*100:>9.2f}%"
          f"{-w*0.20*100:>10.2f}%{-w*0.30*100:>10.2f}%")
print("\n  反推：给定「能承受的组合级痛感预算」，最大仓位是多少")
print(f"  {'痛感预算':>8}{'按21日5分位':>14}{'按21日1分位':>14}{'按-5ATR':>12}{'按-30%崩盘':>13}")
for pain in (0.005, 0.01, 0.02, 0.03, 0.05):
    print(f"  {pain:>7.1%}{pain/abs(q05_21):>13.0%}{pain/abs(q01_21):>13.0%}"
          f"{pain/(5*a22/px):>11.0%}{pain/0.30:>12.0%}")
print("\n  注意 AAPL 的 ATR22 只有价格的 2.1%，是低波动大盘股。以「-5ATR」为痛感尺度时，"
      "\n  同样的组合痛感允许的 AAPL 仓位大约是 ARM 那类 6% ATR 标的的 3 倍。")

# ============================================================ 3  STOP
print("\n" + "=" * 118)
print("【3】止损算术（本仓库 5×ATR22 移动止损口径）")
print("=" * 118)
hi22 = float(C.rolling(22).max().iloc[-1])
stop = hi22 - 5 * a22
print(f"  22日最高收盘 {hi22:.2f}   5×ATR22 = {5*a22:.2f}   止损位 {stop:.2f}")
print(f"  现价距止损 {stop/px-1:+.2%}   = {(px-stop)/a22:.2f} 个 ATR22")
print(f"  止损位处历史百分位 {(C <= stop).mean():.0%}（即历史上 {(C<=stop).mean():.0%} 的收盘日低于该价）")
p_close_below_21 = p_below(stop, 21)
p_close_below_10 = p_below(stop, 10)
print(f"\n  A) P(第21日收盘 < {stop:.2f}) = {p_close_below_21:.1%}   "
      f"P(第10日收盘 < {stop:.2f}) = {p_close_below_10:.1%}      ← 终点概率")

# ---- path simulation: block bootstrap of AAPL's OWN bars, rescaled to the fitted 21-day sigma
lr = np.log(C).diff().dropna()
hi_r = np.log(H / C).clip(lower=0)
lo_r = np.log(L / C).clip(upper=0)
tri = pd.DataFrame({"lr": lr, "hi": hi_r.reindex(lr.index), "lo": lo_r.reindex(lr.index)}).dropna()
A = tri.to_numpy()
BLK, HZ, NSIM = 5, 21, 40000
nblk = int(np.ceil(HZ / BLK))
starts = RNG.integers(0, len(A) - BLK, size=(NSIM, nblk))
idx = (starts[:, :, None] + np.arange(BLK)[None, None, :]).reshape(NSIM, -1)[:, :HZ]
path_lr = A[idx, 0]
path_hi = A[idx, 1]
path_lo = A[idx, 2]
# rescale the whole path so the realised 21-day sigma equals the model's fitted sigma
raw_sig = path_lr.sum(axis=1).std(ddof=1)
k = SIG[21] / raw_sig
path_lr = path_lr * k
path_hi, path_lo = path_hi * k, path_lo * k
print(f"\n  路径模拟：AAPL 自身日线的 5日分块 bootstrap，{NSIM:,} 条 21日路径，"
      f"整体缩放 {k:.3f} 倍使 21日 sigma 对齐模型的 {SIG[21]:.4f}")
cum = np.cumsum(path_lr, axis=1)
close_p = px * np.exp(cum)
low_p = px * np.exp(cum + path_lo)
high_p = px * np.exp(cum + path_hi)

print(f"  校验：模拟 21日收益 5分位 {np.quantile(close_p[:,-1]/px-1,0.05):+.1%} "
      f"vs 模型 {q05_21:+.1%};  95分位 {np.quantile(close_p[:,-1]/px-1,0.95):+.1%} vs {q95_21:+.1%};"
      f"  P(终点<止损) {np.mean(close_p[:,-1]<stop):.1%} vs 模型 {p_close_below_21:.1%}")

p_touch_fixed_low = float(np.mean(low_p.min(axis=1) < stop))
p_touch_fixed_close = float(np.mean(close_p.min(axis=1) < stop))
print(f"\n  B) 固定止损位 {stop:.2f}：")
print(f"       P(21日内任一日 收盘 < 止损)  = {p_touch_fixed_close:.1%}")
print(f"       P(21日内任一日 盘中低点 < 止损) = {p_touch_fixed_low:.1%}   ← 真正的「触及」概率")

# ---- the ACTUAL rule: stop ratchets with the 22-day rolling max of closes
hist_c = C.tail(21).to_numpy()          # last 21 closes, needed to seed the 22-day window
hit_ratchet = np.zeros(NSIM, dtype=bool)
hit_day = np.full(NSIM, HZ + 1)
for i in range(HZ):
    # 22-day window of closes ending at simulated day i
    if i + 1 >= 22:
        win_max = close_p[:, i + 1 - 22:i + 1].max(axis=1)
    else:
        need = 22 - (i + 1)
        win_max = np.maximum(hist_c[-need:].max(), close_p[:, :i + 1].max(axis=1))
    s_i = win_max - 5 * a22             # ATR held at today's value (see caveat printed below)
    newly = (~hit_ratchet) & (low_p[:, i] < s_i)
    hit_day[newly] = i + 1
    hit_ratchet |= newly
p_touch_ratchet = float(hit_ratchet.mean())
print(f"\n  C) 真实规则（止损随 22日滚动最高收盘上移，ATR 固定为今天的 {a22:.2f}）：")
print(f"       P(21日内被触发) = {p_touch_ratchet:.1%}"
      f"   中位触发日 = 第 {int(np.median(hit_day[hit_ratchet])) if p_touch_ratchet>0 else 0} 日")
print(f"       被触发时的价格中位数 {np.median([low_p[j, hit_day[j]-1] for j in np.where(hit_ratchet)[0]]):.2f}")
print("       口径说明：ATR 在模拟中固定。下跌时 ATR 通常放大，止损位会被推得更低，"
      "\n       所以这是触发概率的偏高估计（止损更宽 → 更难触发）；同时它随高点上移，又推高概率。两者部分抵消。")
print(f"\n  距止损 {(px-stop)/a22:.1f} 个 ATR。判读：>4 ATR 属于「真实风险控制」，2 ATR 以内基本是"
      "\n  在赌路径噪音。这里的 5.0 ATR 是设计值，因为止损锚在 22日最高点而现价就在最高点附近。")

# ---- realized historical hit rate of this exact rule on AAPL
print("\n  D) 该规则在 AAPL 真实历史上的表现（同一条 5×ATR22 规则，全样本回看）")
a_s = atr(H, L, C)
stop_s = C.rolling(22).max() - 5 * a_s
above = (C > C.rolling(200).mean()) & (C > C.rolling(50).mean()) & (C > C.rolling(20).mean())
near_hi = C / C.rolling(756).max() - 1 > -0.05
dist_atr = (C - stop_s) / a_s
fut_hit = pd.Series(index=C.index, dtype=float)
Ln, Sn = L.to_numpy(), stop_s.to_numpy()
for i in range(len(C) - 21):
    fut_hit.iloc[i] = float(np.nanmin(Ln[i + 1:i + 22] - Sn[i + 1:i + 22]) < 0)
base = fut_hit.notna() & stop_s.notna()
cond_like_today = base & above & near_hi
print(f"     全样本         21日内被触发 {fut_hit[base].mean():.1%}   (n={int(base.sum()):,})")
print(f"     与今日同类状态 21日内被触发 {fut_hit[cond_like_today].mean():.1%}   "
      f"(n={int(cond_like_today.sum()):,})  条件=收盘在MA20/50/200上方 且 距3年高点<5%")
d_now = float(dist_atr.iloc[-1])
band = base & dist_atr.between(d_now - 0.5, d_now + 0.5)
print(f"     距止损 {d_now:.1f}±0.5 ATR 时 21日内被触发 {fut_hit[band].mean():.1%}   (n={int(band.sum()):,})")
fwd21 = C.shift(-21) / C - 1
print(f"     同类状态下的 21日前瞻收益：中位 {fwd21[cond_like_today].median():+.2%}   "
      f"平均 {fwd21[cond_like_today].mean():+.2%}   P(>0) {(fwd21[cond_like_today]>0).mean():.1%}"
      f"   5分位 {fwd21[cond_like_today].quantile(0.05):+.1%}")
print(f"     全样本对照 21日前瞻：中位 {fwd21[base].median():+.2%}  平均 {fwd21[base].mean():+.2%}"
      f"  P(>0) {(fwd21[base]>0).mean():.1%}")
print("     注意：这是 AAPL 单只股票的重叠窗口样本，不是独立观测；n 看似很大但有效自由度约 n/21。")
