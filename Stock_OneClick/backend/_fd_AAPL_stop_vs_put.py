"""
_fd_AAPL_stop_vs_put.py — the one number that decides between a stop and a put.

A 5xATR22 stop and a 300-strike put both floor the loss around -10.5%. They are not equivalent: the
stop has FALSE POSITIVES (it exits, then the stock recovers) and the put does not. So the honest
comparison is:

    expected cost of the stop's false positives   vs   the put's premium (38bp for 32 days)

Both computed on the same 60,000 simulated paths whose terminal distribution is pinned to
move_prob.py's own calibrated 21-day distribution. Also: check whether 60-day realised vol really is
32.4% (which would make IV30 CHEAP at 0.75x, contradicting the 1.03x reading off RV20 alone) and if
so, what caused it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import move_prob as M

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260914)
SYM, HZ, BLK, NSIM = "AAPL", 21, 5, 60000


def atr(h, l, c, n=22):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


d = yf.download(SYM, period="max", progress=False, auto_adjust=True)
if isinstance(d.columns, pd.MultiIndex):
    d.columns = d.columns.get_level_values(0)
d = d[d.Close > 0].dropna(subset=["Close"])
C, H, L, V = d.Close, d.High, d.Low, d.Volume
px, a22 = float(C.iloc[-1]), float(atr(H, L, C).iloc[-1])
stop0 = float(C.rolling(22).max().iloc[-1]) - 5 * a22

# ---------- why is RV60 so much higher than RV20?
lr = np.log(C).diff()
print("=" * 112)
print("【0】已实现波动的期限：IV 到底贵不贵")
for n in (10, 20, 40, 60, 90, 120, 252):
    print(f"  已实现{n:>3}日年化波动 {float(lr.tail(n).std(ddof=1)*np.sqrt(252)):.1%}")
big = lr.tail(70).abs().sort_values(ascending=False).head(6)
print("\n  最近70个交易日里最大的6个单日变动（找出把 RV60 顶高的那几天）")
for dt, v in big.items():
    print(f"    {dt.date()}  {np.exp(v)-1 if lr[dt]>0 else -(np.exp(v)-1):+.2%}   "
          f"收盘 {float(C[dt]):.2f}")
print("  → IV30 24.4% / RV20 23.7% = 1.03「公允」，但 IV30 / RV60 32.4% = 0.75。")
print("    两个读数不矛盾：波动已经从夏天的高位回落，IV 定在两者之间。")
print("    对「买保护」而言重要的是：保护的定价并没有反映过去两个月真实发生过的波动。")

# ---------- simulate
model = pd.read_pickle(M.MODEL_PATH)
g = M.asset_class(SYM)
f = M.build_features(C, H, L, None, V)
sp = model["groups"][g]["per_h"][HZ]
sig = float(np.exp(M._design(f.iloc[[-1]], sp["cols"]) @ sp["beta"]).item()) * np.sqrt(HZ)
z = np.sort(np.asarray(sp["z"]))
tri = pd.DataFrame({"lr": lr, "hi": np.log(H / C).clip(lower=0),
                    "lo": np.log(L / C).clip(upper=0)}).dropna().to_numpy()
nblk = int(np.ceil(HZ / BLK))
st = RNG.integers(0, len(tri) - BLK, size=(NSIM, nblk))
idx = (st[:, :, None] + np.arange(BLK)[None, None, :]).reshape(NSIM, -1)[:, :HZ]
p_lr, p_hi, p_lo = tri[idx, 0], tri[idx, 1], tri[idx, 2]
k = sig / p_lr.sum(axis=1).std(ddof=1)
p_lr, p_hi, p_lo = p_lr * k, p_hi * k, p_lo * k
cum = np.cumsum(p_lr, axis=1)
T = cum[:, -1]
order = np.argsort(T)
tgt = np.empty(NSIM)
tgt[order] = np.quantile(z, (np.arange(NSIM) + 0.5) / NSIM) * sig
cum = cum + (tgt - T)[:, None] * (np.arange(1, HZ + 1) / HZ)[None, :]
close_p, low_p = px * np.exp(cum), px * np.exp(cum + p_lo)

# ---------- run the actual trailing rule, record exit price and the terminal counterfactual
hist = C.tail(21).to_numpy()
hit = np.zeros(NSIM, bool)
exit_px = np.full(NSIM, np.nan)
hday = np.full(NSIM, HZ + 1)
for i in range(HZ):
    if i + 1 >= 22:
        wmax = close_p[:, i + 1 - 22:i + 1].max(axis=1)
    else:
        need = 22 - (i + 1)
        wmax = np.maximum(hist[-need:].max(), close_p[:, :i + 1].max(axis=1))
    s_i = wmax - 5 * a22
    new = (~hit) & (low_p[:, i] < s_i)
    # fill at the stop level, not at the low: a stop order fills at/near the trigger
    exit_px[new] = np.minimum(s_i[new], close_p[new, i])
    hday[new] = i + 1
    hit |= new

end_px = close_p[:, -1]
r_hold = end_px / px - 1
r_stop = np.where(hit, exit_px / px - 1, r_hold)     # stopped out -> sit in cash to day 21
PUT_K, PUT_C = 300.0, 1.28                          # 2026-10-16 300P, mid of 1.26/1.30, OI 24,817
r_put = (np.maximum(end_px, PUT_K) - PUT_C) / px - 1

print("\n" + "=" * 112)
print(f"【1】三种做法在同一 {NSIM:,} 条路径上的 21日收益分布")
print(f"     止损位 {stop0:.2f} ({stop0/px-1:+.1%})   看跌 K={PUT_K:.0f} 成本 {PUT_C:.2f} "
      f"({PUT_C/px:.2%} of spot, 2026-10-16 到期, OI 24,817)")
tbl = pd.DataFrame({"纯持股": r_hold, "持股+移动止损": r_stop, "持股+300P": r_put})
print(f"\n  {'':>16}{'均值':>9}{'中位':>9}{'标准差':>9}{'1分位':>9}{'5分位':>9}"
      f"{'25分位':>9}{'75分位':>9}{'95分位':>9}{'P(<0)':>9}")
for c in tbl.columns:
    s = tbl[c]
    print(f"  {c:>14}{s.mean()*100:>8.2f}%{s.median()*100:>8.2f}%{s.std()*100:>8.2f}%"
          f"{s.quantile(.01)*100:>8.2f}%{s.quantile(.05)*100:>8.2f}%{s.quantile(.25)*100:>8.2f}%"
          f"{s.quantile(.75)*100:>8.2f}%{s.quantile(.95)*100:>8.2f}%{(s<0).mean()*100:>8.1f}%")

print("\n" + "=" * 112)
print("【2】移动止损的假信号成本（这就是「止损 vs 看跌」的分水岭）")
print(f"  P(21日内被触发) = {hit.mean():.1%}   中位触发日 第 {int(np.median(hday[hit]))} 日")
fp = hit & (end_px > exit_px)
print(f"  被触发的路径里，21日终点价高于出场价的比例 = {fp.sum()/hit.sum():.1%}  ← 假信号率")
give = (end_px[fp] - exit_px[fp]) / px
print(f"  这些假信号平均让你少赚 {give.mean():.2%}（中位 {np.median(give):.2%}，"
      f"95分位 {np.quantile(give,0.95):.2%}）")
print(f"  对全体路径的期望拖累 = {hit.mean()*fp.sum()/hit.sum()*give.mean()*100:.3f}%"
      f"  (= 触发率 × 假信号率 × 平均少赚)")
tp = hit & (end_px <= exit_px)
save = (exit_px[tp] - end_px[tp]) / px
print(f"  真信号（终点更低）比例 {tp.sum()/hit.sum():.1%}，平均帮你避开 {save.mean():.2%}"
      f"（中位 {np.median(save):.2%}）")
net = (r_stop - r_hold).mean()
print(f"  净效果：止损相对纯持股的期望收益差 = {net*100:+.3f}%   "
      f"标准差 {r_stop.std()*100:.2f}% vs {r_hold.std()*100:.2f}%  (降波 {1-r_stop.std()/r_hold.std():.0%})")
print(f"\n  对照：300P 的确定成本 = {PUT_C/px:.2%}（32天），期望收益差 "
      f"{(r_put-r_hold).mean()*100:+.3f}%，降波 {1-r_put.std()/r_hold.std():.0%}")
print(f"  两者的 5分位：止损 {np.quantile(r_stop,0.05)*100:+.2f}%   "
      f"300P {np.quantile(r_put,0.05)*100:+.2f}%   纯持股 {np.quantile(r_hold,0.05)*100:+.2f}%")
print(f"  1分位（真正的尾部）：止损 {np.quantile(r_stop,0.01)*100:+.2f}%   "
      f"300P {np.quantile(r_put,0.01)*100:+.2f}%   纯持股 {np.quantile(r_hold,0.01)*100:+.2f}%")
print("\n  读法：止损与看跌把 5 分位改善到几乎同一水平，但止损用 18% 的出场概率换，"
      "\n  看跌用 38bp 的确定现金换。止损还有一个看跌没有的漏洞：跳空。看跌的下限是合约条款，"
      "\n  止损的下限只是一个委托单。")
gap = (low_p[:, 0] / px - 1)
print(f"  跳空敏感度：模拟中第1日就跌破止损的路径占 {float(np.mean(low_p[:,0]<stop0)):.2%}"
      f"（单日 -10.7% 需要 {abs(stop0/px-1)/(a22/px):.1f} 个 ATR，历史上 AAPL 单日跌幅超此的天数 "
      f"{int((lr < np.log(stop0/px)).sum())} / {len(lr):,}）")
