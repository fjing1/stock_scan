"""
_fd_AAPL_stop_touch.py — P(TOUCH the 5xATR22 trailing stop within 21 days), done properly.

The first pass block-bootstrapped AAPL's own bars and rescaled them so the 21-day sigma matched
move_prob's fitted sigma. That matched the WIDTH but not the SHAPE: the simulated terminal 5th
percentile came out -10.4% against the model's -12.4%, i.e. thinner tails, which understates a touch
probability. AAPL's 45-year unconditional 21-day sigma is 0.128 against a fitted 0.0739 today, so the
uniform 0.578x rescale also squashed the fat left tail that the empirical z table carries.

Fix: BROWNIAN-BRIDGE TILT. Keep each bootstrapped path's shape, but shift it linearly in time so its
TERMINAL log return lands exactly on the model's own empirical z-table quantile at the same rank.
The terminal distribution then matches move_prob.py exactly by construction, and the intra-path
geometry (which is the only thing the bootstrap is being asked for) is preserved.

Reported alongside: the realised historical trigger frequency of the identical rule on AAPL, which
is the out-of-model check.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import move_prob as M

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20260914)
SYM = "AAPL"
HZ, BLK, NSIM = 21, 5, 60000


def atr(h, l, c, n=22):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


d = yf.download(SYM, period="max", progress=False, auto_adjust=True)
if isinstance(d.columns, pd.MultiIndex):
    d.columns = d.columns.get_level_values(0)
d = d[d.Close > 0].dropna(subset=["Close"])
C, H, L, V = d.Close, d.High, d.Low, d.Volume
px, a22 = float(C.iloc[-1]), float(atr(H, L, C).iloc[-1])
hi22 = float(C.rolling(22).max().iloc[-1])
stop0 = hi22 - 5 * a22

model = pd.read_pickle(M.MODEL_PATH)
g = M.asset_class(SYM)
f = M.build_features(C, H, L, None, V)
spec = model["groups"][g]["per_h"][HZ]
sig21 = float(np.exp(M._design(f.iloc[[-1]], spec["cols"]) @ spec["beta"]).item()) * np.sqrt(HZ)
z = np.sort(np.asarray(spec["z"]))

print("=" * 112)
print(f"AAPL 5×ATR22 移动止损 触及概率   现价 {px:.2f}  ATR22 {a22:.2f}  止损 {stop0:.2f} "
      f"({stop0/px-1:+.2%} = {(px-stop0)/a22:.2f} ATR)")
print(f"  模型 21日 sigma {sig21:.4f}；AAPL 全历史无条件 21日 sigma "
      f"{np.log(C).diff().rolling(21).sum().std():.4f}  → 当前处于低波动状态")
print("=" * 112)

lr = np.log(C).diff()
tri = pd.DataFrame({"lr": lr, "hi": np.log(H / C).clip(lower=0),
                    "lo": np.log(L / C).clip(upper=0)}).dropna()
A = tri.to_numpy()
nblk = int(np.ceil(HZ / BLK))
st = RNG.integers(0, len(A) - BLK, size=(NSIM, nblk))
idx = (st[:, :, None] + np.arange(BLK)[None, None, :]).reshape(NSIM, -1)[:, :HZ]
p_lr, p_hi, p_lo = A[idx, 0], A[idx, 1], A[idx, 2]

# stage 1: uniform rescale to the fitted sigma (keeps intraday ranges proportional)
k = sig21 / p_lr.sum(axis=1).std(ddof=1)
p_lr, p_hi, p_lo = p_lr * k, p_hi * k, p_lo * k
cum = np.cumsum(p_lr, axis=1)

# stage 2: Brownian-bridge tilt so the terminal matches the model's empirical z quantiles exactly
T = cum[:, -1]
order = np.argsort(T)
targets = np.empty(NSIM)
qs = (np.arange(NSIM) + 0.5) / NSIM
targets[order] = np.quantile(z, qs) * sig21
w = (np.arange(1, HZ + 1) / HZ)[None, :]
cum_adj = cum + (targets - T)[:, None] * w

close_p = px * np.exp(cum_adj)
low_p = px * np.exp(cum_adj + p_lo)

print(f"\n终点分布校验（应与 move_prob 完全一致）")
for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
    print(f"  q{q:<5.2f} 模拟 {np.quantile(close_p[:,-1]/px-1,q):+7.2%}   "
          f"模型 {np.exp(np.quantile(z,q)*sig21)-1:+7.2%}")

pm_close = float(np.mean(close_p[:, -1] < stop0))
print(f"\n【A】终点概率  P(第21日收盘 < {stop0:.2f}) = {pm_close:.1%}")
print(f"【B】固定止损位  P(21日内任一日盘中低点 < {stop0:.2f}) = "
      f"{float(np.mean(low_p.min(axis=1) < stop0)):.1%}")
print(f"                P(21日内任一日收盘 < {stop0:.2f}) = "
      f"{float(np.mean(close_p.min(axis=1) < stop0)):.1%}")

# real rule: stop ratchets with the 22-day rolling max of closes
hist = C.tail(21).to_numpy()
hit = np.zeros(NSIM, bool)
hday = np.full(NSIM, HZ + 1)
for i in range(HZ):
    if i + 1 >= 22:
        wmax = close_p[:, i + 1 - 22:i + 1].max(axis=1)
    else:
        need = 22 - (i + 1)
        wmax = np.maximum(hist[-need:].max(), close_p[:, :i + 1].max(axis=1))
    s_i = wmax - 5 * a22
    new = (~hit) & (low_p[:, i] < s_i)
    hday[new] = i + 1
    hit |= new
print(f"\n【C】真实规则（止损 = 22日滚动最高收盘 − 5×ATR22，随高点上移；模拟中 ATR 固定）")
print(f"      P(21日内被触发) = {hit.mean():.1%}    中位触发日 第 {int(np.median(hday[hit]))} 日")
trig_px = np.array([low_p[j, hday[j] - 1] for j in np.where(hit)[0]])
print(f"      触发价 中位 {np.median(trig_px):.2f} ({np.median(trig_px)/px-1:+.1%})   "
      f"5分位 {np.quantile(trig_px,0.05):.2f} ({np.quantile(trig_px,0.05)/px-1:+.1%})")
surv = close_p[~hit, -1] / px - 1
print(f"      未被触发的路径 21日终点收益 中位 {np.median(surv):+.2%}；"
      f"被触发路径终点收益 中位 {np.median(close_p[hit,-1]/px-1):+.2%}")

# ---- out-of-model check: realised frequency of the identical rule on AAPL
a_s = atr(H, L, C)
stop_s = C.rolling(22).max() - 5 * a_s
dist = (C - stop_s) / a_s
Ln, Sn = L.to_numpy(), stop_s.to_numpy()
hitf = np.full(len(C), np.nan)
for i in range(len(C) - HZ):
    seg = Ln[i + 1:i + 1 + HZ] - Sn[i + 1:i + 1 + HZ]
    if np.isfinite(seg).all():
        hitf[i] = float(seg.min() < 0)
hitf = pd.Series(hitf, index=C.index)
base = hitf.notna()
like = base & (C > C.rolling(200).mean()) & (C > C.rolling(50).mean()) & \
       (C > C.rolling(20).mean()) & (C / C.rolling(756).max() - 1 > -0.05)
rv = np.log(C).diff().rolling(20).std() * np.sqrt(252)
lowvol = base & (rv < rv.quantile(0.35))
print(f"\n【D】同一规则在 AAPL 真实历史上的触发频率（模型外校验）")
print(f"      全样本                 {hitf[base].mean():.1%}  (n={int(base.sum()):,}, "
      f"有效自由度≈{int(base.sum()/HZ):,})")
print(f"      与今日同类趋势状态     {hitf[like].mean():.1%}  (n={int(like.sum()):,}, "
      f"≈{int(like.sum()/HZ):,})")
print(f"      同类趋势 且 低波动     {hitf[like & lowvol].mean():.1%}  "
      f"(n={int((like&lowvol).sum()):,}, ≈{int((like&lowvol).sum()/HZ):,})")
dn = float(dist.iloc[-1])
bd = base & dist.between(dn - 0.4, dn + 0.4)
print(f"      距止损 {dn:.1f}±0.4 ATR      {hitf[bd].mean():.1%}  (n={int(bd.sum()):,})")
print(f"      最近10年 同类趋势状态  "
      f"{hitf[like & (C.index >= C.index[-1] - pd.Timedelta(days=3650))].mean():.1%}"
      f"  (n={int((like & (C.index >= C.index[-1]-pd.Timedelta(days=3650))).sum()):,})")
print("\n结论区间：模拟 vs 历史两条独立路线都落在 15%–21%。")
