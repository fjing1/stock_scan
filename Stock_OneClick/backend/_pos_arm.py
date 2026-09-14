"""
_pos_arm.py — full position review for a single holding, using every tool built in this repo.

Pulls together: position P&L math, the scan's own Gann/xunlong signal state, the trailing-stop
convention the scan uses (5x ATR22), the move-probability distribution at every horizon, today's
option-IV snapshot, the earnings calendar, and sector/market context.

Usage: ../../vcp_env/bin/python _pos_arm.py SYMBOL COST [SHARES]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

import move_prob as M

warnings.filterwarnings("ignore")


def atr(h, l, c, n=22):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "ARM"
    cost = float(sys.argv[2]) if len(sys.argv) > 2 else 257.0
    shares = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0

    d = yf.download(sym, period="3y", progress=False, auto_adjust=True)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d[d.Close > 0].dropna(subset=["Close"])
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    px = float(c.iloc[-1])

    print("=" * 78)
    print(f"{sym}  持仓复盘   成本 {cost:.2f}   最新 {px:.2f}   数据截至 {c.index[-1].date()}")
    print("=" * 78)

    # ---- position math
    pnl_pct = px / cost - 1
    print(f"\n【仓位】")
    print(f"  盈亏      {pnl_pct*100:+.2f}%   (每 {shares:.0f} 股 {(px-cost)*shares:+,.0f} 美元)")
    print(f"  历史区间  上市至今最高 {float(c.max()):.2f} / 最低 {float(c.min()):.2f}"
          f"   现价距最高 {px/float(c.max())-1:+.1%}")
    print(f"  成本位置  成本在上市以来收盘价的第 {float((c <= cost).mean())*100:.0f} 百分位")

    # ---- trend / technical state, using the scan's own conventions
    ma = {n: float(c.rolling(n).mean().iloc[-1]) for n in (10, 20, 50, 200) if len(c) >= n}
    a22 = float(atr(h, l, c).iloc[-1])
    stop = float(c.rolling(22).max().iloc[-1]) - 5 * a22        # scan's 5xATR22 trailing stop
    dd = c / c.cummax() - 1
    print(f"\n【趋势】")
    for n, mv in ma.items():
        print(f"  MA{n:<4}{mv:>9.2f}   现价{'上' if px > mv else '下'}方 {px/mv-1:+.1%}")
    print(f"  ATR22    {a22:>9.2f}  ({a22/px:.1%} of price)")
    print(f"  5×ATR22 移动止损 {stop:.2f}   现价距止损 {px/stop-1:+.1%}   "
          f"{'成本在止损上方' if cost > stop else '⚠️ 成本已在止损下方'}")
    print(f"  当前回撤  {float(dd.iloc[-1])*100:+.1f}% (自上市以来最深 {float(dd.min())*100:.1f}%)")
    rs = c.diff()
    up, dn = rs.clip(lower=0), -rs.clip(upper=0)
    rsi = 100 - 100 / (1 + up.ewm(alpha=1/14, adjust=False).mean() / dn.ewm(alpha=1/14, adjust=False).mean())
    print(f"  RSI14    {float(rsi.iloc[-1]):>9.0f}")
    r63 = float(c.iloc[-1] / c.iloc[-64] - 1) if len(c) > 64 else float("nan")
    print(f"  近3月     {r63:+.1%}    近1月 {float(c.iloc[-1]/c.iloc[-22]-1):+.1%}"
          f"    近5日 {float(c.iloc[-1]/c.iloc[-6]-1):+.1%}")

    # ---- the scan's own signal engine
    print(f"\n【扫描信号引擎（观海买卖点 / Gann）】")
    try:
        import scan_stocks as S
        from xunlong import XunLongIndicator
        sig = S.scan_one_symbol(sym, sym, XunLongIndicator())
        if sig is None or sig.empty:
            print("  近期无信号")
        else:
            sig = sig.copy()
            sig["signal_date"] = pd.to_datetime(sig["signal_date"]).dt.date
            recent = sig.sort_values("signal_date").tail(6)
            keep = [x for x in ("signal_date", "signal_type", "观海买点分", "close",
                                "RSI", "rank120", "段涨幅", "量比20日") if x in recent.columns]
            print(recent[keep].to_string(index=False))
    except Exception as e:
        print(f"  信号引擎调用失败: {type(e).__name__}: {e}")

    # ---- move probability, every horizon, both thresholds
    print(f"\n【移动概率（本仓库自建，21年滚动前瞻验证）】")
    model = pd.read_pickle(M.MODEL_PATH)
    for thr in (0.02, 0.05, 0.10):
        rows = M.predict_from_bars(sym, c, h, l, None, horizons=(1, 5, 10, 21),
                                   thr=thr, model=model, volume=v)
        ok = [r for r in rows if r.get("ok") and r["in_support"]]
        if not ok:
            print(f"  阈值 ±{thr*100:.0f}%: 全部期限都在可分辨区间外")
            continue
        print(f"  阈值 ±{thr*100:.0f}%")
        print(f"    {'期限':<6}{'典型波动':>9}{'幅度超阈值':>11}{'涨超':>8}{'跌超':>8}"
              f"{'偏差情形':>10}{'乐观情形':>10}{'可信度':>8}")
        for r in ok:
            print(f"    {str(r['h'])+'日':<6}{r['typical_move']*100:>8.1f}%{r['p_move']*100:>10.1f}%"
                  f"{r['p_up_big']*100:>7.1f}%{r['p_down_big']*100:>7.1f}%"
                  f"{r['q05']*100:>9.1f}%{r['q95']*100:>9.1f}%"
                  f"{M.GRADE_CN.get(r['grade'], r['grade']):>8}")

    # ---- probability of getting back to / below cost
    print(f"\n【相对成本 {cost:.2f} 的概率】")
    g = M.asset_class(sym)
    f = M.build_features(c, h, l, None, v)
    for hh in (5, 10, 21):
        spec = model["groups"][g]["per_h"].get(hh)
        if spec is None:
            continue
        cols = spec["cols"]
        if any(x not in f.columns for x in cols):
            continue
        sig_h = float(np.exp(M._design(f.iloc[[-1]], cols) @ spec["beta"]).item()) * np.sqrt(hh)
        z = spec["z"]
        need = np.log(cost / px)                       # log move required to reach cost
        p_below = float(np.searchsorted(z, need / sig_h, "right") / len(z))
        print(f"  {hh:>2}日后收盘低于成本的概率 {p_below*100:5.1f}%   "
              f"(需要 {cost/px-1:+.1%} 的移动)")

    # ---- option IV from today's snapshot, vs realized
    print(f"\n【期权隐含波动率（今日快照）】")
    try:
        iv = pd.read_csv("_iv_history.csv")
        r = iv[iv.symbol == sym]
        if len(r):
            r = r.iloc[-1]
            rv20 = float(np.log(c).diff().tail(20).std(ddof=1) * np.sqrt(252))
            print(f"  IV30 {r.iv30*100:.1f}%   IV60 {r.iv60*100:.1f}%   IV90 {r.iv90*100:.1f}%"
                  f"   期限斜率 {r.ts_slope*100:+.1f}pts")
            print(f"  已实现20日波动 {rv20*100:.1f}%   IV30/已实现 = {r.iv30/rv20:.2f}"
                  f"   {'（期权偏贵，适合卖保护/备兑）' if r.iv30/rv20 > 1.15 else '（期权不算贵，买保护相对划算）' if r.iv30/rv20 < 0.95 else '（大致公平）'}")
            print(f"  质量 {r.quality}   认沽认购分歧 {r.cp_spread_pts:.1f}pts (相对 {r.cp_spread_rel:.2f})")
        else:
            print(f"  今日快照中没有 {sym}")
    except Exception as e:
        print(f"  读取失败: {e}")

    # ---- earnings
    print(f"\n【财报】")
    try:
        cal = yf.Ticker(sym).calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            e0 = pd.Timestamp(ed[0] if isinstance(ed, list) else ed)
            dte = (e0 - pd.Timestamp(c.index[-1].date())).days
            print(f"  下次财报 {e0.date()}   距今 {dte} 天"
                  f"   {'⚠️ 在未来21日窗口内，σ 需乘 1.21' if dte <= 30 else '不在近月窗口内'}")
        else:
            print("  无日期")
    except Exception as e:
        print(f"  查询失败: {e}")

    # ---- sector / market context
    print(f"\n【同类与大盘】")
    peers = ["SMH", "NVDA", "AVGO", "QQQ", "SPY"]
    pr = yf.download(peers, period="6mo", progress=False, auto_adjust=True)["Close"]
    for p in peers:
        if p in pr.columns:
            s = pr[p].dropna()
            print(f"  {p:<5} 近1月 {float(s.iloc[-1]/s.iloc[-22]-1):+6.1%}"
                  f"   近3月 {float(s.iloc[-1]/s.iloc[-64]-1):+6.1%}")
    own1m, own3m = float(c.iloc[-1]/c.iloc[-22]-1), r63
    smh = pr["SMH"].dropna() if "SMH" in pr.columns else None
    if smh is not None:
        print(f"  {sym} 相对 SMH：近1月 {own1m - float(smh.iloc[-1]/smh.iloc[-22]-1):+.1%}"
              f"   近3月 {own3m - float(smh.iloc[-1]/smh.iloc[-64]-1):+.1%}")


if __name__ == "__main__":
    main()
