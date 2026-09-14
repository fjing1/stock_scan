"""
_move_week.py — run the move-probability forecast over the current buy-signal list for the
coming week (h=5 trading sessions).

Batches the bar download instead of calling move_prob.predict() per symbol (which would re-fetch
^VIX 50 times), and looks up each single name's next earnings date so the earnings sigma
multiplier can be applied where the report actually lands inside the window.

Usage: ../../vcp_env/bin/python _move_week.py [--h 5] [--file <tv_buy_today_*.txt>]
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import move_prob as M

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
INDICES = ["SPY", "QQQ", "IWM"]


def read_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line.split(":")[-1].strip())
    return list(dict.fromkeys(out))


def next_earnings(sym: str):
    """Next scheduled earnings date, or None. Best-effort: yfinance's calendar is often empty."""
    try:
        cal = yf.Ticker(sym).calendar
        if isinstance(cal, dict):
            d = cal.get("Earnings Date")
            if isinstance(d, list) and d:
                return pd.Timestamp(d[0])
            if d is not None:
                return pd.Timestamp(d)
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=5)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()

    f = Path(args.file) if args.file else sorted(
        (BASE / "tv_buy_signals").glob("tv_buy_today_2*.txt"))[-1]
    names = read_list(f)
    syms = INDICES + [s for s in names if s not in INDICES]
    print(f"signal list: {f.name}  ({len(names)} names) + {len(INDICES)} indices\n")

    raw = yf.download(syms + ["^VIX"], period="2y", progress=False,
                      auto_adjust=True, group_by="column", threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        C, H, Lo = raw["Close"], raw["High"], raw["Low"]
    else:
        C = raw[["Close"]]; H = raw[["High"]]; Lo = raw[["Low"]]
    vix = pd.to_numeric(C["^VIX"], errors="coerce").dropna() if "^VIX" in C else None

    last_bar = C.drop(columns=["^VIX"], errors="ignore").dropna(how="all").index[-1]
    print(f"last closed bar: {last_bar.date()}   forecast window: next {args.h} sessions\n")

    with ThreadPoolExecutor(max_workers=12) as ex:
        earn = dict(zip(syms, ex.map(next_earnings, syms)))

    model = pd.read_pickle(M.MODEL_PATH)
    rows = []
    for s in syms:
        if s not in C.columns:
            continue
        c = pd.to_numeric(C[s], errors="coerce").dropna()
        c = c[c > 0]
        if len(c) < 120:
            rows.append({"sym": s, "err": "历史不足"})
            continue
        hi = pd.to_numeric(H[s], errors="coerce").reindex(c.index)
        lo = pd.to_numeric(Lo[s], errors="coerce").reindex(c.index)

        ed = earn.get(s)
        in_win = bool(ed is not None and pd.notna(ed)
                      and last_bar < pd.Timestamp(ed).tz_localize(None)
                      <= last_bar + pd.Timedelta(days=int(args.h * 1.45)))
        e_set = {args.h} if in_win else None

        # This ticker's OWN trailing up-rate over the same horizon: a base rate, not a forecast.
        # Included only so the model's structurally-constant P(up) can be compared against the
        # thing people actually mean by "will it go up" -- and so the noise in it is visible.
        fwd2y = (c.shift(-args.h) / c - 1.0).dropna()
        rec = {"sym": s, "cls": M.asset_class(s), "earn": ed, "earn_in": in_win,
               "hist_up": float((fwd2y.tail(504) > 0).mean()) if len(fwd2y) >= 200 else np.nan}
        for thr in (0.02, 0.05):
            r = M.predict_from_bars(s, c, hi, lo, vix if M.asset_class(s) == "index" else None,
                                    horizons=(args.h,), thr=thr, model=model, earnings_in=e_set)
            r = r[0] if r else {}
            tag = f"{int(thr*100)}"
            rec[f"ok{tag}"] = bool(r.get("ok") and r.get("in_support"))
            if r.get("ok"):
                rec.update({f"mv{tag}": r["p_move"], f"up{tag}": r["p_up_big"],
                            f"dn{tag}": r["p_down_big"], f"g{tag}": r["grade"],
                            "typ": r["typical_move"], "q05": r["q05"], "q95": r["q95"],
                            "up_any": r["p_up_any"], "exp_ret": r["exp_return"]})
        rows.append(rec)

    df = pd.DataFrame(rows)
    ok = df[df.get("typ").notna()] if "typ" in df else df

    print("=" * 108)
    print(f"未来 {args.h} 个交易日 波动概率   阈值 ±2% 与 ±5%（只显示落在可分辨区间内的格子）")
    print("=" * 108)
    print(f"{'代码':<8}{'类别':<6}{'典型波动':>9}{'幅度超2%':>11}{'涨超2%':>8}{'跌超2%':>8}"
          f"{'幅度超5%':>11}{'涨超5%':>8}{'跌超5%':>8}{'上涨(任意)':>11}{'自身历史':>9}"
          f"{'期望收益':>10}{'偏差':>8}{'乐观':>8}{'财报':>6}")
    for _, r in ok.sort_values("typ", ascending=False).iterrows():
        def cell(tag, key, w=8):
            return (f"{r[key]*100:>{w}.1f}%" if r.get(f"ok{tag}") and pd.notna(r.get(key))
                    else f"{'—':>{w}}")
        print(f"{r['sym']:<8}{('指数' if r['cls']=='index' else '个股'):<6}"
              f"{r['typ']*100:>8.1f}%{cell('2','mv2',10)}{cell('2','up2')}{cell('2','dn2')}"
              f"{cell('5','mv5',10)}{cell('5','up5')}{cell('5','dn5')}"
              f"{r['up_any']*100:>10.1f}%"
              + (f"{r['hist_up']*100:>8.1f}%" if pd.notna(r.get('hist_up')) else f"{'—':>9}")
              + f"{r['exp_ret']*100:>+9.2f}%"
              + f"{r['q05']*100:>7.1f}%{r['q95']*100:>7.1f}%{('是' if r['earn_in'] else ''):>6}")

    print(f"\n落在可分辨区间内的标的：±2% 共 {int(ok['ok2'].sum())}/{len(ok)}，"
          f"±5% 共 {int(ok['ok5'].sum())}/{len(ok)}")
    n_e = int(ok["earn_in"].sum())
    print(f"下周有财报（波动率已放大 {M.EARN_MULT.get(args.h, 1):.2f} 倍）：{n_e} 个")
    ua = ok.groupby("cls")["up_any"].agg(["min", "max"])
    print("注意：技能全部集中在「幅度超阈值」两列。涨/跌拆分由漂移和偏度决定，不是方向预测。")
    print(f"'上涨(任意)'在同一类别内是常数（指数 {ua.loc['index','min']*100:.1f}%，"
          f"个股 {ua.loc['single','min']*100:.1f}%）——判定点 0/波动率 = 0，与波动率无关，")
    print("即模型对方向没有任何信息。「自身历史」是该标的过去两年同期限的上涨频率，")
    print("属于基准率而非预测，标准误约 ±5 个百分点，跨标的差异基本是噪音。")
    print("「偏差」「乐观」= 第 5 / 第 95 百分位收益。「期望收益」= 分布均值（截尾后）。")
    print("期望收益 ≡ 漂移 × 波动率，波动越大这一列必然越高，是结构使然而非选股依据——")
    print("实证上低波动股票的风险调整后收益反而更好，请勿按此列排序。")


if __name__ == "__main__":
    main()
