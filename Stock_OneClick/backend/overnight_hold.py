#!/usr/bin/env python3
"""
盘中多次扫描 → 收盘前定版：隔夜持仓候选

用法（在 backend 目录下运行）：
    ../../vcp_env/bin/python overnight_hold.py                  # 今天
    ../../vcp_env/bin/python overnight_hold.py --date 2026-09-03
    ../../vcp_env/bin/python overnight_hold.py --top 15 --min-score 60

前提：当天已经跑过 ≥2 次 scan_stocks.py（例如早盘一次 + 收盘前一次）。
每次 scan 都会把当次 RawSignals 落到 history/scan_result_<YYYYMMDD>_<HHMMSS>.xlsx，
本脚本读回同一天的所有快照，用“早盘 → 收盘前”的演化来筛隔夜持仓。

日内多次扫描才能看到的三件事（单次扫描永远看不到）：
  1. 持续性：早盘就在、收盘前还在的信号，不是盘中一根假突破刷出来的。
  2. 升级/降级：预警买入 → 正式买入 是升级；正式买入在收盘前掉回预警是降级。
  3. 收盘走强：同一根未收盘日线，两次扫描之间的价格漂移 = 资金是买进收盘还是抛进收盘。

⚠️ 隔夜分是**启发式加权，未经回测**。它只是把上面三件事量化成一个可排序的数字，
   不是已验证的 alpha。本仓库里“看起来合理但回测没有 edge”的规则已经有一堆
   （MTF 对齐、RSI-MA 交叉、RSI/OBV 背离、预警升级…），这个也要照样先回测再当真。
   要证伪它，需要按本脚本的分档回放历史双扫描日的隔夜收益。
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date as date_cls, datetime, time as time_cls
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = BASE_DIR / "history"
OUT_DIR = BASE_DIR / "overnight_hold"

RESULT_RE = re.compile(r"^scan_result_(\d{8})_(\d{6})\.xlsx$")
TV_RE = re.compile(r"^tv_buy_today_(\d{8})_(\d{6})\.txt$")

# 规则等级：越高越接近“可以真金白银买”的确认度。
# 正式买入=日线 BUY A/0出；二进宫买入点=回踩不破后再次绿柱；预警买入=4H BUY A/0出；
# 第一观察点=低位首绿柱（仅预备观察）。回调买入点是择时参考，不构成等级。
RULE_TIER = {
    "正式买入": 3,
    "二进宫买入点": 2,
    "预警买入": 2,
    "第一观察点": 1,
}
TIER_NAME = {3: "正式买入", 2: "预警/二进宫", 1: "第一观察点", 0: "-"}

# 美股常规交易时段（ET）。用来把盘中运行的量比按已走完的时段比例外推。
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time_cls(9, 30)
MARKET_CLOSE = time_cls(16, 0)
SESSION_MINUTES = 6.5 * 60


# ---------------------------------------------------------------- 读取当天快照


def find_day_runs(day: date_cls, history_dir: Path = HISTORY_DIR) -> list[Path]:
    """按时间正序返回当天所有 scan_result 快照。"""
    stamp = day.strftime("%Y%m%d")
    hits = []
    for p in history_dir.glob(f"scan_result_{stamp}_*.xlsx"):
        m = RESULT_RE.match(p.name)
        if m:
            hits.append((m.group(2), p))
    return [p for _, p in sorted(hits)]


def load_run_signals(path: Path) -> pd.DataFrame:
    """读一次扫描的 RawSignals，只保留该 run 自己 run_date 当天的信号行。"""
    try:
        df = pd.read_excel(path, sheet_name="RawSignals")
    except Exception as e:
        print(f"⚠️ 读不到 RawSignals，跳过 {path.name}：{e}")
        return pd.DataFrame()
    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["signal_type"] = df["signal_type"].fillna("").astype(str)
    df["signal_side"] = df["signal_side"].fillna("").astype(str).str.upper()
    df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce").dt.date
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.date
    for c in ["close", "volume", "vol_ma20", "RSI", "rank120", "Gann_gain_pct",
              "buy_score", "buy_score_raw", "H4_RSI", "H4_FJ"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # 只要“今天这根 bar”的信号：补回的往日信号不参与隔夜判断。
    df = df[df["signal_date"] == df["run_date"]]
    df = df[df["symbol"] != ""]

    run_time = str(df["run_time"].dropna().iloc[0]) if not df["run_time"].dropna().empty else ""
    df.attrs["run_time"] = run_time
    df.attrs["source"] = path.name
    df.attrs["bar_partial"] = bool(df["bar_partial"].fillna(False).any()) if "bar_partial" in df else False
    return df


def build_tv_map(day: date_cls, history_dir: Path = HISTORY_DIR) -> dict[str, str]:
    """
    从当天各次扫描的 tv_buy_today_<ts>.txt 里反查 symbol → TradingView 代码。
    RawSignals 不带 exchange，而这些纯导入清单本来就是同一批 run 写出来的，
    合并全天所有 run 才能覆盖“早盘有、收盘前消失”的票。
    """
    stamp = day.strftime("%Y%m%d")
    out: dict[str, str] = {}
    for p in sorted(history_dir.glob(f"tv_buy_today_{stamp}_*.txt")):
        if not TV_RE.match(p.name):
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            code = line.strip()
            if not code or code.startswith("#"):
                continue
            bare = code.split(":")[-1].strip().upper()
            if bare:
                out.setdefault(bare, code)
    return out


# ---------------------------------------------------------------- 日内演化


def _run_time_to_session_fraction(run_time: str, day: date_cls) -> float:
    """
    run_time 是扫描机本地时钟。换算到 ET，得出这根日线已经走完的时段比例，
    用来把盘中量比外推成全天口径（盘中量比天生偏低，直接比会系统性低估）。
    """
    try:
        hh, mm, ss = (int(x) for x in str(run_time).split(":"))
    except Exception:
        return 1.0
    local_tz = datetime.now().astimezone().tzinfo
    local_dt = datetime.combine(day, time_cls(hh, mm, ss)).replace(tzinfo=local_tz)
    et = local_dt.astimezone(MARKET_TZ)
    open_dt = datetime.combine(et.date(), MARKET_OPEN, tzinfo=MARKET_TZ)
    elapsed = (et - open_dt).total_seconds() / 60.0
    return float(np.clip(elapsed / SESSION_MINUTES, 0.05, 1.0))


def _agg_one_run(df: pd.DataFrame) -> pd.DataFrame:
    """把一次扫描压成 per-symbol 一行。"""
    buys = df[df["signal_side"] == "BUY"].copy()
    sells = set(df.loc[df["signal_side"] == "SELL", "symbol"])
    if buys.empty:
        return pd.DataFrame()

    buys["_tier"] = buys["signal_type"].map(RULE_TIER).fillna(0).astype(int)
    buys["_score"] = buys["buy_score_raw"].fillna(buys["buy_score"])

    rows = []
    for sym, g in buys.groupby("symbol"):
        rules = [r for r in pd.unique(g["signal_type"]) if r]
        # 只有回调买入点（tier 0）的票不算买入信号：它是择时参考，不开仓。
        if int(g["_tier"].max()) == 0:
            continue
        best = g.loc[g["_tier"].idxmax()]
        rows.append({
            "symbol": sym,
            "name": str(best.get("name", "")),
            "板块": str(best.get("板块", "")),
            "tier": int(g["_tier"].max()),
            "rules": " + ".join(rules),
            "score": float(g["_score"].max()) if g["_score"].notna().any() else np.nan,
            "close": float(best["close"]) if pd.notna(best["close"]) else np.nan,
            "volume": float(best["volume"]) if pd.notna(best["volume"]) else np.nan,
            "vol_ma20": float(best["vol_ma20"]) if pd.notna(best["vol_ma20"]) else np.nan,
            "RSI": float(best["RSI"]) if pd.notna(best["RSI"]) else np.nan,
            "rank120": float(best["rank120"]) if pd.notna(best["rank120"]) else np.nan,
            "Gann_gain_pct": float(best["Gann_gain_pct"]) if pd.notna(best["Gann_gain_pct"]) else np.nan,
            "H4_RSI": float(best["H4_RSI"]) if pd.notna(best["H4_RSI"]) else np.nan,
            "H4_FJ": float(best["H4_FJ"]) if pd.notna(best["H4_FJ"]) else np.nan,
            "has_pullback": "回调买入点" in rules,
        })
    out = pd.DataFrame(rows)
    out.attrs["sell_symbols"] = sells
    return out


def build_day_evolution(runs: list[pd.DataFrame], day: date_cls) -> pd.DataFrame:
    """
    合并当天 N 次扫描 → 每个“收盘前那次仍在场”的票一行，带日内演化列。
    收盘前最后一次扫描是基准（价格/量能最接近真实收盘），早盘各次提供确认与漂移。
    """
    aggs = [(_agg_one_run(df), df) for df in runs]
    aggs = [(a, df) for a, df in aggs if not a.empty]
    if not aggs:
        return pd.DataFrame()

    final_agg, final_df = aggs[-1]
    first_agg, first_df = aggs[0]
    n_runs = len(aggs)

    final_sells = final_agg.attrs.get("sell_symbols", set())
    frac = _run_time_to_session_fraction(final_df.attrs.get("run_time", ""), day)

    # 每个 symbol 在几次扫描里出现过
    presence: dict[str, int] = {}
    for a, _ in aggs:
        for sym in a["symbol"]:
            presence[sym] = presence.get(sym, 0) + 1

    first_map = first_agg.set_index("symbol")
    rows = []
    for _, r in final_agg.iterrows():
        sym = r["symbol"]
        in_first = sym in first_map.index
        f = first_map.loc[sym] if in_first else None

        first_close = float(f["close"]) if in_first and pd.notna(f["close"]) else np.nan
        intraday_ret = (
            r["close"] / first_close - 1.0
            if pd.notna(r["close"]) and pd.notna(first_close) and first_close else np.nan
        )
        tier_delta = int(r["tier"] - f["tier"]) if in_first else 0
        score_delta = (
            r["score"] - float(f["score"])
            if in_first and pd.notna(r["score"]) and pd.notna(f["score"]) else np.nan
        )
        vol_ratio = (
            r["volume"] / r["vol_ma20"]
            if pd.notna(r["volume"]) and pd.notna(r["vol_ma20"]) and r["vol_ma20"] else np.nan
        )

        rows.append({
            **{k: r[k] for k in ["symbol", "name", "板块", "tier", "rules", "score",
                                 "close", "RSI", "rank120", "Gann_gain_pct",
                                 "H4_RSI", "H4_FJ", "has_pullback"]},
            "n_runs_seen": presence.get(sym, 1),
            "n_runs_total": n_runs,
            "in_first_run": in_first,
            "first_close": first_close,
            "intraday_ret": intraday_ret,
            "first_tier": int(f["tier"]) if in_first else 0,
            "tier_delta": tier_delta,
            "score_delta": score_delta,
            "vol_ratio": vol_ratio,
            "vol_ratio_proj": vol_ratio / frac if pd.notna(vol_ratio) else np.nan,
            "has_sell_signal": sym in final_sells,
        })

    ev = pd.DataFrame(rows)
    ev.attrs["session_fraction"] = frac

    # 早盘有买入信号、收盘前那次没了 → 单独一档，明确不建议隔夜
    final_syms = set(final_agg["symbol"])
    faded = []
    for a, df in aggs[:-1]:
        for _, r in a.iterrows():
            if r["symbol"] not in final_syms:
                faded.append({
                    "symbol": r["symbol"], "name": r["name"], "板块": r["板块"],
                    "rules": r["rules"], "score": r["score"], "close": r["close"],
                    "last_seen": df.attrs.get("run_time", ""),
                })
    ev.attrs["faded"] = (
        pd.DataFrame(faded).drop_duplicates("symbol", keep="last")
        if faded else pd.DataFrame()
    )
    return ev


# ---------------------------------------------------------------- 隔夜分


def _bucket(value, thresholds: list[tuple[float, float]], default: float) -> float:
    """thresholds 从高到低：value >= 阈值 → 给分。"""
    if pd.isna(value):
        return default
    for cutoff, pts in thresholds:
        if value >= cutoff:
            return pts
    return 0.0


def score_overnight(ev: pd.DataFrame) -> pd.DataFrame:
    """
    透明加权，满分 100，再扣过热分。各分项就是“日内多扫描才能看到的东西”：

      持续性   30  早盘就在 + 收盘前还在 > 只在收盘前冒出来
      规则等级 25  正式买入 > 预警/二进宫 > 第一观察点
      日内演化 20  等级升级 + 分数上行
      收盘走强 15  两次扫描之间的价格漂移（买进收盘 vs 抛进收盘）
      量能确认 10  收盘前量比（按已走时段外推到全天口径）
      过热扣分 -18 RSI 过高 / 段涨幅已经很大 → 隔夜跳空风险
    """
    if ev.empty:
        return ev
    d = ev.copy()
    n_total = int(d["n_runs_total"].iloc[0]) if "n_runs_total" in d else 1

    # 1) 持续性
    def persistence(r):
        if n_total <= 1:
            return 10.0  # 只跑过一次，没有日内确认可言
        if r["n_runs_seen"] >= n_total:
            return 30.0
        if r["in_first_run"]:
            return 25.0
        return 10.0 if r["n_runs_seen"] > 1 else 8.0
    d["p_持续性"] = d.apply(persistence, axis=1)

    # 2) 规则等级
    d["p_等级"] = d["tier"].map({3: 25.0, 2: 16.0, 1: 7.0}).fillna(0.0)

    # 3) 日内演化：等级变化 + 分数变化。收盘前才冒出来的票没有“演化”可言，
    #    不能白拿“维持”那 8 分——它恰恰是本脚本要区分出来的那一类。
    def evolution(r):
        if not r["in_first_run"]:
            return 4.0
        tier_pts = 14.0 if r["tier_delta"] > 0 else (8.0 if r["tier_delta"] == 0 else 0.0)
        sd = r["score_delta"]
        score_pts = 3.0 if pd.isna(sd) else (6.0 if sd >= 2 else (3.0 if sd > -2 else 0.0))
        return tier_pts + score_pts
    d["p_演化"] = d.apply(evolution, axis=1)

    # 4) 收盘走强
    d["p_收盘"] = d["intraday_ret"].map(
        lambda x: 7.0 if pd.isna(x) else _bucket(
            x, [(0.010, 15.0), (0.002, 11.0), (-0.002, 7.0), (-0.010, 3.0)], 7.0
        )
    )

    # 5) 量能确认（全天口径外推）
    d["p_量能"] = d["vol_ratio_proj"].map(
        lambda x: 5.0 if pd.isna(x) else _bucket(x, [(1.5, 10.0), (1.0, 8.0), (0.7, 5.0)], 5.0)
    )

    # 6) 过热扣分
    d["p_过热"] = -(
        d["RSI"].map(lambda x: 0.0 if pd.isna(x) else (8.0 if x >= 70 else (4.0 if x >= 65 else 0.0)))
        + d["Gann_gain_pct"].map(
            lambda x: 0.0 if pd.isna(x) else (7.0 if x >= 0.25 else (3.0 if x >= 0.15 else 0.0))
        )
        + d["has_sell_signal"].map(lambda x: 3.0 if x else 0.0)
    )

    d["隔夜分"] = (
        d[["p_持续性", "p_等级", "p_演化", "p_收盘", "p_量能"]].sum(axis=1) + d["p_过热"]
    ).clip(lower=0, upper=100).round(1)

    # 收盘前还带卖出信号的，直接出局：隔夜方向本身是矛盾的
    d["disqualified"] = d["has_sell_signal"]

    return d.sort_values(["disqualified", "隔夜分", "tier", "symbol"],
                         ascending=[True, False, False, True]).reset_index(drop=True)


# ---------------------------------------------------------------- 报告


def _fmt(v, spec, suffix="", dash="-"):
    v = pd.to_numeric(v, errors="coerce")
    if pd.isna(v):
        return dash
    return format(float(v), spec) + suffix


def format_report(
    scored: pd.DataFrame,
    ev: pd.DataFrame,
    runs: list[pd.DataFrame],
    day: date_cls,
    top: int,
    min_score: float,
) -> tuple[str, str]:
    """返回 (备注版正文, 纯 TradingView 导入正文)。"""
    date_str = day.strftime("%Y-%m-%d")
    times = [df.attrs.get("run_time", "?") for df in runs]
    final_partial = bool(runs[-1].attrs.get("bar_partial", False))
    frac = ev.attrs.get("session_fraction", 1.0) if not ev.empty else 1.0

    L = [
        f"# {date_str} 隔夜持仓候选（当天 {len(runs)} 次扫描"
        + ("的日内演化）" if len(runs) >= 2 else "，无日内演化可比）"),
        f"# 扫描时刻（本地）：{' → '.join(times)}   定版基准=最后一次",
        "# 隔夜分 = 持续性30 + 规则等级25 + 日内演化20 + 收盘走强15 + 量能确认10 − 过热扣分",
        "# ⚠️ 启发式加权，未经回测。只是把“早盘→收盘前的演化”量化排序，不是已验证 edge。",
    ]
    if len(runs) < 2:
        L.append("# ⚠️ 当天只找到 1 次扫描：没有日内确认，持续性分全部按最低给。收盘前再跑一次 scan_stocks.py。")
    if final_partial:
        L.append(
            f"# ⚠️ 最后一次扫描时日线仍未收盘（约走完 {frac:.0%} 时段）："
            "收盘走强只测到该时刻，量比已按时段外推。越贴近 16:00 ET 再跑，定版越准。"
        )
    L.append("")

    ok = scored[~scored["disqualified"]] if not scored.empty else scored
    bad = scored[scored["disqualified"]] if not scored.empty else pd.DataFrame()

    header = (
        "# TV代码 | 隔夜分 | 出现次数 | 规则(收盘前) | 日内变化 | 日内涨跌 | "
        "观海买点分Δ | 量比(外推) | RSI | 段涨幅 | 板块"
    )

    def emit(title: str, rows: pd.DataFrame, note: str = ""):
        L.append(f"# —— {title} ——")
        if note:
            L.append(f"# {note}")
        if rows.empty:
            L.append("# （无）")
            L.append("")
            return
        L.append(header)
        for _, r in rows.iterrows():
            tv = r["tv"]
            if r["in_first_run"] and r["tier_delta"] > 0:
                change = f"升级({TIER_NAME.get(r['first_tier'], '-')}→{TIER_NAME.get(r['tier'], '-')})"
            elif r["in_first_run"] and r["tier_delta"] < 0:
                change = f"降级({TIER_NAME.get(r['first_tier'], '-')}→{TIER_NAME.get(r['tier'], '-')})"
            elif r["in_first_run"]:
                change = "维持"
            else:
                change = "收盘前新增"
            L.append(
                f"{tv} | {r['隔夜分']:.0f} | {int(r['n_runs_seen'])}/{int(r['n_runs_total'])} | "
                f"{r['rules']} | {change} | {_fmt(r['intraday_ret'], '+.2%')} | "
                f"{_fmt(r['score_delta'], '+.0f')} | {_fmt(r['vol_ratio_proj'], '.1f', 'x')} | "
                f"{_fmt(r['RSI'], '.0f')} | {_fmt(r['Gann_gain_pct'], '.1%')} | {r['板块']}"
            )
        L.append("")

    picks = ok[ok["隔夜分"] >= min_score].head(top) if not ok.empty else ok
    rest = ok.drop(picks.index) if not ok.empty else ok

    emit(
        f"A. 隔夜首选（隔夜分 ≥ {min_score:.0f}，取前 {top}）",
        picks,
        "早盘就在、收盘前仍在、且买盘没有在下午撤退的票。",
    )
    emit(
        f"B. 其余在场买入信号（未进前 {top} / 确认不足）",
        rest.head(40) if not rest.empty else rest,
        "收盘前仍有买入信号，但没进首选名额，或持续性、等级、收盘走强其中一项不够。",
    )
    if not bad.empty:
        emit(
            "C. 出局：收盘前同时带卖出信号",
            bad,
            "同一票同时挂买和卖，隔夜方向矛盾 → 不参与隔夜。",
        )

    faded = ev.attrs.get("faded", pd.DataFrame()) if not ev.empty else pd.DataFrame()
    L.append("# —— D. 日内消失（早盘有买入信号，收盘前那次已无）——")
    L.append("# 单看收盘那一次扫描是看不见这批的。盘中信号没撑到收盘 → 不建议隔夜。")
    if faded is None or faded.empty:
        L.append("# （无）")
    else:
        L.append("# TV代码 | 早盘规则 | 早盘观海买点分 | 最后出现时刻 | 板块")
        for _, r in faded.sort_values("score", ascending=False).iterrows():
            L.append(
                f"{r['tv']} | {r['rules']} | {_fmt(r['score'], '.0f')} | "
                f"{r['last_seen']} | {r['板块']}"
            )
    L.append("")

    down = ok[ok["tier_delta"] < 0] if not ok.empty else ok
    L.append("# —— E. 日内降级（等级掉了，仍在场）——")
    if down.empty:
        L.append("# （无）")
    else:
        for _, r in down.iterrows():
            L.append(
                f"{r['tv']} | {TIER_NAME.get(r['first_tier'], '-')} → "
                f"{TIER_NAME.get(r['tier'], '-')} | {_fmt(r['intraday_ret'], '+.2%')} | {r['板块']}"
            )

    pure = "\n".join(picks["tv"].tolist()) + ("\n" if not picks.empty else "")
    return "\n".join(L) + "\n", pure


# ---------------------------------------------------------------- main


def run(day: date_cls, top: int, min_score: float,
        history_dir: Path = HISTORY_DIR, out_dir: Path = OUT_DIR) -> int:
    paths = find_day_runs(day, history_dir)
    if not paths:
        print(f"❌ {day} 没有找到任何 scan_result 快照。先跑 scan_stocks.py。")
        return 1

    runs = [d for d in (load_run_signals(p) for p in paths) if not d.empty]
    if not runs:
        print(f"❌ {day} 的快照里没有当天信号行。")
        return 1

    print(f"📄 当天扫描 {len(runs)} 次：" + ", ".join(
        f"{d.attrs.get('run_time', '?')}({len(d)}行)" for d in runs))
    if len(runs) < 2:
        print("⚠️ 只有 1 次扫描 → 无法做日内确认。收盘前再跑一次 scan_stocks.py 效果才对。")

    ev = build_day_evolution(runs, day)
    if ev.empty:
        print("❌ 收盘前那次扫描没有买入信号，无隔夜候选。")
        return 0

    scored = score_overnight(ev)
    tv_map = build_tv_map(day, history_dir)
    scored["tv"] = scored["symbol"].map(lambda s: tv_map.get(s, s))
    faded = ev.attrs.get("faded", pd.DataFrame())
    if faded is not None and not faded.empty:
        faded["tv"] = faded["symbol"].map(lambda s: tv_map.get(s, s))
        ev.attrs["faded"] = faded

    notes, pure = format_report(scored, ev, runs, day, top, min_score)

    out_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    date_str = day.strftime("%Y-%m-%d")
    ts = f"{day.strftime('%Y%m%d')}_{str(runs[-1].attrs.get('run_time', '')).replace(':', '') or '000000'}"
    targets = [
        (out_dir / f"overnight_hold_{date_str}.txt", notes),
        (history_dir / f"overnight_hold_{ts}.txt", notes),
        (out_dir / f"overnight_hold_tv_{date_str}.txt", pure),
    ]
    # _latest 只在跑当天时更新：用 --date 回看旧日子不应该把 latest 写成过期内容。
    if day == datetime.now().date():
        targets += [
            (out_dir / "overnight_hold_latest.txt", notes),
            (out_dir / "overnight_hold_tv_latest.txt", pure),
        ]
    for path, content in targets:
        path.write_text(content, encoding="utf-8")

    print()
    print(notes)
    n_pick = len(pure.strip().splitlines()) if pure.strip() else 0
    print(f"✅ 隔夜首选 {n_pick} 只（隔夜分 ≥ {min_score:.0f}）")
    print(f"✅ 备注版：{out_dir / f'overnight_hold_{date_str}.txt'}")
    print(f"✅ 纯导入：{out_dir / f'overnight_hold_tv_{date_str}.txt'}")
    print(f"✅ 历史留档：{history_dir / f'overnight_hold_{ts}.txt'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="合并当天多次扫描，给出隔夜持仓候选（需当天 ≥2 次 scan_stocks.py）"
    )
    ap.add_argument("--date", default="", help="YYYY-MM-DD，默认今天")
    ap.add_argument("--top", type=int, default=12, help="首选档最多输出几只（默认12）")
    ap.add_argument("--min-score", type=float, default=65.0, help="进入首选档的隔夜分门槛（默认65）")
    args = ap.parse_args(argv)

    if args.date:
        try:
            day = datetime.strptime(args.date.strip(), "%Y-%m-%d").date()
        except ValueError:
            print(f"❌ --date 格式应为 YYYY-MM-DD，收到：{args.date}")
            return 2
    else:
        day = datetime.now().date()

    return run(day, args.top, args.min_score)


if __name__ == "__main__":
    sys.exit(main())
