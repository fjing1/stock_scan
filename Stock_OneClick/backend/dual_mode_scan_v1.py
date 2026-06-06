from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple

import pandas as pd


@dataclass
class ScanConfig:
    # ---- 通用 ----
    near_pct: float = 0.03
    focus_top_n: int = 5

    # ---- 低位启动 ----
    low_start_lookback: int = 60
    low_zone_quantile: float = 0.30
    low_green_recent_bars: int = 8
    simple_pullback_max_bars: int = 8

    # ---- 二进宫 ----
    anchor_window: int = 3
    impulse_max_bars: int = 60
    retrace_levels: Tuple[float, float, float] = (0.618, 0.5, 0.382)

    # 高高锁定（主升段枢轴高）
    high_lock_drawdown_pct: float = 0.08
    high_lock_atr_len: int = 14
    high_lock_atr_mult: float = 1.8
    high_lock_no_new_high_bars: int = 3
    high_lock_timeout_bars: int = 10
    high_lock_ema_span: int = 8

    # ---- 4H 节奏近低位 ----
    t_low_threshold_4h: float = 10.0


@dataclass
class PromptTag:
    symbol: str
    mode: str
    tag: str
    status: str
    note: str
    current_price: float


@dataclass
class DetailCard:
    symbol: str
    mode: str
    current_price: float
    trigger_reason: str
    daily_state: str
    h4_state: str
    position_state: str
    levels: Dict[str, float]
    risk_note: str


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 or pd.isna(b) else a / b


def safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.replace(0, pd.NA)
    out = a / denom
    return out.fillna(0)


def xsa(series: pd.Series, length: int, weight: int = 1) -> pd.Series:
    out = pd.Series(index=series.index, dtype="float64")
    ma = series.rolling(length).mean()
    for i in range(len(series)):
        if i == 0:
            out.iloc[i] = ma.iloc[i] if not pd.isna(ma.iloc[i]) else series.iloc[i]
        else:
            prev = out.iloc[i - 1]
            src = series.iloc[i]
            if pd.isna(prev):
                out.iloc[i] = ma.iloc[i] if not pd.isna(ma.iloc[i]) else src
            else:
                out.iloc[i] = (src * weight + prev * (length - weight)) / length
    return out


def crossover(a: pd.Series, b: pd.Series | float) -> pd.Series:
    if not isinstance(b, pd.Series):
        b = pd.Series(float(b), index=a.index)
    return (a > b) & (a.shift(1) <= b.shift(1))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    up_rma = up.ewm(alpha=1 / length, adjust=False).mean()
    down_rma = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = up_rma / down_rma.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)


def compute_bbuy(df: pd.DataFrame) -> pd.Series:
    d1 = (df["close"] + df["low"] + df["high"]) / 3.0
    d2 = ema(d1, 6)
    d3 = ema(d2, 5)
    return crossover(d2, d3)


def compute_l2_t_p(df: pd.DataFrame, K: int = 9, D: int = 3, mid_period: int = 58) -> pd.DataFrame:
    high_roll = df["high"].rolling(K).max()
    low_roll = df["low"].rolling(K).min()
    den_k = (high_roll - low_roll).replace(0, 1e-9)

    var1b = (high_roll - df["close"]) / den_k * 100 - 70
    var2b = xsa(var1b, K, 1) + 100
    var3b = (df["close"] - low_roll) / den_k * 100
    var4b = xsa(var3b, D, 1)
    var5b = xsa(var4b, D, 1) + 100
    var6b = var5b - var2b
    trend = (var6b - 45).clip(lower=0)

    low_prev = df["low"].shift(1)
    abs_move = (df["low"] - low_prev).abs()
    up_move = (df["low"] - low_prev).clip(lower=0)
    s_abs = xsa(abs_move.fillna(0), D, 1)
    s_up = xsa(up_move.fillna(0), D, 1)
    var3q = safe_div_series(s_abs, s_up) * 100.0
    tmp = pd.Series(index=df.index, dtype="float64")
    tmp[df["close"].diff() > 0] = var3q[df["close"].diff() > 0] * 10.0
    tmp[df["close"].diff() <= 0] = var3q[df["close"].diff() <= 0] / 10.0
    var4q = ema(tmp.fillna(0), D)

    var5q = df["low"].rolling(30).min()
    var6q = var4q.rolling(30).max()
    sma_mid = sma(df["close"], mid_period)
    var7q = (~sma_mid.isna()).astype(float)
    var8q = (
        ema(pd.Series((df["low"] <= var5q).astype(float) * ((var4q + var6q * 2.0) / 2.0), index=df.index), D)
        / 999.0
        * var7q
    )
    pump = var8q.clip(upper=100).fillna(0)

    out = pd.DataFrame(index=df.index)
    out["t"] = trend.fillna(0)
    out["p"] = pump.fillna(0)
    out["bbuy"] = compute_bbuy(df)
    out["rsi"] = rsi(df["close"])
    return out


def calc_retrace_levels(low_low: float, high_high: float, levels: Tuple[float, float, float]) -> Dict[str, float]:
    price_range = high_high - low_low
    return {f"{lv:.3f}".rstrip("0").rstrip("."): low_low + price_range * lv for lv in levels}


def _compute_atr(df_daily: pd.DataFrame, length: int) -> pd.Series:
    prev_close = df_daily["close"].shift(1)
    tr1 = df_daily["high"] - df_daily["low"]
    tr2 = (df_daily["high"] - prev_close).abs()
    tr3 = (df_daily["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / max(1, length), adjust=False).mean()


def _locate_impulse_high(
    df_daily: pd.DataFrame,
    start_loc: int,
    cfg: ScanConfig,
) -> Tuple[float, pd.Timestamp, str]:
    end_loc = min(len(df_daily) - 1, start_loc + cfg.impulse_max_bars)
    if end_loc <= start_loc:
        idx = df_daily.index[start_loc]
        return float(df_daily["high"].iloc[start_loc]), idx, "fallback_window_max"

    ema8 = ema(df_daily["close"], cfg.high_lock_ema_span)
    atr = _compute_atr(df_daily, cfg.high_lock_atr_len)

    cand_high = float(df_daily["high"].iloc[start_loc])
    cand_loc = start_loc
    bars_since_new_high = 0

    for i in range(start_loc + 1, end_loc + 1):
        h = float(df_daily["high"].iloc[i])
        c = float(df_daily["close"].iloc[i])
        l = float(df_daily["low"].iloc[i])

        if h >= cand_high:
            cand_high = h
            cand_loc = i
            bars_since_new_high = 0
            continue

        bars_since_new_high += 1
        if cand_loc <= start_loc:
            continue

        atr_i = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0
        atr_ratio = safe_div(cfg.high_lock_atr_mult * atr_i, cand_high)
        need_drop = max(cfg.high_lock_drawdown_pct, atr_ratio)
        drawdown = safe_div(cand_high - l, cand_high)
        cond_drawdown = drawdown >= need_drop

        ema_i = float(ema8.iloc[i]) if not pd.isna(ema8.iloc[i]) else c
        cond_structure = bars_since_new_high >= cfg.high_lock_no_new_high_bars and c < ema_i
        cond_timeout = (i - cand_loc) >= cfg.high_lock_timeout_bars

        if cond_drawdown or cond_structure or cond_timeout:
            idx = df_daily.index[cand_loc]
            if cond_drawdown:
                return cand_high, idx, "lock_drawdown"
            if cond_structure:
                return cand_high, idx, "lock_structure"
            return cand_high, idx, "lock_timeout"

    high_window = df_daily["high"].iloc[start_loc : end_loc + 1]
    max_idx = high_window.idxmax()
    return float(high_window.max()), max_idx, "fallback_window_max"


def detect_low_start_mode(df_daily: pd.DataFrame, fg_daily: pd.DataFrame, cfg: ScanConfig) -> Optional[Dict]:
    if len(df_daily) < max(80, cfg.low_start_lookback + 5):
        return None

    close = df_daily["close"]
    recent = close.iloc[-cfg.low_start_lookback:]
    cmin, cmax = recent.min(), recent.max()
    if pd.isna(cmin) or pd.isna(cmax) or cmax <= cmin:
        return None

    current_price = float(close.iloc[-1])
    qline = cmin + (cmax - cmin) * cfg.low_zone_quantile
    in_low_zone = current_price <= qline

    green = fg_daily["bbuy"].fillna(False)
    recent_green_idx = green[green].tail(3).index.tolist()
    if not recent_green_idx:
        return None

    last_green_idx = recent_green_idx[-1]
    bars_since_last_green = len(df_daily.loc[last_green_idx:]) - 1
    if bars_since_last_green > cfg.low_green_recent_bars:
        return None

    second_green = False
    if len(recent_green_idx) >= 2:
        prev_green_idx = recent_green_idx[-2]
        gap = len(df_daily.loc[prev_green_idx:last_green_idx]) - 1
        if 1 <= gap <= cfg.simple_pullback_max_bars:
            second_green = True

    t_now = float(fg_daily["t"].iloc[-1])
    t_prev = float(fg_daily["t"].iloc[-2]) if len(fg_daily) >= 2 else t_now
    low_t = t_now <= 20 or (t_now <= 30 and t_now >= t_prev)

    if not in_low_zone and not low_t:
        return None

    stage = "低位启动-二次绿柱" if second_green else "低位启动-首绿柱"
    status = "重点观察" if second_green else "预备观察"
    note = f"低位区={'是' if in_low_zone else '否'}; t={t_now:.2f}; 最近绿柱有效"

    return {
        "mode": "低位启动",
        "tag": stage,
        "status": status,
        "note": note,
        "current_price": current_price,
        "daily_state": f"t={t_now:.2f}, {'抬升中' if t_now >= t_prev else '回落中'}",
        "h4_state": "待接入4H细化",
        "position_state": "低位区" if in_low_zone else "刚脱离低位",
        "levels": {},
        "trigger_reason": stage,
        "risk_note": "当前为提示，不代表自动买入。",
    }


def detect_rebound_mode(
    df_daily: pd.DataFrame,
    fg_daily: pd.DataFrame,
    df_4h: pd.DataFrame,
    fg_4h: pd.DataFrame,
    cfg: ScanConfig,
) -> Optional[Dict]:
    if len(df_daily) < max(100, cfg.impulse_max_bars + 10):
        return None
    if len(df_4h) < 50:
        return None

    green = fg_daily["bbuy"].fillna(False)
    green_dates = green[green].index.tolist()
    if not green_dates:
        return None

    start_idx = green_dates[-1]
    start_loc = df_daily.index.get_loc(start_idx)

    left = max(0, start_loc - cfg.anchor_window)
    right = min(len(df_daily) - 1, start_loc + cfg.anchor_window)
    low_low = float(df_daily["low"].iloc[left : right + 1].min())

    high_high, high_idx, high_method = _locate_impulse_high(df_daily, start_loc, cfg)

    current_price = float(df_daily["close"].iloc[-1])
    if high_high <= low_low or current_price >= high_high:
        return None

    levels = calc_retrace_levels(low_low, high_high, cfg.retrace_levels)

    t4_now = float(fg_4h["t"].iloc[-1])
    t4_prev = float(fg_4h["t"].iloc[-2]) if len(fg_4h) >= 2 else t4_now
    h4_near_low = t4_now <= cfg.t_low_threshold_4h
    h4_rebound_window = h4_near_low or (t4_now <= cfg.t_low_threshold_4h * 1.5 and t4_now >= t4_prev)

    lv618 = levels.get("0.618")
    lv05 = levels.get("0.5")
    lv0382 = levels.get("0.382")

    if lv618 is None or lv05 is None or lv0382 is None:
        return None

    tag = None
    status = "仅观察"
    if abs(current_price - lv618) / lv618 <= cfg.near_pct:
        tag = "二进宫-0.618"
        status = "临近点位"
    elif abs(current_price - lv05) / lv05 <= cfg.near_pct:
        tag = "二进宫-0.5"
        status = "重点观察"
    elif abs(current_price - lv0382) / lv0382 <= cfg.near_pct:
        tag = "二进宫-0.382"
        status = "重点观察"
    else:
        closest_name, closest_value = min(
            [("0.618", lv618), ("0.5", lv05), ("0.382", lv0382)],
            key=lambda x: abs(current_price - x[1]) / x[1],
        )
        if abs(current_price - closest_value) / closest_value <= cfg.near_pct * 1.8:
            tag = f"二进宫-接近{closest_name}"
            status = "临近点位"

    if tag is None:
        return None

    h4_state = f"4H t={t4_now:.2f}, {'接近低位' if h4_near_low else ('反弹窗口临近' if h4_rebound_window else '仍待压缩')}"
    note = f"低低={low_low:.2f}, 高高={high_high:.2f}({high_method}), 当前={current_price:.2f}"

    return {
        "mode": "二进宫反弹",
        "tag": tag,
        "status": status,
        "note": note,
        "current_price": current_price,
        "daily_state": f"主升锚点={start_idx.date()}, 高高确认={high_idx.date()}",
        "h4_state": h4_state,
        "position_state": _build_rebound_position_text(current_price, lv618, lv05, lv0382, cfg.near_pct),
        "levels": {
            "low_low": round(low_low, 2),
            "high_high": round(high_high, 2),
            "0.618": round(lv618, 2),
            "0.5": round(lv05, 2),
            "0.382": round(lv0382, 2),
        },
        "trigger_reason": tag,
        "risk_note": "当前为盘后提示，需要人工结合结构确认。",
    }


def _build_rebound_position_text(current_price: float, lv618: float, lv05: float, lv0382: float, near_pct: float) -> str:
    checks = [("0.618", lv618), ("0.5", lv05), ("0.382", lv0382)]
    for name, val in checks:
        if abs(current_price - val) / val <= near_pct:
            return f"已到{name}附近"
    nearest_name, nearest_val = min(checks, key=lambda x: abs(current_price - x[1]) / x[1])
    direction = "上方" if current_price > nearest_val else "下方"
    return f"{direction}靠近{nearest_name}"


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
    out = out[required].sort_index().dropna()
    return out


def scan_symbol(symbol: str, df_daily: pd.DataFrame, df_4h: pd.DataFrame, cfg: Optional[ScanConfig] = None) -> Dict[str, Optional[object]]:
    cfg = cfg or ScanConfig()
    df_daily = normalize_ohlcv(df_daily)
    df_4h = normalize_ohlcv(df_4h)

    fg_daily = compute_l2_t_p(df_daily)
    fg_4h = compute_l2_t_p(df_4h)

    low_start = detect_low_start_mode(df_daily, fg_daily, cfg)
    rebound = detect_rebound_mode(df_daily, fg_daily, df_4h, fg_4h, cfg)

    return {"symbol": symbol, "low_start": low_start, "rebound": rebound}


def _detail_status(card: DetailCard, prompt_df: pd.DataFrame) -> str:
    if prompt_df.empty:
        return "仅观察"
    hit = prompt_df[(prompt_df["symbol"] == card.symbol) & (prompt_df["mode"] == card.mode)]
    if hit.empty:
        return "仅观察"
    return str(hit.iloc[0]["status"])


def build_reports(scan_results: List[Dict], cfg: Optional[ScanConfig] = None) -> Tuple[pd.DataFrame, List[DetailCard]]:
    cfg = cfg or ScanConfig()
    prompts: List[PromptTag] = []
    details: List[DetailCard] = []

    for item in scan_results:
        symbol = item["symbol"]
        for key in ["low_start", "rebound"]:
            data = item.get(key)
            if not data:
                continue
            prompts.append(
                PromptTag(
                    symbol=symbol,
                    mode=data["mode"],
                    tag=data["tag"],
                    status=data["status"],
                    note=data["note"],
                    current_price=float(data["current_price"]),
                )
            )
            details.append(
                DetailCard(
                    symbol=symbol,
                    mode=data["mode"],
                    current_price=float(data["current_price"]),
                    trigger_reason=data["trigger_reason"],
                    daily_state=data["daily_state"],
                    h4_state=data["h4_state"],
                    position_state=data["position_state"],
                    levels=data["levels"],
                    risk_note=data["risk_note"],
                )
            )

    prompt_df = pd.DataFrame([asdict(x) for x in prompts])
    if not prompt_df.empty:
        prompt_df = prompt_df.sort_values(["status", "mode", "symbol"], ascending=[True, True, True]).reset_index(drop=True)

    priority = {"重点观察": 0, "临近点位": 1, "预备观察": 2, "仅观察": 3}
    details_sorted = sorted(details, key=lambda d: priority.get(_detail_status(d, prompt_df), 9))
    focus_cards = details_sorted[: cfg.focus_top_n]

    return prompt_df, focus_cards


def render_version2(prompt_df: pd.DataFrame) -> str:
    if prompt_df.empty:
        return "# 版本2总览\n\n今日无符合条件提示。"

    lines = ["# 版本2总览", ""]
    for _, row in prompt_df.iterrows():
        lines.append(f"## {row['symbol']}")
        lines.append(f"- 模式：{row['mode']}")
        lines.append(f"- 提示：{row['tag']}")
        lines.append(f"- 状态：{row['status']}")
        lines.append(f"- 当前价：{row['current_price']:.2f}")
        lines.append(f"- 备注：{row['note']}")
        lines.append("")
    return "\n".join(lines)


def render_version3(cards: List[DetailCard]) -> str:
    if not cards:
        return "# 版本3重点票\n\n今日无重点票。"

    lines = ["# 版本3重点票", ""]
    for card in cards:
        lines.append(f"## {card.symbol}")
        lines.append(f"- 模式：{card.mode}")
        lines.append(f"- 当前价：{card.current_price:.2f}")
        lines.append(f"- 触发原因：{card.trigger_reason}")
        lines.append(f"- 日线状态：{card.daily_state}")
        lines.append(f"- 4H状态：{card.h4_state}")
        lines.append(f"- 位置状态：{card.position_state}")
        if card.levels:
            level_txt = ", ".join([f"{k}={v}" for k, v in card.levels.items()])
            lines.append(f"- 关键位：{level_txt}")
        lines.append(f"- 风险提示：{card.risk_note}")
        lines.append("")
    return "\n".join(lines)


def example_run(data_map: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[str, str]:
    cfg = ScanConfig()
    results = []
    for symbol, frames in data_map.items():
        result = scan_symbol(symbol, frames["daily"], frames["4h"], cfg)
        results.append(result)

    prompt_df, focus_cards = build_reports(results, cfg)
    v2 = render_version2(prompt_df)
    v3 = render_version3(focus_cards)
    return v2, v3


if __name__ == "__main__":
    print("dual_mode_scan_v1.py 已加载。请接入你的 daily / 4h 数据后调用 example_run 或 scan_symbol。")
