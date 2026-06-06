# xunlong.py
import numpy as np
import pandas as pd

# ==============================
# 工具函数
# ==============================

def safe_div(num, den):
    """安全除法：den 为 0 或 NaN 时返回 0"""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.zeros_like(num, dtype="float64")
    mask = (den != 0) & ~np.isnan(den)
    out[mask] = num[mask] / den[mask]
    return out

def hist(series, n):
    """等价 Pine hist(src, n) = src[n]"""
    return series.shift(n)

def xsa(series, length, wei):
    """
    尽量还原你 Pine 里的 xsa 自定义平滑函数。
    兼容传进来的对象是 Series 或 1 列 DataFrame。
    sumf := nz(sumf[1]) - nz(hist(src, len)) + src
    ma   := na(hist(src, len)) ? na : sumf/len
    out  := na(out[1]) ? ma : (src*wei + out[1]*(len-wei)) / len
    """
    # 如果不小心传进来的是 DataFrame，就压成单列 Series
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("xsa expects 1D series, got multi-column DataFrame")
        series = series.iloc[:, 0]

    s = series.astype("float64")
    n = len(s)
    out_vals = np.full(n, np.nan, dtype="float64")
    sumf = 0.0
    prev_out = np.nan

    for i in range(n):
        v = s.iat[i]

        # hist(src, len)
        if i - length >= 0:
            v_hist = s.iat[i - length]
        else:
            v_hist = np.nan

        sumf = (sumf if not np.isnan(sumf) else 0.0) \
               - (v_hist if not np.isnan(v_hist) else 0.0) \
               + (v if not np.isnan(v) else 0.0)

        ma = np.nan if np.isnan(v_hist) else sumf / float(length)

        if np.isnan(prev_out):
            out = ma
        else:
            out = (v * wei + prev_out * (length - wei)) / float(length)

        out_vals[i] = out
        prev_out = out

    return pd.Series(out_vals, index=series.index)

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def sma(series, length):
    return series.rolling(length).mean()

def rma(series, length):
    """近似 Pine ta.rma"""
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()

def highest(series, length):
    return series.rolling(length).max()

def lowest(series, length):
    return series.rolling(length).min()

def nz(series, value=0.0):
    return series.fillna(value)


# ==============================
# 主类：寻龙诀指标
# ==============================

class XunLongIndicator:
    def __init__(
        self,
        K=9,
        D=3,
        MidPeriod=58,
        vol_short=5,
        vol_long=10,
        max_pool_rev=0.2,
        rise_val=18,
        rsi_len=14,
        C_vol_threshold=40.0,
        C_rev_lookback=10,
        C_rev_mid_diff=0.01,
        C_rev_trend_diff_min=0.12,
        C_rev_trend_diff_max=0.40,
        mom4_len=12,
        mom4_smooth=6,
        sell_overheat_threshold=85.0,
        sell_reset_threshold=60.0,
        gann_ema_len=10,
        gann_min_ema_up_bars=3,
        gann_min_gain_pct=0.08,
        gann_buy_green_lookback=8,
        gann_buy_a_rsi_min=35.0,
        gann_buy_a_rsi_low_ceiling=50.0,
    ):
        # 参数基本与 Pine 版本一致
        self.K = K
        self.D = D
        self.MidPeriod = MidPeriod
        self.vol_short = vol_short
        self.vol_long = vol_long
        self.max_pool_rev = max_pool_rev
        self.rise_val = rise_val
        self.rsi_len = rsi_len
        self.C_vol_threshold = C_vol_threshold
        self.C_rev_lookback = C_rev_lookback
        self.C_rev_mid_diff = C_rev_mid_diff
        self.C_rev_trend_diff_min = C_rev_trend_diff_min
        self.C_rev_trend_diff_max = C_rev_trend_diff_max
        self.mom4_len = mom4_len
        self.mom4_smooth = mom4_smooth
        self.sell_overheat_threshold = sell_overheat_threshold
        self.sell_reset_threshold = sell_reset_threshold
        self.gann_ema_len = gann_ema_len
        self.gann_min_ema_up_bars = gann_min_ema_up_bars
        self.gann_min_gain_pct = gann_min_gain_pct
        self.gann_buy_green_lookback = gann_buy_green_lookback
        self.gann_buy_a_rsi_min = gann_buy_a_rsi_min
        self.gann_buy_a_rsi_low_ceiling = gann_buy_a_rsi_low_ceiling

    # --------- 1) L2 Swing (日线) ----------
    def _calc_L2_trend_pump(self, df):
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        K = self.K
        D = self.D

        # denK
        highest_K = highest(high, K)
        lowest_K = lowest(low, K)
        denK = (highest_K - lowest_K).clip(lower=1e-8)

        var1b = (highest_K - close) / denK * 100.0 - 70.0
        var2b = xsa(var1b, K, 1) + 100.0
        var3b = (close - lowest_K) / denK * 100.0
        var4b = xsa(var3b, D, 1)
        var5b = xsa(var4b, D, 1) + 100.0
        var6b = var5b - var2b
        trend = var6b.where(var6b > 45.0, 0.0)

        # pump
        var2q = hist(low, 1)
        var3q = xsa((low - var2q).abs(), D, 1) / xsa((low - var2q).clip(lower=0), D, 1) * 100.0
        change_close = close.diff()
        cond_pos = change_close > 0
        # np.where 返回 ndarray，这里需要保留 index 后再做 EMA
        var4q_input = pd.Series(np.where(cond_pos, var3q * 10.0, var3q / 10.0), index=close.index)
        var4q = ema(var4q_input, D)
        var5q = lowest(low, 30)
        var6q = highest(var4q, 30)
        smaMid = sma(close, self.MidPeriod)
        var7q = (~smaMid.isna()).astype(float)
        tmp = np.where(low <= var5q, (var4q + var6q * 2.0) / 2.0, 0.0)
        var8q = ema(pd.Series(tmp, index=close.index), D) / 999.0 * var7q
        pump = np.minimum(var8q, 100.0)

        return trend, pump

    # --------- 2) EMA/RSI 辅助 ----------
    def _calc_ema_rsi_helpers(self, df):
        close = df["Close"]
        low = df["Low"]
        high = df["High"]

        d1 = (close + low + high) / 3.0
        d2 = ema(d1, 6)
        d3 = ema(d2, 5)
        bbuy = (d2 > d3) & (d2.shift(1) <= d3.shift(1))

        chg = close - hist(close, 1)
        up = chg.clip(lower=0)
        abs_chg = chg.abs()
        varr1 = xsa(up, 6, 1) / xsa(abs_chg, 6, 1) * 100.0
        crossunder_varr1_82 = (varr1.shift(1) >= 82.0) & (varr1 < 82.0)

        return {
            "d2": d2,
            "d3": d3,
            "bbuy": bbuy,
            "varr1": varr1,
            "crossunder_varr1_82": crossunder_varr1_82,
        }

    def _calc_sell_signal(self, df, helpers):
        """
        高位转弱卖出提示：
        - 分金值 >= 85 后进入 armed
        - 当天高位转弱（黄转紫近似）且红K，触发严格首个卖点
        - 若首个卖点后仍处弱势周期，允许继续发出弱势延续卖点
        - 直到分金回落到 60 以下才重新允许下一次提示
        """
        close = df["Close"]
        open_ = df["Open"]
        varr1 = helpers["varr1"]

        high_zone = varr1 >= self.sell_overheat_threshold
        turned_purple = (varr1 < varr1.shift(1)) & (varr1.shift(1) >= varr1.shift(2))
        red_bar = close < open_
        weak_close = close < close.shift(1)
        high_zone_recent = varr1.rolling(4, min_periods=1).max() >= self.sell_overheat_threshold
        followthrough_weak = turned_purple.shift(1, fill_value=False) & (red_bar | weak_close) & (varr1 <= varr1.shift(1))

        primary_vals = np.zeros(len(df), dtype=bool)
        follow_vals = np.zeros(len(df), dtype=bool)
        armed = False
        primary_sold_in_cycle = False
        last_primary_idx = -999
        last_sell_idx = -999

        for i in range(len(df)):
            v = varr1.iat[i]
            if pd.isna(v):
                continue

            if v < self.sell_reset_threshold:
                armed = False
                primary_sold_in_cycle = False
                last_primary_idx = -999

            if high_zone_recent.iat[i]:
                armed = True

            primary_trigger = bool(turned_purple.iat[i] and (red_bar.iat[i] or weak_close.iat[i]))
            follow_trigger = bool(
                primary_sold_in_cycle
                and ((i - last_sell_idx) >= 2)
                and turned_purple.iat[i]
                and (red_bar.iat[i] or weak_close.iat[i] or followthrough_weak.iat[i])
                and (v >= self.sell_reset_threshold)
                and (v <= varr1.shift(1).iat[i] if i > 0 and pd.notna(varr1.shift(1).iat[i]) else True)
            )

            if armed and (not primary_sold_in_cycle) and primary_trigger:
                primary_vals[i] = True
                primary_sold_in_cycle = True
                last_primary_idx = i
                last_sell_idx = i
                continue

            if armed and follow_trigger:
                follow_vals[i] = True
                last_sell_idx = i

        return (
            pd.Series(primary_vals, index=df.index),
            pd.Series(follow_vals, index=df.index),
            varr1,
        )

    # --------- 3) Volume Oscillator ----------
    def _calc_vol_osc(self, df):
        vol = df["Volume"]
        cumVol = vol.cumsum()
        hasVol = (~vol.isna()) & ((cumVol - hist(cumVol, 50)) > 0)

        shortlen = self.vol_short
        longlen = self.vol_long
        max_pool_rev = self.max_pool_rev

        shortEMA = ema(vol, shortlen)
        longEMA = ema(vol, longlen)
        raw_osc = 100.0 * safe_div(shortEMA.values - longEMA.values, longEMA.values)

        osc = pd.Series(np.nan, index=vol.index)
        oscpool = pd.Series(0.0, index=vol.index)

        prev_osc = np.nan
        prev_pool = 0.0

        for i, (idx, hv) in enumerate(hasVol.items()):
            if not hv:
                osc.iat[i] = np.nan
                oscpool.iat[i] = prev_pool
                continue

            cur_osc = raw_osc[i]
            if np.isnan(prev_osc):
                diff = 0.0
            else:
                diff = cur_osc - prev_osc

            if diff > 0 or (prev_pool * max_pool_rev > abs(diff)):
                cur_pool = prev_pool + diff
            else:
                cur_pool = prev_pool

            osc.iat[i] = cur_osc
            oscpool.iat[i] = cur_pool

            prev_osc = cur_osc
            prev_pool = cur_pool

        legacy_rise = hasVol & (oscpool > self.rise_val)
        volosc_shifted = np.where(hasVol, osc.values + 40.0, np.nan)

        return (
            osc,
            oscpool,
            legacy_rise,
            pd.Series(volosc_shifted, index=vol.index),
            pd.Series(hasVol.astype(bool), index=vol.index),
        )

    # --------- 4) 手动 RSI ----------
    def _calc_manual_rsi(self, close):
        change = close.diff()
        up = change.clip(lower=0)
        down = -change.clip(upper=0)
        up_rma = rma(up, self.rsi_len)
        down_rma = rma(down, self.rsi_len)
        rs = safe_div(up_rma.values, down_rma.values)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
        return pd.Series(rsi_val, index=close.index)

    # --------- 5) A / C / C_rev (日线) ----------
    def _calc_ABC_daily(self, df, trend, pump, rsi_val, helpers, volosc_zero_line, hasVol):
        close = df["Close"]
        low = df["Low"]
        vol = df["Volume"]

        ema8 = ema(close, 8)
        ma13 = sma(close, 13)
        ma21 = sma(close, 21)

        # OBV
        prev_close = hist(close, 1)
        obv_delta = np.where(close > prev_close, vol,
                        np.where(close < prev_close, -vol, 0))
        obvS = pd.Series(obv_delta, index=close.index).cumsum()
        ema8v = ema(obvS, 8)

        # A
        t = nz(trend)
        p = nz(pump)
        t_is_zero = t == 0
        p_is_zero = p == 0
        bbuy = helpers["bbuy"]
        rsiCrossUp = (rsi_val > 50) & (rsi_val.shift(1) <= 50)
        A_ok = (t_is_zero & p_is_zero) & (bbuy | (rsi_val > 50) | rsiCrossUp)

        # C
        twoUpOver8 = (close > ema8) & (close.shift(1) > ema8) & (close.shift(2) <= ema8)
        obvCrossUp = (obvS > ema8v) & (obvS.shift(1) <= ema8v.shift(1))
        C_raw = (twoUpOver8 & (ema8 < ma13) & (ma13 < ma21)) | obvCrossUp

        C_ok = C_raw & ((~hasVol) | (volosc_zero_line > self.C_vol_threshold))

        # C_rev
        lookback = self.C_rev_lookback
        mid_diff = self.C_rev_mid_diff
        diff_min = self.C_rev_trend_diff_min
        diff_max = self.C_rev_trend_diff_max

        low_L2 = lowest(low, lookback)
        high_L2 = highest(df["High"], lookback)
        mid_L2 = (low_L2 + high_L2) / 2.0

        mid_ready = (close / mid_L2 - 1.0).abs() < mid_diff
        trend_diff = (high_L2 / low_L2 - 1.0)
        trend_ok = (trend_diff > diff_min) & (trend_diff < diff_max)
        trend_change = (t > 0) & (t.shift(1) <= 0)

        C_rev_ok = mid_ready & trend_ok & trend_change

        return {
            "ema8": ema8,
            "ma13": ma13,
            "ma21": ma21,
            "obvS": obvS,
            "ema8v": ema8v,
            "A_ok": A_ok,
            "C_ok": C_ok,
            "C_rev_ok": C_rev_ok,
        }

    # --------- 6) 4H B / C_4H ----------
    def _calc_4h_B_C(self, df_4h):
        if df_4h is None or df_4h.empty:
            return None

        c = df_4h["Close"]
        o = df_4h["Open"]
        v = df_4h["Volume"]

        # 4H 动能
        delta = c - c.shift(self.mom4_len)
        mom4 = ema(delta, self.mom4_smooth)

        mom4CrossUp0 = (mom4 > 0) & (mom4.shift(1) <= 0)
        B_firstGreen = mom4CrossUp0

        ema8_4 = ema(c, 8)
        ma13_4 = sma(c, 13)
        ma21_4 = sma(c, 21)
        twoUp4 = (c > ema8_4) & (c.shift(1) > ema8_4) & (c.shift(2) <= ema8_4)

        obv_delta4 = np.where(c > c.shift(1), v,
                         np.where(c < c.shift(1), -v, 0))
        obv4 = pd.Series(obv_delta4, index=c.index).cumsum()
        ema8v4 = ema(obv4, 8)
        obvX4 = (obv4 > ema8v4) & (obv4.shift(1) <= ema8v4.shift(1))

        C_ok_4h = (twoUp4 & (ema8_4 < ma13_4) & (ma13_4 < ma21_4)) | obvX4

        h4_helpers = self._calc_ema_rsi_helpers(df_4h)
        h4_rsi = self._calc_manual_rsi(c)
        h4_gann = self._calc_gann_turn_marks(df_4h, min_gain_pct=0.03)

        out = pd.DataFrame(index=df_4h.index)
        out["B_firstGreen"] = B_firstGreen
        out["C_ok_4h"] = C_ok_4h
        out["H4_RSI"] = h4_rsi
        out["H4_FJ_value"] = h4_helpers["varr1"]
        out["H4_Gann_0_birth"] = h4_gann["Gann_0_birth"]
        out["H4_Gann_1_birth"] = h4_gann["Gann_1_birth"]
        return out

    def _calc_gann_turn_marks(self, df, min_gain_pct=None):
        """
        只计算 Gann 的 0出/1出触发日期。
        0出 = EMA 上拐当天，系统识别临时 0；
        1出 = EMA 下拐当天，前一段上涨满足最小根数/涨幅后确认高点。
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        min_gain = self.gann_min_gain_pct if min_gain_pct is None else float(min_gain_pct)

        ema10 = ema(close, self.gann_ema_len)
        ema_up = ema10 > ema10.shift(1)
        ema_down = ema10 < ema10.shift(1)
        ema_turn_up = ema_up & (ema10.shift(1) <= ema10.shift(2))
        ema_turn_down = ema_down & (ema10.shift(1) >= ema10.shift(2))

        n = len(df)
        zero_birth = np.zeros(n, dtype=bool)
        one_birth = np.zeros(n, dtype=bool)
        seg_active = False
        seg_low = np.nan
        seg_high = np.nan
        ema_up_count = 0
        temp_zero = np.nan

        for i in range(n):
            lo = low.iat[i]
            hi = high.iat[i]

            if not seg_active:
                if pd.isna(temp_zero) or lo < temp_zero:
                    temp_zero = lo

            if bool(ema_turn_up.iat[i]) and pd.notna(temp_zero):
                seg_active = True
                seg_low = temp_zero
                seg_high = hi
                ema_up_count = 1
                temp_zero = np.nan
                zero_birth[i] = True
            elif seg_active:
                if bool(ema_up.iat[i]):
                    ema_up_count += 1
                if pd.isna(seg_high) or hi > seg_high:
                    seg_high = hi

            if seg_active and bool(ema_turn_down.iat[i]):
                gain = seg_high / seg_low - 1.0 if pd.notna(seg_low) and seg_low else np.nan
                valid_trend = (
                    pd.notna(gain)
                    and ema_up_count >= self.gann_min_ema_up_bars
                    and gain >= min_gain
                    and seg_high > seg_low
                )
                if valid_trend:
                    one_birth[i] = True
                seg_active = False
                ema_up_count = 0
                seg_low = np.nan
                seg_high = np.nan
                temp_zero = lo

        return {
            "Gann_0_birth": pd.Series(zero_birth, index=df.index),
            "Gann_1_birth": pd.Series(one_birth, index=df.index),
        }

    # --------- 7) GannBox BUY A / 1确认SELL ----------
    def _calc_gannbox_buy_sell(self, df, helpers, rsi_val):
        close = df["Close"]
        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]

        ema10 = ema(close, self.gann_ema_len)
        ema_up = ema10 > ema10.shift(1)
        ema_down = ema10 < ema10.shift(1)
        ema_turn_up = ema_up & (ema10.shift(1) <= ema10.shift(2))
        ema_turn_down = ema_down & (ema10.shift(1) >= ema10.shift(2))

        green_event = helpers["bbuy"].fillna(False).astype(bool)
        green_recent = green_event.rolling(self.gann_buy_green_lookback, min_periods=1).max().astype(bool)
        fj_value = helpers["varr1"]
        fj_repair = (fj_value > fj_value.shift(1)) & (rsi_val > 45)
        rsi_low_turn = (
            (rsi_val > self.gann_buy_a_rsi_min)
            & (rsi_val < self.gann_buy_a_rsi_low_ceiling)
            & (rsi_val > rsi_val.shift(1))
            & (rsi_val.shift(1) <= rsi_val.shift(2))
        )

        n = len(df)
        buy_a = np.zeros(n, dtype=bool)
        sell_1 = np.zeros(n, dtype=bool)
        gann0 = np.full(n, np.nan, dtype="float64")
        gann382 = np.full(n, np.nan, dtype="float64")
        gann50 = np.full(n, np.nan, dtype="float64")
        gann618 = np.full(n, np.nan, dtype="float64")
        gann1 = np.full(n, np.nan, dtype="float64")
        gann1_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        seg_gain = np.full(n, np.nan, dtype="float64")

        seg_active = False
        seg_low = np.nan
        seg_high = np.nan
        seg_high_date = pd.NaT
        ema_up_count = 0
        buy_a_done = False
        segment_has_buy = False
        temp_zero = np.nan

        for i in range(n):
            lo = low.iat[i]
            hi = high.iat[i]
            cl = close.iat[i]

            if not seg_active:
                if pd.isna(temp_zero) or lo < temp_zero:
                    temp_zero = lo

            if bool(ema_turn_up.iat[i]) and pd.notna(temp_zero):
                seg_active = True
                seg_low = temp_zero
                seg_high = hi
                seg_high_date = df.index[i]
                ema_up_count = 1
                buy_a_done = False
                segment_has_buy = False
                temp_zero = np.nan
            elif seg_active:
                if bool(ema_up.iat[i]):
                    ema_up_count += 1
                if pd.isna(seg_high) or hi > seg_high:
                    seg_high = hi
                    seg_high_date = df.index[i]

            if seg_active and pd.notna(seg_low) and pd.notna(seg_high) and seg_high > seg_low:
                gain = seg_high / seg_low - 1.0 if seg_low else np.nan
                seg_gain[i] = gain
                gann0[i] = seg_low
                gann382[i] = seg_low + (seg_high - seg_low) * 0.382
                gann50[i] = seg_low + (seg_high - seg_low) * 0.5
                gann618[i] = seg_low + (seg_high - seg_low) * 0.618
                gann1[i] = seg_high
                gann1_date[i] = np.datetime64(seg_high_date) if pd.notna(seg_high_date) else np.datetime64("NaT")

            buy_a_trigger = bool(ema_turn_up.iat[i] or rsi_low_turn.iat[i])
            launch_confirm = bool(green_recent.iat[i] or fj_repair.iat[i] or cl > ema10.iat[i])
            buy_a_ok = (
                seg_active
                and not buy_a_done
                and buy_a_trigger
                and launch_confirm
                and pd.notna(rsi_val.iat[i])
                and rsi_val.iat[i] > self.gann_buy_a_rsi_min
                and cl >= seg_low
                and ((cl > open_.iat[i]) or (cl > close.shift(1).iat[i] if i > 0 else True))
            )
            if buy_a_ok:
                buy_a[i] = True
                buy_a_done = True
                segment_has_buy = True

            if seg_active and bool(ema_turn_down.iat[i]):
                gain = seg_high / seg_low - 1.0 if pd.notna(seg_low) and seg_low else np.nan
                valid_trend = (
                    pd.notna(gain)
                    and ema_up_count >= self.gann_min_ema_up_bars
                    and gain >= self.gann_min_gain_pct
                    and seg_high > seg_low
                )
                if valid_trend and segment_has_buy:
                    sell_1[i] = True
                    seg_gain[i] = gain
                    gann0[i] = seg_low
                    gann382[i] = seg_low + (seg_high - seg_low) * 0.382
                    gann50[i] = seg_low + (seg_high - seg_low) * 0.5
                    gann618[i] = seg_low + (seg_high - seg_low) * 0.618
                    gann1[i] = seg_high
                    gann1_date[i] = np.datetime64(seg_high_date) if pd.notna(seg_high_date) else np.datetime64("NaT")

                seg_active = False
                ema_up_count = 0
                seg_low = np.nan
                seg_high = np.nan
                seg_high_date = pd.NaT
                buy_a_done = False
                segment_has_buy = False
                temp_zero = lo

        return {
            "Gann_EMA10": ema10,
            "Gann_BUY_A": pd.Series(buy_a, index=df.index),
            "Gann_SELL_1_confirmed": pd.Series(sell_1, index=df.index),
            "Gann_0": pd.Series(gann0, index=df.index),
            "Gann_382": pd.Series(gann382, index=df.index),
            "Gann_50": pd.Series(gann50, index=df.index),
            "Gann_618": pd.Series(gann618, index=df.index),
            "Gann_1": pd.Series(gann1, index=df.index),
            "Gann_1_date": pd.Series(gann1_date, index=df.index),
            "Gann_gain_pct": pd.Series(seg_gain, index=df.index),
        }

    # --------- 7) 公共接口 ----------
    def compute(self, df_daily, df_4h=None):
        """
        df_daily: DataFrame 必须包含 [Open, High, Low, Close, Volume]
        df_4h:    可选 4H 数据（同样列），用于 B / C(4H)
        返回：在 df_daily 基础上增加多列指标 & 信号
        """
        df = df_daily.copy()

        # 1) L2
        trend, pump = self._calc_L2_trend_pump(df)
        df["L2_trend"] = trend
        df["L2_pump"] = pump

        # 2) EMA/RSI 辅助
        helpers = self._calc_ema_rsi_helpers(df)
        sell_signal, sell_follow_signal, fj_value = self._calc_sell_signal(df, helpers)

        # 3) Vol Osc
        osc, oscpool, legacy_rise, volosc_shifted, hasVol = self._calc_vol_osc(df)
        df["VolOsc"] = osc
        df["VolOscPool"] = oscpool
        df["VolLegacyRise"] = legacy_rise
        df["VolOscZeroLine"] = volosc_shifted

        # 4) 主 RSI
        rsi_val = self._calc_manual_rsi(df["Close"])
        df["RSI"] = rsi_val
        df["FJ_value"] = fj_value
        df["SELL_high_weakening"] = sell_signal
        df["SELL_weak_trend_followthrough"] = sell_follow_signal

        # 5) A / C / C_rev
        abc = self._calc_ABC_daily(
            df,
            trend=trend,
            pump=pump,
            rsi_val=rsi_val,
            helpers=helpers,
            volosc_zero_line=volosc_shifted,
            hasVol=hasVol,
        )
        for k, v in abc.items():
            df[k] = v

        # 5.5) GannBox 交易闭环：新 0 后 BUY A，1 确认后 SELL
        gann = self._calc_gannbox_buy_sell(df, helpers=helpers, rsi_val=rsi_val)
        for k, v in gann.items():
            df[k] = v

        # 6) 4H → 按交易日合并
        if df_4h is not None and not df_4h.empty:
            intr = self._calc_4h_B_C(df_4h)
            if intr is not None and not intr.empty:
                intr = intr.copy()
                intr["date"] = intr.index.date
                daily_flags = intr.groupby("date").agg({
                    "B_firstGreen": "max",
                    "C_ok_4h": "max",
                    "H4_Gann_0_birth": "max",
                    "H4_Gann_1_birth": "max",
                    "H4_RSI": "last",
                    "H4_FJ_value": "last",
                })
                df["date"] = df.index.date
                df = df.merge(
                    daily_flags,
                    how="left",
                    left_on="date",
                    right_index=True,
                )
                bfg = df["B_firstGreen"].to_numpy()
                df["B_firstGreen_daily"] = pd.Series(
                    np.where(pd.isna(bfg), False, bfg).astype(bool),
                    index=df.index,
                )
                cok = df["C_ok_4h"].to_numpy()
                df["C_ok_4h_daily"] = pd.Series(
                    np.where(pd.isna(cok), False, cok).astype(bool),
                    index=df.index,
                )
                h4_zero = df.get("H4_Gann_0_birth", pd.Series(False, index=df.index)).to_numpy()
                df["H4_Gann_0_birth_daily"] = pd.Series(
                    np.where(pd.isna(h4_zero), False, h4_zero).astype(bool),
                    index=df.index,
                )
                h4_one = df.get("H4_Gann_1_birth", pd.Series(False, index=df.index)).to_numpy()
                df["H4_Gann_1_birth_daily"] = pd.Series(
                    np.where(pd.isna(h4_one), False, h4_one).astype(bool),
                    index=df.index,
                )
                df["H4_RSI_last"] = pd.to_numeric(df.get("H4_RSI"), errors="coerce")
                df["H4_FJ_last"] = pd.to_numeric(df.get("H4_FJ_value"), errors="coerce")
                df = df.drop(columns=["B_firstGreen", "C_ok_4h", "H4_Gann_0_birth", "H4_Gann_1_birth", "H4_RSI", "H4_FJ_value", "date"], errors="ignore")

        # 6.5) 1D 分金背景 + 4H 0出/1出触发的新策略层
        h4_zero_daily = df.get("H4_Gann_0_birth_daily", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        h4_one_daily = df.get("H4_Gann_1_birth_daily", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        fj_daily = pd.to_numeric(df["FJ_value"], errors="coerce")
        d1_fj_low_context = (fj_daily <= 40) | (fj_daily.rolling(5, min_periods=1).min() <= 40)
        d1_fj_deep_low = fj_daily.rolling(5, min_periods=1).min() <= 10
        d1_fj_not_worse = (fj_daily >= fj_daily.shift(1)) | d1_fj_deep_low
        # NOTE: the original expression was `(A) | (A) & (B)`, which by Python
        # operator precedence is `A | (A & B)` and reduces to just `A` — the
        # second clause was dead code. Keep the effective behavior ("FJ rolled
        # over vs. the prior bar") but state it cleanly. For a stricter two-bar
        # decline, use the commented form instead.
        d1_fj_weak = fj_daily < fj_daily.shift(1)
        # d1_fj_weak = (fj_daily < fj_daily.shift(1)) & (fj_daily.shift(1) < fj_daily.shift(2))
        d1_fj_high_recent = fj_daily.rolling(8, min_periods=1).max() >= 60
        d1_price_weak = (df["Close"] < df["Close"].shift(1)) | (df["Close"] < df.get("ema8", df["Close"]))

        df["BUY_B_D1FJ_low_H4_0birth"] = d1_fj_low_context & d1_fj_not_worse & h4_zero_daily
        df["SHORT_A_D1FJ_weak_H4_1birth"] = h4_one_daily & (d1_fj_weak | d1_price_weak | d1_fj_high_recent)

        # 7) 低位强度 & 新增早期/强买信号（基于你的描述）
        close_min_120 = df["Close"].rolling(120).min()
        close_max_120 = df["Close"].rolling(120).max()
        denom = close_max_120 - close_min_120
        df["Rank120"] = (df["Close"] - close_min_120) / denom.replace(0, np.nan)
        df["VolMA20"] = df["Volume"].rolling(20).mean()

        cond_low = df["Rank120"] <= 0.4
        rsi = df["RSI"]

        # 放宽：RSI>40 且向上；4H 首绿回看 3 天，或 4H C_ok 回看 5 天，或日线收盘站上 ema8
        rsi_turn_up = (rsi > 40) & (rsi > rsi.shift(1))
        # 没有 4H 时给一个同长度 False 的 Series，避免 .fillna 报错
        bfg_src = df["B_firstGreen_daily"] if "B_firstGreen_daily" in df else pd.Series(False, index=df.index)
        c4_src = df["C_ok_4h_daily"] if "C_ok_4h_daily" in df else pd.Series(False, index=df.index)

        bfg_recent = bfg_src.fillna(False).rolling(3, min_periods=1).max().astype(bool)
        c4_recent = c4_src.fillna(False).rolling(5, min_periods=1).max().astype(bool)
        price_above_ema8 = df["Close"] > df["ema8"]
        # 若 RSI 尚未抬头，但 4H 首绿 + 收盘站上 EMA8 也给“早期”提醒
        rsi_bfg_recover = (rsi > 32) & bfg_recent & price_above_ema8
        df["V2_early_turn"] = cond_low & (
            (rsi_turn_up & (bfg_recent | c4_recent | price_above_ema8))
            | rsi_bfg_recover
        )

        # 放宽：低位 + RSI>55 + 放量阈值降至 1.4xMA20 + 收盘高于前收且站上 ema8
        prev_close = df["Close"].shift(1)
        cond_price = (df["Close"] > df["ema8"]) & (df["Close"] > prev_close)
        df["V2_strong_buy"] = (
            cond_low
            & (rsi > 55)
            & (df["Volume"] >= 1.4 * df["VolMA20"])
            & cond_price
        )

        # 新增：低位盘整后，首根放量长阳突破短期箱体/EMA8
        # - 低位：Rank120 <= 0.55
        # - 盘整：近 10 日振幅 < 35%（宽一些，适配长阴后横盘）
        # - 突破：收盘 > 近 10 日高点(不含当日) 且 站上 EMA8；或 “大阳线+站上 EMA8”
        # - 量能：>= 0.7 * MA20（不苛求暴量，但过滤地量）
        # - 动能：RSI>40；若为大阳线则不强求 RSI 方向
        range_high_10 = df["High"].rolling(10).max().shift(1)
        range_low_10 = df["Low"].rolling(10).min().shift(1)
        box_spread = range_high_10 - range_low_10
        box_compact_ratio = pd.Series(
            safe_div(box_spread.values, range_low_10.values),
            index=df.index,
        ).fillna(1.0)
        box_compact = box_compact_ratio < 0.35

        box_break = (df["Close"] > range_high_10) & (df["Close"] > df["ema8"])
        big_green = (df["Close"] > df["Open"]) & (safe_div(df["Close"] - df["Open"], df["Open"]) >= 0.05)
        vol_surge = df["Volume"] >= 0.7 * df["VolMA20"]
        rsi_up = rsi > 40
        df["V2_base_breakout"] = (
            (df["Rank120"] <= 0.55)
            & box_compact
            & (box_break | (big_green & (df["Close"] > df["ema8"])))
            & vol_surge
            & (rsi_up | big_green)
        )

        # 8) 交易确认层：只把“指标触发”升级成更可交易的买卖点
        prev_close = df["Close"].shift(1)
        close_above_ema8 = df["Close"] > df["ema8"]
        ema8_rising = df["ema8"] > df["ema8"].shift(1)
        constructive_bar = (df["Close"] > df["Open"]) | (df["Close"] > prev_close)
        enough_volume = df["Volume"] >= 0.8 * df["VolMA20"]
        not_too_extended = df["Close"] <= 1.12 * df["ema8"]
        rsi_buy_zone = (rsi >= 45) & (rsi <= 72)

        df["BUY_C_confirmed"] = (
            df["C_ok"].fillna(False).astype(bool)
            & (df["Rank120"].fillna(1.0) <= 0.80)
            & close_above_ema8
            & ema8_rising
            & constructive_bar
            & enough_volume
            & not_too_extended
            & rsi_buy_zone
        )

        recent_overheat = df["FJ_value"].rolling(6, min_periods=1).max() >= 80
        fj_falling = df["FJ_value"] < df["FJ_value"].shift(1)
        close_lost_ema8 = df["Close"] < df["ema8"]
        df["SELL_profit_protect"] = (
            recent_overheat
            & fj_falling
            & close_lost_ema8
            & (df["Close"] < prev_close)
            & (rsi < 65)
        )

        df["SELL_trend_break"] = (
            (df["Close"] < df["ma21"])
            & (df["ema8"] < df["ma13"])
            & (rsi < 50)
            & enough_volume
        )

        return df
