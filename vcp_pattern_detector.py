#!/usr/bin/env python3
"""
VCP (Volatility Contraction Pattern) 检测器 - 突破预测版
VCP Pattern Detector with Breakout Prediction - Based on Mark Minervini's methodology

功能特点:
- 检测Mark Minervini的VCP (Volatility Contraction Pattern) 模式
- 识别已突破的VCP和即将突破的潜在VCP (1-3天内)
- 分类输出：突破VCP 和 潜在突破VCP观察清单
- 评分系统：趋势强度 + 整理质量 + 成交量配合 + 突破确认/潜力
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import time
import json
import os
from stock_symbols_2000 import STOCK_SYMBOLS, ETF_SYMBOLS
import warnings
warnings.filterwarnings('ignore')

# ============ VCP检测参数 ============
VCP_CONFIG = {
    "data_period": "1y",  # 需要1年数据来识别VCP模式
    "min_contractions": 3,
    "max_contractions": 5,
    "pullback_thresholds": {
        "first": (8, 15),    # 第1次回调：8-15%
        "second": (4, 8),    # 第2次回调：4-8%
        "third": (2, 4),     # 第3次回调：2-4%
        "fourth": (1, 3)     # 第4次回调：1-3%
    },
    "trend_requirements": {
        "distance_from_52w_high": 25,  # 距离52周新高不超过25%
        "ma150_rising_periods": 20     # 150日MA上升确认期
    },
    "volume_requirements": {
        "contraction_threshold": 0.7,  # 回调期间成交量萎缩到70%以下
        "breakout_threshold": 1.5,     # 突破时成交量放大50%以上
        "dry_up_threshold": 0.5        # 最终阶段成交量干涸到50%以下
    },
    "breakout_prediction": {
        "proximity_threshold": 3.0,    # 距离突破点3%以内为潜在突破
        "tight_range_days": 5,         # 最近5天价格区间收窄
        "volume_drying_threshold": 0.6  # 成交量萎缩到60%以下
    }
}

# 创建结果文件夹
RESULTS_BASE_DIR = "results"
DATE_FOLDER = datetime.now().strftime('%Y%m%d')
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, DATE_FOLDER)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_stock_data_extended(symbol, period="1y"):
    """获取扩展的股票数据用于VCP分析"""
    try:
        data = yf.download(symbol, period=period, interval="1d", 
                         progress=False, auto_adjust=False)
        
        if data is None or len(data) < 100:  # 至少需要100天数据 (降低要求)
            return None
        
        # 处理多级列名问题
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 计算技术指标
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA150'] = data['Close'].rolling(window=150).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        
        # 计算成交量移动平均
        data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_MA50'] = data['Volume'].rolling(window=50).mean()
        
        # 计算52周高低点
        data['High_52W'] = data['High'].rolling(window=252).max()
        data['Low_52W'] = data['Low'].rolling(window=252).min()
        
        # 计算波动率
        data['Returns'] = data['Close'].pct_change()
        data['Volatility_20'] = data['Returns'].rolling(window=20).std()
        
        return data.dropna()
        
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None

def check_trend_confirmation(data):
    """检查趋势确认条件"""
    if len(data) < 150:
        return False, 0, {}
    
    latest = data.iloc[-1]
    details = {}
    score = 0
    
    # 1. 价格高于关键移动平均线 (2分)
    price_above_ma50 = latest['Close'] > latest['MA50']
    price_above_ma150 = latest['Close'] > latest['MA150']
    
    if price_above_ma50 and price_above_ma150:
        score += 2
        details['price_above_mas'] = True
    else:
        details['price_above_mas'] = False
        return False, score, details
    
    # 2. 150日MA呈上升趋势 (1分)
    ma150_current = latest['MA150']
    ma150_20_days_ago = data.iloc[-20]['MA150'] if len(data) >= 20 else ma150_current
    
    ma150_rising = ma150_current > ma150_20_days_ago
    details['ma150_rising'] = ma150_rising
    
    if ma150_rising:
        score += 1
    
    # 3. 距离52周新高的距离 (2分)
    distance_from_high = (latest['High_52W'] - latest['Close']) / latest['High_52W'] * 100
    details['distance_from_52w_high'] = round(distance_from_high, 2)
    
    if distance_from_high <= VCP_CONFIG['trend_requirements']['distance_from_52w_high']:
        if distance_from_high <= 10:  # 距离新高10%以内
            score += 2
        elif distance_from_high <= 25:  # 距离新高25%以内
            score += 1
    
    # 趋势确认需要至少3分
    trend_confirmed = score >= 3
    return trend_confirmed, score, details

def find_swing_points(data, window=10):
    """识别摆动高点和低点"""
    highs = []
    lows = []
    
    for i in range(window, len(data) - window):
        # 检查是否为摆动高点
        if data['High'].iloc[i] == data['High'].iloc[i-window:i+window+1].max():
            highs.append((i, data.index[i], data['High'].iloc[i]))
        
        # 检查是否为摆动低点
        if data['Low'].iloc[i] == data['Low'].iloc[i-window:i+window+1].min():
            lows.append((i, data.index[i], data['Low'].iloc[i]))
    
    return highs, lows

def analyze_contraction_pattern(data):
    """分析收缩整理形态"""
    highs, lows = find_swing_points(data)
    
    if len(highs) < 3 or len(lows) < 3:
        return False, 0, {}
    
    # 获取最近的摆动点
    recent_highs = highs[-5:]  # 最近5个高点
    recent_lows = lows[-5:]    # 最近5个低点
    
    details = {}
    score = 0
    
    # 分析回调幅度
    pullbacks = []
    if len(recent_highs) >= 2:
        for i in range(1, len(recent_highs)):
            high_price = recent_highs[i-1][2]
            # 找到这个高点之后的最低点
            high_date_idx = recent_highs[i-1][0]
            next_high_idx = recent_highs[i][0]
            
            # 在两个高点之间找最低点
            low_in_period = data['Low'].iloc[high_date_idx:next_high_idx].min()
            pullback_pct = (high_price - low_in_period) / high_price * 100
            pullbacks.append(pullback_pct)
    
    details['pullbacks'] = [round(p, 2) for p in pullbacks]
    
    # 检查回调幅度是否递减
    if len(pullbacks) >= 3:
        decreasing_pullbacks = all(pullbacks[i] > pullbacks[i+1] for i in range(len(pullbacks)-1))
        details['decreasing_pullbacks'] = decreasing_pullbacks
        
        if decreasing_pullbacks:
            score += 3
        
        # 检查回调幅度是否在合理范围内
        valid_pullbacks = 0
        thresholds = list(VCP_CONFIG['pullback_thresholds'].values())
        
        for i, pullback in enumerate(pullbacks[:4]):  # 最多检查4次回调
            if i < len(thresholds):
                min_thresh, max_thresh = thresholds[i]
                if min_thresh <= pullback <= max_thresh:
                    valid_pullbacks += 1
        
        details['valid_pullbacks'] = valid_pullbacks
        if valid_pullbacks >= 2:
            score += 2
    
    # 检查波动率收缩
    if len(data) >= 60:
        recent_volatility = data['Volatility_20'].iloc[-20:].mean()
        earlier_volatility = data['Volatility_20'].iloc[-60:-40].mean()
        
        volatility_contraction = recent_volatility < earlier_volatility
        details['volatility_contraction'] = volatility_contraction
        
        if volatility_contraction:
            score += 1
    
    pattern_valid = score >= 3
    return pattern_valid, score, details

def analyze_volume_characteristics(data):
    """分析成交量特征"""
    if len(data) < 50:
        return False, 0, {}
    
    details = {}
    score = 0
    
    # 1. 最近期间成交量萎缩 (2分)
    recent_volume = data['Volume'].iloc[-10:].mean()
    avg_volume = data['Volume_MA50'].iloc[-1]
    
    volume_contraction_ratio = recent_volume / avg_volume
    details['volume_contraction_ratio'] = round(volume_contraction_ratio, 2)
    
    if volume_contraction_ratio < VCP_CONFIG['volume_requirements']['contraction_threshold']:
        score += 2
    elif volume_contraction_ratio < 0.85:
        score += 1
    
    # 2. 成交量干涸 (1分)
    min_volume_recent = data['Volume'].iloc[-20:].min()
    dry_up_ratio = min_volume_recent / avg_volume
    details['dry_up_ratio'] = round(dry_up_ratio, 2)
    
    if dry_up_ratio < VCP_CONFIG['volume_requirements']['dry_up_threshold']:
        score += 1
    
    # 3. 检查是否有突破成交量 (2分)
    latest_volume = data['Volume'].iloc[-1]
    breakout_volume_ratio = latest_volume / avg_volume
    details['breakout_volume_ratio'] = round(breakout_volume_ratio, 2)
    
    if breakout_volume_ratio > VCP_CONFIG['volume_requirements']['breakout_threshold']:
        score += 2
    elif breakout_volume_ratio > 1.2:
        score += 1
    
    volume_valid = score >= 2
    return volume_valid, score, details

def check_breakout_status(data):
    """检查突破状态：已突破 vs 潜在突破"""
    if len(data) < 50:
        return "无效", 0, {}
    
    details = {}
    score = 0
    
    # 1. 识别关键阻力位（前期高点）
    current_price = data['Close'].iloc[-1]
    recent_high = data['High'].iloc[-50:-5].max()  # 排除最近5天，看前期高点
    
    details['current_price'] = round(current_price, 2)
    details['resistance_level'] = round(recent_high, 2)
    
    # 2. 计算距离阻力位的距离
    distance_to_resistance = (recent_high - current_price) / current_price * 100
    details['distance_to_resistance_pct'] = round(distance_to_resistance, 2)
    
    # 3. 判断突破状态
    if current_price > recent_high:
        # 已经突破
        breakout_strength = (current_price - recent_high) / recent_high * 100
        details['breakout_strength'] = round(breakout_strength, 2)
        
        # 检查突破后的维持能力
        days_above_resistance = sum(data['Close'].iloc[-3:] > recent_high)
        details['days_above_resistance'] = days_above_resistance
        
        if breakout_strength > 2 and days_above_resistance >= 2:
            score += 5
            status = "已突破"
        elif breakout_strength > 0.5:
            score += 3
            status = "刚突破"
        else:
            score += 1
            status = "弱突破"
    
    elif distance_to_resistance <= VCP_CONFIG['breakout_prediction']['proximity_threshold']:
        # 接近突破点，检查潜在突破信号
        status = "潜在突破"
        
        # 检查价格区间收窄
        recent_range = data['High'].iloc[-5:].max() - data['Low'].iloc[-5:].min()
        earlier_range = data['High'].iloc[-15:-5].max() - data['Low'].iloc[-15:-5].min()
        range_contraction = recent_range < earlier_range * 0.7
        details['range_contraction'] = range_contraction
        
        if range_contraction:
            score += 2
        
        # 检查成交量萎缩（蓄势待发）
        recent_volume = data['Volume'].iloc[-5:].mean()
        avg_volume = data['Volume_MA20'].iloc[-1]
        volume_drying = recent_volume / avg_volume
        details['volume_drying_ratio'] = round(volume_drying, 2)
        
        if volume_drying < VCP_CONFIG['breakout_prediction']['volume_drying_threshold']:
            score += 2
        
        # 检查是否在关键支撑位之上
        support_level = data['Low'].iloc[-20:].min()
        above_support = current_price > support_level * 1.02  # 高于支撑位2%
        details['above_support'] = above_support
        
        if above_support:
            score += 1
        
        # 潜在突破的评分调整
        if distance_to_resistance <= 1:  # 距离阻力位1%以内
            score += 2
        elif distance_to_resistance <= 2:  # 距离阻力位2%以内
            score += 1
    
    else:
        status = "远离突破"
        score = 0
    
    details['breakout_status'] = status
    return status, score, details

def detect_vcp_pattern(symbol):
    """检测VCP模式的主函数"""
    try:
        # 获取数据
        data = get_stock_data_extended(symbol)
        if data is None:
            return None
        
        # 1. 趋势确认
        trend_ok, trend_score, trend_details = check_trend_confirmation(data)
        if not trend_ok:
            return None
        
        # 2. 整理形态分析
        pattern_ok, pattern_score, pattern_details = analyze_contraction_pattern(data)
        
        # 3. 成交量分析
        volume_ok, volume_score, volume_details = analyze_volume_characteristics(data)
        
        # 4. 突破状态检查
        breakout_status, breakout_score, breakout_details = check_breakout_status(data)
        
        # 计算总分 (最高20分)
        total_score = trend_score + pattern_score + volume_score + breakout_score
        
        # VCP分类
        if breakout_status in ["已突破", "刚突破"]:
            if total_score >= 16:
                vcp_category = "🔥 优秀突破VCP"
            elif total_score >= 12:
                vcp_category = "⭐ 良好突破VCP"
            else:
                vcp_category = "✅ 一般突破VCP"
        elif breakout_status == "潜在突破":
            if total_score >= 14:
                vcp_category = "🎯 高潜力VCP观察"
            elif total_score >= 10:
                vcp_category = "👀 中潜力VCP观察"
            else:
                vcp_category = "📋 低潜力VCP观察"
        else:
            vcp_category = "❌ 无效VCP"
        
        # 构建结果
        result = {
            "symbol": symbol,
            "vcp_score": total_score,
            "vcp_category": vcp_category,
            "breakout_status": breakout_status,
            "current_price": round(data['Close'].iloc[-1], 2),
            "price_change_pct": round((data['Close'].iloc[-1] / data['Close'].iloc[-2] - 1) * 100, 2),
            "trend_confirmed": trend_ok,
            "pattern_valid": pattern_ok,
            "volume_valid": volume_ok,
            "analysis_details": {
                "trend": trend_details,
                "pattern": pattern_details,
                "volume": volume_details,
                "breakout": breakout_details
            },
            "component_scores": {
                "trend_score": trend_score,
                "pattern_score": pattern_score,
                "volume_score": volume_score,
                "breakout_score": breakout_score
            }
        }
        
        return result
        
    except Exception as e:
        print(f"分析 {symbol} VCP模式时出错: {e}")
        return None

def scan_vcp_patterns(symbols, min_score=8):
    """批量扫描VCP模式"""
    print(f"🔍 开始VCP模式扫描（突破预测版）...")
    print(f"   - 扫描股票数量: {len(symbols)}")
    print(f"   - 最低VCP评分: {min_score}")
    print(f"   - 预计用时: {len(symbols) * 3 // 60}分钟")
    print("=" * 60)
    
    results = []
    processed = 0
    errors = 0
    start_time = datetime.now()
    
    # 统计各个筛选条件的通过率
    stats = {
        'data_available': 0,
        'trend_confirmed': 0,
        'pattern_valid': 0,
        'volume_valid': 0,
        'breakout_potential': 0,
        'min_score_met': 0,
        # 组合筛选条件统计
        'selection_1': 0,  # 趋势确认
        'selection_2': 0,  # 趋势确认 + 整理形态
        'selection_3': 0,  # 趋势确认 + 整理形态 + 成交量配合
        'selection_4': 0,  # 全部条件 + 突破潜力
        # 详细条件统计
        'price_above_ma50': 0,
        'price_above_ma150': 0,
        'ma150_rising': 0,
        'within_25pct_high': 0,
        'decreasing_pullbacks': 0,
        'volume_contraction': 0,
        'breakout_volume': 0
    }
    
    for i, symbol in enumerate(symbols):
        try:
            # 进度显示
            if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(symbols) - i - 1) / rate if rate > 0 else 0
                print(f"📈 进度: {i + 1}/{len(symbols)} ({(i + 1)/len(symbols)*100:.1f}%) | "
                      f"发现VCP: {len(results)} | 错误: {errors} | "
                      f"预计剩余: {eta/60:.1f}分钟")
            
            # 获取数据
            data = get_stock_data_extended(symbol)
            if data is not None:
                stats['data_available'] += 1
                
                # 检查各个条件
                trend_ok, trend_score, trend_details = check_trend_confirmation(data)
                if trend_ok:
                    stats['trend_confirmed'] += 1
                
                # 详细趋势条件统计
                if trend_details.get('price_above_mas', False):
                    stats['price_above_ma50'] += 1
                    stats['price_above_ma150'] += 1
                if trend_details.get('ma150_rising', False):
                    stats['ma150_rising'] += 1
                if trend_details.get('distance_from_52w_high', 100) <= 25:
                    stats['within_25pct_high'] += 1
                
                pattern_ok, pattern_score, pattern_details = analyze_contraction_pattern(data)
                if pattern_ok:
                    stats['pattern_valid'] += 1
                
                # 详细形态条件统计
                if pattern_details.get('decreasing_pullbacks', False):
                    stats['decreasing_pullbacks'] += 1
                
                volume_ok, volume_score, volume_details = analyze_volume_characteristics(data)
                if volume_ok:
                    stats['volume_valid'] += 1
                
                # 详细成交量条件统计
                if volume_details.get('volume_contraction_ratio', 1.0) < 0.7:
                    stats['volume_contraction'] += 1
                if volume_details.get('breakout_volume_ratio', 0.0) > 1.5:
                    stats['breakout_volume'] += 1
                
                breakout_status, breakout_score, breakout_details = check_breakout_status(data)
                if breakout_status != "远离突破":
                    stats['breakout_potential'] += 1
                
                total_score = trend_score + pattern_score + volume_score + breakout_score
                if total_score >= min_score:
                    stats['min_score_met'] += 1
                
                # 组合筛选条件统计
                if trend_ok:  # Selection 1: 趋势确认
                    stats['selection_1'] += 1
                    
                    if pattern_ok:  # Selection 2: 趋势确认 + 整理形态
                        stats['selection_2'] += 1
                        
                        if volume_ok:  # Selection 3: 趋势确认 + 整理形态 + 成交量配合
                            stats['selection_3'] += 1
                            
                            if breakout_status != "远离突破":  # Selection 4: 全部条件 + 突破潜力
                                stats['selection_4'] += 1
            
            result = detect_vcp_pattern(symbol)
            processed += 1
            
            if result and result['vcp_score'] >= min_score and result['breakout_status'] != "远离突破":
                results.append(result)
                
                # 实时显示发现的VCP
                category = result['vcp_category']
                score = result['vcp_score']
                price = result['current_price']
                change = result['price_change_pct']
                status = result['breakout_status']
                print(f"{category}: {symbol} ({status}) | {score}分 | ${price} | {change:+.1f}%")
            
            # 避免请求过于频繁
            time.sleep(0.2)
            
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"❌ {symbol} 分析失败: {e}")
    
    # 按评分排序
    results.sort(key=lambda x: x['vcp_score'], reverse=True)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # 显示详细统计
    print(f"\n📊 VCP筛选条件通过率统计:")
    print(f"   📈 数据可用: {stats['data_available']}/{processed} ({stats['data_available']/processed*100:.1f}%)")
    print(f"   🎯 趋势确认: {stats['trend_confirmed']}/{processed} ({stats['trend_confirmed']/processed*100:.1f}%)")
    print(f"   📐 整理形态: {stats['pattern_valid']}/{processed} ({stats['pattern_valid']/processed*100:.1f}%)")
    print(f"   📊 成交量配合: {stats['volume_valid']}/{processed} ({stats['volume_valid']/processed*100:.1f}%)")
    print(f"   🚀 突破潜力: {stats['breakout_potential']}/{processed} ({stats['breakout_potential']/processed*100:.1f}%)")
    print(f"   ⭐ 达到最低评分: {stats['min_score_met']}/{processed} ({stats['min_score_met']/processed*100:.1f}%)")
    
    print(f"\n🔍 详细筛选条件统计:")
    print(f"   📈 Price > MA50 & MA150: {stats['price_above_ma50']}/{processed} ({stats['price_above_ma50']/processed*100:.1f}%)")
    print(f"   📊 MA150 Rising: {stats['ma150_rising']}/{processed} ({stats['ma150_rising']/processed*100:.1f}%)")
    print(f"   🎯 Within 25% of 52W High: {stats['within_25pct_high']}/{processed} ({stats['within_25pct_high']/processed*100:.1f}%)")
    print(f"   📐 Decreasing Pullbacks: {stats['decreasing_pullbacks']}/{processed} ({stats['decreasing_pullbacks']/processed*100:.1f}%)")
    print(f"   📊 Volume Contraction (<70%): {stats['volume_contraction']}/{processed} ({stats['volume_contraction']/processed*100:.1f}%)")
    print(f"   🚀 Breakout Volume (>150%): {stats['breakout_volume']}/{processed} ({stats['breakout_volume']/processed*100:.1f}%)")
    
    print(f"\n🔍 VCP组合筛选条件统计:")
    print(f"   Selection 1 (趋势确认): {stats['selection_1']}/{processed} ({stats['selection_1']/processed*100:.1f}%)")
    print(f"   Selection 2 (趋势+形态): {stats['selection_2']}/{processed} ({stats['selection_2']/processed*100:.1f}%)")
    print(f"   Selection 3 (趋势+形态+成交量): {stats['selection_3']}/{processed} ({stats['selection_3']/processed*100:.1f}%)")
    print(f"   Selection 4 (全部条件+突破): {stats['selection_4']}/{processed} ({stats['selection_4']/processed*100:.1f}%)")
    
    print(f"\n📊 VCP扫描完成统计:")
    print(f"   - 处理股票: {processed}")
    print(f"   - 发现VCP: {len(results)}")
    print(f"   - 错误数量: {errors}")
    print(f"   - 总用时: {total_time/60:.1f}分钟")
    print(f"   - 平均速度: {processed/(total_time/60):.1f}个/分钟")
    print(f"   - VCP发现率: {len(results)/processed*100:.2f}%")
    
    return results

def save_vcp_results(results):
    """保存VCP扫描结果"""
    if not results:
        print("❌ 没有发现符合条件的VCP模式")
        return
    
    # 分类结果
    breakout_vcps = [r for r in results if r['breakout_status'] in ['已突破', '刚突破']]
    potential_vcps = [r for r in results if r['breakout_status'] == '潜在突破']
    
    # 转换为DataFrame
    def create_dataframe(vcp_list):
        df_data = []
        for result in vcp_list:
            row = {
                "股票代码": result['symbol'],
                "VCP评分": result['vcp_score'],
                "VCP分类": result['vcp_category'],
                "突破状态": result['breakout_status'],
                "当前价格": result['current_price'],
                "涨跌幅%": result['price_change_pct'],
                "趋势确认": "✅" if result['trend_confirmed'] else "❌",
                "整理形态": "✅" if result['pattern_valid'] else "❌",
                "成交量配合": "✅" if result['volume_valid'] else "❌",
                "趋势评分": result['component_scores']['trend_score'],
                "形态评分": result['component_scores']['pattern_score'],
                "成交量评分": result['component_scores']['volume_score'],
                "突破评分": result['component_scores']['breakout_score']
            }
            
            # 添加详细分析数据
            details = result['analysis_details']
            if 'trend' in details:
                row["距52周高点%"] = details['trend'].get('distance_from_52w_high', 'N/A')
            if 'pattern' in details:
                row["回调次数"] = len(details['pattern'].get('pullbacks', []))
            if 'volume' in details:
                row["成交量萎缩比"] = details['volume'].get('volume_contraction_ratio', 'N/A')
            if 'breakout' in details:
                if result['breakout_status'] in ['已突破', '刚突破']:
                    row["突破强度%"] = details['breakout'].get('breakout_strength', 'N/A')
                else:
                    row["距阻力位%"] = details['breakout'].get('distance_to_resistance_pct', 'N/A')
                    row["成交量萎缩"] = details['breakout'].get('volume_drying_ratio', 'N/A')
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    # 保存文件
    base_name = f"VCP_Pattern_Scan_{DATE_FOLDER}"
    
    # Excel文件
    excel_path = os.path.join(RESULTS_DIR, f"{base_name}.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        # 突破VCP
        if breakout_vcps:
            df_breakout = create_dataframe(breakout_vcps)
            df_breakout.to_excel(writer, sheet_name="突破VCP", index=False)
        
        # 潜在突破VCP观察清单
        if potential_vcps:
            df_potential = create_dataframe(potential_vcps)
            df_potential.to_excel(writer, sheet_name="潜在突破VCP观察清单", index=False)
        
        # 所有结果
        if results:
            df_all = create_dataframe(results)
            df_all.to_excel(writer, sheet_name="所有VCP结果", index=False)
    
    # 分别保存CSV文件
    if breakout_vcps:
        df_breakout = create_dataframe(breakout_vcps)
        breakout_csv = os.path.join(RESULTS_DIR, f"{base_name}_突破VCP.csv")
        df_breakout.to_csv(breakout_csv, index=False)
    
    if potential_vcps:
        df_potential = create_dataframe(potential_vcps)
        potential_csv = os.path.join(RESULTS_DIR, f"{base_name}_潜在突破VCP观察清单.csv")
        df_potential.to_csv(potential_csv, index=False)
    
    # 详细分析JSON
    json_path = os.path.join(RESULTS_DIR, f"{base_name}_详细分析.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ VCP扫描结果已保存:")
    print(f"📊 Excel文件: {excel_path}")
    if breakout_vcps:
        print(f"🔥 突破VCP: {len(breakout_vcps)}个")
    if potential_vcps:
        print(f"🎯 潜在突破VCP观察清单: {len(potential_vcps)}个")
    print(f"📋 详细分析: {json_path}")
    
    # 显示分类统计
    print(f"\n📊 VCP分类统计:")
    print(f"   🔥 已突破VCP: {len(breakout_vcps)}个")
    print(f"   🎯 潜在突破VCP: {len(potential_vcps)}个")
    
    # 显示前10个最佳VCP
    print(f"\n🏆 前10个最佳VCP模式:")
    print("=" * 90)
    for i, result in enumerate(results[:10]):
        category = result['vcp_category']
        symbol = result['symbol']
        score = result['vcp_score']
        price = result['current_price']
        change = result['price_change_pct']
        status = result['breakout_status']
        print(f"{i+1:2d}. {category} {symbol:>6} ({status}) | {score:2d}分 | ${price:>8.2f} | {change:+6.1f}%")

def main():
    """主函数"""
    print("🎯 VCP (Volatility Contraction Pattern) 突破预测检测器")
    print("基于Mark Minervini理论 - 识别突破VCP和潜在突破VCP观察清单")
    print("=" * 70)
    
    # 选择扫描范围
    print("请选择扫描范围:")
    print("1. 测试模式 (前100个股票)")
    print("2. 完整扫描 (所有股票)")
    print("3. 自定义股票列表")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        symbols = STOCK_SYMBOLS[:100]
        print(f"🧪 测试模式: 扫描前100个股票")
    elif choice == "2":
        symbols = STOCK_SYMBOLS
        print(f"🔍 完整扫描: 扫描{len(STOCK_SYMBOLS)}个股票")
    elif choice == "3":
        custom_symbols = input("请输入股票代码 (用逗号分隔): ").strip().upper().split(',')
        symbols = [s.strip() for s in custom_symbols if s.strip()]
        print(f"📝 自定义扫描: {len(symbols)}个股票")
    else:
        print("❌ 无效选择，使用测试模式")
        symbols = STOCK_SYMBOLS[:100]
    
    # 设置最低评分
    min_score_input = input("请输入最低VCP评分 (默认8分): ").strip()
    min_score = int(min_score_input) if min_score_input.isdigit() else 8
    
    # 开始扫描
    results = scan_vcp_patterns(symbols, min_score)
    
    # 保存结果
    if results:
        save_vcp_results(results)
    else:
        print("❌ 未发现符合条件的VCP模式")

if __name__ == "__main__":
    main()