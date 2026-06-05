#!/usr/bin/env python3
"""
Enhanced VCP Pattern Detector with Updated Selection Criteria
Based on Mark Minervini's Trend Template and Advanced Technical Analysis

Updated Selection Criteria:
1. Trend Template Met (Mark Minervini's 10 criteria)
2. Market Cap > $1 billion
3. Uptrend Nearing Breakout
4. Higher Lows Pattern
5. Volume Contracting
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import time
import json
import os
from stock_symbols_1243 import STOCK_SYMBOLS, ETF_SYMBOLS
import warnings
warnings.filterwarnings('ignore')

# ============ Enhanced VCP Selection Criteria ============
ENHANCED_VCP_CONFIG = {
    "data_period": "2y",  # Need 2 years for weekly data and comprehensive analysis
    "market_cap_min": 1_000_000_000,  # $1 billion minimum market cap
    "trend_template_criteria": {
        "price_above_ma50": True,
        "price_above_ma150": True,
        "price_above_ma200": True,
        "ma50_above_ma150": True,
        "ma50_above_ma200": True,
        "ma150_above_ma200": True,
        "ma200_rising": True,
        "price_within_25pct_high": True,
        "price_above_30week_low": True,
        "relative_strength": True
    },
    "breakout_criteria": {
        "near_100day_high": 10,  # Within 10 candles of 100-day high
        "within_7pct_daily_high": 7.0,  # Within 7% of daily 100-day high
        "within_20pct_weekly_high": 20.0,  # Within 20% of weekly 100-day high
        "below_daily_high": True  # Must be below the high (not broken out yet)
    },
    "higher_lows_periods": [10, 20, 30],  # Check higher lows over these periods
    "volume_contraction_periods": [5, 10, 15, 20, 25, 30]  # Volume contraction lookback periods
}

# Create results folder
RESULTS_BASE_DIR = "results"
DATE_FOLDER = datetime.now().strftime('%Y%m%d')
RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, DATE_FOLDER)
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_enhanced_stock_data(symbol, period="2y"):
    """Get comprehensive stock data for enhanced VCP analysis"""
    try:
        # Get daily data
        daily_data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
        
        # Get weekly data
        weekly_data = yf.download(symbol, period=period, interval="1wk", progress=False, auto_adjust=False)
        
        if daily_data is None or len(daily_data) < 200:
            return None, None
        
        # Handle multi-level columns
        if isinstance(daily_data.columns, pd.MultiIndex):
            daily_data.columns = daily_data.columns.get_level_values(0)
        if isinstance(weekly_data.columns, pd.MultiIndex):
            weekly_data.columns = weekly_data.columns.get_level_values(0)
        
        # Calculate technical indicators for daily data
        daily_data['MA20'] = daily_data['Close'].rolling(window=20).mean()
        daily_data['MA50'] = daily_data['Close'].rolling(window=50).mean()
        daily_data['MA150'] = daily_data['Close'].rolling(window=150).mean()
        daily_data['MA200'] = daily_data['Close'].rolling(window=200).mean()
        
        # Volume indicators
        daily_data['Volume_MA20'] = daily_data['Volume'].rolling(window=20).mean()
        
        # Range calculations
        daily_data['High_100'] = daily_data['High'].rolling(window=100).max()
        daily_data['Low_100'] = daily_data['Low'].rolling(window=100).min()
        daily_data['Low_10'] = daily_data['Low'].rolling(window=10).min()
        daily_data['Low_20'] = daily_data['Low'].rolling(window=20).min()
        daily_data['Low_30'] = daily_data['Low'].rolling(window=30).min()
        
        # Weekly indicators
        if len(weekly_data) > 0:
            weekly_data['High_100'] = weekly_data['High'].rolling(window=100).max()
        
        return daily_data.dropna(), weekly_data.dropna()
        
    except Exception as e:
        print(f"获取 {symbol} 数据失败: {e}")
        return None, None

def get_market_cap(symbol):
    """Get market capitalization for the stock"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        market_cap = info.get('marketCap', 0)
        return market_cap
    except:
        return 0

def check_trend_template(daily_data):
    """Check Mark Minervini's 10-point Trend Template"""
    if len(daily_data) < 200:
        return False, 0, {}
    
    latest = daily_data.iloc[-1]
    details = {}
    score = 0
    criteria_met = 0
    
    # 1. Price above 50-day MA
    price_above_ma50 = latest['Close'] > latest['MA50']
    details['price_above_ma50'] = price_above_ma50
    if price_above_ma50:
        criteria_met += 1
        score += 1
    
    # 2. Price above 150-day MA
    price_above_ma150 = latest['Close'] > latest['MA150']
    details['price_above_ma150'] = price_above_ma150
    if price_above_ma150:
        criteria_met += 1
        score += 1
    
    # 3. Price above 200-day MA
    price_above_ma200 = latest['Close'] > latest['MA200']
    details['price_above_ma200'] = price_above_ma200
    if price_above_ma200:
        criteria_met += 1
        score += 1
    
    # 4. 50-day MA above 150-day MA
    ma50_above_ma150 = latest['MA50'] > latest['MA150']
    details['ma50_above_ma150'] = ma50_above_ma150
    if ma50_above_ma150:
        criteria_met += 1
        score += 1
    
    # 5. 50-day MA above 200-day MA
    ma50_above_ma200 = latest['MA50'] > latest['MA200']
    details['ma50_above_ma200'] = ma50_above_ma200
    if ma50_above_ma200:
        criteria_met += 1
        score += 1
    
    # 6. 150-day MA above 200-day MA
    ma150_above_ma200 = latest['MA150'] > latest['MA200']
    details['ma150_above_ma200'] = ma150_above_ma200
    if ma150_above_ma200:
        criteria_met += 1
        score += 1
    
    # 7. 200-day MA is rising
    ma200_current = latest['MA200']
    ma200_30_days_ago = daily_data.iloc[-30]['MA200'] if len(daily_data) >= 30 else ma200_current
    ma200_rising = ma200_current > ma200_30_days_ago
    details['ma200_rising'] = ma200_rising
    if ma200_rising:
        criteria_met += 1
        score += 1
    
    # 8. Price within 25% of 52-week high
    high_52w = daily_data['High'].rolling(window=252).max().iloc[-1]
    distance_from_high = (high_52w - latest['Close']) / high_52w * 100
    within_25pct_high = distance_from_high <= 25
    details['within_25pct_high'] = within_25pct_high
    details['distance_from_52w_high'] = round(distance_from_high, 2)
    if within_25pct_high:
        criteria_met += 1
        score += 1
    
    # 9. Price above 30% above 52-week low
    low_52w = daily_data['Low'].rolling(window=252).min().iloc[-1]
    above_52w_low_pct = (latest['Close'] - low_52w) / low_52w * 100
    above_30pct_low = above_52w_low_pct >= 30
    details['above_30pct_52w_low'] = above_30pct_low
    details['above_52w_low_pct'] = round(above_52w_low_pct, 2)
    if above_30pct_low:
        criteria_met += 1
        score += 1
    
    # 10. Relative strength (simplified - price performance vs market)
    # Using 3-month performance as proxy
    relative_strength_good = False
    details['relative_strength'] = False
    details['price_performance_3m'] = 0.0
    if len(daily_data) >= 63:
        price_3m_ago = daily_data['Close'].iloc[-63]
        price_performance = (latest['Close'] - price_3m_ago) / price_3m_ago * 100
        relative_strength_good = price_performance > 0  # Simplified: positive 3-month performance
        details['relative_strength'] = relative_strength_good
        details['price_performance_3m'] = round(price_performance, 2)
        if relative_strength_good:
            criteria_met += 1
            score += 1
    
    details['criteria_met'] = criteria_met
    details['total_criteria'] = 10
    
    # Need at least 8 out of 10 criteria for trend template
    trend_template_met = criteria_met >= 8
    
    # Stage 2 identification (stricter criteria)
    stage2_criteria = 0
    if price_above_ma50 and price_above_ma150 and price_above_ma200:
        stage2_criteria += 1
    if ma50_above_ma150 and ma150_above_ma200:
        stage2_criteria += 1
    if ma200_rising:
        stage2_criteria += 1
    if within_25pct_high:
        stage2_criteria += 1
    if above_30pct_low:
        stage2_criteria += 1
    if relative_strength_good:
        stage2_criteria += 1
    
    # Stage 2 requires all 6 key criteria
    is_stage2 = stage2_criteria >= 6
    details['is_stage2'] = is_stage2
    details['stage2_criteria_met'] = stage2_criteria
    
    return trend_template_met, score, details

def check_uptrend_nearing_breakout(daily_data, weekly_data):
    """Check if stock is in uptrend nearing breakout"""
    if len(daily_data) < 100:
        return False, 0, {}
    
    details = {}
    score = 0
    current_price = daily_data['Close'].iloc[-1]
    
    # 1. Current high equals 100-day high within last 10 candles
    high_100_day = daily_data['High_100'].iloc[-1]
    recent_10_high = daily_data['High'].iloc[-10:].max()
    near_100day_high = recent_10_high >= high_100_day * 0.99  # Within 1% tolerance
    details['near_100day_high'] = near_100day_high
    details['high_100_day'] = round(high_100_day, 2)
    details['recent_10_high'] = round(recent_10_high, 2)
    if near_100day_high:
        score += 2
    
    # 2. Current price within 7% of daily 100-day high
    distance_to_daily_high = (high_100_day - current_price) / current_price * 100
    within_7pct_daily = distance_to_daily_high <= 7.0
    details['within_7pct_daily_high'] = within_7pct_daily
    details['distance_to_daily_high'] = round(distance_to_daily_high, 2)
    if within_7pct_daily:
        score += 2
    
    # 3. Current price within 20% of weekly 100-period high
    if len(weekly_data) >= 100:
        weekly_100_high = weekly_data['High_100'].iloc[-1]
        distance_to_weekly_high = (weekly_100_high - current_price) / current_price * 100
        within_20pct_weekly = distance_to_weekly_high <= 20.0
        details['within_20pct_weekly_high'] = within_20pct_weekly
        details['distance_to_weekly_high'] = round(distance_to_weekly_high, 2)
        details['weekly_100_high'] = round(weekly_100_high, 2)
        if within_20pct_weekly:
            score += 1
    
    # 4. Current price below daily high (not broken out yet)
    below_daily_high = current_price <= high_100_day
    details['below_daily_high'] = below_daily_high
    if below_daily_high:
        score += 1
    
    breakout_ready = score >= 4
    return breakout_ready, score, details

def check_higher_lows(daily_data):
    """Check for higher lows pattern"""
    if len(daily_data) < 50:
        return False, 0, {}
    
    details = {}
    score = 0
    periods = ENHANCED_VCP_CONFIG['higher_lows_periods']
    
    for period in periods:
        if len(daily_data) >= period * 2:
            current_low = daily_data['Low'].iloc[-period:].min()
            previous_low = daily_data['Low'].iloc[-period*2:-period].min()
            
            higher_low = current_low > previous_low
            details[f'higher_low_{period}d'] = higher_low
            details[f'current_low_{period}d'] = round(current_low, 2)
            details[f'previous_low_{period}d'] = round(previous_low, 2)
            
            if higher_low:
                score += 1
    
    # Need at least 2 out of 3 higher low patterns
    higher_lows_confirmed = score >= 2
    return higher_lows_confirmed, score, details

def check_volume_contracting(daily_data):
    """Check for volume contraction pattern"""
    if len(daily_data) < 50:
        return False, 0, {}
    
    details = {}
    score = 0
    periods = ENHANCED_VCP_CONFIG['volume_contraction_periods']
    
    current_volume_ma = daily_data['Volume_MA20'].iloc[-1]
    
    contracting_count = 0
    for period in periods:
        if len(daily_data) >= period + 1:
            past_volume_ma = daily_data['Volume_MA20'].iloc[-(period+1)]
            volume_contracting = current_volume_ma < past_volume_ma
            
            details[f'volume_contracting_{period}d'] = volume_contracting
            details[f'current_volume_ma'] = round(current_volume_ma, 0)
            details[f'volume_ma_{period}d_ago'] = round(past_volume_ma, 0)
            
            if volume_contracting:
                contracting_count += 1
    
    # Need at least 3 out of 6 volume contraction signals
    volume_contracting_confirmed = contracting_count >= 3
    details['contracting_signals'] = contracting_count
    details['total_signals'] = len(periods)
    
    if volume_contracting_confirmed:
        score = contracting_count
    
    return volume_contracting_confirmed, score, details

def enhanced_vcp_scan(symbol, *, market_cap=None, daily_data=None, weekly_data=None):
    """Enhanced VCP pattern detection with new criteria.

    market_cap / daily_data / weekly_data may be passed in by the caller to
    avoid duplicate yfinance fetches. If None, they are fetched here.
    """
    try:
        # Get market cap (skip the network call if caller already has it)
        if market_cap is None:
            market_cap = get_market_cap(symbol)
        if market_cap < ENHANCED_VCP_CONFIG['market_cap_min']:
            return None

        # Get data (skip the network call if caller already has it)
        if daily_data is None:
            daily_data, weekly_data = get_enhanced_stock_data(symbol)
        if daily_data is None:
            return None
        
        # 1. Check Trend Template
        trend_template_met, trend_score, trend_details = check_trend_template(daily_data)
        if not trend_template_met:
            return None
        
        # 2. Check Uptrend Nearing Breakout
        breakout_ready, breakout_score, breakout_details = check_uptrend_nearing_breakout(daily_data, weekly_data)
        
        # 3. Check Higher Lows
        higher_lows_ok, higher_lows_score, higher_lows_details = check_higher_lows(daily_data)
        
        # 4. Check Volume Contracting
        volume_contracting_ok, volume_score, volume_details = check_volume_contracting(daily_data)
        
        # Calculate total score
        total_score = trend_score + breakout_score + higher_lows_score + volume_score
        
        # Check if stock is in Stage 2
        is_stage2 = trend_details.get('is_stage2', False)
        
        # Determine VCP category with Stage 2 identification
        if is_stage2 and total_score >= 20:
            vcp_category = "🚀 Stage2优秀VCP"
        elif is_stage2 and total_score >= 15:
            vcp_category = "📈 Stage2良好VCP"
        elif is_stage2 and total_score >= 10:
            vcp_category = "✅ Stage2一般VCP"
        elif total_score >= 20:
            vcp_category = "🔥 优秀增强VCP"
        elif total_score >= 15:
            vcp_category = "⭐ 良好增强VCP"
        elif total_score >= 10:
            vcp_category = "✅ 一般增强VCP"
        else:
            vcp_category = "📋 潜在增强VCP"
        
        # Build result
        result = {
            "symbol": symbol,
            "market_cap": market_cap,
            "market_cap_billions": round(market_cap / 1_000_000_000, 2),
            "total_score": total_score,
            "vcp_category": vcp_category,
            "current_price": round(daily_data['Close'].iloc[-1], 2),
            "price_change_pct": round((daily_data['Close'].iloc[-1] / daily_data['Close'].iloc[-2] - 1) * 100, 2),
            "criteria_met": {
                "trend_template": trend_template_met,
                "breakout_ready": breakout_ready,
                "higher_lows": higher_lows_ok,
                "volume_contracting": volume_contracting_ok
            },
            "component_scores": {
                "trend_score": trend_score,
                "breakout_score": breakout_score,
                "higher_lows_score": higher_lows_score,
                "volume_score": volume_score
            },
            "analysis_details": {
                "trend_template": trend_details,
                "breakout": breakout_details,
                "higher_lows": higher_lows_details,
                "volume": volume_details
            }
        }
        
        return result
        
    except Exception as e:
        print(f"分析 {symbol} 增强VCP模式时出错: {e}")
        return None

def scan_enhanced_vcp_patterns(symbols, min_score=10):
    """Scan for enhanced VCP patterns with new criteria"""
    print(f"🎯 增强VCP模式扫描 - 基于Mark Minervini趋势模板")
    print(f"   - 扫描股票数量: {len(symbols)}")
    print(f"   - 最低评分: {min_score}")
    print(f"   - 市值要求: ≥$1B")
    print("=" * 70)
    
    results = []
    processed = 0
    errors = 0
    start_time = datetime.now()
    
    # Enhanced statistics
    stats = {
        'data_available': 0,
        'market_cap_qualified': 0,
        'trend_template_met': 0,
        'stage2_stocks': 0,
        'breakout_ready': 0,
        'higher_lows_confirmed': 0,
        'volume_contracting': 0,
        'min_score_met': 0,
        # Detailed trend template stats
        'price_above_ma50': 0,
        'price_above_ma150': 0,
        'price_above_ma200': 0,
        'ma_alignment_correct': 0,
        'ma200_rising': 0,
        'within_25pct_high': 0,
        'above_30pct_low': 0,
        'relative_strength_good': 0
    }
    
    for i, symbol in enumerate(symbols):
        try:
            # Progress display
            if (i + 1) % 25 == 0 or (i + 1) == len(symbols):
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(symbols) - i - 1) / rate if rate > 0 else 0
                print(f"📈 进度: {i + 1}/{len(symbols)} ({(i + 1)/len(symbols)*100:.1f}%) | "
                      f"发现增强VCP: {len(results)} | 错误: {errors} | "
                      f"预计剩余: {eta/60:.1f}分钟")
            
            # Get market cap first for quick filtering
            market_cap = get_market_cap(symbol)
            daily_data = None
            weekly_data = None
            if market_cap >= ENHANCED_VCP_CONFIG['market_cap_min']:
                stats['market_cap_qualified'] += 1

                # Get data and analyze
                daily_data, weekly_data = get_enhanced_stock_data(symbol)
                if daily_data is not None:
                    stats['data_available'] += 1
                    
                    # Check each criteria for statistics
                    trend_met, trend_score, trend_details = check_trend_template(daily_data)
                    if trend_met:
                        stats['trend_template_met'] += 1
                    
                    # Stage 2 statistics
                    if trend_details.get('is_stage2', False):
                        stats['stage2_stocks'] += 1
                    
                    # Detailed trend template statistics
                    if trend_details.get('price_above_ma50', False):
                        stats['price_above_ma50'] += 1
                    if trend_details.get('price_above_ma150', False):
                        stats['price_above_ma150'] += 1
                    if trend_details.get('price_above_ma200', False):
                        stats['price_above_ma200'] += 1
                    if (trend_details.get('ma50_above_ma150', False) and
                        trend_details.get('ma150_above_ma200', False)):
                        stats['ma_alignment_correct'] += 1
                    if trend_details.get('ma200_rising', False):
                        stats['ma200_rising'] += 1
                    if trend_details.get('within_25pct_high', False):
                        stats['within_25pct_high'] += 1
                    if trend_details.get('above_30pct_52w_low', False):
                        stats['above_30pct_low'] += 1
                    if trend_details.get('relative_strength', False):
                        stats['relative_strength_good'] += 1
                    
                    breakout_ready, _, _ = check_uptrend_nearing_breakout(daily_data, weekly_data)
                    if breakout_ready:
                        stats['breakout_ready'] += 1
                    
                    higher_lows_ok, _, _ = check_higher_lows(daily_data)
                    if higher_lows_ok:
                        stats['higher_lows_confirmed'] += 1
                    
                    volume_ok, _, _ = check_volume_contracting(daily_data)
                    if volume_ok:
                        stats['volume_contracting'] += 1
            
            # Full analysis (reuse pre-fetched market_cap and data — no extra network calls)
            result = enhanced_vcp_scan(
                symbol,
                market_cap=market_cap,
                daily_data=daily_data,
                weekly_data=weekly_data,
            )
            processed += 1
            
            if result and result['total_score'] >= min_score:
                results.append(result)
                stats['min_score_met'] += 1
                
                # Display found VCP - detailed breakdown only for top scores (21+)
                category = result['vcp_category']
                score = result['total_score']
                price = result['current_price']
                change = result['price_change_pct']
                market_cap_b = result['market_cap_billions']
                
                # Get criteria details
                criteria = result['criteria_met']
                scores = result['component_scores']
                details = result['analysis_details']
                
                # Check if this is a Stage 2 stock
                is_stage2 = details['trend_template'].get('is_stage2', False)
                stage2_indicator = " [Stage2]" if is_stage2 else ""
                
                print(f"\n{category}: {symbol} | 总分:{score}/30 | ${price} ({change:+.1f}%) | 市值${market_cap_b:.1f}B{stage2_indicator}")
                
                # Show detailed breakdown only for high-scoring stocks (21+ points)
                if score >= 21:
                    
                    print(f"   📊 评分详情: 趋势{scores['trend_score']}/10 + 突破{scores['breakout_score']}/6 + 低点{scores['higher_lows_score']}/3 + 成交量{scores['volume_score']}/6")
                    
                    # Trend Template Details (10 criteria)
                    trend_details = details['trend_template']
                    trend_status = []
                    trend_status.append("✅MA50" if trend_details.get('price_above_ma50') else "❌MA50")
                    trend_status.append("✅MA150" if trend_details.get('price_above_ma150') else "❌MA150")
                    trend_status.append("✅MA200" if trend_details.get('price_above_ma200') else "❌MA200")
                    trend_status.append("✅MA排列" if trend_details.get('ma50_above_ma150') and trend_details.get('ma150_above_ma200') else "❌MA排列")
                    trend_status.append("✅MA200上升" if trend_details.get('ma200_rising') else "❌MA200上升")
                    trend_status.append("✅25%高点" if trend_details.get('within_25pct_high') else "❌25%高点")
                    trend_status.append("✅30%低点" if trend_details.get('above_30pct_52w_low') else "❌30%低点")
                    trend_status.append("✅相对强度" if trend_details.get('relative_strength') else "❌相对强度")
                    print(f"   🎯 趋势模板({trend_details.get('criteria_met', 0)}/10): {' '.join(trend_status)}")
                    
                    # Breakout Details
                    breakout_details = details['breakout']
                    breakout_status = []
                    breakout_status.append("✅近期高点" if breakout_details.get('near_100day_high') else "❌近期高点")
                    breakout_status.append("✅7%日线" if breakout_details.get('within_7pct_daily_high') else "❌7%日线")
                    breakout_status.append("✅20%周线" if breakout_details.get('within_20pct_weekly_high') else "❌20%周线")
                    breakout_status.append("✅未突破" if breakout_details.get('below_daily_high') else "❌已突破")
                    print(f"   🚀 接近突破: {' '.join(breakout_status)}")
                    
                    # Higher Lows Details
                    higher_lows_details = details['higher_lows']
                    higher_lows_status = []
                    higher_lows_status.append("✅10日" if higher_lows_details.get('higher_low_10d') else "❌10日")
                    higher_lows_status.append("✅20日" if higher_lows_details.get('higher_low_20d') else "❌20日")
                    higher_lows_status.append("✅30日" if higher_lows_details.get('higher_low_30d') else "❌30日")
                    print(f"   📈 更高低点: {' '.join(higher_lows_status)}")
                    
                    # Volume Contraction Details
                    volume_details = details['volume']
                    contracting_signals = volume_details.get('contracting_signals', 0)
                    total_signals = volume_details.get('total_signals', 6)
                    volume_status = []
                    for period in [5, 10, 15, 20, 25, 30]:
                        if volume_details.get(f'volume_contracting_{period}d'):
                            volume_status.append(f"✅{period}日")
                        else:
                            volume_status.append(f"❌{period}日")
                    print(f"   📊 成交量萎缩({contracting_signals}/{total_signals}): {' '.join(volume_status)}")
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"❌ {symbol} 分析失败: {e}")
    
    # Sort results by score
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Display comprehensive statistics
    print(f"\n📊 增强VCP筛选条件通过率统计:")
    print(f"   💰 市值≥$1B: {stats['market_cap_qualified']}/{processed} ({stats['market_cap_qualified']/processed*100:.1f}%)")
    print(f"   📈 数据可用: {stats['data_available']}/{processed} ({stats['data_available']/processed*100:.1f}%)")
    print(f"   🎯 趋势模板达标: {stats['trend_template_met']}/{processed} ({stats['trend_template_met']/processed*100:.1f}%)")
    print(f"   🚀 Stage2股票: {stats['stage2_stocks']}/{processed} ({stats['stage2_stocks']/processed*100:.1f}%)")
    print(f"   🚀 接近突破: {stats['breakout_ready']}/{processed} ({stats['breakout_ready']/processed*100:.1f}%)")
    print(f"   📈 更高低点: {stats['higher_lows_confirmed']}/{processed} ({stats['higher_lows_confirmed']/processed*100:.1f}%)")
    print(f"   📊 成交量萎缩: {stats['volume_contracting']}/{processed} ({stats['volume_contracting']/processed*100:.1f}%)")
    print(f"   ⭐ 达到最低评分: {stats['min_score_met']}/{processed} ({stats['min_score_met']/processed*100:.1f}%)")
    
    print(f"\n🔍 Mark Minervini趋势模板详细统计:")
    print(f"   📈 Price > MA50: {stats['price_above_ma50']}/{processed} ({stats['price_above_ma50']/processed*100:.1f}%)")
    print(f"   📈 Price > MA150: {stats['price_above_ma150']}/{processed} ({stats['price_above_ma150']/processed*100:.1f}%)")
    print(f"   📈 Price > MA200: {stats['price_above_ma200']}/{processed} ({stats['price_above_ma200']/processed*100:.1f}%)")
    print(f"   📊 MA Alignment Correct: {stats['ma_alignment_correct']}/{processed} ({stats['ma_alignment_correct']/processed*100:.1f}%)")
    print(f"   📈 MA200 Rising: {stats['ma200_rising']}/{processed} ({stats['ma200_rising']/processed*100:.1f}%)")
    print(f"   🎯 Within 25% of High: {stats['within_25pct_high']}/{processed} ({stats['within_25pct_high']/processed*100:.1f}%)")
    print(f"   📈 Above 30% of Low: {stats['above_30pct_low']}/{processed} ({stats['above_30pct_low']/processed*100:.1f}%)")
    print(f"   💪 Relative Strength Good: {stats['relative_strength_good']}/{processed} ({stats['relative_strength_good']/processed*100:.1f}%)")
    
    print(f"\n📊 扫描完成统计:")
    print(f"   - 处理股票: {processed}")
    print(f"   - 发现增强VCP: {len(results)}")
    print(f"   - 错误数量: {errors}")
    print(f"   - 总用时: {total_time/60:.1f}分钟")
    print(f"   - 平均速度: {processed/(total_time/60):.1f}个/分钟")
    print(f"   - 增强VCP发现率: {len(results)/processed*100:.2f}%")
    
    # Save top VCP results to markdown file
    save_vcp_results_to_markdown(results)
    
    return results

def save_vcp_results_to_markdown(results):
    """Save top VCP results to markdown file in results folder"""
    if not results:
        return
    
    # Create filename with current date
    current_date = datetime.now().strftime('%Y%m%d')
    filename = f"{current_date}-vcp.md"
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Sort results by score (highest first)
    top_results = sorted(results, key=lambda x: x['total_score'], reverse=True)
    
    # Generate markdown content
    markdown_content = f"""# VCP Pattern Analysis Report - {datetime.now().strftime('%Y-%m-%d')}

## 📊 Enhanced VCP (Volatility Contraction Pattern) Detection Results
**Based on Mark Minervini's Trend Template + Advanced Technical Analysis**

### 🎯 Scan Summary
- **Total Stocks Analyzed**: {len(results)}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Methodology**: Mark Minervini 10-Point Trend Template + VCP Pattern Recognition
- **Minimum Score**: 10/30 points

---

## 🏆 Top VCP Candidates

"""
    
    # Add top 20 results with detailed breakdown
    for i, result in enumerate(top_results[:20], 1):
        symbol = result['symbol']
        category = result['vcp_category']
        score = result['total_score']
        price = result['current_price']
        change = result['price_change_pct']
        market_cap_b = result['market_cap_billions']
        
        # Check if Stage 2
        is_stage2 = result['analysis_details']['trend_template'].get('is_stage2', False)
        stage2_badge = " 🚀 **STAGE 2**" if is_stage2 else ""
        
        # Component scores
        scores = result['component_scores']
        trend_score = scores['trend_score']
        breakout_score = scores['breakout_score']
        higher_lows_score = scores['higher_lows_score']
        volume_score = scores['volume_score']
        
        markdown_content += f"""### {i}. **{symbol}** - {category}{stage2_badge}

**📈 Stock Info:**
- **Price**: ${price} ({change:+.1f}%)
- **Market Cap**: ${market_cap_b:.1f}B
- **Total Score**: {score}/30 points

**📊 Component Scores:**
- **Trend Template**: {trend_score}/10
- **Breakout Readiness**: {breakout_score}/6
- **Higher Lows**: {higher_lows_score}/3
- **Volume Contraction**: {volume_score}/6

"""
        
        # Add detailed analysis for top 10 stocks
        if i <= 10:
            details = result['analysis_details']
            
            # Trend Template Details
            trend_details = details['trend_template']
            markdown_content += f"""**🎯 Mark Minervini Trend Template ({trend_details.get('criteria_met', 0)}/10):**
- Price > MA50: {'✅' if trend_details.get('price_above_ma50') else '❌'}
- Price > MA150: {'✅' if trend_details.get('price_above_ma150') else '❌'}
- Price > MA200: {'✅' if trend_details.get('price_above_ma200') else '❌'}
- MA Alignment: {'✅' if trend_details.get('ma50_above_ma150') and trend_details.get('ma150_above_ma200') else '❌'}
- MA200 Rising: {'✅' if trend_details.get('ma200_rising') else '❌'}
- Within 25% of High: {'✅' if trend_details.get('within_25pct_high') else '❌'}
- Above 30% of Low: {'✅' if trend_details.get('above_30pct_52w_low') else '❌'}
- Relative Strength: {'✅' if trend_details.get('relative_strength') else '❌'}

"""
            
            # Breakout Analysis
            breakout_details = details['breakout']
            markdown_content += f"""**🚀 Breakout Readiness:**
- Near 100-day High: {'✅' if breakout_details.get('near_100day_high') else '❌'}
- Within 7% Daily High: {'✅' if breakout_details.get('within_7pct_daily_high') else '❌'}
- Within 20% Weekly High: {'✅' if breakout_details.get('within_20pct_weekly_high') else '❌'}
- Below Resistance: {'✅' if breakout_details.get('below_daily_high') else '❌'}

"""
            
            # Higher Lows Pattern
            higher_lows_details = details['higher_lows']
            markdown_content += f"""**📈 Higher Lows Pattern:**
- 10-day Higher Lows: {'✅' if higher_lows_details.get('higher_low_10d') else '❌'}
- 20-day Higher Lows: {'✅' if higher_lows_details.get('higher_low_20d') else '❌'}
- 30-day Higher Lows: {'✅' if higher_lows_details.get('higher_low_30d') else '❌'}

"""
            
            # Volume Analysis
            volume_details = details['volume']
            contracting_signals = volume_details.get('contracting_signals', 0)
            total_signals = volume_details.get('total_signals', 6)
            markdown_content += f"""**📊 Volume Contraction ({contracting_signals}/{total_signals}):**
- 5-day: {'✅' if volume_details.get('volume_contracting_5d') else '❌'}
- 10-day: {'✅' if volume_details.get('volume_contracting_10d') else '❌'}
- 15-day: {'✅' if volume_details.get('volume_contracting_15d') else '❌'}
- 20-day: {'✅' if volume_details.get('volume_contracting_20d') else '❌'}
- 25-day: {'✅' if volume_details.get('volume_contracting_25d') else '❌'}
- 30-day: {'✅' if volume_details.get('volume_contracting_30d') else '❌'}

---

"""
        else:
            markdown_content += "---\n\n"
    
    # Add methodology section
    markdown_content += f"""## 📚 Methodology

### Mark Minervini's 10-Point Trend Template
1. **Price above 50-day MA**: Stock price must be above 50-day moving average
2. **Price above 150-day MA**: Stock price must be above 150-day moving average
3. **Price above 200-day MA**: Stock price must be above 200-day moving average
4. **MA Alignment**: 50-day > 150-day > 200-day moving averages
5. **MA200 Rising**: 200-day moving average trending upward
6. **Within 25% of High**: Current price within 25% of 52-week high
7. **Above 30% of Low**: Current price at least 30% above 52-week low
8. **Relative Strength**: Positive 3-month price performance

### VCP Pattern Components
- **Breakout Readiness** (6 points): Proximity to breakout levels
- **Higher Lows Pattern** (3 points): Constructive pullback pattern
- **Volume Contraction** (6 points): Decreasing volume during consolidation

### Stage 2 Identification
Stocks meeting all 6 core Stage 2 criteria are marked as **STAGE 2** candidates:
- All moving averages properly aligned
- 200-day MA rising
- Near 52-week highs
- Strong relative performance

---

*Generated by Enhanced VCP Pattern Detector*
*Based on Mark Minervini's "Think & Trade Like a Champion" methodology*
"""
    
    # Write to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"\n💾 VCP分析报告已保存: {filepath}")
    except Exception as e:
        print(f"❌ 保存VCP报告失败: {e}")

def main():
    """Main function for enhanced VCP scanning"""
    print("🎯 增强VCP (Volatility Contraction Pattern) 检测器")
    print("基于Mark Minervini趋势模板 + 高级技术分析")
    print("=" * 70)
    
    # Scan options
    print("请选择扫描范围:")
    print("1. 测试模式 (前50个股票)")
    print("2. 完整扫描 (所有股票)")
    print("3. 自定义股票列表")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        symbols = STOCK_SYMBOLS[:50]
        print(f"🧪 测试模式: 扫描前50个股票")
    elif choice == "2":
        symbols = STOCK_SYMBOLS
        print(f"🔍 完整扫描: 扫描{len(STOCK_SYMBOLS)}个股票")
    elif choice == "3":
        custom_symbols = input("请输入股票代码 (用逗号分隔): ").strip().upper().split(',')
        symbols = [s.strip() for s in custom_symbols if s.strip()]
        print(f"📝 自定义扫描: {len(symbols)}个股票")
    else:
        print("❌ 无效选择，使用测试模式")
        symbols = STOCK_SYMBOLS[:50]
    
    # Set minimum score
    min_score_input = input("请输入最低评分 (默认10分): ").strip()
    min_score = int(min_score_input) if min_score_input.isdigit() else 10
    
    # Start scanning
    results = scan_enhanced_vcp_patterns(symbols, min_score)
    
    # Display results
    if results:
        print(f"\n🏆 发现的增强VCP模式:")
        print("=" * 90)
        for i, result in enumerate(results[:10]):
            category = result['vcp_category']
            symbol = result['symbol']
            score = result['total_score']
            price = result['current_price']
            change = result['price_change_pct']
            market_cap_b = result['market_cap_billions']
            print(f"{i+1:2d}. {category} {symbol:>6} | {score:2d}分 | ${price:>8.2f} ({change:+6.1f}%) | ${market_cap_b:.1f}B")
    else:
        print("❌ 未发现符合条件的增强VCP模式")

if __name__ == "__main__":
    main()