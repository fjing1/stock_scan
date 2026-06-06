#!/usr/bin/env python3
"""
Debug VCP detector to understand why no patterns are being found
"""

from vcp_pattern_detector import detect_vcp_pattern, get_stock_data_extended, check_trend_confirmation, analyze_contraction_pattern, analyze_volume_characteristics, check_breakout_status

def debug_vcp_analysis(symbol):
    """Debug individual VCP analysis"""
    print(f"\n🔍 Debug Analysis for {symbol}")
    print("=" * 50)
    
    try:
        # Get data
        data = get_stock_data_extended(symbol)
        if data is None:
            print(f"❌ No data available for {symbol}")
            return
        
        print(f"✅ Data loaded: {len(data)} days")
        print(f"📊 Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        print(f"📈 Current price: ${data['Close'].iloc[-1]:.2f}")
        
        # 1. Trend Analysis
        print(f"\n1️⃣ Trend Confirmation:")
        trend_ok, trend_score, trend_details = check_trend_confirmation(data)
        print(f"   Score: {trend_score}/5 | Confirmed: {trend_ok}")
        print(f"   Price above MAs: {trend_details.get('price_above_mas', False)}")
        print(f"   MA150 rising: {trend_details.get('ma150_rising', False)}")
        print(f"   Distance from 52W high: {trend_details.get('distance_from_52w_high', 'N/A')}%")
        
        # 2. Pattern Analysis
        print(f"\n2️⃣ Contraction Pattern:")
        pattern_ok, pattern_score, pattern_details = analyze_contraction_pattern(data)
        print(f"   Score: {pattern_score}/6 | Valid: {pattern_ok}")
        print(f"   Pullbacks: {pattern_details.get('pullbacks', [])}")
        print(f"   Decreasing pullbacks: {pattern_details.get('decreasing_pullbacks', False)}")
        print(f"   Valid pullbacks: {pattern_details.get('valid_pullbacks', 0)}")
        print(f"   Volatility contraction: {pattern_details.get('volatility_contraction', False)}")
        
        # 3. Volume Analysis
        print(f"\n3️⃣ Volume Characteristics:")
        volume_ok, volume_score, volume_details = analyze_volume_characteristics(data)
        print(f"   Score: {volume_score}/5 | Valid: {volume_ok}")
        print(f"   Volume contraction ratio: {volume_details.get('volume_contraction_ratio', 'N/A')}")
        print(f"   Dry up ratio: {volume_details.get('dry_up_ratio', 'N/A')}")
        print(f"   Breakout volume ratio: {volume_details.get('breakout_volume_ratio', 'N/A')}")
        
        # 4. Breakout Status
        print(f"\n4️⃣ Breakout Status:")
        breakout_status, breakout_score, breakout_details = check_breakout_status(data)
        print(f"   Score: {breakout_score}/5 | Status: {breakout_status}")
        print(f"   Current price: ${breakout_details.get('current_price', 'N/A')}")
        print(f"   Resistance level: ${breakout_details.get('resistance_level', 'N/A')}")
        print(f"   Distance to resistance: {breakout_details.get('distance_to_resistance_pct', 'N/A')}%")
        
        # Total Score
        total_score = trend_score + pattern_score + volume_score + breakout_score
        print(f"\n📊 Total VCP Score: {total_score}/20")
        
        if total_score >= 3:
            print(f"✅ This stock has some VCP characteristics!")
        else:
            print(f"❌ This stock doesn't meet VCP criteria")
            
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")

# Test with a few specific stocks
test_symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

print("🎯 VCP Debug Analysis")
print("Daily timeframe, 1-year data period")
print("=" * 60)

for symbol in test_symbols:
    debug_vcp_analysis(symbol)
    print("\n" + "="*60)