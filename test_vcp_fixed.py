#!/usr/bin/env python3
"""
Test the fixed VCP detector
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vcp_pattern_detector import detect_vcp_pattern

def test_vcp_fixed():
    """Test VCP detector with fixed data requirements"""
    test_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
    
    print("🎯 Testing Fixed VCP Detector")
    print("Daily timeframe, reduced data requirements")
    print("=" * 60)
    
    results = []
    for symbol in test_symbols:
        print(f"\n🔍 Testing {symbol}...")
        try:
            result = detect_vcp_pattern(symbol)
            if result:
                results.append(result)
                print(f"✅ {symbol}: VCP Score = {result['vcp_score']}")
                print(f"   Category: {result['vcp_category']}")
                print(f"   Status: {result['breakout_status']}")
                print(f"   Price: ${result['current_price']}")
                print(f"   Trend: {result['trend_confirmed']}")
                print(f"   Pattern: {result['pattern_valid']}")
                print(f"   Volume: {result['volume_valid']}")
            else:
                print(f"❌ {symbol}: No VCP pattern detected (failed trend confirmation)")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    print(f"\n📊 Summary:")
    print(f"   Tested: {len(test_symbols)} stocks")
    print(f"   Results: {len(results)} with VCP analysis")
    
    if results:
        print(f"\n🏆 Best Results:")
        results.sort(key=lambda x: x['vcp_score'], reverse=True)
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. {result['symbol']}: {result['vcp_score']} points - {result['vcp_category']}")

if __name__ == "__main__":
    test_vcp_fixed()