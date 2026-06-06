#!/usr/bin/env python3
"""
Test VCP detector with specific stocks that might have VCP patterns
"""

from vcp_pattern_detector import detect_vcp_pattern

# Test with some popular stocks that might have VCP patterns
test_symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 
    "NVDA", "META", "NFLX", "CRM", "ADBE",
    "SHOP", "ROKU", "ZM", "PTON", "SNOW",
    "PLTR", "COIN", "RBLX", "UPST", "AFRM"
]

print("🎯 Testing VCP Pattern Detection on Specific Stocks")
print("=" * 60)

results = []
for symbol in test_symbols:
    print(f"\n🔍 Analyzing {symbol}...")
    try:
        result = detect_vcp_pattern(symbol)
        if result:
            results.append(result)
            category = result['vcp_category']
            score = result['vcp_score']
            status = result['breakout_status']
            price = result['current_price']
            change = result['price_change_pct']
            print(f"✅ {symbol}: {category} | Score: {score} | Status: {status} | ${price} ({change:+.1f}%)")
        else:
            print(f"❌ {symbol}: No VCP pattern detected")
    except Exception as e:
        print(f"❌ {symbol}: Error - {e}")

print(f"\n📊 Summary:")
print(f"   - Tested stocks: {len(test_symbols)}")
print(f"   - VCP patterns found: {len(results)}")

if results:
    print(f"\n🏆 Best VCP Patterns Found:")
    results.sort(key=lambda x: x['vcp_score'], reverse=True)
    for i, result in enumerate(results[:5]):
        symbol = result['symbol']
        category = result['vcp_category']
        score = result['vcp_score']
        status = result['breakout_status']
        print(f"{i+1}. {symbol}: {category} (Score: {score}, Status: {status})")