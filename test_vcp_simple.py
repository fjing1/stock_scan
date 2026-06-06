#!/usr/bin/env python3
"""
Simple VCP test script to verify dependencies and basic functionality
"""

import sys
print("Python version:", sys.version)

try:
    import yfinance as yf
    print("✅ yfinance imported successfully")
except ImportError as e:
    print("❌ yfinance import failed:", e)

try:
    import pandas as pd
    print("✅ pandas imported successfully")
except ImportError as e:
    print("❌ pandas import failed:", e)

try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print("❌ numpy import failed:", e)

try:
    import ta
    print("✅ ta imported successfully")
except ImportError as e:
    print("❌ ta import failed:", e)

# Test basic data fetch
try:
    print("\n🔍 Testing data fetch for AAPL...")
    data = yf.download("AAPL", period="1mo", progress=False)
    print(f"✅ Data fetched successfully: {len(data)} rows")
    print(f"Latest close price: ${data['Close'].iloc[-1]:.2f}")
except Exception as e:
    print("❌ Data fetch failed:", e)

print("\n✅ Basic dependency test completed")