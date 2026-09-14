import yfinance as yf, pandas as pd, numpy as np, pickle, datetime as dt
tk = ["ARM","AAPL","NVDA","SMH","QQQ","SPY","QCOM","AVGO","MU","SOXX","TSM"]
df = yf.download(tk, start="2023-09-01", end="2026-09-16", auto_adjust=True, progress=False, group_by="column")
px = df["Close"].copy()
print("last 4 rows:")
print(px.tail(4).to_string())
print("\nindex tz/type:", px.index[-1], type(px.index[-1]))
px.to_pickle("_arm_apple_panel.pkl")
# also raw volume to detect partial bar
vol = df["Volume"].copy(); vol.to_pickle("_arm_apple_vol.pkl")
print("\nARM vol last 6:"); print(vol["ARM"].tail(6).to_string())
