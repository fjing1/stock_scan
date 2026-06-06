from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import ssl

import pandas as pd
import yfinance as yf

import scan_stocks as scan


CACHE_DIR = Path(os.getenv("STOCK_DASHBOARD_CACHE_DIR", str(Path(__file__).resolve().parent.parent / ".dashboard_cache")))
LOCAL_ENV_FILE = Path(__file__).resolve().parent / "dashboard_env.local"


def _load_local_env_file() -> None:
    if not LOCAL_ENV_FILE.exists():
        return
    for raw_line in LOCAL_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env_file()
PROVIDER_NAME = os.getenv("STOCK_DASHBOARD_DATA_PROVIDER", "yfinance").strip().lower()
REQUEST_DELAY_SECONDS = float(os.getenv("STOCK_DASHBOARD_REQUEST_DELAY_SECONDS", "0.15"))
CACHE_TTL_SECONDS = int(os.getenv("STOCK_DASHBOARD_CACHE_TTL_SECONDS", "240"))
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("STOCK_DASHBOARD_POLYGON_API_KEY")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or os.getenv("STOCK_DASHBOARD_ALPACA_API_KEY")
ALPACA_SECRET_KEY = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
    or os.getenv("STOCK_DASHBOARD_ALPACA_SECRET_KEY")
)
try:
    import certifi

    HTTPS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    HTTPS_CONTEXT = ssl.create_default_context()


def _open_url(request_or_url, timeout: int = 30):
    return urlopen(request_or_url, timeout=timeout, context=HTTPS_CONTEXT)


class DashboardDataProvider:
    def __init__(self, name: str | None = None):
        self.name = (name or PROVIDER_NAME or "yfinance").strip().lower()
        self._memory_cache: dict[tuple[str, str, str], tuple[float, pd.DataFrame | None]] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def label(self) -> str:
        if self.name == "alpaca":
            return "Alpaca IEX"
        if self.name == "polygon":
            return "Polygon"
        return "yfinance-dashboard"

    def download_daily(self, symbol: str, period: str = "1y") -> pd.DataFrame | None:
        return self._cached(symbol, "1d", period, lambda: self._download_daily_uncached(symbol, period))

    def download_4h(self, symbol: str, period: str = "90d") -> pd.DataFrame | None:
        return self._cached(symbol, "4h", period, lambda: self._download_4h_uncached(symbol, period))

    def download_1m(self, symbol: str, period: str = "5d") -> pd.DataFrame | None:
        return self._cached(symbol, "1m", period, lambda: self._download_intraday_uncached(symbol, period, interval="1m"))

    def download_5m(self, symbol: str, period: str = "5d") -> pd.DataFrame | None:
        return self._cached(symbol, "5m", period, lambda: self._download_intraday_uncached(symbol, period, interval="5m"))

    def download_many_daily(self, symbols: list[str], period: str = "1y") -> dict[str, pd.DataFrame | None]:
        if self.name == "alpaca":
            data = self._alpaca_bars_many(symbols, timeframe="1Day", period=period)
            self._store_many_cache(data, interval="1d", period=period)
            return data
        data = {symbol: self.download_daily(symbol, period=period) for symbol in symbols}
        self._store_many_cache(data, interval="1d", period=period)
        return data

    def download_many_4h(self, symbols: list[str], period: str = "90d") -> dict[str, pd.DataFrame | None]:
        if self.name == "alpaca":
            data = self._alpaca_bars_many(symbols, timeframe="4Hour", period=period)
            self._store_many_cache(data, interval="4h", period=period)
            return data
        data = {symbol: self.download_4h(symbol, period=period) for symbol in symbols}
        self._store_many_cache(data, interval="4h", period=period)
        return data

    def _cached(self, symbol: str, interval: str, period: str, fetcher) -> pd.DataFrame | None:
        key = (str(symbol).upper(), interval, period)
        now = time.time()
        cached = self._memory_cache.get(key)
        if cached and now - cached[0] <= CACHE_TTL_SECONDS:
            df = cached[1]
            return None if df is None else df.copy()

        df = fetcher()
        self._memory_cache[key] = (now, None if df is None else df.copy())
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        return df

    def _store_many_cache(self, data: dict[str, pd.DataFrame | None], interval: str, period: str) -> None:
        now = time.time()
        for symbol, df in data.items():
            key = (str(symbol).upper(), interval, period)
            self._memory_cache[key] = (now, None if df is None else df.copy())

    def _download_daily_uncached(self, symbol: str, period: str) -> pd.DataFrame | None:
        if self.name == "alpaca":
            return self._alpaca_bars(symbol, timeframe="1Day", period=period)
        if self.name == "polygon":
            return self._polygon_aggs(symbol, multiplier=1, timespan="day", period=period)
        return self._yf_download(symbol, period=period, interval="1d")

    def _download_4h_uncached(self, symbol: str, period: str) -> pd.DataFrame | None:
        if self.name == "alpaca":
            return self._alpaca_bars(symbol, timeframe="4Hour", period=period)
        if self.name == "polygon":
            return self._polygon_aggs(symbol, multiplier=4, timespan="hour", period=period)
        return self._yf_download(symbol, period=period, interval="4h")

    def _download_intraday_uncached(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        if self.name == "alpaca":
            timeframe = "1Min" if interval == "1m" else "5Min"
            return self._alpaca_bars(symbol, timeframe=timeframe, period=period)
        if self.name == "polygon":
            multiplier = 1 if interval == "1m" else 5
            return self._polygon_aggs(symbol, multiplier=multiplier, timespan="minute", period=period)
        return self._yf_download(symbol, period=period, interval=interval)

    def _yf_download(self, symbol: str, period: str, interval: str) -> pd.DataFrame | None:
        yf_symbol = scan.to_yfinance_symbol(symbol)
        df = yf.download(yf_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
        df = scan.normalize_yf_df(df)
        if df.empty:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    def _polygon_aggs(self, symbol: str, multiplier: int, timespan: str, period: str) -> pd.DataFrame | None:
        if not POLYGON_API_KEY:
            raise RuntimeError("STOCK_DASHBOARD_DATA_PROVIDER=polygon 需要设置 POLYGON_API_KEY。")
        ticker = _polygon_ticker(symbol)
        start = _period_start_date(period)
        end = date.today()
        params = urlencode(
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": "50000",
                "apiKey": POLYGON_API_KEY,
            }
        )
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{start.isoformat()}/{end.isoformat()}?{params}"
        )
        with _open_url(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("status") not in {"OK", "DELAYED"}:
            raise RuntimeError(f"Polygon返回异常：{payload.get('status')} {payload.get('error') or payload.get('message')}")
        results = payload.get("results") or []
        if not results:
            return None
        idx = pd.to_datetime([r["t"] for r in results], unit="ms", utc=True).tz_convert("America/New_York")
        df = pd.DataFrame(
            {
                "Open": [r.get("o") for r in results],
                "High": [r.get("h") for r in results],
                "Low": [r.get("l") for r in results],
                "Close": [r.get("c") for r in results],
                "Volume": [r.get("v") for r in results],
            },
            index=idx,
        )
        return df.dropna()

    def _alpaca_bars(self, symbol: str, timeframe: str, period: str) -> pd.DataFrame | None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("STOCK_DASHBOARD_DATA_PROVIDER=alpaca 需要设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY。")
        ticker = _alpaca_ticker(symbol)
        start = _period_start_date(period)
        end = date.today() + timedelta(days=1)
        params = urlencode(
            {
                "timeframe": timeframe,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": "raw",
                "feed": "iex",
                "limit": "10000",
            }
        )
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars?{params}"
        request = Request(
            url,
            headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
                "Accept": "application/json",
            },
        )
        with _open_url(request, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        bars = payload.get("bars") or []
        if not bars:
            return None
        idx = pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert("America/New_York")
        df = pd.DataFrame(
            {
                "Open": [b.get("o") for b in bars],
                "High": [b.get("h") for b in bars],
                "Low": [b.get("l") for b in bars],
                "Close": [b.get("c") for b in bars],
                "Volume": [b.get("v") for b in bars],
            },
            index=idx,
        )
        return df.dropna()

    def _alpaca_bars_many(self, symbols: list[str], timeframe: str, period: str) -> dict[str, pd.DataFrame | None]:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("STOCK_DASHBOARD_DATA_PROVIDER=alpaca 需要设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY。")
        clean_symbols = []
        seen = set()
        for symbol in symbols:
            ticker = _alpaca_ticker(symbol)
            if ticker and ticker not in seen:
                clean_symbols.append(ticker)
                seen.add(ticker)
        out: dict[str, pd.DataFrame | None] = {symbol: None for symbol in clean_symbols}
        start = _period_start_date(period)
        end = date.today() + timedelta(days=1)
        for chunk_start in range(0, len(clean_symbols), 100):
            chunk = clean_symbols[chunk_start : chunk_start + 100]
            page_token = None
            while True:
                params_data = {
                    "symbols": ",".join(chunk),
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "adjustment": "raw",
                    "feed": "iex",
                    "limit": "10000",
                }
                if page_token:
                    params_data["page_token"] = page_token
                params = urlencode(params_data)
                url = f"https://data.alpaca.markets/v2/stocks/bars?{params}"
                request = Request(
                    url,
                    headers={
                        "APCA-API-KEY-ID": ALPACA_API_KEY,
                        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
                        "Accept": "application/json",
                    },
                )
                with _open_url(request, timeout=30) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                bars_by_symbol = payload.get("bars") or {}
                for ticker, bars in bars_by_symbol.items():
                    df = _bars_to_df(bars)
                    if df is None:
                        continue
                    if out.get(ticker) is None:
                        out[ticker] = df
                    else:
                        out[ticker] = pd.concat([out[ticker], df]).sort_index()
                page_token = payload.get("next_page_token")
                if not page_token:
                    break
        return out


def _period_start_date(period: str) -> date:
    text = str(period or "").strip().lower()
    today = date.today()
    if text.endswith("d"):
        return today - timedelta(days=int(text[:-1] or "1"))
    if text.endswith("mo"):
        return today - timedelta(days=31 * int(text[:-2] or "1"))
    if text.endswith("y"):
        return today - timedelta(days=366 * int(text[:-1] or "1"))
    return today - timedelta(days=366)


def _polygon_ticker(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if ":" in sym:
        sym = sym.split(":", 1)[1]
    # Polygon uses SPY, QQQ, AAPL. Class shares use dash, not dot.
    return sym.replace(".", "-")


def _alpaca_ticker(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    if ":" in sym:
        sym = sym.split(":", 1)[1]
    # Alpaca stock symbols use dot class notation, e.g. BRK.B.
    return sym.replace("-", ".")


def _bars_to_df(bars: list[dict] | None) -> pd.DataFrame | None:
    if not bars:
        return None
    idx = pd.to_datetime([b["t"] for b in bars], utc=True).tz_convert("America/New_York")
    df = pd.DataFrame(
        {
            "Open": [b.get("o") for b in bars],
            "High": [b.get("h") for b in bars],
            "Low": [b.get("l") for b in bars],
            "Close": [b.get("c") for b in bars],
            "Volume": [b.get("v") for b in bars],
        },
        index=idx,
    )
    return df.dropna()
