# Stock OneClick Dashboard Data Source

The realtime dashboard uses a separate data-provider layer from the daily close scanner.

## Default prototype

By default it uses:

```bash
STOCK_DASHBOARD_DATA_PROVIDER=yfinance
```

This is useful for testing the dashboard UI, but it is still the same upstream provider family as the daily scanner. It reduces code coupling, not IP/provider-level rate-limit risk.

## Preferred isolated setup

Use Polygon for the 5-minute dashboard and keep the daily close scanner unchanged:

```bash
export STOCK_DASHBOARD_DATA_PROVIDER=polygon
export POLYGON_API_KEY="your_polygon_key"
open /Users/ben/Desktop/Stock_OneClick/backend/run_dashboard.command
```

The daily close scanner still runs from:

```bash
/Users/ben/Desktop/Stock_OneClick/backend/scan_stocks.py
```

## Throttle controls

```bash
export STOCK_DASHBOARD_REFRESH_SECONDS=300
export STOCK_DASHBOARD_REQUEST_DELAY_SECONDS=0.15
export STOCK_DASHBOARD_CACHE_TTL_SECONDS=240
```

If the dashboard provider starts rate-limiting, increase request delay or reduce the scan pool with:

```bash
export STOCK_DASHBOARD_LIMIT=50
```
