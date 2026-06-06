#!/bin/bash
cd "$(dirname "$0")"
if [ -f "./dashboard_env.local" ]; then
  source "./dashboard_env.local"
fi
export STOCK_DASHBOARD_DATA_PROVIDER="${STOCK_DASHBOARD_DATA_PROVIDER:-alpaca}"
exec /usr/bin/env python3 intraday_dashboard_app.py
