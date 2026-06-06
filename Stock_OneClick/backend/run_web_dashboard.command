#!/bin/bash
cd "$(dirname "$0")"
export STOCK_DASHBOARD_DATA_PROVIDER="${STOCK_DASHBOARD_DATA_PROVIDER:-alpaca}"
exec /usr/bin/env python3 web_dashboard.py
