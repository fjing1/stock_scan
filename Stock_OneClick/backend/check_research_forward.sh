#!/bin/bash
# Health check for the research-forward scheduled driver.
# Read-only: prints whether the launchd agent is registered, when it last ran,
# whether the ledger is growing, and any recent errors. Safe to run anytime.
#
#   ./check_research_forward.sh
set -uo pipefail

LABEL="com.stockscan.research-forward"
STORE_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/reports/research_forward"
PY="/Users/feijing/github.com/stock_scan/vcp_env/bin/python"
BACKEND_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LEDGER="${STORE_DIR}/ledger.csv"
STAMP="${STORE_DIR}/.last_run_et"
LOG="${STORE_DIR}/driver.log"

ok(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m!\033[0m %s\n' "$*"; }
bad(){ printf '  \033[31m✗\033[0m %s\n' "$*"; }

echo "=== research-forward scheduler health  ($(TZ=America/New_York date '+%Y-%m-%d %H:%M %Z')) ==="

# 1) agent registered?
echo "[agent]"
if launchctl list 2>/dev/null | grep -q "$LABEL"; then
  line="$(launchctl list | grep "$LABEL")"
  status="$(echo "$line" | awk '{print $2}')"
  ok "registered ($LABEL)"
  if [ "$status" = "0" ] || [ "$status" = "-" ]; then ok "last exit status: ${status:-none}"; else warn "last exit status: $status (non-zero)"; fi
else
  bad "NOT registered — load it in YOUR terminal:"
  echo "      launchctl bootstrap gui/\$(id -u) $PLIST"
fi
[ -f "$PLIST" ] && ok "plist present" || bad "plist missing: $PLIST"

# 2) last successful run (ET stamp) + freshness vs today
echo "[last run]"
if [ -f "$STAMP" ]; then
  last="$(cat "$STAMP")"; today="$(TZ=America/New_York date +%F)"
  ok "last completed ET date: $last"
  if [ "$last" = "$today" ]; then
    ok "ran today"
  else
    # how many weekdays behind?
    warn "has not completed today ($today) yet — normal before 15:30 ET or on weekends/holidays"
  fi
else
  warn "no stamp yet — driver hasn't completed a full run (or first run pending)"
fi

# 3) ledger growth
echo "[ledger]"
if [ -f "$LEDGER" ]; then
  "$PY" - "$LEDGER" <<'PY'
import sys, pandas as pd, numpy as np
L = pd.read_csv(sys.argv[1])
n = len(L)
d = pd.to_numeric(L.get("days_filled"), errors="coerce").fillna(0).astype(int)
with_fwd = int((d >= 1).sum()); complete = int(L.get("complete").fillna(False).astype(bool).sum()) if "complete" in L else 0
dates = sorted(L["signal_date"].astype(str).unique())
print(f"  \033[32m✓\033[0m {n} signals over {len(dates)} trading dates ({dates[0]}..{dates[-1]})")
print(f"  \033[32m✓\033[0m with >=1 fwd day: {with_fwd} | complete(D14): {complete} | days_filled median {int(d.median()) if n else 0}")
# readiness hint
if with_fwd < 200 or len(dates) < 20:
    print(f"  \033[33m!\033[0m still accruing — need more dates/regimes before report is a valid edge test")
else:
    print(f"  \033[32m✓\033[0m enough breadth to run: {sys.argv[0] if False else 'research_forward.py report'}")
PY
else
  warn "no ledger yet: $LEDGER"
fi

# 4) recent errors in driver log
echo "[recent log]"
if [ -f "$LOG" ]; then
  tail -n 3 "$LOG" | sed 's/^/  /'
  fails="$(grep -c "FAILED" "$LOG" 2>/dev/null | head -1 | tr -d '[:space:]')"
  [ -z "$fails" ] && fails=0
  [ "$fails" -gt 0 ] && warn "$fails FAILED lines in driver.log (investigate)" || ok "no FAILED lines logged"
else
  warn "no driver.log yet"
fi
echo "=== done ==="
