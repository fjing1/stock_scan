#!/bin/bash
# Health check for the scheduled twice-daily scan driver (open+30min / close-10min).
# Read-only: prints whether the launchd agent is registered, when it last ran each slot,
# VPN status, and any recent errors. Safe to run anytime.
#
#   ./check_scheduled_scan.sh
set -uo pipefail

LABEL="com.stockscan.scheduled-scan"
STORE_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/reports/scheduled_scan"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="${STORE_DIR}/driver.log"

ok(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn(){ printf '  \033[33m!\033[0m %s\n' "$*"; }
bad(){ printf '  \033[31m✗\033[0m %s\n' "$*"; }

echo "=== scheduled-scan health  ($(TZ=America/New_York date '+%Y-%m-%d %H:%M %Z')) ==="

echo "[agent]"
if [ "$(launchctl list 2>/dev/null | grep -c "$LABEL")" -gt 0 ]; then
  status="$(launchctl list | grep "$LABEL" | awk '{print $2}')"
  ok "registered ($LABEL)"
  if [ "$status" = "0" ] || [ "$status" = "-" ]; then ok "last exit status: ${status:-none}"; else warn "last exit status: $status (non-zero)"; fi
else
  bad "NOT registered — load it in YOUR terminal:"
  echo "      launchctl bootstrap gui/\$(id -u) $PLIST"
fi
[ -f "$PLIST" ] && ok "plist present" || bad "plist missing: $PLIST"

echo "[VPN]"
# grep -c (not -q) -- see the note in run_scheduled_scan.sh's vpn_connected(): with
# `set -o pipefail`, grep -q's early exit sends SIGPIPE to scutil and makes the pipeline
# exit non-zero even on a match.
if [ "$(scutil --nc list 2>/dev/null | grep -c '(Connected)')" -gt 0 ]; then
  scutil --nc list 2>/dev/null | grep '(Connected)' | sed 's/^/  /' | while read -r l; do ok "$l"; done
else
  warn "no VPN service currently connected — scheduled fires will self-skip until it reconnects"
fi

echo "[today's slots]"
today="$(TZ=America/New_York date +%F)"
for slot in open close; do
  f="${STORE_DIR}/.last_run_${slot}_${today}"
  if [ -f "$f" ]; then ok "$slot: ran today ($(cat "$f"))"; else warn "$slot: not yet today"; fi
done

echo "[recent log]"
if [ -f "$LOG" ]; then
  tail -n 6 "$LOG" | sed 's/^/  /'
else
  warn "no driver.log yet"
fi
echo "=== done ==="
