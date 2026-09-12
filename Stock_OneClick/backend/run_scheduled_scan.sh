#!/bin/bash
# Scheduled driver for the daily stock scan (scan_stocks.py), twice a trading day:
#   - 30 min after the NYSE open  (10:00 ET)
#   - 10 min before the NYSE close (15:50 ET)
#
# Mirrors the self-gating pattern already used by run_research_forward.sh: launchd fires
# this frequently (every few minutes) and the script itself decides whether to actually run,
# so it's correct regardless of the machine's local timezone/DST. It only runs the scan when
# ALL of these hold:
#   * US weekday (Mon-Fri, evaluated in America/New_York) -- does NOT know about NYSE
#     holidays, so it will still fire (and scan stale/no-change data) on market holidays.
#   * at/near one of the two target ET times (a few-minute tolerance window)
#   * not already run for that (date, slot) -- stamp file dedupes repeat launchd fires
#     inside the same window
#   * the corporate VPN is connected (scutil --nc list shows a "(Connected)" service) --
#     yfinance calls need to egress through it on this machine
#   * the machine is awake (implicit: launchd simply doesn't fire a sleeping Mac, and this
#     script does not attempt to catch up missed fires after a sleep/wake)
#
# Flags:  --force    ignore weekday/time/VPN/stamp gates (manual run)
#         --dry-run  evaluate gates and print the decision, run nothing
set -uo pipefail

BACKEND_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend"
PY="/Users/feijing/github.com/stock_scan/vcp_env/bin/python"
STORE_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/reports/scheduled_scan"
LOG="${STORE_DIR}/driver.log"
LOCK="${STORE_DIR}/.driver.lock"

OPEN_TARGET=$((10*60 + 0))     # 10:00 ET = 30 min after 09:30 ET open, in minutes-since-midnight
CLOSE_TARGET=$((15*60 + 50))   # 15:50 ET = 10 min before 16:00 ET close
TOLERANCE_MIN=6                # launchd polls every 5 min; allow a bit of slack

FORCE=0; DRY=0
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --dry-run) DRY=1 ;;
  esac
done

mkdir -p "$STORE_DIR"
log(){ echo "[$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "$LOG"; }

ET_DATE="$(TZ=America/New_York date '+%Y-%m-%d')"
ET_DOW="$(TZ=America/New_York date '+%u')"   # 1=Mon..7=Sun
ET_H="$(TZ=America/New_York date '+%H')"; ET_H=$((10#$ET_H))
ET_M="$(TZ=America/New_York date '+%M')"; ET_M=$((10#$ET_M))
ET_MINUTES=$(( ET_H * 60 + ET_M ))   # minutes since midnight ET
ET_HHMM=$(( ET_H * 100 + ET_M ))     # for human-readable logging only

vpn_connected() {
  # NOTE: don't use `grep -q` here -- with `set -o pipefail` above, grep -q's early exit
  # once it finds a match sends SIGPIPE back to scutil, which makes the PIPELINE's exit
  # status non-zero (141) even though grep DID match. Confirmed empirically: this made
  # the check always report "not connected" regardless of actual VPN state. grep -c reads
  # the full input so the pipeline exits cleanly.
  [ "$(scutil --nc list 2>/dev/null | grep -c '(Connected)')" -gt 0 ]
}

# which slot (if any) are we within tolerance of?
slot=""
for pair in "open:$OPEN_TARGET" "close:$CLOSE_TARGET"; do
  name="${pair%%:*}"; target="${pair##*:}"
  diff=$(( ET_MINUTES - target ))
  [ "$diff" -lt 0 ] && diff=$(( -diff ))
  if [ "$diff" -le "$TOLERANCE_MIN" ]; then slot="$name"; fi
done

STAMP="${STORE_DIR}/.last_run_${slot:-none}_${ET_DATE}"

decision="RUN"; reason="weekday $ET_DOW, ET $ET_HHMM, slot=${slot:-none}"
if [ "$FORCE" -ne 1 ]; then
  if [ "$ET_DOW" -gt 5 ]; then decision="SKIP"; reason="weekend (ET dow $ET_DOW)"; fi
  if [ "$decision" = "RUN" ] && [ -z "$slot" ]; then decision="SKIP"; reason="not near a target time (ET $ET_HHMM)"; fi
  if [ "$decision" = "RUN" ] && [ -f "$STAMP" ]; then decision="SKIP"; reason="already ran slot=$slot for ET date $ET_DATE"; fi
  if [ "$decision" = "RUN" ] && ! vpn_connected; then decision="SKIP"; reason="VPN not connected (scutil --nc list has no '(Connected)' entry)"; fi
fi

if [ "$DRY" -eq 1 ]; then log "DRY-RUN -> would $decision ($reason)"; exit 0; fi
if [ "$decision" = "SKIP" ]; then log "skip: $reason"; exit 0; fi

if ! mkdir "$LOCK" 2>/dev/null; then log "another scan driver run is in progress (lock held) — exiting"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

log "START scheduled scan ($reason)"
cd "$BACKEND_DIR" || { log "cannot cd to backend"; exit 1; }
if STOCK_ONECLICK_NO_OPEN=1 "$PY" scan_stocks.py >>"$LOG" 2>&1; then
  log "    scan ok"
  echo "$ET_DATE" > "$STAMP"
else
  log "    scan FAILED (see log) — will retry next scheduled slot"; exit 1
fi

log "DONE slot=$slot for ET date $ET_DATE"
tail -n 3000 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG"
exit 0
