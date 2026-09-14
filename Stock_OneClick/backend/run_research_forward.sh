#!/bin/bash
# Daily driver for the isolated research-pool forward-tracking harness.
#
# Runs ONCE per US trading day at the 30-min-before-close CUTOVER (>= 15:30 ET) to
# (1) take a fresh isolated research scan attributed to that trading day and
# (2) ingest+fill forward returns into reports/research_forward/ledger.csv. Never
# touches the live book / lifecycle (research_scan.py is read-only; research_forward
# writes only under reports/research_forward/).
#
# NOTE: at 15:30 ET the daily bar is not final — the D0 signal is captured on a
# near-complete (provisional) bar so it lands on THAT trading day (your requested
# same-day cutover). Forward returns are still computed from FINALIZED daily closes
# on later runs, so fwd_d1..d14 stay correct; only the D0 entry snapshot is as-of
# 15:30. If a day's 15:30 fire is missed, a later fire the same evening still runs
# (post-close fallback) so the trading day isn't lost.
#
# Self-gating so it is correct no matter the machine's local timezone / DST:
#   * only on US weekdays (Mon-Fri, evaluated in America/New_York)
#   * only at/after 15:30 ET (the cutover)
#   * at most once per ET trading date (stamp file)
#   * single-instance lock (mkdir is atomic)
#
# Flags:  --force   ignore weekday/time/stamp gates (manual run)
#         --dry-run  evaluate gates and print the decision, run nothing
set -uo pipefail

BACKEND_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/backend"
PY="/Users/feijing/github.com/stock_scan/vcp_env/bin/python"
STORE_DIR="/Users/feijing/github.com/stock_scan/Stock_OneClick/reports/research_forward"
SCAN_CSV="${STORE_DIR}/latest_scan.csv"
LOG="${STORE_DIR}/driver.log"
STAMP="${STORE_DIR}/.last_run_et"
LOCK="${STORE_DIR}/.driver.lock"

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
ET_HHMM="$(TZ=America/New_York date '+%H%M')"; ET_HHMM=$((10#$ET_HHMM))   # e.g. 1530

CUTOVER=1530   # 30 min before the 16:00 ET close
decision="RUN"; reason="weekday $ET_DOW, ET $ET_HHMM, date $ET_DATE"
if [ "$FORCE" -ne 1 ]; then
  if [ "$ET_DOW" -gt 5 ]; then decision="SKIP"; reason="weekend (ET dow $ET_DOW)"; fi
  if [ "$decision" = "RUN" ] && [ "$ET_HHMM" -lt "$CUTOVER" ]; then decision="SKIP"; reason="before cutover (ET $ET_HHMM < $CUTOVER)"; fi
  if [ "$decision" = "RUN" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP" 2>/dev/null)" = "$ET_DATE" ]; then
    decision="SKIP"; reason="already ran for ET date $ET_DATE"
  fi
fi

if [ "$DRY" -eq 1 ]; then log "DRY-RUN -> would $decision ($reason)"; exit 0; fi
if [ "$decision" = "SKIP" ]; then log "skip: $reason"; exit 0; fi

# single-instance lock
if ! mkdir "$LOCK" 2>/dev/null; then log "another driver run is in progress (lock held) — exiting"; exit 0; fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

log "START research forward driver ($reason)"
cd "$BACKEND_DIR" || { log "cannot cd to backend"; exit 1; }
export STOCK_ONECLICK_DOWNLOAD_WORKERS="${STOCK_ONECLICK_DOWNLOAD_WORKERS:-6}"

log "1/2 isolated research scan -> $SCAN_CSV"
if "$PY" research_scan.py --out "$SCAN_CSV" >>"$LOG" 2>&1; then
  log "    scan ok"
else
  log "    scan FAILED (see log) — aborting this run, will retry next schedule"; exit 1
fi

log "2/2 ingest + fill forward returns"
if "$PY" research_forward.py update --from-csv "$SCAN_CSV" >>"$LOG" 2>&1; then
  log "    update ok"
else
  log "    update FAILED (see log)"; exit 1
fi

echo "$ET_DATE" > "$STAMP"
log "DONE for ET date $ET_DATE"
# keep the log from growing unbounded
tail -n 2000 "$LOG" > "${LOG}.tmp" 2>/dev/null && mv "${LOG}.tmp" "$LOG"
exit 0
