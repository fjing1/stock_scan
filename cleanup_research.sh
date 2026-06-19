#!/usr/bin/env bash
# cleanup_research.sh — remove intermediate / regenerable artifacts from the
# strategy-research sessions, while preserving the deliverables and the repro harness.
#
# Safe & idempotent: only targets known session artifacts by explicit path. Re-runnable.
# Tracked files it removes will show as deletions in `git status` — commit them afterward.
#
# REMOVES:
#   • backtest panel caches (Stock_OneClick/reports/exit_cache/*.pkl, 250MB+, regenerable)
#   • one-off scratch research scripts (superseded; findings live in reports/ + RESEARCH.md)
#   • __pycache__ dirs under backend/ and tradingview_scripts/
#   • this session's /tmp logs
# KEEPS (deliverables + cited repro):
#   • integrated code: Stock_OneClick/backend/scan_stocks.py, xunlong.py, *.pine
#   • reports: Stock_OneClick/reports/*.md and *.csv, tradingview_scripts/RESEARCH.md
#   • repro harness cited by the docs: _exit_build.py, _exit_backtest.py,
#     _swing_research.py, _swing_basket.py   (regenerate the caches with these)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BK="$REPO/Stock_OneClick/backend"
removed=0

say() { echo "  rm  $1"; removed=$((removed+1)); }

echo "== cleanup_research.sh =="

# 1) regenerable backtest caches
for f in "$REPO/Stock_OneClick/reports/exit_cache/panel.pkl" \
         "$REPO/Stock_OneClick/reports/exit_cache/panel_full.pkl"; do
  [ -e "$f" ] && { rm -f "$f"; say "${f#$REPO/}"; }
done
rmdir "$REPO/Stock_OneClick/reports/exit_cache" 2>/dev/null && echo "  rmdir Stock_OneClick/reports/exit_cache (empty)"

# 2) one-off scratch research scripts (KEEP the 4 cited-repro scripts; remove the rest)
for s in _aapl_strategy.py _aapl_refine.py _aapl_cashonly.py \
         _exit_vs_spy.py _score_alpha.py _regime_now.py \
         _swing_mr.py _swing_regime.py; do
  [ -e "$BK/$s" ] && { rm -f "$BK/$s"; say "Stock_OneClick/backend/$s"; }
done

# 3) __pycache__ under the research dirs
for d in "$BK" "$REPO/tradingview_scripts"; do
  find "$d" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
done
echo "  rm  __pycache__ (backend, tradingview_scripts)"

# 4) this session's /tmp logs
for t in /tmp/scan_combo_check.log /tmp/rebuild_full.log /tmp/exit_bt_10.txt; do
  [ -e "$t" ] && { rm -f "$t"; say "$t"; }
done

echo "== done: removed $removed item(s). Kept repro: _exit_build.py, _exit_backtest.py, _swing_research.py, _swing_basket.py + all reports."
echo "   Rebuild caches anytime:  cd Stock_OneClick/backend && EXIT_BT_UNIVERSE=full ../../vcp_env/bin/python _exit_build.py"
