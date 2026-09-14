"""Probe: can yfinance serve MORE than 60 days of 5-minute bars via chunked
start/end windows walking backwards in time?

This is the cheapest possible answer to the intraday-history problem, so test it
explicitly before touching any vendor that needs a key.

Strategy: request consecutive 55-day windows going backwards from today and
record, for each window, how many bars came back and what date span they cover.
If Yahoo enforces the 60-day cap as "no data older than 60 days" then every
window beyond the first will return 0 rows. If the cap is merely "max 60 days
per request" then we can stitch years together.
"""
import sys
import time
import datetime as dt

import pandas as pd
import yfinance as yf

SYM = "SPY"
INTERVAL = sys.argv[1] if len(sys.argv) > 1 else "5m"
WINDOW_DAYS = 55
N_WINDOWS = 10

today = dt.date.today()
rows = []
print(f"=== yfinance chunk probe: {SYM} interval={INTERVAL} "
      f"window={WINDOW_DAYS}d x {N_WINDOWS} ===")
print(f"today={today}")

for i in range(N_WINDOWS):
    end = today - dt.timedelta(days=i * WINDOW_DAYS)
    start = end - dt.timedelta(days=WINDOW_DAYS)
    t0 = time.time()
    try:
        df = yf.download(
            SYM,
            start=start.isoformat(),
            end=end.isoformat(),
            interval=INTERVAL,
            progress=False,
            auto_adjust=False,
            prepost=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{i}] {start}..{end}  EXC {type(exc).__name__}: {exc}")
        rows.append(dict(i=i, req_start=start, req_end=end, n=0,
                         got_start=None, got_end=None, err=str(exc)[:120]))
        continue
    el = time.time() - t0
    n = 0 if df is None else len(df)
    if n:
        gs, ge = df.index[0], df.index[-1]
        print(f"[{i}] req {start}..{end}  n={n:6d}  got {gs} .. {ge}  ({el:.2f}s)")
        rows.append(dict(i=i, req_start=start, req_end=end, n=n,
                         got_start=str(gs), got_end=str(ge), err=""))
    else:
        print(f"[{i}] req {start}..{end}  n=0  EMPTY  ({el:.2f}s)")
        rows.append(dict(i=i, req_start=start, req_end=end, n=0,
                         got_start=None, got_end=None, err="empty"))
    time.sleep(1.0)

out = pd.DataFrame(rows)
print("\n=== SUMMARY ===")
print(out.to_string(index=False))
tot = int(out["n"].sum())
nonempty = out[out["n"] > 0]
print(f"\ntotal bars stitched: {tot}")
if len(nonempty):
    print(f"oldest bar reached : {nonempty['got_start'].min()}")
    print(f"newest bar reached : {nonempty['got_end'].max()}")
    print(f"windows with data  : {len(nonempty)} / {N_WINDOWS}")
out.to_csv(f"_data_probe_yf_chunk_{INTERVAL}.csv", index=False)
