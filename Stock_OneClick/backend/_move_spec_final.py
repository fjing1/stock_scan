"""
_move_spec_final.py -- last numbers for the spec: display quantiles, grid-representation error,
reliability table, refusal-cell quantification, and the honest test of the up/down split.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
SHRINK = 40.0
COLS = {"index": [0, 1, 2, 4, 6], "single": [0, 1, 2, 3, 4, 5, 6]}
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)


def main():
    art = pd.read_pickle("_move_spec_ship2.pkl")
    print("=== DISPLAY CONSTANTS: z-table quantiles (full-sample fit) ===")
    QS = [1, 5, 10, 16, 25, 50, 75, 84, 90, 95, 99]
    for g in ("index", "single"):
        for hz in HORIZONS:
            z = art[(g, "MAIN", hz)]["z"]
            q = np.percentile(z, QS)
            tsd = z[(z > np.percentile(z, 0.5)) & (z < np.percentile(z, 99.5))].std()
            iqs = (q[6] - q[4]) / 1.349
            print(f"{g:<7}h={hz:<3} iqs={iqs:.4f} sd_trim0.5%={tsd:.4f} "
                  f"q10={q[2]:+.4f} q16={q[3]:+.4f} q50={q[5]:+.4f} q84={q[7]:+.4f} "
                  f"q90={q[8]:+.4f}  (q84-q16)/2={(q[7]-q[3])/2:.4f}")

    print("\n=== QUANTILE-GRID REPRESENTATION ERROR (1001-pt grid vs full sorted array) ===")
    rng = np.random.default_rng(1)
    for g in ("index", "single"):
        for hz in HORIZONS:
            z = np.asarray(art[(g, "MAIN", hz)]["z"], dtype=np.float64)
            for npts in (201, 1001, 4001):
                pr = np.linspace(0.0, 1.0, npts)
                grid = np.quantile(z, pr)
                xs = rng.uniform(-4, 4, 20000)
                exact = np.searchsorted(z, xs, side="right") / len(z)
                approx = np.interp(xs, grid, pr)
                if npts == 1001:
                    print(f"{g:<7}h={hz:<3} npts={npts} max|dF|={np.abs(exact-approx).max():.5f} "
                          f"mean|dF|={np.abs(exact-approx).mean():.6f}")
    log("display done")


if __name__ == "__main__":
    main()
