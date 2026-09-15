"""BETA STRUCTURE: does the market trade AAPL as a derivative of the AI complex, of its sector,
of China, or of the market? OLS with HC1 robust t-stats (no statsmodels in this env).

Univariate first, then a joint regression where every non-market factor is ORTHOGONALIZED to SPY
so its coefficient measures marginal explanatory power rather than repackaged market beta."""
import numpy as np, pandas as pd, yfinance as yf, sys, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

TICKERS = ["AAPL","SPY","QQQ","XLK","SMH","NVDA","MSFT","GOOGL","AVGO","XLY","MCHI","FXI","TLT","UUP","TSM"]
px = yf.download(TICKERS, start="2018-01-01", end="2026-09-13", auto_adjust=True,
                 progress=False, threads=True)["Close"]
px = px.dropna(how="all")
print("price rows", len(px), px.index.min().date(), px.index.max().date())
print("last close:\n", px.tail(1).T.round(2).to_string())
r = np.log(px).diff().dropna(how="all")
r.to_pickle("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_rets.pkl")

def ols(y, X, names):
    """returns coef, robust t, R2, n"""
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    # HC1
    S = (X * e[:, None]).T @ (X * e[:, None])
    V = XtXi @ S @ XtXi * n / (n - k)
    se = np.sqrt(np.diag(V))
    t = b / se
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, t, r2, n, ["const"] + names

def run(window_label, sub):
    sub = sub.dropna()
    y = sub["AAPL"].values
    print(f"\n{'='*118}\n{window_label}   n={len(sub)}  ({sub.index.min().date()} -> {sub.index.max().date()})\n{'='*118}")
    print(f"{'factor':<10} {'univar beta':>12} {'t':>8} {'R2':>8}   |  {'marg beta (orth to SPY)':>24} {'t':>8} {'dR2 vs SPY-only':>16}")
    b0, t0, r2_spy, n0, _ = ols(y, sub[["SPY"]].values, ["SPY"])
    for f in [c for c in sub.columns if c not in ("AAPL",)]:
        b, t, r2, n, nm = ols(y, sub[[f]].values, [f])
        if f == "SPY":
            print(f"{f:<10} {b[1]:>12.3f} {t[1]:>8.2f} {r2:>8.3f}   |  {'(is the market)':>24}")
            continue
        # orthogonalize f on SPY
        Xs = np.column_stack([np.ones(len(sub)), sub["SPY"].values])
        bb, *_ = np.linalg.lstsq(Xs, sub[f].values, rcond=None)
        fres = sub[f].values - Xs @ bb
        b2, t2, r22, _, _ = ols(y, np.column_stack([sub["SPY"].values, fres]), ["SPY", f+"_orth"])
        print(f"{f:<10} {b[1]:>12.3f} {t[1]:>8.2f} {r2:>8.3f}   |  {b2[2]:>24.3f} {t2[2]:>8.2f} {r22-r2_spy:>16.4f}")
    # joint: SPY + XLK_orth + NVDA_orth + MCHI_orth
    cols = ["XLK","NVDA","MCHI","TLT"]
    Xs = np.column_stack([np.ones(len(sub)), sub["SPY"].values])
    orth = []
    for c in cols:
        bb, *_ = np.linalg.lstsq(Xs, sub[c].values, rcond=None)
        orth.append(sub[c].values - Xs @ bb)
    Xj = np.column_stack([sub["SPY"].values] + orth)
    bj, tj, r2j, _, nmj = ols(y, Xj, ["SPY"] + [c+"_orth" for c in cols])
    print(f"\n  JOINT  R2={r2j:.3f}   (SPY-only R2={r2_spy:.3f})")
    for nm, bv, tv in zip(nmj, bj, tj):
        print(f"    {nm:<14} beta {bv:>8.3f}   t {tv:>7.2f}")

cols = ["AAPL","SPY","QQQ","XLK","SMH","NVDA","MSFT","GOOGL","AVGO","XLY","MCHI","FXI","TLT","UUP","TSM"]
sub = r[cols]
run("FULL 2018-01 .. 2026-09", sub)
run("LAST 3 YEARS", sub.loc["2023-09-14":])
run("LAST 1 YEAR", sub.loc["2025-09-14":])
run("LAST 3 MONTHS (the recent move)", sub.loc["2026-06-14":])
