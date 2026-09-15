"""ADVERSARIAL VERIFICATION of: 'market prices AAPL as an AI-complex HEDGE; AI transmission
channel is wrong in SIGN.'  Evidence offered = HC1 OLS on daily log rets, AAPL-SPY on SMH-SPY.

Attacks:
 A. replicate exactly
 B. economic magnitude: dR2 in the 3y/1y windows (claim only reported dR2 for the FULL window,
    where it was ~0 -- suspicious omission)
 C. CONTROL BASKET: is a negative SMH_orth loading SPECIFIC to AAPL, or does every non-semi
    large cap show it?  If KO/PG/JNJ also load -0.2..-0.4, the 'AI hedge' claim reduces to
    'AAPL is not a semiconductor company' -- true but empty.
 D. FREQUENCY: weekly / monthly. A structural 'hedge' relationship must survive aggregation.
 E. LEVELS: a hedge should be negatively related in CUMULATIVE returns, not just daily diffs.
"""
import numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
B = "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/"

def ols(y, X, names):
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b; n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    V = XtXi @ S @ XtXi * n / (n - k)
    t = b / np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return b, t, r2, n

def marg(y, spy, f):
    """orthogonalise f on spy, then y ~ spy + f_orth. return beta, t, dR2 vs spy-only"""
    Xs = np.column_stack([np.ones(len(spy)), spy])
    bb, *_ = np.linalg.lstsq(Xs, f, rcond=None)
    fres = f - Xs @ bb
    _, _, r2s, _ = ols(y, spy.reshape(-1,1), ["SPY"])
    b, t, r2, n = ols(y, np.column_stack([spy, fres]), ["SPY","f_orth"])
    return b[2], t[2], r2 - r2s, n

# ---------- data ----------
CONTROL = ["KO","PG","JNJ","WMT","PEP","MRK","MCD","HD","UNH","COST","XLP","XLV","XLU","JPM","V",
           "MSFT","GOOGL","META","AMZN","AAPL","SPY","SMH","NVDA","QQQ","XLK"]
px = yf.download(CONTROL, start="2018-01-01", end="2026-09-13", auto_adjust=True,
                 progress=False, threads=True)["Close"].dropna(how="all")
px.to_pickle(B+"_fd_AAPL_verify_px.pkl")
r = np.log(px).diff().dropna()
print("rows", len(r), r.index.min().date(), "->", r.index.max().date())

WINDOWS = {"FULL 2018-2026": (None, None),
           "LAST 3Y": ("2023-09-14", None),
           "LAST 1Y": ("2025-09-14", None),
           "2018-2023 (pre)": (None, "2023-12-31")}

# ---------- A + B : replicate AAPL, and report dR2 the claim omitted ----------
print("\n" + "="*110)
print("A/B  AAPL marginal loading on the AI complex, WITH dR2 (economic magnitude)")
print("="*110)
print(f"{'window':<18} {'factor':<8} {'beta':>8} {'t':>8} {'dR2':>10} {'n':>6}")
for lab,(s,e) in WINDOWS.items():
    sub = r.loc[s:e]
    for f in ["SMH","NVDA","XLK","QQQ"]:
        b,t,d,n = marg(sub["AAPL"].values, sub["SPY"].values, sub[f].values)
        print(f"{lab:<18} {f+'_orth':<8} {b:>8.3f} {t:>8.2f} {d:>10.4f} {n:>6}")

# ---------- C : CONTROL BASKET -- is it AAPL-specific? ----------
print("\n" + "="*110)
print("C  SAME regression for every large cap.  If all negative -> nothing AAPL-specific.")
print("="*110)
names = [c for c in r.columns if c not in ("SPY","SMH","NVDA","QQQ","XLK")]
for lab,(s,e) in [("LAST 3Y",("2023-09-14",None)),("LAST 1Y",("2025-09-14",None))]:
    sub = r.loc[s:e]
    rows=[]
    for nmv in names:
        b,t,d,n = marg(sub[nmv].values, sub["SPY"].values, sub["SMH"].values)
        b2,t2,d2,_ = marg(sub[nmv].values, sub["SPY"].values, sub["NVDA"].values)
        rows.append((nmv,b,t,d,b2,t2,d2))
    df=pd.DataFrame(rows,columns=["ticker","SMH_orth","t","dR2","NVDA_orth","t_nv","dR2_nv"]).sort_values("SMH_orth")
    print(f"\n--- {lab}  (n={len(sub)}) ---")
    print(df.round(3).to_string(index=False))
    neg=(df.SMH_orth<0).sum()
    print(f"  >>> {neg}/{len(df)} of these large caps have NEGATIVE SMH_orth loading."
          f"  AAPL rank = {list(df.ticker).index('AAPL')+1} of {len(df)} (1=most negative)")
    print(f"  >>> median SMH_orth across non-AI names = "
          f"{df[~df.ticker.isin(['MSFT','GOOGL','META','AMZN','AAPL'])].SMH_orth.median():.3f}")

# ---------- D : FREQUENCY AGGREGATION ----------
print("\n" + "="*110)
print("D  FREQUENCY TEST. Daily rotation vs structural relationship.")
print("="*110)
lp = np.log(px)
for freq,flab in [("W-FRI","weekly"),("ME","monthly")]:
    rf = lp.resample(freq).last().diff().dropna()
    for lab,(s,e) in [("FULL",(None,None)),("LAST 3Y",("2023-09-14",None))]:
        sub=rf.loc[s:e]
        if len(sub)<20: continue
        b,t,d,n = marg(sub["AAPL"].values, sub["SPY"].values, sub["SMH"].values)
        b2,t2,d2,_ = marg(sub["AAPL"].values, sub["SPY"].values, sub["NVDA"].values)
        print(f"{flab:<8} {lab:<8} n={n:<5} SMH_orth {b:>7.3f} (t {t:>6.2f}, dR2 {d:>7.4f})   "
              f"NVDA_orth {b2:>7.3f} (t {t2:>6.2f}, dR2 {d2:>7.4f})")

# ---------- E : LEVELS.  Does AAPL actually behave like a hedge? ----------
print("\n" + "="*110)
print("E  LEVELS. A hedge goes DOWN when the hedged thing goes UP.")
print("="*110)
for lab,s in [("1Y","2025-09-12"),("3Y","2023-09-14"),("since 2024","2024-01-02")]:
    seg = px.loc[s:]
    tot = (seg.iloc[-1]/seg.iloc[0]-1)*100
    print(f"{lab:<12} AAPL {tot['AAPL']:+7.1f}%   SPY {tot['SPY']:+7.1f}%   SMH {tot['SMH']:+7.1f}%   "
          f"NVDA {tot['NVDA']:+7.1f}%   | AAPL-SPY {tot['AAPL']-tot['SPY']:+6.1f}pp  SMH-SPY {tot['SMH']-tot['SPY']:+6.1f}pp")
# monthly-excess correlation in levels
mo = lp.resample("ME").last().diff().dropna()
for lab,s in [("FULL",None),("since 2024","2024-01-01")]:
    m=mo.loc[s:]
    c=np.corrcoef(m["AAPL"]-m["SPY"], m["SMH"]-m["SPY"])[0,1]
    print(f"  corr(monthly AAPL-SPY, monthly SMH-SPY) {lab}: {c:+.3f}  (n={len(m)})")
