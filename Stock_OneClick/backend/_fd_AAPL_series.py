"""Dump AAPL point-in-time quarterly/annual series from SEC XBRL (companyfacts) with date-matched YoY.
Every printed number carries tag + form + accession so it is auditable."""
import sys, pandas as pd, numpy as np
sys.path.insert(0, "/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import _fund_data as fd

pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
panel = fd.load()
a = panel[panel.symbol == "AAPL"]
print("concepts:", sorted(a.concept.unique()))

def dump(concept, annual, n=24, label=""):
    s = fd.series(panel, "AAPL", concept, annual=annual)
    if s.empty:
        print(f"\n[{concept} annual={annual}] EMPTY"); return s
    s = s.tail(n).copy()
    s["val_m"] = s.val/1e6
    print(f"\n=== {concept} annual={annual} {label} (n={len(fd.series(panel,'AAPL',concept,annual=annual))}) ===")
    print(s[["end","val_m","tag","form","filed","accn","days"]].to_string(index=False))
    return s

rq = dump("revenue", False, 24, "quarterly revenue $M")
ra = dump("revenue", True, 24, "annual revenue $M")

# date-matched YoY on quarterly revenue
def yoy_table(s, tol=45):
    s = s.sort_values("end").reset_index(drop=True)
    out=[]
    for i,r in s.iterrows():
        tgt = r.end - pd.Timedelta(days=365)
        cand = s[(s.end - tgt).abs() <= pd.Timedelta(days=tol)]
        cand = cand[cand.end < r.end - pd.Timedelta(days=180)]
        if cand.empty: out.append((r.end, r.val/1e6, np.nan, np.nan, r.days, r.accn)); continue
        b = cand.iloc[(cand.end - tgt).abs().argsort()[:1]].iloc[0]
        out.append((r.end, r.val/1e6, b.end, r.val/b.val-1, r.days, r.accn))
    return pd.DataFrame(out, columns=["end","rev_m","base_end","yoy","days","accn"])

full_q = fd.series(panel, "AAPL", "revenue", annual=False)
t = yoy_table(full_q)
print("\n=== date-matched quarterly revenue YoY (last 20) ===")
print(t.tail(20).to_string(index=False, formatters={"yoy": lambda v: f"{v*100:+.1f}%" if pd.notna(v) else "-"}))
