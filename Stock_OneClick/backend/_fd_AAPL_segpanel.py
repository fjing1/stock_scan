"""Build a quarterly revenue panel by product category and by geographic segment,
deriving fiscal Q4 = FY - 9M. All values from XBRL instances (see _fd_AAPL_segments.py)."""
import pandas as pd, numpy as np, json
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
d = pd.read_pickle("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_segments.pkl")
d = d[d.ndims == 1].copy()
d["mem"] = d.member

def panel(axis):
    x = d[d.axis == axis].copy()
    # 3-month facts
    q3 = x[(x.days >= 80) & (x.days <= 100)].drop_duplicates(subset=["end","mem"], keep="first")
    # 12-month and 9-month facts, to derive fiscal Q4
    fy = x[(x.days >= 355) & (x.days <= 375)].drop_duplicates(subset=["end","mem"], keep="first")
    m9 = x[(x.days >= 265) & (x.days <= 285)].drop_duplicates(subset=["end","mem"], keep="first")
    rows = q3[["end","mem","val","accn"]].assign(src="3M")
    derived = []
    for _, f in fy.iterrows():
        # matching 9M ending ~3 months before fiscal year end, same member
        cand = m9[(m9.mem == f.mem) & (m9.end < f.end) & (m9.end > f.end - pd.Timedelta(days=130))]
        if len(cand):
            derived.append({"end": f.end, "mem": f.mem, "val": f.val - cand.iloc[0].val,
                            "accn": f.accn + "|9M:" + cand.iloc[0].accn, "src": "FY-9M"})
    out = pd.concat([rows, pd.DataFrame(derived)], ignore_index=True)
    out = out.drop_duplicates(subset=["end","mem"], keep="first").sort_values(["mem","end"])
    return out

for axis in ("ProductOrServiceAxis", "StatementBusinessSegmentsAxis"):
    p = panel(axis)
    w = p.pivot(index="end", columns="mem", values="val") / 1e6
    w = w.sort_index()
    print(f"\n{'='*140}\n{axis}: quarterly net sales, $M\n{'='*140}")
    print(w.round(0).to_string())
    yoy = w / w.shift(4) - 1
    print(f"\n--- YoY % (4 quarters back; verify index spacing) ---")
    print((yoy*100).round(1).to_string())
    p.to_csv(f"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_seg_{axis}.csv", index=False)
    print("\nsrc/accn provenance (last 8 rows):")
    print(p.sort_values("end").tail(12).to_string(index=False))
