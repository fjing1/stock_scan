"""Geographic segment quarterly net sales. FY2026 10-Qs tag segment revenue with TWO axes
(ConsolidationItemsAxis=OperatingSegmentsMember + StatementBusinessSegmentsAxis), so a
one-dimension filter silently truncates the series at 2025-06-28."""
import pandas as pd, numpy as np, json
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 300)
d = pd.read_pickle("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_segments.pkl")
d["dd"] = d.dims.apply(json.loads)
seg = d[d.dd.apply(lambda x: "StatementBusinessSegmentsAxis" in x)].copy()
seg["mem"] = seg.dd.apply(lambda x: x["StatementBusinessSegmentsAxis"])
seg["other"] = seg.dd.apply(lambda x: {k:v for k,v in x.items() if k!="StatementBusinessSegmentsAxis"})
print("secondary axes on segment facts:", seg.other.astype(str).value_counts().to_dict())
q3 = seg[(seg.days>=80)&(seg.days<=100)].drop_duplicates(subset=["end","mem"], keep="first")
fy = seg[(seg.days>=355)&(seg.days<=375)].drop_duplicates(subset=["end","mem"], keep="first")
m9 = seg[(seg.days>=265)&(seg.days<=285)].drop_duplicates(subset=["end","mem"], keep="first")
rows = q3[["end","mem","val","accn"]].assign(src="3M").to_dict("records")
for _, f in fy.iterrows():
    c = m9[(m9.mem==f.mem)&(m9.end<f.end)&(m9.end>f.end-pd.Timedelta(days=130))]
    if len(c):
        rows.append({"end": f.end, "mem": f.mem, "val": f.val-c.iloc[0].val,
                     "accn": f.accn+"|"+c.iloc[0].accn, "src": "FY-9M"})
p = pd.DataFrame(rows).drop_duplicates(subset=["end","mem"], keep="first")
w = (p.pivot(index="end", columns="mem", values="val")/1e6).sort_index()
w["TOTAL"] = w.sum(axis=1)
print("\n=== geographic segment quarterly net sales $M ===")
print(w.round(0).to_string())
print("\n=== YoY % ===")
print(((w/w.shift(4)-1)*100).round(1).to_string())
print("\n=== share of total ===")
print((w.div(w.TOTAL, axis=0)*100).round(1).to_string())
p.to_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_geo.csv", index=False)

# incremental attribution for the three accelerating quarters, product categories
pc = pd.read_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_seg_ProductOrServiceAxis.csv", parse_dates=["end"])
wp = (pc.pivot(index="end", columns="mem", values="val")/1e6).sort_index()
cats = ["IPhoneMember","MacMember","IPadMember","WearablesHomeandAccessoriesMember","ServiceMember"]
print("\n=== incremental revenue attribution (YoY $M and % of total increment) ===")
for qend in [pd.Timestamp("2025-12-27"), pd.Timestamp("2026-03-28"), pd.Timestamp("2026-06-27")]:
    i = wp.index.get_loc(qend); base = wp.index[i-4]
    tot_inc = wp.loc[qend, cats].sum() - wp.loc[base, cats].sum()
    print(f"\n{qend.date()} vs {base.date()}: total increment {tot_inc:+,.0f}M")
    for c in cats:
        inc = wp.loc[qend, c] - wp.loc[base, c]
        print(f"   {c:<36} {wp.loc[base,c]:>9,.0f} -> {wp.loc[qend,c]:>9,.0f}  {inc:>+9,.0f}M  {inc/tot_inc*100:>+6.1f}% of increment  YoY {wp.loc[qend,c]/wp.loc[base,c]-1:>+6.1%}")

# 2-year stacked CAGR to strip base effects
print("\n=== 2-year stacked CAGR (strips easy-base effect) ===")
tot = wp[cats].sum(axis=1)
for qend in wp.index[-8:]:
    i = wp.index.get_loc(qend)
    if i < 8: continue
    b8 = wp.index[i-8]
    print(f"  {qend.date()} vs {b8.date()}: total {tot.loc[b8]:>9,.0f} -> {tot.loc[qend]:>9,.0f}  2yr CAGR {(tot.loc[qend]/tot.loc[b8])**0.5-1:>+6.1%}"
          f"   iPhone 2yr CAGR {(wp.loc[qend,'IPhoneMember']/wp.loc[b8,'IPhoneMember'])**0.5-1:>+6.1%}"
          f"   Services 2yr CAGR {(wp.loc[qend,'ServiceMember']/wp.loc[b8,'ServiceMember'])**0.5-1:>+6.1%}")
