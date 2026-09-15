"""(a) Upper bound on the FX contribution to the reported acceleration, using actual average
       spot rates over Apple's own fiscal quarters weighted by its disclosed segment revenue.
       Assumption stated: 100% local-currency invoicing outside the Americas -> UPPER bound.
   (b) R&D step-up and operating leverage from XBRL."""
import sys, numpy as np, pandas as pd, yfinance as yf, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/feijing/github.com/stock_scan/Stock_OneClick/backend")
import _fund_data as fd
pd.set_option("display.width",220)

FX = {"EUR":"EURUSD=X","CNY":"CNYUSD=X","JPY":"JPYUSD=X","GBP":"GBPUSD=X","KRW":"KRWUSD=X","INR":"INRUSD=X","AUD":"AUDUSD=X","CAD":"CADUSD=X"}
px = yf.download(list(FX.values()), start="2024-01-01", end="2026-07-01", progress=False, auto_adjust=True)["Close"]
QTRS = {"Q3FY26":("2026-03-29","2026-06-27"), "Q3FY25":("2025-03-30","2025-06-28"),
        "Q2FY26":("2025-12-28","2026-03-28"), "Q2FY25":("2024-12-29","2025-03-29"),
        "Q1FY26":("2025-09-28","2025-12-27"), "Q1FY25":("2024-09-29","2024-12-28")}
avg = {q: px.loc[a:b].mean() for q,(a,b) in QTRS.items()}
A = pd.DataFrame(avg).T
print("=== average spot (USD per unit of local ccy), Apple fiscal quarters ===")
print(A.round(5).to_string())
print("\n=== YoY change in average rate (positive = local ccy stronger = USD revenue tailwind) ===")
for cur,tic in FX.items():
    for q,qb in [("Q3FY26","Q3FY25"),("Q2FY26","Q2FY25"),("Q1FY26","Q1FY25")]:
        pass
ch = pd.DataFrame({q: A.loc[q]/A.loc[qb]-1 for q,qb in [("Q3FY26","Q3FY25"),("Q2FY26","Q2FY25"),("Q1FY26","Q1FY25")]})
ch.index = [k for k,v in FX.items()]
print((ch*100).round(1).to_string())

# segment weights from Apple's own disclosure
geo = pd.read_csv("/Users/feijing/github.com/stock_scan/Stock_OneClick/backend/_fd_AAPL_geo.csv", parse_dates=["end"])
w = (geo.pivot(index="end",columns="mem",values="val")/1e6).sort_index()
MAP = {  # crude currency basket per segment; Americas treated as USD -> conservative (lowers the bound)
 "EuropeSegmentMember": {"EUR":0.55,"GBP":0.20,"INR":0.15,"USD":0.10},
 "GreaterChinaSegmentMember": {"CNY":1.00},
 "JapanSegmentMember": {"JPY":1.00},
 "RestOfAsiaPacificSegmentMember": {"KRW":0.25,"AUD":0.30,"INR":0.25,"USD":0.20},
 "AmericasSegmentMember": {"USD":0.88,"CAD":0.12}}
print("\n=== UPPER-BOUND FX contribution to reported YoY total revenue growth ===")
for q,(qe,qb) in {"Q3FY26":("2026-06-27","2025-06-28"),"Q2FY26":("2026-03-28","2025-03-29"),
                  "Q1FY26":("2025-12-27","2024-12-28")}.items():
    base_tot = w.loc[qb].sum(); contrib = 0.0
    for seg, basket in MAP.items():
        segw = w.loc[qb, seg]/base_tot
        for cur, share in basket.items():
            if cur == "USD": continue
            contrib += segw*share*ch.loc[cur, q]
    rep = w.loc[qe].sum()/base_tot - 1
    print(f"  {q}: reported total growth {rep:+.1%}   FX upper bound {contrib:+.2%}   "
          f"=> growth ex-FX at least {rep-contrib:+.1%}   ({contrib/rep*100:.0f}% of the growth at most)")

panel = fd.load()
print(f"\n{'='*100}\nR&D and operating leverage (XBRL quarterly, $M)\n{'='*100}")
def q(c):
    s = fd.series(panel,"AAPL",c,annual=False); return s.set_index("end").val/1e6
rnd, rev, oi, gp = q("rnd"), q("revenue"), q("operating_income"), q("gross_profit")
t = pd.DataFrame({"rev":rev,"rnd":rnd,"gp":gp,"oi":oi}).dropna().tail(13)
t["rnd_pct"] = t.rnd/t.rev*100; t["gm_pct"]=t.gp/t.rev*100; t["om_pct"]=t.oi/t.rev*100
t["rnd_yoy"] = t.rnd.pct_change(4)*100; t["rev_yoy"]=t.rev.pct_change(4)*100
print(t.round(1).to_string())
