"""Map which option-IV-related hosts are reachable through this machine's egress proxy."""
import concurrent.futures as cf, requests, time
H={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
URLS=[
 # CBOE
 ("cboe cdn opt chain","https://cdn.cboe.com/api/global/delayed_quotes/options/AAPL.json"),
 ("cboe www","https://www.cboe.com/"),
 ("cboe datashop","https://datashop.cboe.com/"),
 ("cboe vix hist csv","https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"),
 # dolt / public datasets
 ("dolthub api","https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master?q=show+tables"),
 ("dolthub www","https://www.dolthub.com/"),
 ("kaggle","https://www.kaggle.com/"),
 ("huggingface api ds","https://huggingface.co/api/datasets?search=option%20implied%20volatility&limit=5"),
 ("github search","https://api.github.com/search/repositories?q=option+chain+implied+volatility+dataset&per_page=3"),
 ("zenodo","https://zenodo.org/api/records?q=implied+volatility&size=3"),
 ("figshare","https://api.figshare.com/v2/articles?search_for=implied%20volatility&page_size=3"),
 # vendors
 ("orats","https://api.orats.io/datav2/hist/dailies?ticker=AAPL"),
 ("ivolatility","https://www.ivolatility.com/"),
 ("optionmetrics","https://optionmetrics.com/"),
 ("marketdata.app","https://api.marketdata.app/v1/options/expirations/AAPL/"),
 ("tradier sandbox","https://sandbox.tradier.com/v1/markets/quotes?symbols=AAPL"),
 ("polygon","https://api.polygon.io/v3/reference/options/contracts?underlying_ticker=AAPL&limit=2"),
 ("finnhub","https://finnhub.io/api/v1/stock/option-chain?symbol=AAPL"),
 ("twelvedata","https://api.twelvedata.com/time_series?symbol=AAPL&interval=1day&outputsize=2"),
 ("eodhd","https://eodhd.com/api/mp/unicornbay/options/eod?filter[underlying_symbol]=AAPL"),
 ("intrinio","https://api-v2.intrinio.com/options/expirations/AAPL"),
 ("barchart ondemand","https://ondemand.websol.barchart.com/getEquityOptions.json?symbol=AAPL"),
 ("nasdaq datalink","https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT.json?rows=2"),
 ("nasdaq api opt","https://api.nasdaq.com/api/quote/AAPL/option-chain?assetclass=stocks&limit=5"),
 ("fred csv VIXCLS","https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"),
 ("stlouisfed api","https://api.stlouisfed.org/fred/series?series_id=VIXCLS"),
 ("wayback avail cboe","https://archive.org/wayback/available?url=cboe.com/publish/ScheduledTask/MktData/datahouse/vxaplcurrent.csv"),
 ("optionsdx","https://www.optionsdx.com/"),
 ("historicaloptiondata","https://historicaloptiondata.com/"),
 ("discountoptiondata","https://discountoptiondata.com/"),
 ("deribit","https://www.deribit.com/api/v2/public/get_index_price?index_name=btc_usd"),
 ("alpaca opt","https://data.alpaca.markets/v1beta1/options/snapshots/AAPL"),
 ("schwab","https://api.schwabapi.com/"),
 ("yahoo opt","https://query2.finance.yahoo.com/v7/finance/options/AAPL"),
 ("stooq","https://stooq.com/q/d/l/?s=aapl.us&i=d"),
]
def probe(t):
    lab,u=t
    try:
        t0=time.time(); r=requests.get(u,headers=H,timeout=25,allow_redirects=True); dt=time.time()-t0
        return (lab,r.status_code,len(r.content),round(dt,2),r.text[:110].replace("\n"," "),u)
    except Exception as e:
        msg=str(e)
        blocked = "403 Forbidden" in msg
        return (lab,"BLOCKED-403" if blocked else "ERR:"+type(e).__name__,0,0,msg[:90],u)
with cf.ThreadPoolExecutor(12) as ex:
    res=list(ex.map(probe,URLS))
print(f"{'label':24} {'status':14} {'bytes':>9} {'s':>6}  preview")
for lab,st,n,dt,prev,u in res:
    print(f"{lab:24} {str(st):14} {n:>9} {dt:>6}  {prev[:100]}")
print("\nREACHABLE (non-403):")
for lab,st,n,dt,prev,u in res:
    if st!="BLOCKED-403": print(f"  {lab:24} {st} {u}")
