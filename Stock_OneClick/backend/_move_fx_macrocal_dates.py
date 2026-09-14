"""
_move_fx_macrocal_dates.py -- assemble + CROSS-VALIDATE a macro event date list and cache it to
_move_fx_macro_dates.json.  Run once; the study reads only the JSON.

Why cross-validation and not one source: a wrong date list silently corrupts the whole study, so
every date used here must appear in >=2 INDEPENDENT scrapes of the primary publisher
(federalreserve.gov for FOMC, bls.gov for CPI), or be explicitly flagged single-sourced.

SOURCES (all fetched 2026-09-12; federalreserve.gov and bls.gov are blocked by this machine's
egress proxy, so every source below is a third-party mirror/scrape of them):
  FOMC
   A raw.githubusercontent.com/marcburri/ScrapeFOMC/HEAD/FOMCData.csv        1936-2016 Fed scrape
   B raw.githubusercontent.com/tobiasi/FOMCscrape/HEAD/FOMC_dates.csv        1940-2018 Fed scrape
   C raw.githubusercontent.com/henryrweiland/Central-Bank-Meeting-Dates/HEAD/
       individual_central_banks/fed_meeting_dates.csv                        1975-2023
   D raw.githubusercontent.com/elenev/FOMCMeetings.jl/HEAD/test/fomccalendars.htm
       = a verbatim mirror of federalreserve.gov/monetarypolicy/fomccalendars.htm  2019-2024
   E raw.githubusercontent.com/borisjoffe/FOMC-dates.js/HEAD/dates.csv       2021-2024
   F raw.githubusercontent.com/Ftariq17/fed-day-backtester/HEAD/main.py      2020-2025
   G en.wikipedia.org/wiki/History_of_Federal_Open_Market_Committee_actions  rate-decision dates
   H raw.githubusercontent.com/xiaoxxcc/fomc-calendar/HEAD/fomc_calendar.json
       daily automated scrape of the Fed calendar page                       2026-2028
  CPI (BLS 08:30 ET release of the Consumer Price Index)
   P raw.githubusercontent.com/abusadat/CPI-release-dates/HEAD/cpi_releases.csv   2010-2025
   Q raw.githubusercontent.com/nthoang84/cpi-releases/HEAD/data/cpi_200801_202509.csv 2008-2025

We use the DECISION day (last day of the meeting), which is when the statement is released, not
the first day of a two-day meeting.
"""
from __future__ import annotations

import html as H
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

OUT = Path(__file__).with_name("_move_fx_macro_dates.json")
TMP = Path("/tmp")

URLS = {
    "fomc_marcburri": "https://raw.githubusercontent.com/marcburri/ScrapeFOMC/HEAD/FOMCData.csv",
    "fomc_tobiasi": "https://raw.githubusercontent.com/tobiasi/FOMCscrape/HEAD/FOMC_dates.csv",
    "fomc_hw": "https://raw.githubusercontent.com/henryrweiland/Central-Bank-Meeting-Dates/HEAD/individual_central_banks/fed_meeting_dates.csv",
    "fomc_elenev": "https://raw.githubusercontent.com/elenev/FOMCMeetings.jl/HEAD/test/fomccalendars.htm",
    "fomc_boris": "https://raw.githubusercontent.com/borisjoffe/FOMC-dates.js/HEAD/dates.csv",
    "fomc_ftariq": "https://raw.githubusercontent.com/Ftariq17/fed-day-backtester/HEAD/main.py",
    "fomc_xiao": "https://raw.githubusercontent.com/xiaoxxcc/fomc-calendar/HEAD/fomc_calendar.json",
    "fomc_ics": "https://raw.githubusercontent.com/Lim2Wolf/FOMC-calendar/HEAD/fomc_calendar/fomc-calendar.ics",
    "fomc_wiki": "https://en.wikipedia.org/wiki/History_of_Federal_Open_Market_Committee_actions",
    "cpi_abusadat": "https://raw.githubusercontent.com/abusadat/CPI-release-dates/HEAD/cpi_releases.csv",
    "cpi_nthoang": "https://raw.githubusercontent.com/nthoang84/cpi-releases/HEAD/data/cpi_200801_202509.csv",
}
EXT = {"fomc_elenev": "htm", "fomc_wiki": "html", "fomc_xiao": "json", "fomc_ftariq": "py",
       "fomc_ics": "ics"}


def grab(k):
    p = TMP / f"_mfx_{k}.{EXT.get(k, 'csv')}"
    if not p.exists() or p.stat().st_size < 100:
        subprocess.run(["curl", "-sL", "-m", "60", "-A", "Mozilla/5.0", URLS[k], "-o", str(p)],
                       check=True)
    return p


def d(x):
    return pd.DatetimeIndex(sorted(set(pd.to_datetime(pd.Series(list(x)).dropna()).dt.normalize())))


# ------------------------------------------------------------------ FOMC per-source parsers
def fomc_sources():
    S = {}
    a = pd.read_csv(grab("fomc_marcburri"))
    a = a[a["Meeting"].astype(str).str.lower() == "meeting"]
    S["A_marcburri"] = d(a["End"])

    b = pd.read_csv(grab("fomc_tobiasi"))
    b = b[b["Scheduled"] == 1]
    S["B_tobiasi"] = d(pd.to_datetime(b["End"], format="%d/%m/%Y"))

    c = pd.read_csv(grab("fomc_hw"))
    S["C_henryrweiland"] = d(c.loc[(c["fed_meeting_date_indicator"] == 1) &
                                   (c["fed_non_scheduled_meeting_indicator"] == 0), "date"])

    # D: the Fed's own fomccalendars.htm, mirrored verbatim. Each meeting row carries a
    # <div class="fomc-meeting__month">January</div> and a
    # <div class="fomc-meeting__date">29-30</div> (or "30-1" spanning a month end, or "15*").
    # The panel heading "<YYYY> FOMC Meetings" gives the year. We take the LAST day = decision day.
    h = grab("fomc_elenev").read_text(encoding="utf-8", errors="ignore")
    MON = ("January February March April May June July August September October November "
           "December").split()
    tokens = re.findall(
        r'(?:<a id="\d+">(\d{4}) FOMC Meetings</a>)'
        r'|(?:fomc-meeting__month[^>]*>\s*(?:<strong>)?\s*([A-Za-z/]+))'
        r'|(?:fomc-meeting__date[^>]*>\s*([^<]+)<)', h)
    cur_y, cur_m, got, unsched = None, None, [], []
    for y, mo, dd in tokens:
        if y:
            cur_y = int(y)
        elif mo:
            cur_m = mo.strip()
        elif dd and cur_y and cur_m:
            s = dd.strip()
            star = "*" in s
            s = s.replace("*", "").strip()
            parts = [p for p in re.split(r"[-–]", s) if p.strip()]
            last = parts[-1].strip()
            if not last.isdigit():
                continue
            # month label is "January", "Apr/May", "Jan/Feb", ... -> resolve each token
            months = []
            for tok in re.split(r"/", cur_m):
                full = [m for m in MON if m.lower().startswith(tok.strip().lower()[:3])]
                if full:
                    months.append(full[0])
            if not months:
                continue
            mo_last = months[-1]
            # a "30-1" range inside a SINGLE labelled month wraps into the next month
            wrap = False
            if len(parts) > 1 and len(months) == 1 and int(parts[0]) > int(last):
                i = MON.index(mo_last)
                mo_last, wrap = MON[(i + 1) % 12], (i == 11)
            yy = cur_y + (1 if wrap else 0)
            # "Jan/Feb" panels belong to the panel year; a Dec/Jan wrap rolls the year
            if months[0] == "December" and mo_last == "January":
                yy = cur_y + 1
            try:
                got.append(pd.Timestamp(f"{yy}-{MON.index(mo_last)+1}-{int(last)}"))
            except ValueError:
                continue
            _ = star, unsched
    S["D_fed_mirror"] = d(got)

    # I: a second automated Fed-calendar scrape published as an .ics (covers 2026+)
    ics = grab("fomc_ics").read_text(errors="ignore")
    blocks = ics.split("BEGIN:VEVENT")[1:]
    ii = []
    for b in blocks:
        if "SUMMARY:Fed FOMC Meeting" not in b:
            continue
        m = re.search(r"DTSTART;VALUE=DATE:(\d{8})", b)
        e = re.search(r"DTEND;VALUE=DATE:(\d{8})", b)
        if not m:
            continue
        st = pd.Timestamp(m.group(1))
        # ics DTEND for all-day events is exclusive; decision day = DTEND-1, else start
        ii.append(pd.Timestamp(e.group(1)) - pd.Timedelta(days=1) if e else st)
    S["I_lim2wolf_ics"] = d(ii)

    e = pd.read_csv(grab("fomc_boris"), header=None)
    S["E_borisjoffe"] = d(e[0])

    f = grab("fomc_ftariq").read_text(errors="ignore")
    # the hardcoded FOMC list in that file; keep only 8-per-year plausible dates, drop the
    # two obvious non-event dates used as backtest bounds (start/end of sample)
    cand = d(re.findall(r"20\d{2}-\d{2}-\d{2}", f))
    S["F_ftariq"] = d([x for x in cand if x not in
                       (pd.Timestamp("2019-12-01"), pd.Timestamp("2025-12-31"))])

    g = grab("fomc_wiki").read_text(encoding="utf-8", errors="ignore")
    wd = []
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", g, re.S):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)
        t = [H.unescape(re.sub("<[^>]+>", "", c)).strip() for c in cells]
        if t and re.match(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$", t[0]):
            wd.append(t[0])
    S["G_wikipedia"] = d(wd)

    j = json.loads(grab("fomc_xiao").read_text())
    S["H_xiao_fedscrape"] = d([m["end_date"] for m in j["meetings"]])
    return S


def cpi_sources():
    S = {}
    p = pd.read_csv(grab("cpi_abusadat"))
    S["P_abusadat"] = d(p["Date"])
    q = pd.read_csv(grab("cpi_nthoang"))
    S["Q_nthoang"] = d(pd.to_datetime(q["ReleaseDate"]).dt.normalize())
    return S


def consensus(S, lo="2001-01-01", hi="2026-12-31", name=""):
    """Keep a date iff >=2 sources that COVER that date's year agree on it. Report disagreements."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    span = {k: (v.min(), v.max()) for k, v in S.items()}
    alld = d(sorted(set().union(*[set(v) for v in S.values()])))
    alld = alld[(alld >= lo) & (alld <= hi)]
    keep, single, disagree = [], [], []
    for x in alld:
        covering = [k for k, (a, b) in span.items() if a <= x <= b]
        votes = [k for k in covering if x in S[k]]
        if len(votes) >= 2:
            keep.append(x)
            if len(votes) < len(covering):
                disagree.append((x.date().isoformat(), sorted(votes),
                                 sorted(set(covering) - set(votes))))
        elif len(votes) == 1:
            single.append((x.date().isoformat(), votes[0], sorted(covering)))
    print(f"\n=== {name} ===")
    for k, v in S.items():
        print(f"  {k:<20} n={len(v):>5}  {v.min().date()} .. {v.max().date()}")
    print(f"  consensus (>=2 covering sources agree): {len(keep)} dates "
          f"{keep[0].date()} .. {keep[-1].date()}")
    print(f"  single-sourced within a covered span (DROPPED): {len(single)}")
    for s in single[:25]:
        print("     ", s)
    print(f"  partial disagreements among covering sources (KEPT): {len(disagree)}")
    for s in disagree[:25]:
        print("     ", s)
    byy = pd.Series(1, index=pd.DatetimeIndex(keep)).groupby(pd.DatetimeIndex(keep).year).sum()
    print("  per year:", {int(k): int(v) for k, v in byy.items()})
    return pd.DatetimeIndex(keep), byy


def main():
    Sf = fomc_sources()
    fomc, fy = consensus(Sf, name="FOMC decision days")
    Sc = cpi_sources()
    cpi, cy = consensus(Sc, name="CPI release days")

    # coverage spans: years where at least 2 sources cover the WHOLE year
    def cov_years(S, expect):
        ok = []
        for y in range(2001, 2027):
            ya, yb = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
            n = sum(1 for v in S.values() if v.min() <= ya and v.max() >= yb)
            if n >= 2:
                ok.append(y)
        return ok

    out = {
        "generated": "2026-09-12",
        "note": "Assembled by _move_fx_macrocal_dates.py. Every date below appears in >=2 "
                "independent third-party scrapes of the primary publisher "
                "(federalreserve.gov / bls.gov), both of which are blocked by this machine's "
                "egress proxy. Dates are the EVENT day: FOMC = last day of the meeting "
                "(statement day); CPI = BLS 08:30 ET release day.",
        "sources": URLS,
        "fomc": [x.date().isoformat() for x in fomc],
        "fomc_full_coverage_years": cov_years(Sf, 8),
        "cpi": [x.date().isoformat() for x in cpi],
        "cpi_full_coverage_years": cov_years(Sc, 12),
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}  fomc={len(fomc)} cpi={len(cpi)}")
    print("  fomc full-coverage years:", out["fomc_full_coverage_years"])
    print("  cpi  full-coverage years:", out["cpi_full_coverage_years"])


if __name__ == "__main__":
    main()
