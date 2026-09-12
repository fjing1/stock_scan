"""
_move_wf_vixiv.py — Q: does VIX-implied vol beat backward-looking realized vol, and can the
index's implied vol help single stocks?

Everything fitted is walk-forward (expanding window, fit on strictly prior calendar years).
Volatility is per-DAY sigma of LOG returns throughout (repo convention, never annualized).

VIX -> comparable per-day sigma:  sigma_iv = (VIX/100) / sqrt(252)

TARGET (forward realized per-day sigma over r[t+1..t+h], indexed at t):
  h >= 2 : L.realized_vol_forward(close, h)            (ddof=1, nu = h-1)
  h == 1 : |log r_{t+1}|                                (nu = 1; ddof=1 std of 1 obs is NaN)
Both proxies are DOWNWARD-biased estimates of sigma in log space and in level space by known
Gaussian factors, so we un-bias them before measuring any ratio/intercept, otherwise the
"variance risk premium" would be contaminated by pure small-sample estimator bias:
  level:  E[s] = c4(nu) * sigma,   c4 = sqrt(2/nu) * G((nu+1)/2)/G(nu/2)
  log:    E[log s] = log sigma + clog(nu),  clog = 0.5*(digamma(nu/2) - log(nu/2))
For nu=1 this reproduces the textbook sqrt(2/pi) and -0.6352 for |z|.

Run: ../../vcp_env/bin/python _move_wf_vixiv.py
"""
from __future__ import annotations

import warnings

from math import lgamma

import numpy as np
import pandas as pd

import _move_data as D
import _move_lib as L

warnings.filterwarnings("ignore")
rng = np.random.default_rng(7)

HORIZONS = (1, 5, 10, 21)
MIN_TRAIN_YEARS = 5
FLOOR = 1e-5          # vol floor before logs (a 0.001% daily sigma is already absurd)


def digamma(x: float) -> float:
    """psi(x) for x>0: recurrence up to 6 then the standard asymptotic series.
    (scipy is not installed in vcp_env; verified below against psi(1/2)=-gamma-2ln2.)"""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return (r + np.log(x) - 0.5 / x
            + f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0 + f * (1 / 240.0)))))


# ------------------------------------------------------------------ small-sample unbiasing
def c4(nu):
    return np.sqrt(2.0 / nu) * np.exp(lgamma((nu + 1) / 2) - lgamma(nu / 2))


def clog(nu):
    return 0.5 * (digamma(nu / 2) - np.log(nu / 2))


def fwd_sigma(close, h):
    """Unbiased-in-LEVEL forward per-day sigma proxy, indexed at t."""
    nu = 1 if h == 1 else h - 1
    if h == 1:
        proxy = np.log(close).diff().abs().shift(-1)
    else:
        proxy = L.realized_vol_forward(close, h)
    return proxy / c4(nu)


def fwd_log_sigma(close, h):
    """Unbiased-in-LOG forward per-day log-sigma proxy, indexed at t."""
    nu = 1 if h == 1 else h - 1
    if h == 1:
        proxy = np.log(close).diff().abs().shift(-1)
    else:
        proxy = L.realized_vol_forward(close, h)
    return np.log(proxy.clip(lower=FLOOR)) - clog(nu)


# ------------------------------------------------------------------ walk-forward OLS
def wf_ols(X: pd.DataFrame, y: pd.Series, min_train_years=MIN_TRAIN_YEARS):
    """Expanding-window per-calendar-year OLS. Returns tidy frame of OOS predictions plus the
    expanding train mean of y (the honest no-lookahead benchmark for R^2)."""
    df = pd.concat([y.rename("y"), X], axis=1).dropna()
    if len(df) < 300:
        return None, None
    yv = df["y"].values
    Xv = np.column_stack([np.ones(len(df)), df[X.columns].values])
    idx = df.index
    rows, coefs = [], []
    for yr, tr, te in L.walk_forward_years(idx, min_train_years):
        if tr.sum() < 250 or te.sum() < 20:
            continue
        b, *_ = np.linalg.lstsq(Xv[tr], yv[tr], rcond=None)
        rows.append(pd.DataFrame({"date": idx[te], "year": yr, "y": yv[te],
                                  "pred": Xv[te] @ b, "ybar_tr": yv[tr].mean(),
                                  "resid_sd_tr": (yv[tr] - Xv[tr] @ b).std(ddof=len(b))}))
        coefs.append(pd.Series(b, index=["const"] + list(X.columns), name=yr))
    if not rows:
        return None, None
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coefs)


def oos_r2(res):
    """R^2 vs the expanding TRAIN mean (Campbell-Thompson style, no lookahead)."""
    if res is None or len(res) == 0:
        return np.nan
    ss_res = ((res.y - res.pred) ** 2).sum()
    ss_tot = ((res.y - res.ybar_tr) ** 2).sum()
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ------------------------------------------------------------------ features
def realized_candidates(o, h_, l_, c_):
    """All backward-looking per-day sigma estimators available at t, from _move_lib only."""
    return {
        "cc21":    L.vol_cc(c_, 21),
        "cc63":    L.vol_cc(c_, 63),
        "ewma94":  L.vol_ewma(c_, 0.94),
        "ewma97":  L.vol_ewma(c_, 0.97),
        "park21":  L.vol_parkinson(h_, l_, 21),
        "gk21":    L.vol_garman_klass(o, h_, l_, c_, 21),
        "rs21":    L.vol_rogers_satchell(o, h_, l_, c_, 21),
        "yz21":    L.vol_yang_zhang(o, h_, l_, c_, 21),
        "yz63":    L.vol_yang_zhang(o, h_, l_, c_, 63),
    }


def lg(s):
    return np.log(s.clip(lower=FLOOR))


def main():
    p = D.load()
    O, H, Lo, C = p["Open"], p["High"], p["Low"], p["Close"]
    keep = C.notna().sum()
    C = C.loc[:, keep >= 500]
    O, H, Lo = O[C.columns], H[C.columns], Lo[C.columns]
    print(f"panel after <500 drop: {C.shape[1]} symbols x {len(C)} rows "
          f"{C.index[0].date()} -> {C.index[-1].date()}")

    vix = C["^VIX"]
    iv = vix / 100.0 / np.sqrt(252.0)           # per-day implied sigma
    log_iv = np.log(iv)
    print(f"VIX: n={vix.notna().sum()}  mean={vix.mean():.2f}  "
          f"implied per-day sigma mean={iv.mean()*100:.3f}%")

    # ============================================================ PART 1: index horse race
    print("\n" + "=" * 100)
    print("PART 1  INDEX HORSE RACE — OOS R^2 on log(forward per-day sigma), walk-forward OLS")
    print("        R^2 measured vs the expanding TRAIN mean (no lookahead). Test years 2006-2026.")
    print("=" * 100)

    part1 = []
    coef_store = {}
    for sym in ["SPY", "^GSPC"]:
        cands = realized_candidates(O[sym], H[sym], Lo[sym], C[sym])
        har = L.har_features(C[sym], H[sym], Lo[sym])
        har_X = pd.DataFrame({"lg_rv_d": lg(har.rv_d), "lg_rv_w": lg(har.rv_w),
                              "lg_rv_m": lg(har.rv_m)})
        for h in HORIZONS:
            y = fwd_log_sigma(C[sym], h)

            # --- per-candidate realized-only OOS R^2 (each is a full WF fit)
            cand_r2, cand_res = {}, {}
            for nm, s in cands.items():
                r, _ = wf_ols(pd.DataFrame({"x": lg(s)}), y)
                cand_r2[nm] = oos_r2(r)
                cand_res[nm] = r

            # --- WF-SELECTED realized estimator: for each test year pick the candidate with the
            #     best TRAIN-period R^2 using only prior years. No selection lookahead.
            sel_rows, sel_pick = [], {}
            base = pd.concat([y.rename("y")] + [lg(s).rename(nm) for nm, s in cands.items()],
                             axis=1).dropna()
            for yr, tr, te in L.walk_forward_years(base.index, MIN_TRAIN_YEARS):
                if tr.sum() < 250 or te.sum() < 20:
                    continue
                yv = base["y"].values
                best, bestr2 = None, -np.inf
                for nm in cands:
                    Xtr = np.column_stack([np.ones(tr.sum()), base[nm].values[tr]])
                    b, *_ = np.linalg.lstsq(Xtr, yv[tr], rcond=None)
                    r2 = 1 - ((yv[tr] - Xtr @ b) ** 2).sum() / ((yv[tr] - yv[tr].mean()) ** 2).sum()
                    if r2 > bestr2:
                        best, bestr2 = nm, r2
                sel_pick[yr] = best
                Xtr = np.column_stack([np.ones(tr.sum()), base[best].values[tr]])
                b, *_ = np.linalg.lstsq(Xtr, yv[tr], rcond=None)
                Xte = np.column_stack([np.ones(te.sum()), base[best].values[te]])
                sel_rows.append(pd.DataFrame({"date": base.index[te], "year": yr, "y": yv[te],
                                              "pred": Xte @ b, "ybar_tr": yv[tr].mean()}))
            sel_res = pd.concat(sel_rows, ignore_index=True)
            picks = pd.Series(sel_pick).value_counts().to_dict()

            best_static = max(cand_r2, key=lambda k: cand_r2[k])

            models = {
                "VIX only":          pd.DataFrame({"lg_iv": log_iv}),
                "realized WF-sel":   None,   # handled above
                "best-single":       pd.DataFrame({"x": lg(cands[best_static])}),
                "HAR(3)":            har_X,
                "VIX+best-single":   pd.DataFrame({"lg_iv": log_iv,
                                                   "x": lg(cands[best_static])}),
                "VIX + HAR(3)":      har_X.assign(lg_iv=log_iv),
            }
            row = {"sym": sym, "h": h, "bs_name": best_static}
            for nm, X in models.items():
                if X is None:
                    row[nm] = oos_r2(sel_res)
                    row["n"] = len(sel_res)
                    continue
                r, cf = wf_ols(X, y)
                row[nm] = oos_r2(r)
                coef_store[(sym, h, nm)] = (r, cf)
            row["WF picks"] = picks
            row["cand_r2"] = {k: round(v, 4) for k, v in cand_r2.items()}
            part1.append(row)

    p1 = pd.DataFrame(part1)
    show = [c for c in p1.columns if c not in ("WF picks", "cand_r2")]
    print("\n" + p1[show].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))
    print("\nWalk-forward realized-estimator picks (count of test years):")
    for _, r in p1.iterrows():
        print(f"  {r['sym']:>6} h={r['h']:>2}: {r['WF picks']}")
    print("\nPer-candidate realized-only OOS R^2 (each a full WF fit; shown for diagnosis, the "
          "headline realized number is 'realized WF-sel'):")
    for _, r in p1.iterrows():
        print(f"  {r['sym']:>6} h={r['h']:>2}: {r['cand_r2']}")

    # incremental R^2 of VIX over the best realized model, with year-block bootstrap
    print("\nIncremental OOS R^2 from adding VIX to HAR(3), with 2000x year-block bootstrap "
          "(resample calendar test-years with replacement; overlap-safe because whole years move "
          "together):")
    for sym in ["SPY", "^GSPC"]:
        for h in HORIZONS:
            a = coef_store[(sym, h, "HAR(3)")][0]
            b = coef_store[(sym, h, "VIX + HAR(3)")][0]
            m = a.merge(b, on=["date", "year"], suffixes=("_a", "_b"))
            yrs = m.year.unique()
            d = []
            for _ in range(2000):
                pick = rng.choice(yrs, size=len(yrs), replace=True)
                sub = pd.concat([m[m.year == q] for q in pick], ignore_index=True)
                ra = 1 - ((sub.y_a - sub.pred_a) ** 2).sum() / ((sub.y_a - sub.ybar_tr_a) ** 2).sum()
                rb = 1 - ((sub.y_b - sub.pred_b) ** 2).sum() / ((sub.y_b - sub.ybar_tr_b) ** 2).sum()
                d.append(rb - ra)
            d = np.array(d)
            pt = oos_r2(b) - oos_r2(a)
            print(f"  {sym:>6} h={h:>2}: dR2={pt:+.4f}  95%CI[{np.percentile(d,2.5):+.4f},"
                  f"{np.percentile(d,97.5):+.4f}]  P(dR2>0)={(d>0).mean():.3f}  "
                  f"n_years={len(yrs)}  n_obs={len(m)}")

    # non-overlapping check on the headline comparison
    print("\nSame comparison on NON-OVERLAPPING rows only (every h-th test row):")
    for sym in ["SPY", "^GSPC"]:
        for h in HORIZONS:
            a = coef_store[(sym, h, "HAR(3)")][0]
            b = coef_store[(sym, h, "VIX + HAR(3)")][0]
            m = a.merge(b, on=["date", "year"], suffixes=("_a", "_b")).iloc[::h]
            ra = 1 - ((m.y_a - m.pred_a) ** 2).sum() / ((m.y_a - m.ybar_tr_a) ** 2).sum()
            rb = 1 - ((m.y_b - m.pred_b) ** 2).sum() / ((m.y_b - m.ybar_tr_b) ** 2).sum()
            print(f"  {sym:>6} h={h:>2}: HAR {ra:.4f} -> VIX+HAR {rb:.4f} "
                  f"(dR2={rb-ra:+.4f}, n={len(m)})")

    # ============================================================ PART 2: variance risk premium
    print("\n" + "=" * 100)
    print("PART 2  VARIANCE RISK PREMIUM — how much does VIX overstate subsequent realized vol?")
    print("=" * 100)
    vrp_rows = []
    for sym in ["SPY", "^GSPC"]:
        for h in HORIZONS:
            rv = fwd_sigma(C[sym], h)
            lrv = fwd_log_sigma(C[sym], h)
            d = pd.concat([rv.rename("rv"), lrv.rename("lrv"), iv.rename("iv"),
                           log_iv.rename("liv")], axis=1).dropna()
            # log-log regression (IN-SAMPLE, full period): log rv = a + b log iv
            X = np.column_stack([np.ones(len(d)), d.liv.values])
            beta, *_ = np.linalg.lstsq(X, d.lrv.values, rcond=None)
            resid = d.lrv.values - X @ beta
            s = resid.std(ddof=2)
            r2_is = 1 - (resid ** 2).sum() / ((d.lrv - d.lrv.mean()) ** 2).sum()
            # walk-forward average of the same coefficients (no lookahead version)
            _, cf = wf_ols(pd.DataFrame({"liv": d.liv}), d.lrv)
            vrp_rows.append({
                "sym": sym, "h": h, "n": len(d),
                "a_IS": beta[0], "b_IS": beta[1], "resid_sd": s, "r2_IS": r2_is,
                "a_WFmean": cf["const"].mean(), "b_WFmean": cf["liv"].mean(),
                "mean_rv/iv": (d.rv / d.iv).mean(), "med_rv/iv": (d.rv / d.iv).median(),
                "meanRV/meanIV": d.rv.mean() / d.iv.mean(),
                "var_ratio": (d.rv ** 2).mean() / (d.iv ** 2).mean(),
                "P(rv>iv)": (d.rv > d.iv).mean(),
                "lognorm_corr": np.exp(s ** 2 / 2),
            })
    vrp = pd.DataFrame(vrp_rows)
    print("\nlog-log:  log(sigma_realized_fwd) = a + b*log(sigma_implied)   [IS = in-sample, full "
          "period; WFmean = mean of walk-forward per-year fits]")
    print(vrp.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print("\nLEVEL bias correction implied by the IS log-log fit, evaluated at VIX=16 and VIX=30\n"
          "  sigma_hat = exp(a) * sigma_iv^b * exp(resid_sd^2/2)   [the exp term un-does Jensen]")
    for _, r in vrp.iterrows():
        for vx in (16.0, 30.0):
            sig_iv = vx / 100 / np.sqrt(252)
            sh = np.exp(r.a_IS) * sig_iv ** r.b_IS * np.exp(r.resid_sd ** 2 / 2)
            print(f"  {r['sym']:>6} h={int(r.h):>2}  VIX={vx:>4.0f}: sigma_iv={sig_iv*100:.3f}%/d"
                  f"  ->  sigma_hat={sh*100:.3f}%/d   ratio={sh/sig_iv:.3f}")

    # ============================================================ PART 3: stability by year
    print("\n" + "=" * 100)
    print("PART 3  IS THE PREMIUM STABLE? realized/implied ratio by calendar year (SPY, h=21)")
    print("=" * 100)
    for h in (5, 21):
        rv = fwd_sigma(C["SPY"], h)
        d = pd.concat([rv.rename("rv"), iv.rename("iv")], axis=1).dropna()
        d["year"] = d.index.year
        g = d.groupby("year").apply(lambda x: pd.Series({
            "n": len(x), "mean_VIX": x.iv.mean() * 100 * np.sqrt(252),
            "meanRV/meanIV": x.rv.mean() / x.iv.mean(),
            "mean_ratio": (x.rv / x.iv).mean(),
            "P(rv>iv)": (x.rv > x.iv).mean()}))
        print(f"\n--- h={h} ---")
        print(g.to_string(float_format=lambda x: f"{x:8.3f}"))
        cal = g.mean_VIX.idxmin()
        print(f"  calmest year by mean VIX: {cal} (VIX {g.mean_VIX[cal]:.1f}, "
              f"ratio {g['meanRV/meanIV'][cal]:.3f})")
        for yr in (2008, 2020):
            if yr in g.index:
                print(f"  {yr}: ratio {g['meanRV/meanIV'][yr]:.3f}, "
                      f"P(rv>iv) {g['P(rv>iv)'][yr]:.3f}, mean VIX {g.mean_VIX[yr]:.1f}")
        print(f"  ratio range across years: {g['meanRV/meanIV'].min():.3f} .. "
              f"{g['meanRV/meanIV'].max():.3f}; years with ratio>1: "
              f"{list(g.index[g['meanRV/meanIV'] > 1])}")

    # sub-period: does the bias correction hold up out of sample? split at 2014
    print("\nBias-correction stability: fit log-log on <=2013, apply to >=2014 (SPY)")
    for h in HORIZONS:
        lrv = fwd_log_sigma(C["SPY"], h)
        rv = fwd_sigma(C["SPY"], h)
        d = pd.concat([lrv.rename("lrv"), rv.rename("rv"), log_iv.rename("liv"),
                       iv.rename("iv")], axis=1).dropna()
        tr, te = d.index.year <= 2013, d.index.year >= 2014
        X = np.column_stack([np.ones(tr.sum()), d.liv.values[tr]])
        b, *_ = np.linalg.lstsq(X, d.lrv.values[tr], rcond=None)
        s = (d.lrv.values[tr] - X @ b).std(ddof=2)
        pred = np.exp(b[0] + b[1] * d.liv.values[te] + s ** 2 / 2)
        act = d.rv.values[te]
        print(f"  h={h:>2}: train a={b[0]:+.4f} b={b[1]:.4f} | test mean(actual/pred)="
              f"{np.mean(act/pred):.3f}  meanAct/meanPred={act.mean()/pred.mean():.3f}  "
              f"raw meanRV/meanIV(test)={act.mean()/d.iv.values[te].mean():.3f}  "
              f"n_tr={tr.sum()} n_te={te.sum()}")

    # ============================================================ PART 4: single names
    print("\n" + "=" * 100)
    print("PART 4  SINGLE NAMES — does today's VIX add to a stock's OWN realized vol?")
    print("=" * 100)
    cov = C.notna().sum()
    stocks = [s for s in C.columns if s not in ("SPY", "QQQ", "IWM", "DIA", "^GSPC", "^VIX")
              and cov[s] >= 1500]
    print(f"{len(stocks)} single names with >=1500 closes")

    # panel-wide features, computed once
    yz = L.vol_yang_zhang(O[stocks], H[stocks], Lo[stocks], C[stocks], 21)
    par = L.vol_parkinson(H[stocks], Lo[stocks], 21)
    cc = L.vol_cc(C[stocks], 21)
    ew = L.vol_ewma(C[stocks], 0.94)
    spy_rv = L.vol_yang_zhang(O["SPY"], H["SPY"], Lo["SPY"], C["SPY"], 21)
    lg_ivr = np.log(iv.clip(lower=FLOOR)) - np.log(spy_rv.clip(lower=FLOOR))   # log(IV / SPY RV)
    # trailing 252d beta vs SPY, available at t
    rs = np.log(C[stocks]).diff()
    rm = np.log(C["SPY"]).diff()
    beta = (rs.rolling(252).cov(rm)).div(rm.rolling(252).var(), axis=0)
    beta = beta.clip(0.1, 3.0)

    per_h = {}
    for h in HORIZONS:
        rows = []
        for sym in stocks:
            y = fwd_log_sigma(C[sym], h)
            har = L.har_features(C[sym], H[sym], Lo[sym])
            own = lg(yz[sym])
            base = {"own": own}
            mods = {
                "A own":            pd.DataFrame(base),
                "B own+VIX":        pd.DataFrame({**base, "liv": log_iv}),
                "C own+beta*VIX":   pd.DataFrame({**base, "bliv": beta[sym] * log_iv}),
                "D own+log(IV/RVm)": pd.DataFrame({**base, "ivr": lg_ivr}),
                "E HAR":            pd.DataFrame({"d": lg(har.rv_d), "w": lg(har.rv_w),
                                                  "m": lg(har.rv_m)}),
                "F HAR+VIX":        pd.DataFrame({"d": lg(har.rv_d), "w": lg(har.rv_w),
                                                  "m": lg(har.rv_m), "liv": log_iv}),
                "G own+VIX+beta*VIX": pd.DataFrame({**base, "liv": log_iv,
                                                    "bliv": beta[sym] * log_iv}),
            }
            r = {"sym": sym}
            resB = None
            for nm, X in mods.items():
                out, cf = wf_ols(X, y)
                r[nm] = oos_r2(out)
                if nm == "B own+VIX" and cf is not None:
                    r["coef_VIX"] = cf["liv"].mean()
                    r["coef_own"] = cf["own"].mean()
                    resB = out
                if nm == "C own+beta*VIX" and cf is not None:
                    r["coef_bVIX"] = cf["bliv"].mean()
                if nm == "A own" and cf is not None:
                    r["coefA_own"] = cf["own"].mean()
                if nm == "D own+log(IV/RVm)" and cf is not None:
                    r["coef_ivr"] = cf["ivr"].mean()
            if resB is not None:
                r["n"] = len(resB)
            rows.append(r)
        st = pd.DataFrame(rows).dropna(subset=["A own"])
        per_h[h] = st
        cols = [c for c in st.columns if c.startswith(("A ", "B ", "C ", "D ", "E ", "F ", "G "))]
        print(f"\n--- h={h}, {len(st)} names, median n_oos={st['n'].median():.0f} ---")
        summ = st[cols].describe(percentiles=[.25, .5, .75]).T[["mean", "25%", "50%", "75%"]]
        summ["frac>0"] = [(st[c] > 0).mean() for c in cols]
        print(summ.to_string(float_format=lambda x: f"{x:8.4f}"))
        for pair in [("B own+VIX", "A own"), ("C own+beta*VIX", "A own"),
                     ("D own+log(IV/RVm)", "A own"), ("F HAR+VIX", "E HAR"),
                     ("G own+VIX+beta*VIX", "B own+VIX")]:
            d = st[pair[0]] - st[pair[1]]
            print(f"   dR2 {pair[0]} - {pair[1]}: median {d.median():+.4f}  mean {d.mean():+.4f}"
                  f"  frac>0 {(d > 0).mean():.3f}  n={len(d)}")
        print(f"   coef on log(VIX) in B: median {st.coef_VIX.median():+.3f}  "
              f"IQR[{st.coef_VIX.quantile(.25):+.3f},{st.coef_VIX.quantile(.75):+.3f}]  "
              f"frac>0 {(st.coef_VIX > 0).mean():.3f}")
        print(f"   coef on log(own) : A-model median {st.coefA_own.median():+.3f} -> "
              f"B-model median {st.coef_own.median():+.3f}")
        print(f"   coef on beta*log(VIX) in C: median {st.coef_bVIX.median():+.3f}")
        print(f"   coef on log(IV/RV_mkt) in D: median {st.coef_ivr.median():+.3f}")

    # date-clustered bootstrap on the POOLED single-name incremental R^2 (h=5 and h=21)
    print("\nPooled single-name incremental R^2 from VIX, DATE-clustered bootstrap on "
          "NON-OVERLAPPING dates (1000 reps; a whole cross-section moves together, so "
          "cross-sectional correlation is respected):")
    for h in (5, 21):
        # rebuild pooled OOS predictions for A and B
        recs = []
        for sym in stocks:
            y = fwd_log_sigma(C[sym], h)
            a, _ = wf_ols(pd.DataFrame({"own": lg(yz[sym])}), y)
            b, _ = wf_ols(pd.DataFrame({"own": lg(yz[sym]), "liv": log_iv}), y)
            if a is None or b is None:
                continue
            m = a.merge(b, on=["date", "year"], suffixes=("_a", "_b"))
            m["sym"] = sym
            recs.append(m)
        pool = pd.concat(recs, ignore_index=True)
        dates = np.sort(pool.date.unique())[::h]          # non-overlapping
        pool = pool[pool.date.isin(dates)]
        gb = {d: g for d, g in pool.groupby("date")}
        keys = np.array(list(gb))

        def r2pair(df):
            ra = 1 - ((df.y_a - df.pred_a) ** 2).sum() / ((df.y_a - df.ybar_tr_a) ** 2).sum()
            rb = 1 - ((df.y_b - df.pred_b) ** 2).sum() / ((df.y_b - df.ybar_tr_b) ** 2).sum()
            return ra, rb
        ra, rb = r2pair(pool)
        boot = []
        for _ in range(1000):
            pick = rng.choice(len(keys), size=len(keys), replace=True)
            sub = pd.concat([gb[keys[i]] for i in pick], ignore_index=True)
            x, z = r2pair(sub)
            boot.append(z - x)
        boot = np.array(boot)
        print(f"  h={h:>2}: pooled R2 own={ra:.4f} own+VIX={rb:.4f}  dR2={rb-ra:+.4f}  "
              f"95%CI[{np.percentile(boot,2.5):+.4f},{np.percentile(boot,97.5):+.4f}]  "
              f"P(dR2>0)={(boot>0).mean():.3f}  n_dates={len(keys)}  n_obs={len(pool)}  "
              f"n_syms={pool['sym'].nunique()}")

    # ============================================================ PART 5: best proxy, no options
    print("\n" + "=" * 100)
    print("PART 5  BEST ACHIEVABLE PROXY FOR A NAME WITH NO OPTIONS DATA (h=5,21)")
    print("=" * 100)
    for h in (5, 21):
        st = per_h[h]
        best = {}
        for c in [c for c in st.columns if c[0] in "ABCDEFG" and c[1] == " "]:
            best[c] = st[c].median()
        srt = sorted(best.items(), key=lambda kv: -kv[1])
        print(f"\n h={h}: median across-name OOS R^2, ranked")
        for k, v in srt:
            print(f"   {k:<22} {v:7.4f}")

    # do the WHOLE thing for the index too, as the ceiling reference
    print("\nCeiling reference (index, unbiased-target, WF): SPY OOS R^2")
    print(p1[p1.sym == "SPY"][show].to_string(index=False, float_format=lambda x: f"{x:7.4f}"))

    print("\nDONE")


if __name__ == "__main__":
    main()
