"""
Desk analytics on top of structured_products.py

The base script answers "which coupon issues the note at par?". This module adds
the questions a structuring desk asks next:

    1. Skew          what does the downside skew do to the autocall coupon,
                     given that the 60% protection barrier is a deep OTM put?
    2. Sales sheet   probability of early redemption at each observation,
                     expected life, probability of capital loss, scenarios
                     at maturity, and a one-page French product sheet
    3. Worst-of      Athena autocall on the worst of Euro Stoxx 50, CAC 40
                     and DAX, correlated by Cholesky, and coupon vs correlation
    4. Funding       issuer funding spread on the zero-coupon leg
    5. Hedging       delta-hedging backtest of the autocall, on simulated paths
                     (model check) and on SPY history (live mode)

Run:   python desk_analytics.py           (live market data if yfinance can reach Yahoo)
       SP_OFFLINE=1 python desk_analytics.py   (explicit parameters, flagged as such)

Every input that is not read from the market is declared below as an explicit
parameter and printed as such in the outputs.

Diptakshi Sen - personal project, 2026
"""

import os
from datetime import datetime, timedelta
from math import exp, log, sqrt, pi

import numpy as np

import structured_products as sp
from structured_products import (
    S0, R, Q, VOL, T, NOTIONAL, FEE, PATHS, SEED, OBS_PER_YEAR,
    AUTOCALL_BARRIER, PDI_BARRIER, MKT, black_scholes, norm_cdf, simulate_paths,
    autocall_payoff, zc,
)

N_OBS = int(OBS_PER_YEAR * T)

# ------------------------------------------------- EXPLICIT PARAMETERS -------
FUNDING_SPREAD = 0.0060        # issuer funding spread over the risk-free rate (60 bp),
                               # illustrative: replace with the issuer's own curve
EUR_RATE = 0.0225              # EUR 5-year rate for the worst-of, explicit assumption
EU_TICKERS = {"Euro Stoxx 50": "^STOXX50E", "CAC 40": "^FCHI", "DAX": "^GDAXI"}
EU_DIVIDENDS = {"Euro Stoxx 50": 0.030, "CAC 40": 0.030, "DAX": 0.000}
# the DAX is a total-return (performance) index: dividends are reinvested, so q = 0

SKEW_FALLBACK_SPREAD = 0.08    # 60%-strike vol minus ATM vol, used only offline
EU_FALLBACK = {"vols": [0.17, 0.17, 0.18],
               "corr": [[1.00, 0.93, 0.92], [0.93, 1.00, 0.88], [0.92, 0.88, 1.00]]}

HEDGE_PATHS_MC = 10_000        # Monte Carlo paths per revaluation inside the backtest
HEDGE_EVERY = 5                # rebalance every 5 trading days
TRADING_DAYS = 252


def offline():
    return bool(os.environ.get("SP_OFFLINE"))


def pct(x):
    return f"{x:.2%}"


# =============================================================================
# Shared Monte Carlo helpers
# =============================================================================
def coupon_from_paths(paths_S, S_fix=S0, disc_r=R, fee=FEE, extra=0.0,
                      target=NOTIONAL, lo=0.0, hi=0.80):
    """Annual coupon that prices the note at `target` on a fixed set of paths.

    `extra` is a value added to the Monte Carlo price that does not depend on
    the coupon (the skew adjustment). `disc_r` discounts the note's cash flows,
    so it can carry the issuer funding spread while the paths stay risk-neutral.
    """
    for _ in range(45):
        mid = (lo + hi) / 2
        price, _ = autocall_payoff(paths_S, mid, S_fix, disc_r, T, NOTIONAL, fee)
        if price + extra < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# =============================================================================
# 1. SKEW ON THE PROTECTION BARRIER
# =============================================================================
def load_skew():
    """Implied vol at the 60% strike, and the local slope of the smile there.

    Read on the puts of the longest listed SPY expiry. The skew is applied as a
    spread over the base ATM vol, so it stays consistent with structured_products.
    """
    K = PDI_BARRIER * S0
    try:
        if offline():
            raise RuntimeError("offline mode (SP_OFFLINE)")
        import yfinance as yf
        tk = yf.Ticker(sp.UNDERLYING)
        expiry = sorted(tk.options)[-1]
        tau = (datetime.strptime(expiry, "%Y-%m-%d").date()
               - datetime.utcnow().date()).days / 365.0
        puts = tk.option_chain(expiry).puts.dropna(subset=["impliedVolatility"])
        puts = puts[(puts["impliedVolatility"] > 0.01) & (puts["impliedVolatility"] < 2.0)]
        k = puts["strike"].to_numpy(float)
        iv = puts["impliedVolatility"].to_numpy(float)
        if k.min() > K or k.max() < S0:
            raise ValueError("chain does not cover the 60% strike")

        def vol_at(x):
            return float(np.interp(x, k, iv))

        h = 0.05 * S0
        spread = vol_at(K) - vol_at(S0)
        slope = (vol_at(K + h) - vol_at(K - h)) / (2 * h)
        return {"vol_60": VOL + spread, "slope": slope, "spread": spread, "tenor": tau,
                "source": f"SPY puts, expiry {expiry}", "live": True}
    except Exception as exc:
        spread = SKEW_FALLBACK_SPREAD
        return {"vol_60": VOL + spread, "slope": -spread / (S0 - K), "spread": spread,
                "tenor": MKT.get("vol_tenor", 2.0),
                "source": f"Explicit skew spread of {spread:.0%} ({exc})", "live": False}


def downside_leg_bs(sigma, slope=0.0, S=S0, S_fix=S0, r=R, q=Q, T_=T):
    """European value of the protection leg N (S_T/S_fix - 1) 1{S_T < K}.

    It replicates as -N/S_fix [ Put(K) + (S_fix - K) DigitalPut(K) ].
    With a smile, the digital put is dP/dK = e^{-rT} N(-d2) + vega * dsigma/dK.
    """
    K = PDI_BARRIER * S_fix
    put = black_scholes(S, K, T_, r, q, sigma, "put")["price"]
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T_) / (sigma * sqrt(T_))
    d2 = d1 - sigma * sqrt(T_)
    vega_unit = S * exp(-q * T_) * exp(-0.5 * d1 * d1) / sqrt(2 * pi) * sqrt(T_)
    digital_put = exp(-r * T_) * norm_cdf(-d2) + vega_unit * slope
    return -NOTIONAL / S_fix * (put + (S_fix - K) * digital_put)


def downside_leg_mc(paths_S, S_fix=S0, r=R, only_alive=True):
    """Monte Carlo value of the same leg, with or without the no-call condition."""
    final = paths_S[:, -1]
    hit = final < PDI_BARRIER * S_fix
    if only_alive:
        never_called = (paths_S[:, :-1] < AUTOCALL_BARRIER * S_fix).all(axis=1)
        hit = hit & never_called
    leg = np.where(hit, NOTIONAL * (final / S_fix - 1.0), 0.0) * exp(-r * T)
    return float(leg.mean())


def skew_analysis(paths_S=None):
    """Coupon with and without skew on the 60% barrier.

    The autocall leg is the European leg restricted to paths never called. The
    skew correction is the European repricing difference, scaled by the share
    of the European leg that survives to maturity in the Monte Carlo.
    """
    if paths_S is None:
        paths_S = simulate_paths(N_OBS)
    sk = load_skew()
    leg_mc_alive = downside_leg_mc(paths_S, only_alive=True)
    leg_mc_euro = downside_leg_mc(paths_S, only_alive=False)
    survival = leg_mc_alive / leg_mc_euro
    leg_flat = downside_leg_bs(VOL)
    leg_skew = downside_leg_bs(sk["vol_60"], sk["slope"])
    adj = survival * (leg_skew - leg_flat)
    c_flat = coupon_from_paths(paths_S)
    c_skew = coupon_from_paths(paths_S, extra=adj)
    return {**sk, "leg_mc_euro": leg_mc_euro, "leg_bs_flat": leg_flat,
            "leg_bs_skew": leg_skew, "survival": survival, "adjustment": adj,
            "coupon_flat": c_flat, "coupon_skew": c_skew}


# =============================================================================
# 2. SALES SHEET
# =============================================================================
def sales_sheet(coupon_pa, paths_S=None, S_fix=S0):
    """Risk-neutral redemption profile of the autocall."""
    if paths_S is None:
        paths_S = simulate_paths(N_OBS)
    n = paths_S.shape[0]
    alive = np.ones(n, dtype=bool)
    rows = []
    for i in range(N_OBS):
        called = alive & (paths_S[:, i] >= AUTOCALL_BARRIER * S_fix)
        rows.append({"obs": i + 1, "t": (i + 1) / OBS_PER_YEAR,
                     "p_call": called.mean(),
                     "redemption": NOTIONAL * (1 + coupon_pa * (i + 1) / OBS_PER_YEAR)})
        alive &= ~called
    final = paths_S[:, -1] / S_fix
    p_protected = (alive & (final >= PDI_BARRIER)).mean()
    breached = alive & (final < PDI_BARRIER)
    p_loss = breached.mean()
    avg_loss_level = float(final[breached].mean()) if breached.any() else float("nan")
    exp_life = sum(r["p_call"] * r["t"] for r in rows) + (p_protected + p_loss) * T
    cumulative = np.cumsum([r["p_call"] for r in rows])
    scenarios = []
    for lvl in (1.30, 1.10, 1.00, 0.90, 0.75, 0.60, 0.59, 0.45, 0.30):
        if lvl >= AUTOCALL_BARRIER:
            pay = NOTIONAL * (1 + coupon_pa * T)
            txt = "called at the last observation, all coupons paid"
        elif lvl >= PDI_BARRIER:
            pay = NOTIONAL
            txt = "capital returned, no coupon"
        else:
            pay = NOTIONAL * lvl
            txt = "capital loss, indexed on the underlying"
        scenarios.append((lvl, pay, txt))
    return {"rows": rows, "cumulative": cumulative, "p_protected": p_protected,
            "p_loss": p_loss, "avg_loss_level": avg_loss_level,
            "expected_life": exp_life, "scenarios": scenarios, "coupon": coupon_pa}


# =============================================================================
# 3. WORST-OF ON EUROPEAN INDICES
# =============================================================================
def load_eu():
    """One year of daily closes: realised vols and correlation matrix.

    Yahoo does not publish index option chains, so realised volatility stands in
    for implied volatility here. That is a stated simplification.
    """
    names = list(EU_TICKERS)
    try:
        if offline():
            raise RuntimeError("offline mode (SP_OFFLINE)")
        import yfinance as yf
        import pandas as pd
        closes = {}
        for n in names:
            h = yf.Ticker(EU_TICKERS[n]).history(period="1y")["Close"]
            h.index = h.index.tz_localize(None).normalize()
            closes[n] = h
        df = pd.DataFrame(closes).dropna()
        if len(df) < 150:
            raise ValueError("not enough common dates")
        lr = np.diff(np.log(df.to_numpy()), axis=0)
        vols = lr.std(axis=0, ddof=1) * sqrt(TRADING_DAYS)
        corr = np.corrcoef(lr.T)
        src = (f"Yahoo Finance, 1-year realised vols and correlations to "
               f"{df.index[-1].date()} ({len(df)} common dates)")
        return {"names": names, "vols": vols, "corr": corr, "source": src, "live": True}
    except Exception as exc:
        return {"names": names, "vols": np.array(EU_FALLBACK["vols"]),
                "corr": np.array(EU_FALLBACK["corr"]),
                "source": f"Explicit vols and correlations ({exc})", "live": False}


def simulate_basket(vols, corr, divs, r=EUR_RATE, paths=PATHS, n_obs=N_OBS,
                    T_=T, seed=SEED):
    """Correlated GBM performances (S/S0) at the observation dates.

    Shape (paths, n_obs, n_assets). Cholesky on the correlation matrix,
    antithetic variates as in the single-underlying engine.
    """
    vols, divs = np.asarray(vols, float), np.asarray(divs, float)
    corr = np.asarray(corr, float)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:             # correlation 1: singular, use the eigen root
        w, v = np.linalg.eigh(corr)
        L = v * np.sqrt(np.clip(w, 0.0, None))
    rng = np.random.default_rng(seed)
    dt = T_ / n_obs
    z = rng.standard_normal((paths // 2, n_obs, len(vols)))
    z = np.concatenate([z, -z], axis=0) @ L.T
    incr = (r - divs - 0.5 * vols ** 2) * dt + vols * sqrt(dt) * z
    return np.exp(np.cumsum(incr, axis=1))


def flat_corr(n, rho):
    c = np.full((n, n), rho)
    np.fill_diagonal(c, 1.0)
    return c


def worst_of_analysis():
    eu = load_eu()
    divs = [EU_DIVIDENDS[n] for n in eu["names"]]
    perf = simulate_basket(eu["vols"], eu["corr"], divs)
    worst = perf.min(axis=2)
    c_wo = coupon_from_paths(worst, S_fix=1.0, disc_r=EUR_RATE)
    single = {}
    for j, n in enumerate(eu["names"]):
        single[n] = coupon_from_paths(perf[:, :, j], S_fix=1.0, disc_r=EUR_RATE)
    by_rho = []
    for rho in (0.50, 0.70, 0.85, 0.95, 1.00):
        p = simulate_basket(eu["vols"], flat_corr(3, rho), divs).min(axis=2)
        by_rho.append((rho, coupon_from_paths(p, S_fix=1.0, disc_r=EUR_RATE)))
    return {**eu, "divs": divs, "coupon_worst_of": c_wo, "coupon_single": single,
            "by_rho": by_rho}


# =============================================================================
# 4. ISSUER FUNDING SPREAD
# =============================================================================
def participation_funded(spread, S=S0, r=R, q=Q, sigma=VOL):
    """Capital-protected note when the zero-coupon is the issuer's own bond."""
    protection = NOTIONAL * exp(-(r + spread) * T)
    call = black_scholes(S, S, T, r, q, sigma, "call")["price"]
    return (NOTIONAL + FEE - protection) / (NOTIONAL * call / S)


def rc_coupon_funded(spread, S=S0, r=R, q=Q, sigma=VOL):
    put = black_scholes(S, S, T, r, q, sigma, "put")["price"]
    return (NOTIONAL + FEE + NOTIONAL * put / S) / exp(-(r + spread) * T) - NOTIONAL


def funding_analysis(paths_S=None):
    if paths_S is None:
        paths_S = simulate_paths(N_OBS)
    out = []
    for s in (0.0, 0.0030, FUNDING_SPREAD, 0.0100):
        out.append({"spread": s,
                    "participation": participation_funded(s),
                    "rc_coupon_pa": rc_coupon_funded(s) / NOTIONAL / T,
                    "autocall_coupon": coupon_from_paths(paths_S, disc_r=R + s)})
    return out


# =============================================================================
# 5. DELTA-HEDGING BACKTEST
# =============================================================================
OBS_TIMES = np.array([(i + 1) / OBS_PER_YEAR for i in range(N_OBS)])
OBS_DAYS = np.rint(OBS_TIMES * TRADING_DAYS).astype(int)


def autocall_value(S, t, coupon_pa, S_fix, sigma, r=R, q=Q, n_paths=HEDGE_PATHS_MC,
                   seed=SEED):
    """Value at time t of an autocall still alive, given the fixing level S_fix."""
    future = OBS_TIMES[OBS_TIMES > t + 1e-9]
    first = N_OBS - len(future)
    rng = np.random.default_rng(seed)
    dts = np.diff(np.concatenate([[t], future]))
    z = rng.standard_normal((n_paths // 2, len(future)))
    z = np.concatenate([z, -z], axis=0)
    paths = S * np.exp(np.cumsum((r - q - 0.5 * sigma ** 2) * dts
                                 + sigma * np.sqrt(dts) * z, axis=1))
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    cpo = coupon_pa * NOTIONAL / OBS_PER_YEAR
    for j, ti in enumerate(future):
        i = first + j
        disc = exp(-r * (ti - t))
        called = alive & (paths[:, j] >= AUTOCALL_BARRIER * S_fix)
        pv[called] += (NOTIONAL + cpo * (i + 1)) * disc
        alive &= ~called
        if i == N_OBS - 1:
            lvl = paths[:, j] / S_fix
            pv[alive & (lvl >= PDI_BARRIER)] += NOTIONAL * disc
            br = alive & (lvl < PDI_BARRIER)
            pv[br] += NOTIONAL * lvl[br] * disc
    return float(pv.mean())


def autocall_delta(S, t, coupon_pa, S_fix, sigma, r=R, q=Q):
    h = 0.01 * S
    up = autocall_value(S + h, t, coupon_pa, S_fix, sigma, r, q)
    dn = autocall_value(S - h, t, coupon_pa, S_fix, sigma, r, q)
    return (up - dn) / (2 * h)


def hedge_backtest(prices, coupon_pa, sigma, r=R, q=Q):
    """The desk sells the note at model value and delta-hedges it.

    Rebalanced every HEDGE_EVERY trading days and on each observation date.
    Cash earns r, the stock position earns the dividend yield q. The result is
    the desk's P&L in points of notional, in present value at issue.
    """
    prices = np.asarray(prices, float)
    S_fix = prices[0]
    dt = 1.0 / TRADING_DAYS
    v0 = autocall_value(S_fix, 0.0, coupon_pa, S_fix, sigma, r, q)
    delta = autocall_delta(S_fix, 0.0, coupon_pa, S_fix, sigma, r, q)
    cash = v0 - delta * S_fix
    obs_set = {int(d): i for i, d in enumerate(OBS_DAYS)}
    last = min(len(prices) - 1, int(OBS_DAYS[-1]))
    outcome, track = "still alive at the end of the data", []
    for k in range(1, last + 1):
        S = prices[k]
        cash = cash * exp(r * dt) + delta * prices[k - 1] * q * dt
        t = k * dt
        if k in obs_set:
            i = obs_set[k]
            if S >= AUTOCALL_BARRIER * S_fix:
                pay = NOTIONAL * (1 + coupon_pa * (i + 1) / OBS_PER_YEAR)
                pnl = cash + delta * S - pay
                outcome = f"called at observation {i + 1} ({t:.2f} years)"
                return _hedge_result(pnl, r, t, outcome, track, v0)
            if i == N_OBS - 1:
                lvl = S / S_fix
                pay = NOTIONAL if lvl >= PDI_BARRIER else NOTIONAL * lvl
                pnl = cash + delta * S - pay
                outcome = f"matured at {lvl:.0%} of the initial level"
                return _hedge_result(pnl, r, t, outcome, track, v0)
        if k % HEDGE_EVERY == 0 or k in obs_set:
            new_delta = autocall_delta(S, t, coupon_pa, S_fix, sigma, r, q)
            cash -= (new_delta - delta) * S
            delta = new_delta
            track.append((t, S / S_fix, delta))
    t = last * dt
    pnl = cash + delta * prices[last] - autocall_value(prices[last], t, coupon_pa,
                                                       S_fix, sigma, r, q)
    return _hedge_result(pnl, r, t, outcome + " (marked to model)", track, v0)


def _hedge_result(pnl, r, t, outcome, track, v0):
    return {"pnl": pnl * exp(-r * t), "outcome": outcome, "end_t": t,
            "track": track, "v0": v0}


def simulated_hedges(coupon_pa, n=60, sigma_true=VOL, seed=7):
    """Backtests on paths simulated with the model's own dynamics (model check).

    With sigma_true equal to the pricing vol the average P&L should sit close
    to zero; the spread around it is the discrete-hedging error.
    """
    rng = np.random.default_rng(seed)
    days = int(OBS_DAYS[-1])
    dt = 1.0 / TRADING_DAYS
    res = []
    for _ in range(n):
        z = rng.standard_normal(days)
        path = S0 * np.exp(np.concatenate([[0.0], np.cumsum(
            (R - Q - 0.5 * sigma_true ** 2) * dt + sigma_true * sqrt(dt) * z)]))
        res.append(hedge_backtest(path, coupon_pa, VOL)["pnl"])
    return np.array(res)


def historical_hedge(coupon_rule="solve"):
    """Live mode only: issue the note on SPY five years ago and hedge it.

    The pricing vol is the realised vol of the year before issue, so nothing
    after the issue date is used to price or hedge.
    """
    if offline():
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(sp.UNDERLYING).history(period="7y")["Close"]
        hist.index = hist.index.tz_localize(None)
        start = hist.index[-1] - timedelta(days=int(365.25 * T))
        pre = hist[hist.index < start].tail(TRADING_DAYS)
        post = hist[hist.index >= start]
        lr = np.diff(np.log(pre.to_numpy()))
        sigma = float(lr.std(ddof=1) * sqrt(TRADING_DAYS))
        paths = simulate_paths(N_OBS, sigma=sigma)
        c = coupon_from_paths(paths, fee=0.0)
        res = hedge_backtest(post.to_numpy(), c, sigma)
        return {**res, "issue": post.index[0].date(), "sigma": sigma, "coupon": c}
    except Exception as exc:
        print(f"[!] Historical hedge skipped ({exc}).")
        return None


# =============================================================================
# CONTROLS
# =============================================================================
def controls(paths_S=None):
    if paths_S is None:
        paths_S = simulate_paths(N_OBS)
    out = []
    mc = downside_leg_mc(paths_S, only_alive=False)
    bs = downside_leg_bs(VOL)
    out.append(("Protection leg: Monte Carlo vs put + digital replication", mc, bs, mc - bs))
    out.append(("Funding spread 0: participation vs base script",
                participation_funded(0.0), sp.solve_participation(),
                participation_funded(0.0) - sp.solve_participation()))
    vols, divs = [VOL] * 3, [Q] * 3
    one = coupon_from_paths(simulate_basket(vols, flat_corr(3, 1.0), divs, r=R).min(axis=2),
                            S_fix=1.0, disc_r=R)
    base = coupon_from_paths(simulate_paths(N_OBS, S=1.0), S_fix=1.0)
    out.append(("Worst-of at correlation 1 vs single underlying (coupon)", one, base, one - base))
    return out


# =============================================================================
# OUTPUT
# =============================================================================
def run_all():
    paths = simulate_paths(N_OBS)
    res = {"skew": skew_analysis(paths)}
    res["sales"] = sales_sheet(res["skew"]["coupon_skew"], paths)
    res["worst_of"] = worst_of_analysis()
    res["funding"] = funding_analysis(paths)
    c_nofee = coupon_from_paths(paths, fee=0.0)
    res["hedge_sim"] = simulated_hedges(c_nofee)
    res["hedge_sim_hi"] = simulated_hedges(c_nofee, sigma_true=VOL + 0.06, seed=8)
    res["hedge_hist"] = historical_hedge()
    res["hedge_coupon"] = c_nofee
    res["controls"] = controls(paths)
    return res


def summary_lines(res):
    sk, ss, wo = res["skew"], res["sales"], res["worst_of"]
    L = ["=" * 70, "DESK ANALYTICS - ATHENA AUTOCALL", "=" * 70, "",
         f"Market data      {MKT['source']}",
         f"Spot {S0:.2f} | rate {pct(R)} | dividend {pct(Q)} | ATM vol {pct(VOL)} "
         f"(read at {MKT['vol_tenor']:.1f}y, held flat to {T:.0f}y)", ""]

    L += ["1. Skew on the 60% protection barrier", "-" * 56,
          f"  Skew source                         {sk['source']}",
          f"  Vol at the 60% strike               {pct(sk['vol_60']):>12}  (ATM {pct(VOL)})",
          f"  Smile slope at 60%                  {sk['slope'] * S0:>12.4f}  vol per 100% of spot",
          f"  Protection leg, flat vol            {sk['leg_bs_flat']:>12.2f} points",
          f"  Protection leg, with skew           {sk['leg_bs_skew']:>12.2f} points",
          f"  Share of the leg alive at maturity  {pct(sk['survival']):>12}",
          f"  Skew adjustment to the note         {sk['adjustment']:>12.2f} points",
          f"  Coupon at par, flat vol             {pct(sk['coupon_flat']):>12}",
          f"  Coupon at par, with skew            {pct(sk['coupon_skew']):>12}", ""]

    L += ["2. Sales sheet (risk-neutral probabilities, coupon with skew)", "-" * 56]
    for r_, cum in zip(ss["rows"], ss["cumulative"]):
        if r_["obs"] in (1, 2, 4, 8, 12, 16, 20):
            L.append(f"  Obs {r_['obs']:>2} ({r_['t']:.2f}y)  call {pct(r_['p_call']):>7}  "
                     f"cumulative {pct(cum):>7}  pays {r_['redemption']:.2f}")
    L += [f"  Not called, capital protected       {pct(ss['p_protected']):>12}",
          f"  Capital loss at maturity            {pct(ss['p_loss']):>12}",
          f"  Average final level if loss         {pct(ss['avg_loss_level']):>12}",
          f"  Expected life                       {ss['expected_life']:>12.2f} years",
          "  Scenarios at maturity (not called before)"]
    for lvl, pay, txt in ss["scenarios"]:
        L.append(f"    final {lvl:>5.0%}  pays {pay:>7.2f}  {txt}")
    L.append("")

    L += ["3. Worst-of autocall: Euro Stoxx 50, CAC 40, DAX", "-" * 56,
          f"  Data                                {wo['source']}",
          f"  EUR rate (explicit)                 {pct(EUR_RATE):>12}"]
    for n, v, d in zip(wo["names"], wo["vols"], wo["divs"]):
        L.append(f"  {n:<14} vol {pct(v):>7}  dividend {pct(d):>6}  "
                 f"single coupon {pct(wo['coupon_single'][n]):>7}")
    c = wo["corr"]
    L.append(f"  Correlations  SX5E/CAC {c[0][1]:.2f}  SX5E/DAX {c[0][2]:.2f}  CAC/DAX {c[1][2]:.2f}")
    L.append(f"  Worst-of coupon at par              {pct(wo['coupon_worst_of']):>12}")
    for rho, cp in wo["by_rho"]:
        L.append(f"    flat correlation {rho:.2f}              {pct(cp):>12}")
    L.append("")

    L += ["4. Issuer funding spread", "-" * 56,
          "  spread   participation   RC coupon p.a.   autocall coupon"]
    for f in res["funding"]:
        L.append(f"  {f['spread'] * 1e4:>4.0f} bp   {pct(f['participation']):>12}   "
                 f"{pct(f['rc_coupon_pa']):>14}   {pct(f['autocall_coupon']):>15}")
    L.append("")

    hs, hh = res["hedge_sim"], res["hedge_sim_hi"]
    L += ["5. Delta hedging (desk short the note, priced without margin)", "-" * 56,
          f"  Coupon hedged                       {pct(res['hedge_coupon']):>12}",
          f"  Simulated, realised vol = {pct(VOL)}   mean {hs.mean():+.2f}  "
          f"std {hs.std(ddof=1):.2f} points ({len(hs)} paths)",
          f"  Simulated, realised vol = {pct(VOL + 0.06)}   mean {hh.mean():+.2f}  "
          f"std {hh.std(ddof=1):.2f} points ({len(hh)} paths)"]
    if res["hedge_hist"]:
        h = res["hedge_hist"]
        L.append(f"  SPY issued {h['issue']}, vol {pct(h['sigma'])}, coupon {pct(h['coupon'])}: "
                 f"{h['outcome']}, P&L {h['pnl']:+.2f} points")
    else:
        L.append("  Historical SPY backtest: live mode only (not run)")
    L.append("")

    L += ["6. Controls", "-" * 56]
    for label, a, b, d in res["controls"]:
        L += [f"  {label}", f"    {a:>14.6f}   {b:>14.6f}   diff  {d:>+12.2e}"]
    L += ["", "=" * 70]
    return L


def charts(res, folder="."):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    ss = res["sales"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar([r["t"] for r in ss["rows"]], [r["p_call"] for r in ss["rows"]], width=0.18,
           color="#1F4E8C")
    ax.set_xlabel("Observation (years)")
    ax.set_ylabel("Risk-neutral probability of call")
    ax.set_title("Autocall: when the note is expected to be called")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    files.append(os.path.join(folder, "out_call_probabilities.png"))
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    wo = res["worst_of"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot([r for r, _ in wo["by_rho"]], [c for _, c in wo["by_rho"]], marker="o",
            color="#1F4E8C")
    ax.set_xlabel("Flat correlation between the three indices")
    ax.set_ylabel("Coupon at par (p.a.)")
    ax.set_title("Worst-of autocall: the investor is short correlation")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    files.append(os.path.join(folder, "out_coupon_vs_correlation.png"))
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(res["hedge_sim"], bins=10, alpha=0.75, label=f"realised vol {VOL:.0%}",
            color="#1F4E8C")
    ax.hist(res["hedge_sim_hi"], bins=10, alpha=0.6, label=f"realised vol {VOL + 0.06:.0%}",
            color="#C0504D")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Desk P&L after delta hedging (points of notional)")
    ax.set_title("Delta-hedging the autocall: hedging error and the desk's long-vol position")
    ax.legend()
    fig.tight_layout()
    files.append(os.path.join(folder, "out_hedge_pnl.png"))
    fig.savefig(files[-1], dpi=150)
    plt.close(fig)
    return files


def write_reports(res, files):
    try:
        from docx import Document
        from docx.shared import Pt, Inches
    except ImportError:
        return []
    out = []

    # English analytics report
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Desk analytics: Athena autocall", 0)
    doc.add_paragraph("Diptakshi Sen | personal project, Python")
    if not MKT["live"]:
        doc.add_paragraph("This run used explicit parameters, not market data. A connected "
                          "run of the same code replaces them with Yahoo Finance inputs.")
    for f in files:
        doc.add_picture(f, width=Inches(6.0))
    p = doc.add_paragraph()
    r = p.add_run("\n".join(summary_lines(res)))
    r.font.name, r.font.size = "Consolas", Pt(8)
    out.append("desk_analytics_report.docx")
    doc.save(out[-1])

    # French one-page product sheet for sales
    ss, sk = res["sales"], res["skew"]
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(9.5)
    doc.add_heading("Fiche produit – Autocall Athena 5 ans", 1)
    doc.add_paragraph("Document pédagogique issu d'un projet personnel : ni une offre, ni un "
                      "conseil en investissement.")
    t = doc.add_table(rows=0, cols=2)
    t.style = "Light Grid Accent 1"
    terms = [
        ("Sous-jacent", f"{sp.UNDERLYING} (niveau initial {S0:.2f})"),
        ("Maturité", f"{T:.0f} ans, observations trimestrielles"),
        ("Rappel anticipé", f"si le sous-jacent ≥ {AUTOCALL_BARRIER:.0%} du niveau initial "
                            f"à une date d'observation"),
        ("Coupon", f"{ss['coupon']:.2%} par an, cumulé et versé au rappel"),
        ("Protection", f"capital remboursé à l'échéance si le sous-jacent ≥ "
                       f"{PDI_BARRIER:.0%} ; en dessous, perte égale à la baisse"),
        ("Prix d'émission", f"100 % (marge de structuration incluse : {FEE:.0f} points)"),
        ("Données", MKT["source"] if MKT["live"] else
         "Paramètres explicites (pas de données de marché pour cette exécution)"),
        ("Volatilité", f"{VOL:.2%} à la monnaie, {sk['vol_60']:.2%} au strike 60 % (skew)"),
    ]
    for k, v in terms:
        row = t.add_row().cells
        row[0].text, row[1].text = k, v
    doc.add_heading("Profil de remboursement (probabilités risque-neutres)", 2)
    for r_, cum in zip(ss["rows"], ss["cumulative"]):
        if r_["obs"] in (1, 4, 8, 12, 16, 20):
            doc.add_paragraph(f"Année {r_['t']:.2f} : rappel {r_['p_call']:.1%} "
                              f"(cumulé {cum:.1%}), remboursement {r_['redemption']:.2f} %",
                              style="List Bullet")
    doc.add_paragraph(f"Durée de vie espérée : {ss['expected_life']:.2f} ans. Probabilité de "
                      f"perte en capital : {ss['p_loss']:.1%}, niveau final moyen dans ce cas : "
                      f"{ss['avg_loss_level']:.0%}.")
    doc.add_heading("Scénarios à l'échéance (sans rappel anticipé)", 2)
    fr = {"called at the last observation, all coupons paid": "rappel à la dernière date, "
          "tous les coupons versés", "capital returned, no coupon": "capital remboursé, "
          "pas de coupon", "capital loss, indexed on the underlying": "perte en capital, "
          "indexée sur le sous-jacent"}
    for lvl, pay, txt in ss["scenarios"]:
        doc.add_paragraph(f"Niveau final {lvl:.0%} : {pay:.2f} % – {fr[txt]}", style="List Bullet")
    doc.add_heading("Risques", 2)
    doc.add_paragraph("Perte en capital possible sous 60 %. Risque de crédit de l'émetteur. "
                      "Liquidité secondaire non garantie. Les probabilités ci-dessus sont "
                      "risque-neutres : ce sont des prix, pas des prévisions.")
    out.append("fiche_produit_autocall_fr.docx")
    doc.save(out[-1])
    return out


def main():
    res = run_all()
    for line in summary_lines(res):
        print(line)
    files = charts(res)
    for f in write_reports(res, files) + files:
        print(f"written: {f}")


if __name__ == "__main__":
    main()
