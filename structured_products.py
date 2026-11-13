"""
Structured products pricing on live market data
Capital-protected note, reverse convertible, autocall

Diptakshi Sen - personal project, 2026

The pricing inputs are not assumed. They are pulled from the market.

    Spot                SPY (S&P 500 ETF), last close
    Risk-free rate      ^FVX, US 5-year Treasury yield (same tenor as the note)
    Dividend yield      implied, backed out of put-call parity on the
                        at-the-money option chain
    Volatility          implied, interpolated at the money across the option
                        chain, with the observed term structure
    Realised volatility computed on one year of history, for comparison

Products
    1. Capital-protected note   ZC + call  -> solve for the participation rate
    2. Reverse convertible      ZC - put   -> solve for the coupon
    3. Autocall (Athena)        path-dependent payoff, Monte Carlo
                                -> solve for the coupon

Controls: vanilla Monte Carlo against Black-Scholes, put-call parity, and a
return to par after solving. The Greeks of the structure are obtained by bump
and revalue.

Run:  pip install yfinance numpy python-docx
      python structured_products_market_en.py
"""

import os
from math import exp, log, sqrt, erf
from datetime import datetime
import numpy as np

# --------------------------------------------------------- 0. MARKET DATA ---
UNDERLYING = "SPY"          # S&P 500 ETF, deep and liquid option chain
RATE_TICKER = "^FVX"        # US 5-year yield, matches the 5-year maturity of the note

NOTIONAL = 100.0    # face value of the note
FEE = 2.0           # structuring margin and distribution fee, in points
T = 5.0             # maturity of the note, in years

PATHS = 100_000     # Monte Carlo paths
SEED = 12

OBS_PER_YEAR = 4            # quarterly observations
AUTOCALL_BARRIER = 1.00     # called if S >= 100% of the initial level
PDI_BARRIER = 0.60          # capital protected at maturity if S >= 60%


def _atm_interp(df, spot, column="impliedVolatility"):
    """At-the-money value, interpolated between the two strikes around spot."""
    d = df.dropna(subset=[column]).sort_values("strike")
    d = d[(d[column] > 0.01) & (d[column] < 2.0)]
    if len(d) < 2:
        return None
    below = d[d["strike"] <= spot].tail(1)
    above = d[d["strike"] >= spot].head(1)
    if len(below) == 0 or len(above) == 0:
        return float(d.iloc[(d["strike"] - spot).abs().argsort().iloc[0]][column])
    k0, v0 = float(below["strike"].iloc[0]), float(below[column].iloc[0])
    k1, v1 = float(above["strike"].iloc[0]), float(above[column].iloc[0])
    if k1 == k0:
        return v0
    w = (spot - k0) / (k1 - k0)
    return v0 + w * (v1 - v0)


def _mid(row):
    bid, ask = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
    if bid > 0 and ask > 0:
        return 0.5 * (bid + ask)
    return float(row.get("lastPrice", 0) or 0)


def load_market():
    """Pull spot, rate, implied dividend and implied volatility from the market.

    The dividend yield is backed out of at-the-money put-call parity:
        q = -(1/T) ln[ (C - P + K e^-rT) / S ]
    which avoids assuming it. If the download fails, the function falls back on
    an explicit parameter set, flagged as such in the report.
    """
    try:
        if os.environ.get("SP_OFFLINE"):
            raise RuntimeError("offline mode (SP_OFFLINE)")
        import yfinance as yf

        tk = yf.Ticker(UNDERLYING)
        hist = tk.history(period="1y")
        if hist.empty:
            raise ValueError("price history unavailable")
        spot = float(hist["Close"].iloc[-1])
        logret = np.diff(np.log(hist["Close"].to_numpy()))
        realised = float(logret.std(ddof=1) * sqrt(252))

        rate_hist = yf.Ticker(RATE_TICKER).history(period="1mo")
        r = float(rate_hist["Close"].iloc[-1]) / 100.0

        today = datetime.utcnow().date()
        expiries = []
        for e in tk.options:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            tau = (d - today).days / 365.0
            if tau > 0.08:
                expiries.append((tau, e))
        if not expiries:
            raise ValueError("no usable expiry")
        expiries.sort()

        # at-the-money volatility term structure
        term, q_est = [], []
        sample = [expiries[0], expiries[len(expiries) // 2], expiries[-1]]
        for tau, e in sample:
            ch = tk.option_chain(e)
            iv_c = _atm_interp(ch.calls, spot)
            iv_p = _atm_interp(ch.puts, spot)
            ivs = [v for v in (iv_c, iv_p) if v]
            if ivs:
                term.append((tau, float(np.mean(ivs)), e))

            # implied dividend from put-call parity at the nearest strike
            try:
                calls = ch.calls.copy()
                puts = ch.puts.copy()
                k = float(calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]["strike"])
                c = _mid(calls[calls["strike"] == k].iloc[0])
                p = _mid(puts[puts["strike"] == k].iloc[0])
                if c > 0 and p > 0:
                    inner = (c - p + k * exp(-r * tau)) / spot
                    if inner > 0:
                        q_est.append(-log(inner) / tau)
            except Exception:
                pass

        if not term:
            raise ValueError("implied volatility unavailable")

        sigma = term[-1][1]                      # longest listed expiry
        vol_tenor = term[-1][0]                  # flat extrapolation beyond this tenor
        q = float(np.median(q_est)) if q_est else 0.0
        q = min(max(q, 0.0), 0.08)               # sanity bound

        src = (f"Yahoo Finance, {UNDERLYING} as of {hist.index[-1].date()}, "
               f"rate {RATE_TICKER}")
        return {"spot": spot, "r": r, "q": q, "sigma": sigma, "realised": realised,
                "term": term, "source": src, "live": True,
                "vol_tenor": vol_tenor}

    except Exception as exc:
        print(f"[!] Market data unavailable ({exc}). Explicit parameters used.")
        return {"spot": 100.0, "r": 0.030, "q": 0.020, "sigma": 0.180,
                "realised": 0.165, "term": [(1.0, 0.170, "n/a"), (2.0, 0.180, "n/a")],
                "vol_tenor": 2.0,
                "source": "Explicit parameters (no market data available)",
                "live": False}


MKT = load_market()
S0 = MKT["spot"]
R = MKT["r"]
Q = MKT["q"]
VOL = MKT["sigma"]


# ------------------------------------------------------- VANILLA BUILDING ---
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(S, K, T_, r, q, sigma, kind="call"):
    """Price and Greeks of a European option (Black-Scholes-Merton)."""
    if T_ <= 0:
        payoff = max(S - K, 0) if kind == "call" else max(K - S, 0)
        return {"price": payoff, "delta": 0.0, "vega": 0.0}
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T_) / (sigma * sqrt(T_))
    d2 = d1 - sigma * sqrt(T_)
    disc, div = exp(-r * T_), exp(-q * T_)
    if kind == "call":
        price = S * div * norm_cdf(d1) - K * disc * norm_cdf(d2)
        delta = div * norm_cdf(d1)
    else:
        price = K * disc * norm_cdf(-d2) - S * div * norm_cdf(-d1)
        delta = -div * norm_cdf(-d1)
    pdf = exp(-0.5 * d1 * d1) / sqrt(2 * np.pi)
    vega = S * div * pdf * sqrt(T_) / 100      # per volatility point
    return {"price": price, "delta": delta, "vega": vega}


def zc(T_, r=R):
    """Zero-coupon price: the cost of capital protection."""
    return exp(-r * T_)


# ------------------------------------------------------------ SIMULATION ----
def simulate_paths(n_obs, paths=PATHS, S=S0, r=R, q=Q, sigma=VOL, T_=T, seed=SEED):
    """Underlying paths at the observation dates, under the risk-neutral
    measure. Antithetic variates to reduce variance."""
    rng = np.random.default_rng(seed)
    dt = T_ / n_obs
    z = rng.standard_normal((paths // 2, n_obs))
    z = np.concatenate([z, -z], axis=0)
    incr = (r - q - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z
    return S * np.exp(np.cumsum(incr, axis=1))


# ---------------------------------- 1. CAPITAL-PROTECTED NOTE (ZC + CALL) ----
def capital_protected(participation, S=S0, r=R, q=Q, sigma=VOL, T_=T,
                      notional=NOTIONAL, fee=FEE):
    """Value of a capital-protected note: ZC + participation x ATM call."""
    protection = notional * zc(T_, r)
    call = black_scholes(S, S, T_, r, q, sigma, "call")["price"]
    upside = participation * notional * call / S
    return protection + upside - fee


def solve_participation(target=NOTIONAL, **kw):
    """Participation rate that allows the note to be issued at par."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if capital_protected(mid, **kw) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ------------------------------------------ 2. REVERSE CONVERTIBLE (ZC - PUT) --
def reverse_convertible(coupon, S=S0, r=R, q=Q, sigma=VOL, T_=T,
                        notional=NOTIONAL, fee=FEE):
    """High-coupon note: the investor is short an at-the-money put."""
    redemption = (notional + coupon) * zc(T_, r)
    put = black_scholes(S, S, T_, r, q, sigma, "put")["price"]
    return redemption - notional * put / S - fee


def solve_coupon_rc(target=NOTIONAL, **kw):
    lo, hi = 0.0, 300.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if reverse_convertible(mid, **kw) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# --------------------------------------------------- 3. AUTOCALL (MONTE CARLO) --
def autocall_payoff(paths_S, coupon_pa, S_fix=S0, r=R, T_=T, notional=NOTIONAL, fee=0.0):
    """Value of an Athena autocall from paths that have already been simulated.

    The barriers are fixed at the level observed at issue (S_fix), not at the
    current spot. That is what makes a spot shock move the price, and therefore
    what makes the delta exist at all.

    At each observation: called if S >= the call barrier, redeemed at par plus
    the accrued coupons. At maturity with no call: capital returned at par if
    S >= the protection barrier, otherwise indexed on the performance.
    """
    n_paths, n_obs = paths_S.shape
    dt = T_ / n_obs
    alive = np.ones(n_paths, dtype=bool)
    pv = np.zeros(n_paths)
    coupon_per_obs = coupon_pa * notional / OBS_PER_YEAR

    for i in range(n_obs):
        t = (i + 1) * dt
        called = alive & (paths_S[:, i] >= AUTOCALL_BARRIER * S_fix)
        if i == n_obs - 1:
            final = alive & ~called
            protected = final & (paths_S[:, i] >= PDI_BARRIER * S_fix)
            breached = final & ~protected
            pv[called] += (notional + coupon_per_obs * (i + 1)) * exp(-r * t)
            pv[protected] += notional * exp(-r * t)
            pv[breached] += notional * (paths_S[breached, i] / S_fix) * exp(-r * t)
        else:
            pv[called] += (notional + coupon_per_obs * (i + 1)) * exp(-r * t)
        alive = alive & ~called

    return float(pv.mean()) - fee, float(pv.std(ddof=1) / sqrt(n_paths))


def autocall_price(coupon_pa, S=S0, S_fix=S0, r=R, q=Q, sigma=VOL, T_=T,
                   notional=NOTIONAL, paths=PATHS, seed=SEED, fee=0.0):
    """Autocall price: simulate, then evaluate the payoff."""
    n_obs = int(OBS_PER_YEAR * T_)
    paths_S = simulate_paths(n_obs, paths, S, r, q, sigma, T_, seed)
    return autocall_payoff(paths_S, coupon_pa, S_fix, r, T_, notional, fee)


def solve_coupon_autocall(target=NOTIONAL, fee=FEE, S=S0, S_fix=S0, r=R, q=Q,
                          sigma=VOL, T_=T, paths=PATHS, seed=SEED):
    """Annual coupon that prices the note at par, margin included.

    The paths do not depend on the coupon, so they are simulated once and the
    bisection runs on the payoff alone. The same draw is reused at every
    iteration, which is what makes the solve stable.
    """
    n_obs = int(OBS_PER_YEAR * T_)
    paths_S = simulate_paths(n_obs, paths, S, r, q, sigma, T_, seed)
    lo, hi = 0.0, 0.60
    for _ in range(40):
        mid = (lo + hi) / 2
        price, _ = autocall_payoff(paths_S, mid, S_fix, r, T_, NOTIONAL, fee)
        if price < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------- GREEKS BY BUMP AND REVALUE ---
def autocall_greeks(coupon_pa, bump_spot=1.0, bump_vol=0.01):
    """Delta and vega of an autocall by bump and revalue, as done on a desk."""
    base, _ = autocall_price(coupon_pa)
    up, _ = autocall_price(coupon_pa, S=S0 + bump_spot, S_fix=S0)
    down, _ = autocall_price(coupon_pa, S=S0 - bump_spot, S_fix=S0)
    vol_up, _ = autocall_price(coupon_pa, sigma=VOL + bump_vol)
    return {
        "price": base,
        "delta": (up - down) / (2 * bump_spot),
        "gamma": (up - 2 * base + down) / (bump_spot ** 2),
        "vega": (vol_up - base) / (bump_vol * 100),      # per volatility point
    }


# ------------------------------------------------------------- VALIDATION ---
def validate():
    """Independent controls, run before any quote is produced."""
    out = []

    # 1. Vanilla Monte Carlo against Black-Scholes
    n_obs = int(OBS_PER_YEAR * T)
    paths_S = simulate_paths(n_obs)
    mc_call = float(np.maximum(paths_S[:, -1] - S0, 0).mean() * exp(-R * T))
    bs_call = black_scholes(S0, S0, T, R, Q, VOL, "call")["price"]
    out.append(("ATM call, Monte Carlo vs Black-Scholes", mc_call, bs_call,
                mc_call - bs_call))

    # 2. Put-call parity
    call = black_scholes(S0, S0, T, R, Q, VOL, "call")["price"]
    put = black_scholes(S0, S0, T, R, Q, VOL, "put")["price"]
    lhs = call - put
    rhs = S0 * exp(-Q * T) - S0 * exp(-R * T)
    out.append(("Put-call parity: C - P vs Se^-qT - Ke^-rT", lhs, rhs, lhs - rhs))

    # 3. Protected note decomposition: the price must come back to par
    p = solve_participation()
    out.append(("Protected note back at par after solving", capital_protected(p),
                NOTIONAL, capital_protected(p) - NOTIONAL))

    return out


# ----------------------------------------------------------------- OUTPUT ---
def pct(x):
    return f"{x:.2%}"


def summary_lines():
    L = []
    L.append("=" * 70)
    L.append("STRUCTURED PRODUCTS PRICING")
    L.append("=" * 70)
    L.append("")
    L.append("Market parameters")
    L.append("-" * 56)
    L.append(f"  Source                              {MKT['source']}")
    L.append(f"  Underlying (index level)            {S0:>12.2f}")
    L.append(f"  Risk-free rate                      {pct(R):>12}")
    L.append(f"  Dividend yield                      {pct(Q):>12}")
    L.append(f"  Implied volatility (market)         {pct(VOL):>12}")
    L.append(f"  Volatility read at tenor            {MKT['vol_tenor']:>12.2f} years, held flat")
    L.append(f"  Realised volatility, 1 year         {pct(MKT['realised']):>12}")
    L.append(f"  Volatility premium                  {pct(VOL - MKT['realised']):>12}")
    L.append(f"  Maturity                            {T:>12.1f} years")
    L.append(f"  Margin and fees                     {FEE:>12.2f} points")
    L.append("")
    L.append("  At-the-money volatility term structure")
    for tau, iv, exp_ in MKT["term"]:
        L.append(f"    {tau:>5.2f} year(s)  {pct(iv):>8}   expiry {exp_}")
    L.append("")

    L.append("1. 100% capital-protected note (ZC + ATM call)")
    L.append("-" * 56)
    protection = NOTIONAL * zc(T)
    call = black_scholes(S0, S0, T, R, Q, VOL, "call")["price"]
    part = solve_participation()
    L.append(f"  Cost of protection (zero-coupon)    {protection:>12.2f}")
    L.append(f"  Option budget available             {NOTIONAL - protection - FEE:>12.2f}")
    L.append(f"  ATM call price                      {call:>12.2f}")
    L.append(f"  Participation rate at par           {pct(part):>12}")
    L.append("")
    L.append("  Sensitivity of the participation rate")
    for r_ in (0.01, 0.02, 0.03, 0.04):
        pr = solve_participation(r=r_)
        L.append(f"    rate {pct(r_):>6}                        {pct(pr):>12}")
    for v_ in (0.14, 0.18, 0.22, 0.26):
        pv_ = solve_participation(sigma=v_)
        L.append(f"    volatility {pct(v_):>6}                  {pct(pv_):>12}")
    L.append("")

    L.append("2. Reverse convertible (ZC - ATM put)")
    L.append("-" * 56)
    cpn = solve_coupon_rc()
    put = black_scholes(S0, S0, T, R, Q, VOL, "put")["price"]
    L.append(f"  Price of the ATM put sold           {put:>12.2f}")
    L.append(f"  Total coupon at par                 {cpn:>12.2f} points")
    L.append(f"  Annualised                          {pct(cpn / NOTIONAL / T):>12}")
    L.append("")

    L.append("3. Athena autocall, quarterly observations")
    L.append("-" * 56)
    L.append(f"  Call barrier                        {pct(AUTOCALL_BARRIER):>12}")
    L.append(f"  Protection barrier at maturity      {pct(PDI_BARRIER):>12}")
    cpn_ac = solve_coupon_autocall()
    price, se = autocall_price(cpn_ac, fee=FEE)
    L.append(f"  Annual coupon at par                {pct(cpn_ac):>12}")
    L.append(f"  Resulting price (margin included)   {price:>12.2f}")
    L.append(f"  Monte Carlo standard error          {se:>12.4f}")
    L.append("")
    L.append("  Sensitivity of the quoted coupon")
    for v_ in (0.14, 0.18, 0.22, 0.26):
        c_ = solve_coupon_autocall(sigma=v_)
        L.append(f"    volatility {pct(v_):>6}                  {pct(c_):>12}")
    for r_ in (0.01, 0.02, 0.03, 0.04):
        c_ = solve_coupon_autocall(r=r_)
        L.append(f"    rate {pct(r_):>6}                        {pct(c_):>12}")
    L.append("")

    L.append("4. Autocall Greeks (bump and revalue)")
    L.append("-" * 56)
    g = autocall_greeks(cpn_ac)
    L.append(f"  Price excluding margin              {g['price']:>12.2f}")
    L.append(f"  Delta                               {g['delta']:>12.4f}")
    L.append(f"  Gamma                               {g['gamma']:>12.4f}")
    L.append(f"  Vega (per volatility point)         {g['vega']:>12.4f}")
    L.append("")

    L.append("5. Controls")
    L.append("-" * 56)
    for label, a, b, diff in validate():
        L.append(f"  {label}")
        L.append(f"    {a:>14.6f}   {b:>14.6f}   diff  {diff:>+12.2e}")
    L.append("")
    L.append("=" * 70)
    return L


def write_docx(path="structured_products_report.docx"):
    try:
        from docx import Document
        from docx.shared import Pt, Cm, RGBColor
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        print("[!] python-docx missing - Word report skipped. pip install python-docx")
        return None

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Calibri", Pt(10.5)
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Cm(1.9)
        sec.left_margin = sec.right_margin = Cm(2.1)

    def shade(par, colour):
        el = OxmlElement("w:shd")
        el.set(qn("w:val"), "clear")
        el.set(qn("w:fill"), colour)
        par._p.get_or_add_pPr().append(el)

    def h1(t):
        p = doc.add_paragraph()
        r = p.add_run(t)
        r.bold, r.font.size = True, Pt(16)
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
        p.paragraph_format.space_after = Pt(2)

    def h2(t):
        p = doc.add_paragraph()
        r = p.add_run(t)
        r.bold, r.font.size = True, Pt(12)
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(13), Pt(4)
        pb = OxmlElement("w:pBdr")
        bt = OxmlElement("w:bottom")
        bt.set(qn("w:val"), "single")
        bt.set(qn("w:sz"), "6")
        bt.set(qn("w:color"), "888888")
        pb.append(bt)
        p._p.get_or_add_pPr().append(pb)

    def para(t, size=10.5, after=6):
        p = doc.add_paragraph()
        r = p.add_run(t)
        r.font.size = Pt(size)
        p.paragraph_format.space_after = Pt(after)
        return p

    def bullet(label, t):
        p = doc.add_paragraph(style="List Bullet")
        if label:
            r = p.add_run(label + " ")
            r.bold, r.font.size = True, Pt(10.5)
        r = p.add_run(t)
        r.font.size = Pt(10.5)
        p.paragraph_format.space_after = Pt(3)

    def mono(lines):
        for line in lines:
            p = doc.add_paragraph()
            r = p.add_run(line if line else " ")
            r.font.name, r.font.size = "Consolas", Pt(7.5)
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            pf = p.paragraph_format
            pf.space_before = pf.space_after = Pt(0)
            pf.line_spacing = 1.0
            shade(p, "F4F4F4")

    part = solve_participation()
    cpn_rc = solve_coupon_rc()
    cpn_ac = solve_coupon_autocall()
    g = autocall_greeks(cpn_ac)

    h1("Structured Products Pricing")
    para("Diptakshi Sen  |  personal project, Python  |  dependency: numpy  |  "
         "diptakshi.sen@kedgebs.com", size=9.5, after=10)

    h2("Purpose")
    para("Assemble the building blocks of a valuation library - zero-coupon bonds, vanilla "
         "options with Greeks, a rate curve - into structured products, and answer the question "
         "a quote actually asks: which participation rate, or which coupon, allows the note to "
         "be issued at par once the margin is built in.")

    h2("Products covered")
    bullet("100% capital-protected note -", "a zero-coupon bond for the protection, an "
           "at-the-money call for the upside. The participation rate is solved for a price at par.")
    bullet("Reverse convertible -", "the investor sells an at-the-money put; the coupon is "
           "solved for a price at par.")
    bullet("Athena autocall -", "quarterly observations, call barrier at 100%, capital protected "
           "at 60% at maturity. The payoff is path-dependent, so it is valued by Monte Carlo with "
           "antithetic variates; the annual coupon is solved by bisection.")
    bullet("Greeks -", "delta, gamma and vega of the autocall obtained by bump and revalue, since "
           "a path-dependent payoff has no closed form.")

    h2("Results")
    bullet("Capital-protected note -", f"the protection costs {NOTIONAL * zc(T):.2f} points out "
           f"of {NOTIONAL:.0f}, which leaves {NOTIONAL - NOTIONAL * zc(T) - FEE:.2f} points of "
           f"option budget and gives a participation rate of {pct(part)}.")
    bullet("Reverse convertible -", f"a total coupon of {cpn_rc:.2f} points, or "
           f"{pct(cpn_rc / NOTIONAL / T)} a year, in exchange for the put sold.")
    bullet("Autocall -", f"an annual coupon of {pct(cpn_ac)} for an issue at par, delta "
           f"{g['delta']:.4f} and vega {g['vega']:.4f} per volatility point.")

    h2("Two points of method")
    bullet("", "The participation rate is not a marketing number, it is a residual: it depends on "
           "what is left after the protection has been bought. When rates fall, the zero-coupon "
           "costs more, the option budget shrinks and the participation rate drops. The "
           "sensitivity table shows this explicitly.")
    bullet("", "The autocall makes the investor a seller of volatility: the coupon rises with "
           "implied volatility, because the 60% protection barrier becomes riskier. That is the "
           "relationship a desk has to explain to a client when the quote moves from one day to "
           "the next.")

    if MKT["live"]:
        note = (f"On the data. Source: {MKT['source']}. The spot is the last close, the risk-free "
                f"rate is the 5-year yield (same tenor as the note), the volatility is read at the money off the option "
                f"chain, and the dividend yield is backed out of put-call parity rather than "
                f"assumed. Volatility assumption: the at-the-money volatility of the longest listed "
                f"expiry ({MKT['vol_tenor']:.1f} years) is held flat to the {T:.0f}-year maturity, and "
                f"the same volatility is used at every strike. Skew is therefore ignored here; "
                f"desk_analytics.py measures what it does to the autocall coupon.")
    else:
        note = (f"On the data. The program queries Yahoo Finance ({UNDERLYING} for the spot and "
                f"the option chain, {RATE_TICKER} for the rate) and backs the dividend yield out "
                f"of put-call parity. This particular run had no network access: the figures "
                f"above come from the explicit fallback parameters (spot {S0:.0f}, rate {R:.2%}, "
                f"dividend {Q:.2%}, volatility {VOL:.2%}), and the program flags this itself at "
                f"the top of its output. A connected run replaces those values with no other "
                f"change. Volatility assumption: the at-the-money volatility of the longest listed expiry "
                f"({MKT['vol_tenor']:.1f} years) is held flat to the {T:.0f}-year maturity and used at "
                f"every strike, so skew is ignored here; desk_analytics.py measures its effect.")
    p = para(note, size=10)
    shade(p, "FFF8E1")

    h2("Program output")
    mono(summary_lines())

    doc.save(path)
    return path


def main():
    for line in summary_lines():
        print(line)
    path = write_docx()
    if path:
        print(f"Word report written: {path}")


if __name__ == "__main__":
    main()
