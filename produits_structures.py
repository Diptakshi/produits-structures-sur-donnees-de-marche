"""
Cotation de produits structures sur donnees de marche
Notes a capital protege, reverse convertible, autocall

Diptakshi Sen - projet personnel, 2026

Les parametres de cotation ne sont pas poses : ils sont extraits du marche.

    Spot                SPY (ETF S&P 500), dernier cours
    Taux sans risque    ^FVX, rendement du 5 ans americain (meme maturite que la note)
    Dividende           implicite, deduit de la parite call-put sur la chaine
                        d'options a la monnaie
    Volatilite          implicite, interpolee a la monnaie sur la chaine
                        d'options, avec la structure par terme observee
    Volatilite realisee calculee sur un an d'historique, pour comparaison

Produits traites
    1. Note a capital protege   ZC + call  -> on resout le taux de participation
    2. Reverse convertible      ZC - put   -> on resout le coupon
    3. Autocall (Athena)        payoff chemin-dependant, Monte-Carlo
                                -> on resout le coupon

Controles : Monte-Carlo vanille contre Black-Scholes, parite call-put, et
retour au pair apres resolution. Les grecques du structure sont obtenues par
choc et revalorisation.

Lancer :  pip install yfinance numpy python-docx
          python structured_products_market.py
"""

import os
from math import exp, log, sqrt, erf
from datetime import datetime
import numpy as np

# ------------------------------------------------------ 0. DONNEES DE MARCHE --
UNDERLYING = "SPY"          # ETF S&P 500, chaine d'options tres liquide
RATE_TICKER = "^FVX"        # rendement 5 ans americain, aligne sur la maturite de la note

NOTIONAL = 100.0    # valeur nominale de la note
FEE = 2.0           # marge de structuration et frais de distribution, en points
T = 5.0             # maturite de la note, en annees

PATHS = 100_000     # trajectoires Monte-Carlo
SEED = 12

OBS_PER_YEAR = 4            # observations trimestrielles
AUTOCALL_BARRIER = 1.00     # rappel si S >= 100 % du niveau initial
PDI_BARRIER = 0.60          # protection du capital a l'echeance si S >= 60 %


def _atm_interp(df, spot, column="impliedVolatility"):
    """Valeur a la monnaie, interpolee entre les deux strikes encadrant le spot."""
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
    """Extrait spot, taux, dividende implicite et volatilite implicite du marche.

    Le dividende est deduit de la parite call-put a la monnaie :
        q = -(1/T) ln[ (C - P + K e^-rT) / S ]
    ce qui evite de le supposer. En cas d'echec du telechargement, on retombe sur
    un jeu de parametres explicite, signale comme tel dans le rapport.
    """
    try:
        if os.environ.get("SP_OFFLINE"):
            raise RuntimeError("offline mode (SP_OFFLINE)")
        import yfinance as yf

        tk = yf.Ticker(UNDERLYING)
        hist = tk.history(period="1y")
        if hist.empty:
            raise ValueError("historique indisponible")
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
            raise ValueError("aucune echeance exploitable")
        expiries.sort()

        # structure par terme de la volatilite a la monnaie
        term, q_est = [], []
        sample = [expiries[0], expiries[len(expiries) // 2], expiries[-1]]
        for tau, e in sample:
            ch = tk.option_chain(e)
            iv_c = _atm_interp(ch.calls, spot)
            iv_p = _atm_interp(ch.puts, spot)
            ivs = [v for v in (iv_c, iv_p) if v]
            if ivs:
                term.append((tau, float(np.mean(ivs)), e))

            # dividende implicite par parite call-put au strike le plus proche
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
            raise ValueError("volatilite implicite indisponible")

        sigma = term[-1][1]                      # echeance la plus longue cotee
        vol_tenor = term[-1][0]                  # extrapolation plate au-dela de cette echeance
        q = float(np.median(q_est)) if q_est else 0.0
        q = min(max(q, 0.0), 0.08)               # garde-fou

        src = (f"Yahoo Finance, {UNDERLYING} au {hist.index[-1].date()}, "
               f"taux {RATE_TICKER}")
        return {"spot": spot, "r": r, "q": q, "sigma": sigma, "realised": realised,
                "term": term, "source": src, "live": True,
                "vol_tenor": vol_tenor}

    except Exception as exc:
        print(f"[!] Donnees de marche indisponibles ({exc}). Parametres explicites utilises.")
        return {"spot": 100.0, "r": 0.030, "q": 0.020, "sigma": 0.180,
                "realised": 0.165, "term": [(1.0, 0.170, "n/a"), (2.0, 0.180, "n/a")],
                "vol_tenor": 2.0,
                "source": "Paramètres explicites (aucune donnée de marché disponible)",
                "live": False}


MKT = load_market()
S0 = MKT["spot"]
R = MKT["r"]
Q = MKT["q"]
VOL = MKT["sigma"]


# --------------------------------------------------------- OUTILS VANILLES --
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes(S, K, T_, r, q, sigma, kind="call"):
    """Prix et grecques d'une option europeenne (Black-Scholes-Merton)."""
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
    vega = S * div * pdf * sqrt(T_) / 100      # par point de volatilite
    return {"price": price, "delta": delta, "vega": vega}


def zc(T_, r=R):
    """Prix d'un zero-coupon : le cout de la protection du capital."""
    return exp(-r * T_)


# ------------------------------------------------------------- SIMULATION ---
def simulate_paths(n_obs, paths=PATHS, S=S0, r=R, q=Q, sigma=VOL, T_=T, seed=SEED):
    """Trajectoires du sous-jacent aux dates d'observation, sous la probabilite
    risque-neutre. Variables antithetiques pour reduire la variance."""
    rng = np.random.default_rng(seed)
    dt = T_ / n_obs
    z = rng.standard_normal((paths // 2, n_obs))
    z = np.concatenate([z, -z], axis=0)
    incr = (r - q - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z
    return S * np.exp(np.cumsum(incr, axis=1))


# ------------------------------------- 1. NOTE A CAPITAL PROTEGE (ZC + CALL) --
def capital_protected(participation, S=S0, r=R, q=Q, sigma=VOL, T_=T,
                      notional=NOTIONAL, fee=FEE):
    """Valeur d'une note a capital protege : ZC + participation x call ATM."""
    protection = notional * zc(T_, r)
    call = black_scholes(S, S, T_, r, q, sigma, "call")["price"]
    upside = participation * notional * call / S
    return protection + upside - fee


def solve_participation(target=NOTIONAL, **kw):
    """Taux de participation qui permet d'emettre la note au pair."""
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if capital_protected(mid, **kw) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------- 2. REVERSE CONVERTIBLE (ZC - PUT) --
def reverse_convertible(coupon, S=S0, r=R, q=Q, sigma=VOL, T_=T,
                        notional=NOTIONAL, fee=FEE):
    """Note a coupon eleve : l'investisseur vend un put au sous-jacent."""
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


# ------------------------------------------------- 3. AUTOCALL (MONTE-CARLO) --
def autocall_payoff(paths_S, coupon_pa, S_fix=S0, r=R, T_=T, notional=NOTIONAL, fee=0.0):
    """Valeur d'un autocall Athena a partir de trajectoires deja simulees.

    Les barrieres sont fixees au niveau constate a l'emission (S_fix), pas au
    spot courant : c'est ce qui fait qu'un choc de spot deplace le prix, et donc
    que le delta existe.

    A chaque observation : rappel si S >= barriere de rappel, remboursement au
    pair majore des coupons cumules. A l'echeance sans rappel : capital rendu au
    pair si S >= barriere de protection, sinon indexe sur la performance.
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
    """Prix d'un autocall : simulation puis evaluation du payoff."""
    n_obs = int(OBS_PER_YEAR * T_)
    paths_S = simulate_paths(n_obs, paths, S, r, q, sigma, T_, seed)
    return autocall_payoff(paths_S, coupon_pa, S_fix, r, T_, notional, fee)


def solve_coupon_autocall(target=NOTIONAL, fee=FEE, S=S0, S_fix=S0, r=R, q=Q,
                          sigma=VOL, T_=T, paths=PATHS, seed=SEED):
    """Coupon annuel donnant une emission au pair, marge incluse.

    Les trajectoires ne dependent pas du coupon : on simule une seule fois, puis
    on resout par dichotomie sur le payoff. Le tirage est donc identique a chaque
    iteration, ce qui rend la resolution stable.
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


# ------------------------------------------------------ GRECQUES PAR CHOC ----
def autocall_greeks(coupon_pa, bump_spot=1.0, bump_vol=0.01):
    """Delta et vega d'un autocall par choc et revalorisation, comme sur un desk."""
    base, _ = autocall_price(coupon_pa)
    up, _ = autocall_price(coupon_pa, S=S0 + bump_spot, S_fix=S0)
    down, _ = autocall_price(coupon_pa, S=S0 - bump_spot, S_fix=S0)
    vol_up, _ = autocall_price(coupon_pa, sigma=VOL + bump_vol)
    return {
        "prix": base,
        "delta": (up - down) / (2 * bump_spot),
        "gamma": (up - 2 * base + down) / (bump_spot ** 2),
        "vega": (vol_up - base) / (bump_vol * 100),      # par point de vol
    }


# ------------------------------------------------------------ VALIDATION ----
def validate():
    """Controles independants avant toute cotation."""
    out = []

    # 1. Monte-Carlo vanille contre Black-Scholes
    n_obs = int(OBS_PER_YEAR * T)
    paths_S = simulate_paths(n_obs)
    mc_call = float(np.maximum(paths_S[:, -1] - S0, 0).mean() * exp(-R * T))
    bs_call = black_scholes(S0, S0, T, R, Q, VOL, "call")["price"]
    out.append(("Call ATM, Monte-Carlo vs Black-Scholes", mc_call, bs_call,
                mc_call - bs_call))

    # 2. Parite call-put
    call = black_scholes(S0, S0, T, R, Q, VOL, "call")["price"]
    put = black_scholes(S0, S0, T, R, Q, VOL, "put")["price"]
    lhs = call - put
    rhs = S0 * exp(-Q * T) - S0 * exp(-R * T)
    out.append(("Parité call-put : C - P vs Se^-qT - Ke^-rT", lhs, rhs, lhs - rhs))

    # 3. Decomposition de la note protegee : le prix doit revenir au pair
    p = solve_participation()
    out.append(("Note protégée au pair après résolution", capital_protected(p),
                NOTIONAL, capital_protected(p) - NOTIONAL))

    return out


# ------------------------------------------------------------------ SORTIE ---
def pct(x):
    return f"{x:.2%}"


def summary_lines():
    L = []
    L.append("=" * 70)
    L.append("COTATION DE PRODUITS STRUCTURES")
    L.append("=" * 70)
    L.append("")
    L.append("Paramètres de marché")
    L.append("-" * 56)
    L.append(f"  Source                              {MKT['source']}")
    L.append(f"  Sous-jacent (indice)                {S0:>12.2f}")
    L.append(f"  Taux sans risque                    {pct(R):>12}")
    L.append(f"  Dividende                           {pct(Q):>12}")
    L.append(f"  Volatilité implicite (marché)       {pct(VOL):>12}")
    L.append(f"  Volatilité lue à l'échéance         {MKT['vol_tenor']:>12.2f} ans, maintenue plate")
    L.append(f"  Volatilité réalisée 1 an            {pct(MKT['realised']):>12}")
    L.append(f"  Prime de volatilité                 {pct(VOL - MKT['realised']):>12}")
    L.append(f"  Maturité                            {T:>12.1f} ans")
    L.append(f"  Marge et frais                      {FEE:>12.2f} points")
    L.append("")
    L.append("  Structure par terme de la volatilité à la monnaie")
    for tau, iv, exp_ in MKT["term"]:
        L.append(f"    {tau:>5.2f} an(s)   {pct(iv):>8}   échéance {exp_}")
    L.append("")

    L.append("1. Note à capital protégé 100 % (ZC + call ATM)")
    L.append("-" * 56)
    protection = NOTIONAL * zc(T)
    call = black_scholes(S0, S0, T, R, Q, VOL, "call")["price"]
    part = solve_participation()
    L.append(f"  Coût de la protection (zéro-coupon) {protection:>12.2f}")
    L.append(f"  Budget optionnel disponible         {NOTIONAL - protection - FEE:>12.2f}")
    L.append(f"  Prix du call ATM                    {call:>12.2f}")
    L.append(f"  Taux de participation au pair       {pct(part):>12}")
    L.append("")
    L.append("  Sensibilité de la participation")
    for r_ in (0.01, 0.02, 0.03, 0.04):
        pr = solve_participation(r=r_)
        L.append(f"    taux {pct(r_):>6}                        {pct(pr):>12}")
    for v_ in (0.14, 0.18, 0.22, 0.26):
        pv_ = solve_participation(sigma=v_)
        L.append(f"    volatilité {pct(v_):>6}                  {pct(pv_):>12}")
    L.append("")

    L.append("2. Reverse convertible (ZC - put ATM)")
    L.append("-" * 56)
    cpn = solve_coupon_rc()
    put = black_scholes(S0, S0, T, R, Q, VOL, "put")["price"]
    L.append(f"  Prix du put ATM vendu               {put:>12.2f}")
    L.append(f"  Coupon total au pair                {cpn:>12.2f} points")
    L.append(f"  Soit en rythme annuel               {pct(cpn / NOTIONAL / T):>12}")
    L.append("")

    L.append("3. Autocall Athena, observations trimestrielles")
    L.append("-" * 56)
    L.append(f"  Barrière de rappel                  {pct(AUTOCALL_BARRIER):>12}")
    L.append(f"  Barrière de protection à l'échéance {pct(PDI_BARRIER):>12}")
    cpn_ac = solve_coupon_autocall()
    price, se = autocall_price(cpn_ac, fee=FEE)
    L.append(f"  Coupon annuel au pair               {pct(cpn_ac):>12}")
    L.append(f"  Prix obtenu (marge incluse)         {price:>12.2f}")
    L.append(f"  Erreur type Monte-Carlo             {se:>12.4f}")
    L.append("")
    L.append("  Sensibilité du coupon coté")
    for v_ in (0.14, 0.18, 0.22, 0.26):
        c_ = solve_coupon_autocall(sigma=v_)
        L.append(f"    volatilité {pct(v_):>6}                  {pct(c_):>12}")
    for r_ in (0.01, 0.02, 0.03, 0.04):
        c_ = solve_coupon_autocall(r=r_)
        L.append(f"    taux {pct(r_):>6}                        {pct(c_):>12}")
    L.append("")

    L.append("4. Grecques de l'autocall (choc et revalorisation)")
    L.append("-" * 56)
    g = autocall_greeks(cpn_ac)
    L.append(f"  Prix hors marge                     {g['prix']:>12.2f}")
    L.append(f"  Delta                               {g['delta']:>12.4f}")
    L.append(f"  Gamma                               {g['gamma']:>12.4f}")
    L.append(f"  Vega (par point de volatilité)      {g['vega']:>12.4f}")
    L.append("")

    L.append("5. Contrôles")
    L.append("-" * 56)
    for label, a, b, diff in validate():
        L.append(f"  {label}")
        L.append(f"    {a:>14.6f}   {b:>14.6f}   ecart {diff:>+12.2e}")
    L.append("")
    L.append("=" * 70)
    return L


def write_docx(path="rapport_produits_structures_fr.docx"):
    try:
        from docx import Document
        from docx.shared import Pt, Cm, RGBColor
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        print("[!] python-docx absent - rapport Word ignore. pip install python-docx")
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

    h1("Cotation de produits structurés")
    para("Diptakshi Sen  |  projet personnel, Python  |  dépendance : numpy  |  "
         "diptakshi.sen@kedgebs.com", size=9.5, after=10)

    h2("Objet")
    para("Assembler les briques d'une bibliothèque de valorisation — zéro-coupon, options "
         "vanilles avec grecques, courbe de taux — en produits structurés, et répondre à la "
         "question que pose une cotation : quel taux de participation, ou quel coupon, permet "
         "d'émettre la note au pair une fois la marge intégrée.")

    h2("Produits traités")
    bullet("Note à capital protégé 100 % —", "zéro-coupon pour la protection, call à la monnaie "
           "pour l'indexation. Le taux de participation est résolu pour un prix au pair.")
    bullet("Reverse convertible —", "l'investisseur vend un put à la monnaie ; le coupon est "
           "résolu pour un prix au pair.")
    bullet("Autocall de type Athena —", "observations trimestrielles, rappel à 100 %, protection "
           "du capital à 60 % à l'échéance. Payoff chemin-dépendant, donc valorisation par "
           "Monte-Carlo avec variables antithétiques ; le coupon annuel est résolu par dichotomie.")
    bullet("Grecques —", "delta, gamma et vega de l'autocall obtenus par choc et revalorisation, "
           "faute de formule fermée sur un payoff chemin-dépendant.")

    h2("Résultats")
    bullet("Note à capital protégé —", f"la protection coûte {NOTIONAL * zc(T):.2f} points sur "
           f"{NOTIONAL:.0f}, ce qui laisse {NOTIONAL - NOTIONAL * zc(T) - FEE:.2f} points de budget "
           f"optionnel et donne une participation de {pct(part)}.")
    bullet("Reverse convertible —", f"coupon total de {cpn_rc:.2f} points, soit "
           f"{pct(cpn_rc / NOTIONAL / T)} par an, en contrepartie du put vendu.")
    bullet("Autocall —", f"coupon annuel de {pct(cpn_ac)} pour une émission au pair, "
           f"delta {g['delta']:.4f} et vega {g['vega']:.4f} par point de volatilité.")

    h2("Deux points de méthode")
    bullet("", "Le taux de participation n'est pas un argument commercial mais un résidu : il "
           "dépend de ce qui reste après l'achat de la protection. Quand les taux baissent, le "
           "zéro-coupon coûte plus cher, le budget optionnel diminue et la participation tombe. "
           "Le tableau de sensibilité le montre explicitement.")
    bullet("", "L'autocall rend l'investisseur vendeur de volatilité : le coupon monte avec la "
           "volatilité implicite, parce que la barrière de protection à 60 % devient plus "
           "risquée. C'est la relation qu'un desk explique à un client quand la cotation "
           "change d'un jour à l'autre.")

    if MKT["live"]:
        note = (f"Sur les données. Source : {MKT['source']}. Le spot vient de la dernière "
                f"clôture, le taux sans risque du rendement 5 ans (même maturité que la note), la volatilité de la chaîne "
                f"d'options à la monnaie, et le dividende est déduit de la parité call-put "
                f"plutôt que supposé. Hypothèse de volatilité : la volatilité à la monnaie de l'échéance cotée la "
                f"plus longue ({MKT['vol_tenor']:.1f} ans) est maintenue plate jusqu'à {T:.0f} ans et "
                f"utilisée à tous les strikes ; le skew est donc ignoré ici, desk_analytics.py en "
                f"mesure l'effet sur le coupon de l'autocall.")
    else:
        note = (f"Sur les données. Le programme interroge Yahoo Finance ({UNDERLYING} pour le "
                f"spot et la chaîne d'options, {RATE_TICKER} pour le taux) et déduit le "
                f"dividende de la parité call-put. Cette exécution n'a pas eu accès au réseau : "
                f"les chiffres ci-dessus proviennent des paramètres explicites de repli "
                f"(spot {S0:.0f}, taux {R:.2%}, dividende {Q:.2%}, volatilité {VOL:.2%}), et le "
                f"programme le signale lui-même en tête de sortie. Une exécution connectée "
                f"remplace ces valeurs sans autre modification. Hypothèse de volatilité : la volatilité à la monnaie de "
                f"l'échéance cotée la plus longue ({MKT['vol_tenor']:.1f} ans) est maintenue plate "
                f"jusqu'à {T:.0f} ans et utilisée à tous les strikes ; le skew est ignoré ici, "
                f"desk_analytics.py en mesure l'effet.")
    p = para(note, size=10)
    shade(p, "FFF8E1")

    h2("Sortie du programme")
    mono(summary_lines())

    doc.save(path)
    return path


def main():
    for line in summary_lines():
        print(line)
    path = write_docx()
    if path:
        print(f"Rapport Word ecrit : {path}")


if __name__ == "__main__":
    main()
