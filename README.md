# Structured products pricing: capital-protected note, reverse convertible, Athena autocall

Python pricer for three equity structured products on the S&P 500 (SPY), plus the
analytics a structuring desk adds before quoting: skew on the protection barrier, a sales
sheet, a worst-of on European indices, the issuer funding spread and a delta-hedging
backtest.

*Version française en bas de page.*

## What it does

| Product / analysis | Method | Output |
|---|---|---|
| 100% capital-protected note | Zero-coupon + ATM call (Black-Scholes) | Participation rate at par |
| Reverse convertible | Zero-coupon − ATM put | Coupon at par |
| Athena autocall, 5y, quarterly | Monte Carlo, 100,000 paths, antithetic | Coupon at par, delta, gamma, vega |
| Skew on the 60% barrier | Put + digital put replication, smile slope | Coupon with and without skew |
| Sales sheet | Monte Carlo redemption profile | Call probability per date, expected life, loss probability, scenarios, French product sheet |
| Worst-of autocall | Euro Stoxx 50, CAC 40, DAX, Cholesky | Coupon, coupon vs correlation |
| Issuer funding | Note discounted at r + spread | Participation and coupons vs spread |
| Delta hedging | Weekly rebalancing, bump-and-revalue delta | Desk P&L on simulated paths and on SPY history |

Market inputs come from Yahoo Finance: SPY spot and option chain, the 5-year US yield
(`^FVX`), a dividend yield backed out of put-call parity, and implied volatility read off
the chain. See [METHOD.md](METHOD.md) for the formulas and the choices behind them.

## Run

```bash
pip install -r requirements.txt
python structured_products.py      # base pricer, English report
python produits_structures.py      # same pricer, French report
python desk_analytics.py           # skew, sales sheet, worst-of, funding, hedging
pytest -q                          # 12 checks, run on explicit parameters
```

Without network access every script falls back to explicit parameters and says so at the
top of its output. `SP_OFFLINE=1` forces that mode.

## Results in this repository

> The committed outputs were produced **offline, on explicit parameters** (spot 100,
> rate 3.00%, dividend 2.00%, ATM vol 18.00%, skew +8 vol points at the 60% strike,
> illustrative European vols and correlations). They show how the engine behaves, not
> today's market. Running the scripts with internet access replaces every input with
> Yahoo Finance data and regenerates the same files.

| Result (explicit parameters) | Value |
|---|---|
| Capital-protected note, participation at par | 97.17% |
| Reverse convertible, coupon | 6.49% p.a. |
| Autocall coupon at par, flat vol | 13.89% p.a. |
| Autocall coupon at par, with skew on the 60% barrier | 17.23% p.a. |
| Probability of call at the first observation | 49.3% |
| Expected life / probability of capital loss | 1.30 years / 5.4% |
| Worst-of (SX5E, CAC, DAX) coupon | 17.74% p.a. |
| Worst-of coupon at correlation 0.50 / 1.00 | 27.02% / 14.79% |
| Autocall coupon with a 60 bp funding spread | 15.12% p.a. |
| Hedging P&L, realised vol 18% / 24% (60 paths) | −0.42 / +2.19 points |

What the numbers say:

- **Skew raises the coupon.** The investor sells the 60% put; with skew that put is worth
  more, so the note can pay more at par.
- **The worst-of investor is short correlation.** Lower correlation makes the worst
  performer worse, and the coupon has to rise to compensate.
- **Funding matters.** A wider issuer spread cheapens the zero-coupon and frees budget for
  the option.
- **The desk is long volatility against the investor.** When realised volatility exceeds
  the pricing volatility, the delta-hedged desk makes money.

## Controls

- Monte Carlo ATM call against Black-Scholes; put-call parity.
- Each note returns to par after solving.
- Protection leg by Monte Carlo against the put + digital replication.
- Worst-of at correlation 1 against the single-underlying autocall.
- Funding spread 0 against the base pricer.
- Delta hedging under the model: average P&L close to zero.

## Limits

- Volatility beyond the longest listed expiry is extrapolated flat.
- The skew adjustment is first order (level and slope at the barrier), not a local- or
  stochastic-volatility model.
- The worst-of uses realised, not implied, volatilities and correlations, and an explicit
  EUR rate.
- Probabilities are risk-neutral: they are prices, not forecasts.
- The funding spread is illustrative; a real quote uses the issuer's curve.

## Files

| File | Content |
|---|---|
| `structured_products.py` / `produits_structures.py` | Base pricer (EN / FR) |
| `desk_analytics.py` | Skew, sales sheet, worst-of, funding, hedging |
| `tests/` | pytest suite |
| `*_report.docx`, `rapport_*.docx` | Base reports |
| `desk_analytics_report.docx` | Desk analytics with charts |
| `fiche_produit_autocall_fr.docx` | One-page French product sheet for sales |
| `out_*.png`, `out_desk_analytics_sample.txt` | Charts and console output |

---

## Version française

Pricer Python de trois produits structurés actions sur le S&P 500 (SPY) : note à capital
protégé (zéro-coupon + call), reverse convertible (zéro-coupon − put) et autocall Athena
5 ans à observations trimestrielles (Monte Carlo, 100 000 trajectoires). Le coupon ou le
taux de participation est résolu pour une émission au pair, marge de 2 points incluse.

`desk_analytics.py` ajoute :

- **le skew sur la barrière de protection à 60 %**, par réplication put + put digital ;
- **une fiche produit** pour la vente : probabilités de rappel, durée de vie espérée,
  probabilité de perte, scénarios à l'échéance ;
- **un autocall worst-of** sur Euro Stoxx 50, CAC 40 et DAX, avec la sensibilité du coupon
  à la corrélation ;
- **le spread de financement de l'émetteur** ;
- **un backtest de couverture en delta**.

Les taux viennent du rendement américain à 5 ans (`^FVX`), même maturité que la note. Les
résultats publiés ici ont été produits hors connexion sur paramètres explicites ; une
exécution connectée les remplace par des données Yahoo Finance.
