# Method notes

## Simulation

Geometric Brownian motion under the risk-neutral measure, on the quarterly observation grid:

```
S(t+dt) = S(t) · exp[(r − q − ½σ²)·dt + σ·√dt·Z]
```

100,000 paths with antithetic variates: the second half of the sample uses `−Z`, which
halves the standard error on symmetric payoffs at no extra cost.

## The barrier bug worth documenting

An early version scaled the autocall barriers with the shocked spot when computing delta.
The payoff then became homogeneous of degree one in spot, delta came out **exactly zero**,
and the number looked plausible enough to miss.

The fix is to separate the *fixing* level from the *current* level:

```python
def autocall_payoff(paths_S, coupon_pa, S_fix=S0, ...):
    called    = alive & (paths_S[:, i] >= AUTOCALL_BARRIER * S_fix)
    protected = final & (paths_S[:, i] >= PDI_BARRIER      * S_fix)
```

Barriers are struck at issue and stay there. A spot shock then moves the price, which is
what makes delta exist at all.

## Solving for the coupon

The paths do not depend on the coupon, so they are simulated **once** outside the bisection
and reused at every iteration. This makes the solve both fast (18s instead of a 2-minute
timeout) and stable: the same draw is used throughout, so the objective is monotonic in the
coupon rather than noisy.

## Greeks

Bump and revalue, with the same random seed on every revaluation so that the difference
measures the bump and not the sampling noise.

- Delta: central difference on a ±1 point spot bump
- Gamma: second central difference on the same bump
- Vega: forward difference on a 1 volatility point bump

Vega on the autocall is **negative**: the investor is short volatility through the barrier.

## Implied dividend

At-the-money put-call parity gives

```
q = −(1/T)·ln[ (C − P + K·e^(−rT)) / S ]
```

The estimate is taken at the nearest strike on each sampled expiry and the median is used,
bounded to [0%, 8%] as a sanity guard against a bad quote.

## Rate and volatility inputs

The risk-free rate is the US **5-year** Treasury yield (`^FVX`), the same tenor as the
note. An earlier version used the 10-year yield, which overstated the discount on a
5-year zero-coupon in a normal curve.

The volatility is the at-the-money implied volatility of the longest listed SPY expiry,
held flat out to five years and used at every strike. Both halves of that sentence are
assumptions: the term structure beyond the last listed expiry is not observed, and the
smile is ignored. `desk_analytics.py` measures what the second one costs.

## Skew on the protection barrier (desk_analytics.py)

The 60% protection is a down-and-in put sold by the investor at maturity. Its European
leg replicates exactly:

```
N (S_T/S_0 − 1) 1{S_T < K}  =  −N/S_0 · [ Put(K) + (S_0 − K) · DigitalPut(K) ]
```

With a smile, the digital put is the strike derivative of the put price,
`e^(−rT) N(−d2) + vega · dσ/dK`. The leg is repriced at the 60%-strike volatility and
smile slope, and the difference with the flat-vol leg is scaled by the share of that leg
that survives to maturity without an early call (read from the Monte Carlo). The control
checks the replication against Monte Carlo.

This is a first-order adjustment, not a local-volatility model: it captures the level and
slope of the smile at the barrier, not the effect of the smile on the call probabilities.

## Worst-of

Three correlated GBMs (Cholesky on the correlation matrix, antithetic variates); the
payoff is the base autocall applied to the worst performance. Yahoo does not publish index
option chains, so one-year **realised** volatilities and correlations stand in for implied
ones. The DAX is a total-return index, hence a zero dividend yield.

## Funding

The note is the issuer's debt: its cash flows are discounted at `r + funding spread`, while
the paths stay risk-neutral at `r`. A wider spread makes the zero-coupon cheaper and leaves
more budget for the option, so participation and coupons rise.

## Delta hedging

The desk sells the note at model value (no margin), buys delta shares, rebalances every
five trading days and on every observation date, earns `r` on cash and `q` on the shares.
Delta is a central bump of ±1% with common random numbers. On paths simulated with the
pricing volatility the average P&L is close to zero; the dispersion is the cost of hedging
discretely around barriers. With realised volatility above the pricing volatility the desk
gains, because the investor is short volatility and the desk holds the other side.
