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
