from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from scipy.interpolate import CubicSpline
from scipy.stats import norm

# Delta convention used throughout this module: "call-equivalent delta" in (0, 1).
# Put quotes are converted via delta_call = 1 + delta_put (the standard spot-delta
# put-call parity relationship, ignoring the small e^{-r_f T} correction — a
# documented v1 simplification). This keeps a single strictly-increasing delta
# axis to interpolate/spline against: 10C=0.10, 25C=0.25, ATM=0.50, 25P=0.75, 10P=0.90.
# Higher call-delta => lower strike (more ITM from the call's perspective).


@dataclass(frozen=True)
class SmilePoint:
    delta: float  # call-equivalent delta, in (0, 1)
    vol: float


def build_smile_points(atm_vol: float, rr25: float, bf25: float, rr10: float, bf10: float) -> list:
    """Build the 5-point smile (10C, 25C, ATM, 25P, 10P) from ATM/RR/BF quotes,
    using the standard first-order approximation from the README:
        vol(25C) = atm_vol + bf25 + rr25/2      vol(25P) = atm_vol + bf25 - rr25/2
        vol(10C) = atm_vol + bf10 + rr10/2      vol(10P) = atm_vol + bf10 - rr10/2
    Returned in ascending call-delta order, ready to feed to interpolate_smile.
    """
    vol_10c = atm_vol + bf10 + rr10 / 2
    vol_25c = atm_vol + bf25 + rr25 / 2
    vol_25p = atm_vol + bf25 - rr25 / 2
    vol_10p = atm_vol + bf10 - rr10 / 2

    return [
        SmilePoint(delta=0.10, vol=vol_10c),
        SmilePoint(delta=0.25, vol=vol_25c),
        SmilePoint(delta=0.50, vol=atm_vol),
        SmilePoint(delta=0.75, vol=vol_25p),
        SmilePoint(delta=0.90, vol=vol_10p),
    ]


def interpolate_smile(points: list) -> Callable[[float], float]:
    """Cubic spline through the 5 smile points, over call-delta in (0, 1).
    SABR/SVI parametric fits are deferred to a later milestone."""
    deltas = [p.delta for p in points]
    vols = [p.vol for p in points]
    spline = CubicSpline(deltas, vols, bc_type="natural")

    def smile_fn(delta_call: float) -> float:
        return float(spline(delta_call))

    return smile_fn


class ConvergenceError(RuntimeError):
    pass


def solve_strike_for_delta(target_delta: float, S: float, r_d: float, r_f: float,
                            T: float, vol_fn: Callable[[float], float]):
    """Delta -> strike. Since this module's smile is parametrized directly by
    call-delta, vol is a direct lookup (vol_fn(target_delta)) and the strike
    then follows in closed form by inverting d1 = (ln(S/K) + (r_d-r_f+vol^2/2)T)
    / (vol*sqrt(T)) for K, using delta_call = e^{-r_f T} * N(d1) => d1 =
    N^{-1}(target_delta * e^{r_f T}). No fixed-point iteration is needed in this
    direction — see solve_delta_for_strike for the direction that does need one.
    Returns (strike, vol).
    """
    vol = vol_fn(target_delta)
    dd1 = norm.ppf(target_delta * math.exp(r_f * T))
    K = S * math.exp(-dd1 * vol * math.sqrt(T) + (r_d - r_f + 0.5 * vol**2) * T)
    return K, vol


def solve_delta_for_strike(K: float, S: float, r_d: float, r_f: float, T: float,
                            vol_fn: Callable[[float], float], vol_guess: float = None,
                            tol: float = 1e-6, max_iter: int = 20):
    """Strike -> delta (and implied vol). Genuinely circular: vol depends on
    delta via the smile, and delta depends on vol via the Black-Scholes delta
    formula, so this is solved by fixed-point iteration:
        1. guess vol (default: ATM vol)
        2. compute delta_call(K, vol) = e^{-r_f T} * N(d1(S,K,r_d,r_f,vol,T))
        3. look up vol_new = vol_fn(delta_call)
        4. stop when |vol_new - vol| < tol, else vol = vol_new and repeat
    Returns (delta_call, vol).
    """
    vol = vol_guess if vol_guess is not None else vol_fn(0.50)
    for _ in range(max_iter):
        dd1 = (math.log(S / K) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
        delta_call = math.exp(-r_f * T) * norm.cdf(dd1)
        delta_call = min(max(delta_call, 1e-6), 1 - 1e-6)
        vol_new = vol_fn(delta_call)
        if abs(vol_new - vol) < tol:
            return delta_call, vol_new
        vol = vol_new
    raise ConvergenceError(f"solve_delta_for_strike did not converge within {max_iter} iterations")
