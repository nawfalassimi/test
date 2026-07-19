from __future__ import annotations

import math

from scipy.stats import norm


def norm_cdf(x: float) -> float:
    return norm.cdf(x)


def norm_pdf(x: float) -> float:
    return norm.pdf(x)


def d1(S: float, K: float, r_d: float, r_f: float, vol: float, T: float) -> float:
    return (math.log(S / K) + (r_d - r_f + 0.5 * vol**2) * T) / (vol * math.sqrt(T))


def d2(S: float, K: float, r_d: float, r_f: float, vol: float, T: float) -> float:
    return d1(S, K, r_d, r_f, vol, T) - vol * math.sqrt(T)
