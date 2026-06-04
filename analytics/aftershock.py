"""
Aftershock Probability Model
Implements Omori-Utsu decay law + Bath's law for aftershock forecasting.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OmoriParams:
    """Omori-Utsu law parameters: λ(t) = K / (t + c)^p"""
    K: float = 0.1       # productivity (events per unit time)
    c: float = 0.05      # time offset (days)
    p: float = 1.1       # decay exponent (usually 0.9–1.5)
    b: float = 1.0       # Gutenberg-Richter b-value
    a: float = 1.0       # productivity scaling


def estimate_omori_params(mainshock_mag: float) -> OmoriParams:
    """
    Estimate Omori parameters from mainshock magnitude using empirical scaling.
    Reference: Reasenberg & Jones (1989), Ogata (1988).
    """
    # Productivity scales with magnitude: log10(K) ≈ a + b*M
    a = -1.67   # empirical constant
    b = 0.91    # b-value estimate
    K = 10 ** (a + b * mainshock_mag)
    p = 1.08    # average global p-value
    c = 0.05    # days

    return OmoriParams(K=K, c=c, p=p, b=b, a=a)


def omori_rate(t_days: np.ndarray, params: OmoriParams) -> np.ndarray:
    """
    Compute Omori-Utsu aftershock rate λ(t) at times t (days after mainshock).
    """
    return params.K / (t_days + params.c) ** params.p


def baths_law_largest_aftershock(mainshock_mag: float) -> float:
    """
    Bath's law: largest aftershock ≈ mainshock - 1.2 magnitude units.
    """
    return mainshock_mag - 1.2


def expected_aftershocks(
    mainshock_mag: float,
    min_mag: float = 2.0,
    days: int = 30,
) -> dict:
    """
    Forecast number of aftershocks above min_mag over the next `days` days.

    Uses modified Omori law integrated over time + Gutenberg-Richter for
    magnitude distribution.
    """
    params = estimate_omori_params(mainshock_mag)
    t = np.linspace(0, days, 1000)
    dt = t[1] - t[0]

    # Rate of M ≥ min_mag aftershocks
    # Scale by G-R: fraction of events ≥ min_mag
    # N(≥M) = 10^(a - b*M)  →  fraction = 10^(-b*(min_mag - mainshock_mag + 1.2))
    b = params.b
    fraction_above_min = 10 ** (-b * max(0, min_mag - (mainshock_mag - 1.2)))

    rates = omori_rate(t, params) * fraction_above_min
    cumulative = np.cumsum(rates * dt)

    # Daily breakdown
    daily = []
    for day in range(1, days + 1):
        t_day = np.linspace(day - 1, day, 100)
        r = omori_rate(t_day, params) * fraction_above_min
        daily.append(float(np.trapezoid(r, t_day)))

    return {
        "mainshock_mag": mainshock_mag,
        "largest_expected": round(baths_law_largest_aftershock(mainshock_mag), 1),
        "expected_total": round(float(cumulative[-1]), 1),
        "days": days,
        "daily_counts": [round(d, 2) for d in daily[:days]],
        "cumulative": cumulative.tolist(),
        "time_axis": t.tolist(),
        "peak_rate_day1": round(float(rates[:33].mean()), 3),
        "params": params,
    }


def probability_larger_event(
    mainshock_mag: float,
    target_mag: float,
    hours: int = 24,
) -> float:
    """
    Estimate probability that an event ≥ target_mag occurs within `hours`.
    Based on STEP model approximation (Gerstenberger et al. 2005).
    """
    params = estimate_omori_params(mainshock_mag)
    t = np.linspace(0, hours / 24, 500)
    dt = t[1] - t[0]

    # Rate of events ≥ target_mag
    b = params.b
    fraction = 10 ** (-b * max(0, target_mag - baths_law_largest_aftershock(mainshock_mag)))
    rates = omori_rate(t, params) * fraction

    # Poisson: P(at least 1) = 1 - exp(-lambda)
    expected = float(np.trapezoid(rates, t))
    prob = 1 - np.exp(-expected)
    return round(float(np.clip(prob, 0, 1)), 4)


def aftershock_zone_radius(mainshock_mag: float) -> float:
    """
    Estimate aftershock zone radius in km using Wells & Coppersmith (1994) scaling.
    """
    # log(rupture_length_km) ≈ -2.44 + 0.59*M
    rupture_length = 10 ** (-2.44 + 0.59 * mainshock_mag)
    # Aftershock zone ≈ 2x rupture length
    return round(rupture_length * 2, 1)


def full_aftershock_report(mainshock_mag: float, mainshock_depth: float = 10) -> dict:
    """Comprehensive aftershock analysis for a given mainshock."""
    forecast = expected_aftershocks(mainshock_mag, min_mag=2.0, days=30)
    forecast_m3 = expected_aftershocks(mainshock_mag, min_mag=3.0, days=7)

    return {
        "mainshock_magnitude": mainshock_mag,
        "mainshock_depth_km": mainshock_depth,
        "largest_aftershock": baths_law_largest_aftershock(mainshock_mag),
        "aftershock_zone_radius_km": aftershock_zone_radius(mainshock_mag),
        "prob_m5_next_24h": probability_larger_event(mainshock_mag, target_mag=5.0, hours=24),
        "prob_m6_next_24h": probability_larger_event(mainshock_mag, target_mag=6.0, hours=24),
        "expected_m2plus_30days": forecast["expected_total"],
        "expected_m3plus_7days": forecast_m3["expected_total"],
        "daily_forecast_30d": forecast["daily_counts"],
        "omori_params": {
            "K": round(forecast["params"].K, 5),
            "c": forecast["params"].c,
            "p": forecast["params"].p,
        },
    }
