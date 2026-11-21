"""Dynamic (time-explicit) climate characterization.

Absolute global warming potentials (AGWP, W m-2 yr kg-1) follow the impulse-response
formulation used by the ``dynamic_characterization`` package (IPCC AR6 Ch. 7 values;
Joos et al. 2013 CO2 IRF). The same functional forms are re-implemented here in
vectorized form. The research archive checks them against the package.

Like the package, the dynamic metric excludes climate-carbon feedbacks, so the
implied GWP100 of CH4 and HFC-32 is slightly below the AR6 tabulated values
(which include the feedback). Static results always use the AR6 table values.
"""
from __future__ import annotations

import numpy as np

M_AIR = 28.97
M_ATM = 5.135e18  # kg


def _re_kg(re_ppb: float, molar_mass: float) -> float:
    return re_ppb * M_AIR / molar_mass * 1e9 / M_ATM


RE_CO2 = _re_kg(1.33e-5, 44.01)
RE_CH4 = _re_kg(5.7e-4, 16.04)
RE_N2O = _re_kg(2.8e-3, 44.01)
# HFC-32 (CH2F2): AR6 Table 7.SM.7 lifetime 5.4 yr, radiative efficiency 0.11 W m-2 ppb-1
RE_HFC32 = _re_kg(0.11, 52.02)
TAU = {"CH4": 11.8, "N2O": 109.0, "HFC32": 5.4}


def agwp_co2(h: np.ndarray | float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    a0, a1, a2, a3 = 0.2173, 0.2240, 0.2824, 0.2763
    t1, t2, t3 = 394.4, 36.54, 4.304
    irf_int = (a0 * h + a1 * t1 * (1 - np.exp(-h / t1)) + a2 * t2 * (1 - np.exp(-h / t2))
               + a3 * t3 * (1 - np.exp(-h / t3)))
    return RE_CO2 * np.where(h > 0, irf_int, 0.0)


def _agwp_decay(h, re, tau):
    h = np.asarray(h, dtype=float)
    return re * tau * np.where(h > 0, 1 - np.exp(-h / tau), 0.0)


def agwp(gas: str, h) -> np.ndarray:
    if gas in ("CO2", "CO2e"):
        return agwp_co2(h)
    if gas == "CH4":
        return _agwp_decay(h, RE_CH4, TAU["CH4"])
    if gas == "N2O":
        return _agwp_decay(h, RE_N2O, TAU["N2O"])
    if gas == "HFC32":
        return _agwp_decay(h, RE_HFC32, TAU["HFC32"])
    if gas == "R290":
        return np.zeros_like(np.asarray(h, dtype=float))
    raise KeyError(gas)


def implied_gwp(gas: str, horizon: float = 100.0) -> float:
    return float(agwp(gas, horizon) / agwp_co2(horizon))


def crf_at(emission_years: np.ndarray, emissions_kg: dict[str, np.ndarray], end_year: float,
           mid_year: bool = True) -> float:
    """Cumulative radiative forcing (W m-2 yr) at ``end_year`` from annual pulses.

    Emissions in year y are treated as a pulse at y + 0.5 (``mid_year``).
    """
    t0 = emission_years + (0.5 if mid_year else 0.0)
    h = np.clip(end_year - t0, 0, None)
    return float(sum(np.sum(e * agwp(g, h)) for g, e in emissions_kg.items()))


def crf_path(emission_years: np.ndarray, emissions_kg: dict[str, np.ndarray],
             eval_years: np.ndarray) -> np.ndarray:
    return np.array([crf_at(emission_years, emissions_kg, y) for y in eval_years])
