"""Calibration of the individual technology-choice model.

The alternative-specific constants (ASCs) of the hybrid and all-electric heat pump
(relative to the gas boiler, ASC = 0) are solved so that the expected choice shares of a
synthetic *reference population* match national 2025 replacement shares:

    target_hybrid  = 43,000 hybrid sales / ~430,000 replacements      (derived, ~10%)
    target_AE      = all-electric sales to existing homes / replacements (derived, ~7%)

Reference population: owner-occupied terraced houses (CBS mean gas use 770 m3/yr),
all options technically feasible, 2026 prices and subsidies, descriptive norm equal to the
national heat-pump-to-dwelling ratio (~0.099), beliefs at the prior mean.
"""
from __future__ import annotations

import numpy as np

from . import technologies as tech
from .background import background_series


def reference_population(p: dict, n: int = 4000, seed: int = 99) -> dict:
    rng = np.random.default_rng([seed, 2002])
    cv = p["housing.gas_use_cv"]
    s2 = np.log(1 + cv ** 2)
    gas = p["behavior.calib_gas_m3"] * rng.lognormal(-s2 / 2, np.sqrt(s2), n)
    q_sh, q_dhw = tech.useful_heat(p, gas)
    env = rng.beta(p["behavior.env_beta_a"], p["behavior.env_beta_b"], n)
    disc = p["behavior.discount_rate_median"] * rng.lognormal(0, p["behavior.discount_sigma"], n)
    return {"q_sh": q_sh, "q_dhw": q_dhw, "env": env, "disc": disc}


def systematic_utility(p: dict, pop: dict, bg: dict, year_idx: int = 0, soc: float | None = None,
                       subsidy: float | None = None) -> np.ndarray:
    """Utility without ASCs for (BOILER, HYBRID, AE), shape (n, 3)."""
    fl = tech.energy_flows(p, pop["q_sh"], pop["q_dhw"])
    op = tech.operational_cost(p, fl)
    ghg = tech.operational_ghg(p, fl, bg, year_idx)
    sub = p["prices.subsidy_share_hp"] if subsidy is None else subsidy
    cap = tech.capex(p, subsidy_ind=sub)
    crf = tech.crf(pop["disc"], p["behavior.horizon_years"])
    soc = p["behavior.calib_social_share"] if soc is None else soc
    opts = [tech.BOILER, tech.HYBRID, tech.AE]
    V = np.empty((len(crf), 3))
    for k, j in enumerate(opts):
        cost = cap[j] * crf + op[j]
        V[:, k] = (p["behavior.beta_cost"] * cost / 100.0
                   + p["behavior.beta_env"] * pop["env"] * (ghg[tech.BOILER] - ghg[j]))
        if j in tech.HEAT_PUMPS:
            V[:, k] += p["behavior.beta_soc"] * soc
    return V


def shares(V: np.ndarray, asc: np.ndarray) -> np.ndarray:
    U = V + asc[None, :]
    U -= U.max(axis=1, keepdims=True)
    e = np.exp(U)
    return (e / e.sum(axis=1, keepdims=True)).mean(axis=0)


def calibrate_asc(p: dict, tol: float = 1e-10, max_iter: int = 500) -> dict:
    """Return {'asc': array(3), 'shares': fitted, 'targets': targets, 'iterations': k}."""
    bg = background_series(p)
    V = systematic_utility(p, reference_population(p), bg)
    t_h, t_ae = p["behavior.target_share_hybrid_2025"], p["behavior.target_share_allelectric_2025"]
    target = np.array([1 - t_h - t_ae, t_h, t_ae])
    asc = np.zeros(3)
    for k in range(max_iter):
        s = shares(V, asc)
        step = np.log(target) - np.log(s)
        step -= step[0]
        asc += step
        if np.max(np.abs(step)) < tol:
            break
    return {"asc": asc, "shares": shares(V, asc), "targets": target, "iterations": k + 1}
