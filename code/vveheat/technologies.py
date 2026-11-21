"""Heating technologies: energy flows, costs and embodied quantities per apartment.

Technology codes:
    0 BOILER      individual gas condensing boiler (space heating + hot water)
    1 HYBRID      individual hybrid heat pump (air-water HP + new condensing boiler)
    2 AE          individual all-electric air-water heat pump with hot-water tank
    3 COLL_HP     collective air-water heat pump (R290), building-level distribution
    4 DH          district heat (connection + substation + in-building works)
    5 COLL_BOILER collective gas boiler (existing 'blokverwarming')
"""
from __future__ import annotations

import numpy as np

BOILER, HYBRID, AE, COLL_HP, DH, COLL_BOILER = range(6)
NAMES = ["Gas boiler", "Hybrid HP", "All-electric HP", "Collective HP", "District heat",
         "Collective gas boiler"]
SHORT = ["boiler_ind", "hybrid_ind", "allelectric_ind", "hp_collective", "district_heat",
         "boiler_collective"]
INDIVIDUAL = (BOILER, HYBRID, AE)
COLLECTIVE = (COLL_HP, DH, COLL_BOILER)
HEAT_PUMPS = (HYBRID, AE, COLL_HP)
GAS_FREE = (AE, COLL_HP, DH)
KWH_PER_GJ = 277.7778


def useful_heat(p: dict, gas_m3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Space-heating and hot-water useful heat (GJ/yr) implied by reference boiler gas use."""
    lhv = p["gas.lhv_MJ_per_m3"] / 1000.0
    dhw, cook = p["housing.share_dhw"], p["housing.share_cooking"]
    q_sh = gas_m3 * (1 - dhw - cook) * lhv * p["technologies.boiler_ind.eta_SH"]
    q_dhw = gas_m3 * dhw * lhv * p["technologies.boiler_ind.eta_DHW"]
    return q_sh, q_dhw


def energy_flows(p: dict, q_sh: np.ndarray, q_dhw: np.ndarray) -> dict[str, np.ndarray]:
    """Annual delivered energy per apartment for every technology.

    Returns arrays of shape (6, n): gas (m3), elec_SH (kWh), elec_DHW (kWh), dh (GJ).
    """
    t = "technologies."
    lhv = p["gas.lhv_MJ_per_m3"] / 1000.0
    n = q_sh.shape[0]
    gas = np.zeros((6, n)); esh = np.zeros((6, n)); edhw = np.zeros((6, n)); dh = np.zeros((6, n))
    eta_sh, eta_dhw = p[t + "boiler_ind.eta_SH"], p[t + "boiler_ind.eta_DHW"]
    gas[BOILER] = (q_sh / eta_sh + q_dhw / eta_dhw) / lhv
    cov = p[t + "hybrid_ind.coverage_SH"]
    gas[HYBRID] = ((1 - cov) * q_sh / eta_sh + q_dhw / eta_dhw) / lhv
    esh[HYBRID] = cov * q_sh / p[t + "hybrid_ind.spf_SH"] * KWH_PER_GJ
    esh[AE] = q_sh / p[t + "allelectric_ind.spf_SH"] * KWH_PER_GJ
    edhw[AE] = q_dhw / p[t + "allelectric_ind.spf_DHW"] * KWH_PER_GJ
    loss = 1 + p[t + "hp_collective.dist_loss"]
    esh[COLL_HP] = q_sh / p[t + "hp_collective.spf_SH"] * KWH_PER_GJ * loss
    edhw[COLL_HP] = q_dhw / p[t + "hp_collective.spf_DHW"] * KWH_PER_GJ * loss
    dh[DH] = q_sh + q_dhw
    closs = 1 + p[t + "boiler_collective.dist_loss"]
    gas[COLL_BOILER] = (q_sh / p[t + "boiler_collective.eta_SH"]
                        + q_dhw / p[t + "boiler_collective.eta_DHW"]) / lhv * closs
    return {"gas": gas, "elec_SH": esh, "elec_DHW": edhw, "dh": dh}


def operational_cost(p: dict, flows: dict[str, np.ndarray]) -> np.ndarray:
    """Annual energy + maintenance cost (EUR/yr), shape (6, n). Real 2026 prices."""
    pr = "prices."
    gas_cost = flows["gas"] * p[pr + "gas_per_m3"] + (flows["gas"] > 0) * p[pr + "gas_fixed_per_yr"]
    el_cost = (flows["elec_SH"] + flows["elec_DHW"]) * p[pr + "elec_per_kwh"]
    cost = gas_cost + el_cost
    cost[DH] = cost[BOILER] * p[pr + "dh_cost_ratio"]
    m = "technologies.maintenance."
    maint = np.array([p["technologies.boiler_ind.maintenance"], p[m + "hybrid_ind"],
                      p[m + "allelectric_ind"], p[m + "hp_collective"], p[m + "district_heat"],
                      p[m + "boiler_collective"]])
    return cost + maint[:, None]


def operational_ghg(p: dict, flows: dict[str, np.ndarray], bg: dict, year_idx: int) -> np.ndarray:
    """Perceived operational GHG (t CO2e/yr) per technology using current-year factors.
    Households are assumed to perceive operational emissions only (no embodied/refrigerant).
    """
    g = bg
    gas_ef = (g["gas_CO2"][year_idx] + g["gas_CH4"][year_idx] * p["gwp100_ar6.CH4_fossil"]
              + g["gas_chain"][year_idx])
    e = (flows["gas"] * gas_ef + flows["elec_SH"] * g["elec_SH"][year_idx]
         + flows["elec_DHW"] * g["elec"][year_idx] + flows["dh"] * g["dh"][year_idx])
    return e / 1000.0


def gross_capex(p: dict) -> np.ndarray:
    t = "technologies."
    out = np.zeros(6)
    out[BOILER] = p[t + "boiler_ind.capex"]
    out[HYBRID] = p[t + "hybrid_ind.capex_with_boiler"]
    out[AE] = p[t + "allelectric_ind.capex"]
    out[COLL_HP] = p[t + "hp_collective.capex_per_apt"]
    out[DH] = p[t + "district_heat.capex_owner"]
    out[COLL_BOILER] = p[t + "boiler_collective.capex_per_apt"]
    return out


def subsidy_per_apartment(p: dict, subsidy_ind: float = 0.0, subsidy_coll: float = 0.0,
                          extra_coll: float = 0.0) -> np.ndarray:
    """Grant per apartment (EUR) for each technology (6,). Single source of truth for both the
    owner's net cost and the public outlay. Every grant is capped at the gross cost.
    """
    g = gross_capex(p)
    out = np.zeros(6)
    out[HYBRID] = (g[HYBRID] - g[BOILER]) * subsidy_ind
    out[AE] = g[AE] * subsidy_ind
    out[COLL_HP] = g[COLL_HP] * min(subsidy_coll + extra_coll, 0.9)
    out[DH] = p["technologies.district_heat.connection_grant"] + extra_coll * g[DH]
    return np.minimum(out, g)


def capex(p: dict, n_apt: np.ndarray | float = 1.0, subsidy_ind: float = 0.0,
          subsidy_coll: float = 0.0, extra_coll: float = 0.0) -> np.ndarray:
    return gross_capex(p) - subsidy_per_apartment(p, subsidy_ind, subsidy_coll, extra_coll)


def crf(rate: np.ndarray, horizon: float) -> np.ndarray:
    """Capital recovery factor."""
    r = np.asarray(rate, dtype=float)
    return r / (1 - (1 + r) ** (-horizon))


def mean_life(p: dict) -> np.ndarray:
    t = "technologies."
    return np.array([p[t + "boiler_ind.life"], p[t + "hybrid_ind.life"], p[t + "allelectric_ind.life"],
                     p[t + "hp_collective.life"], p[t + "district_heat.substation_life"],
                     p[t + "boiler_collective.life"]], dtype=float)


def collective_kw(p: dict, tech: int, n_apt: int) -> float:
    if tech == COLL_HP:
        return p["technologies.hp_collective.kw_per_apt"] * n_apt
    if tech == COLL_BOILER:
        return p["technologies.boiler_collective.kw_per_apt"] * n_apt
    return 0.0
