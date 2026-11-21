"""Time-explicit, attributional life-cycle GHG assessment (ISO 14040/44-aligned scope).

Goal and scope
  Functional unit : supply of space heating and domestic hot water to all apartments of the
                    synthetic neighborhood, at their reference demand, each year 2026-2050.
  Reference flows : GJ of useful heat delivered and dwellings served per year (both identical
                    across policies within a paired seed, so comparisons are at equal service).
  Boundary        : energy supply chains (cradle-to-grave for electricity, gas incl. upstream
                    methane, green-gas chain, district heat at end user), heating equipment
                    (A1-A5 production/installation, C1-C4 end-of-life), heat-network connection
                    pipes and substations (A1-A5; substation C at retirement), in-building
                    conversion works, refrigerant leakage. Cut-offs: end-of-life of shared
                    heat-network pipes (service life well beyond the horizon), module D credits,
                    building envelope, cooking, appliances, grid infrastructure beyond lifecycle
                    factors.
  Indicator       : climate change. Static GWP100 (IPCC AR6, incl. climate-carbon feedback)
                    and time-explicit cumulative radiative forcing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import dynamic
from .abm import POST_YEARS, Result
from .background import YEARS, background_series

GASES = ("CO2", "CH4", "HFC32", "CO2e")

# activity -> (unit, life-cycle stage category)
ACTIVITIES = {
    "gas_supply_combustion": ("m3", "Gas: combustion & supply chain"),
    "elec_space_heating": ("kWh", "Electricity: heat pumps (space heating)"),
    "elec_hot_water": ("kWh", "Electricity: heat pumps (hot water)"),
    "district_heat": ("GJ", "District heat supply"),
    "prod_boiler_ind": ("unit", "Equipment: production & installation"),
    "prod_hp_hybrid_unit": ("unit", "Equipment: production & installation"),
    "prod_hp_ae": ("unit", "Equipment: production & installation"),
    "prod_hp_collective": ("kW", "Equipment: production & installation"),
    "prod_boiler_collective": ("kW", "Equipment: production & installation"),
    "dh_connection": ("apartment", "Heat-network connection & substations"),
    "dh_substation": ("apartment", "Heat-network connection & substations"),
    "eol_dh_substation": ("apartment", "Equipment: end-of-life"),
    "retrofit_collective_hp": ("apartment", "In-building conversion works"),
    "retrofit_dh": ("apartment", "In-building conversion works"),
    "repair_ind": ("unit", "Equipment: production & installation"),
    "repair_coll": ("kW", "Equipment: production & installation"),
    "eol_boiler_ind": ("unit", "Equipment: end-of-life"),
    "eol_hp_hybrid_unit": ("unit", "Equipment: end-of-life"),
    "eol_hp_ae": ("unit", "Equipment: end-of-life"),
    "eol_hp_collective": ("kW", "Equipment: end-of-life"),
    "eol_boiler_collective": ("kW", "Equipment: end-of-life"),
    "refrigerant_R32": ("kg", "Refrigerant emissions"),
    "refrigerant_R290": ("kg", "Refrigerant emissions"),
}
ACT_LIST = list(ACTIVITIES)
CATEGORIES = list(dict.fromkeys(v[1] for v in ACTIVITIES.values()))


def _demand(a: dict, p: dict) -> dict[str, np.ndarray]:
    lk, le = p["technologies.refrigerant.leak_annual"], p["technologies.refrigerant.leak_eol"]
    return {
        "gas_supply_combustion": a["gas_m3"],
        "elec_space_heating": a["elec_SH_kWh"],
        "elec_hot_water": a["elec_DHW_kWh"],
        "district_heat": a["dh_GJ"],
        # a hybrid installation = HP unit + new condensing boiler
        "prod_boiler_ind": a["inst_boiler"] + a["inst_hybrid"],
        "prod_hp_hybrid_unit": a["inst_hybrid"],
        "prod_hp_ae": a["inst_ae"],
        "prod_hp_collective": a["inst_collhp_kW"],
        "prod_boiler_collective": a["inst_collboiler_kW"],
        "dh_connection": a["dh_connect"],
        "dh_substation": a["dh_substation"],
        "eol_dh_substation": a["eol_dh_substation"],
        "retrofit_collective_hp": a["retrofit_collhp"],
        "retrofit_dh": a["retrofit_dh"],
        "repair_ind": a["repair_ind"],
        "repair_coll": a["repair_coll_kW"],
        "eol_boiler_ind": a["eol_boiler"] + a["eol_hybrid"],
        "eol_hp_hybrid_unit": a["eol_hybrid"],
        "eol_hp_ae": a["eol_ae"],
        "eol_hp_collective": a["eol_collhp_kW"],
        "eol_boiler_collective": a["eol_collboiler_kW"],
        "refrigerant_R32": lk * a["ref_stock_R32"] + le * a["ref_eol_R32"],
        "refrigerant_R290": lk * a["ref_stock_R290"] + le * a["ref_eol_R290"],
    }


@dataclass
class Inventory:
    """Activity-year demand table (the ABM-LCA coupling interface)."""
    within: dict[str, np.ndarray]   # 2026-2050
    post: dict[str, np.ndarray]     # 2051-2080 (committed)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for per, d, yrs in (("2026-2050", self.within, YEARS), ("committed", self.post, POST_YEARS)):
            for act, arr in d.items():
                for y, v in zip(yrs, arr):
                    if v != 0:
                        rows.append((act, ACTIVITIES[act][0], int(y), per, float(v)))
        return pd.DataFrame(rows, columns=["activity", "unit", "year", "period", "amount"])


def inventory(res: Result, p: dict) -> Inventory:
    return Inventory(_demand(res.act, p), _demand(res.post, p))


def equipment_base(p: dict) -> dict[str, float]:
    """Cradle-to-gate (A1-A3) emissions per activity unit, kg CO2e, before scaling."""
    t = "technologies."
    hc = t + "hp_collective."
    return {
        "boiler_ind": p[t + "boiler_ind.embodied"],
        "hp_hybrid_unit": p[t + "hybrid_ind.hp_embodied"],
        "hp_ae": p[t + "allelectric_ind.embodied"],
        "hp_collective": p[hc + "embodied_per_kw"] * (1 + p[hc + "balance_of_plant"]),
        "boiler_collective": p[t + "boiler_collective.embodied_per_kw"],
        "dh_substation": p[t + "district_heat.substation"],
    }


def unit_factors(p: dict, dh_allocation: str = "dutch") -> dict[str, dict[str, np.ndarray]]:
    """Per-gas emission factor (kg per activity unit) for each year 2026-2050 and post-2050.

    Returns {activity: {gas: array(len(YEARS)+len(POST_YEARS))}}.
    """
    bg = background_series(p, dh_allocation=dh_allocation)
    allyears = np.concatenate([YEARS, POST_YEARS])
    nY = len(allyears)

    def ext(a):
        return np.concatenate([a, np.full(len(POST_YEARS), a[-1])])

    z = np.zeros(nY)
    f: dict[str, dict[str, np.ndarray]] = {a: {g: z.copy() for g in GASES} for a in ACT_LIST}
    f["gas_supply_combustion"]["CO2"] = ext(bg["gas_CO2"])
    f["gas_supply_combustion"]["CH4"] = ext(bg["gas_CH4"])
    f["gas_supply_combustion"]["CO2e"] = ext(bg["gas_chain"])
    f["elec_space_heating"]["CO2e"] = ext(bg["elec_SH"])
    f["elec_hot_water"]["CO2e"] = ext(bg["elec"])
    f["district_heat"]["CO2e"] = ext(bg["dh"])

    t = "technologies."
    scale = p[t + "embodied_scale"]
    decline = (1 - p[t + "embodied_decline"]) ** np.clip(allyears - 2026, 0, 2050 - 2026)
    a45, cshare = p[t + "embodied_A4A5_share"], p[t + "embodied_C_share"]
    base = equipment_base(p)
    prod = lambda b: base[b] * scale * decline * (1 + a45)
    eol = lambda b: np.full(nY, base[b] * scale * cshare)
    f["prod_boiler_ind"]["CO2e"] = prod("boiler_ind")
    f["prod_hp_hybrid_unit"]["CO2e"] = prod("hp_hybrid_unit")
    f["prod_hp_ae"]["CO2e"] = prod("hp_ae")
    f["prod_hp_collective"]["CO2e"] = prod("hp_collective")
    f["prod_boiler_collective"]["CO2e"] = prod("boiler_collective")
    f["dh_substation"]["CO2e"] = prod("dh_substation")
    dh = t + "district_heat."
    pipe = p[dh + "pipe_m_per_apt"] * p[dh + "pipe_kgco2e_per_m"] * p[dh + "civil_works_factor"]
    f["dh_connection"]["CO2e"] = pipe * scale * decline + prod("dh_substation")
    f["retrofit_collective_hp"]["CO2e"] = p[t + "hp_collective.building_retrofit"] * scale * decline
    f["retrofit_dh"]["CO2e"] = p[dh + "building_retrofit"] * scale * decline
    f["eol_dh_substation"]["CO2e"] = eol("dh_substation")
    f["repair_ind"]["CO2e"] = p[t + "repair_share"] * prod("boiler_ind")
    f["repair_coll"]["CO2e"] = p[t + "repair_share"] * prod("boiler_collective")
    f["eol_boiler_ind"]["CO2e"] = eol("boiler_ind")
    f["eol_hp_hybrid_unit"]["CO2e"] = eol("hp_hybrid_unit")
    f["eol_hp_ae"]["CO2e"] = eol("hp_ae")
    f["eol_hp_collective"]["CO2e"] = eol("hp_collective")
    f["eol_boiler_collective"]["CO2e"] = eol("boiler_collective")
    f["refrigerant_R32"]["HFC32"] = np.ones(nY)
    f["refrigerant_R290"]["CO2e"] = np.full(nY, p["gwp100_ar6.R290"])
    return f


def gwp100_weights(p: dict) -> dict[str, float]:
    return {"CO2": 1.0, "CH4": p["gwp100_ar6.CH4_fossil"], "HFC32": p["gwp100_ar6.HFC32"], "CO2e": 1.0}


@dataclass
class Impacts:
    by_act_gas: np.ndarray     # (n_act, n_gas, nY_all) kg
    years_all: np.ndarray
    n_within: int
    gwp: dict

    @property
    def gwp100_by_act(self) -> np.ndarray:   # (n_act, nY_all) kg CO2e
        w = np.array([self.gwp[g] for g in GASES])
        return np.einsum("agy,g->ay", self.by_act_gas, w)

    @property
    def annual(self) -> np.ndarray:          # (nY_all,) kg CO2e
        return self.gwp100_by_act.sum(axis=0)

    def by_category(self) -> pd.DataFrame:
        g = self.gwp100_by_act
        df = pd.DataFrame(g.T, index=self.years_all, columns=ACT_LIST)
        return df.T.groupby([ACTIVITIES[a][1] for a in ACT_LIST]).sum().T.reindex(columns=CATEGORIES)

    def cumulative_within(self) -> float:
        return float(self.annual[: self.n_within].sum())

    def committed(self) -> float:
        return float(self.annual[self.n_within:].sum())

    def gas_totals(self, within: bool = True) -> dict[str, np.ndarray]:
        sl = slice(0, self.n_within) if within else slice(None)
        return {g: self.by_act_gas[:, k, sl].sum(axis=0) for k, g in enumerate(GASES)}

    def co2e_aggregate_share(self) -> float:
        """Share of in-horizon GWP100 carried by pre-aggregated CO2e flows (electricity, DH,
        EPD-based equipment, green gas). The dynamic metric treats these as CO2 pulses."""
        g = self.by_act_gas[:, :, : self.n_within].sum(axis=(0, 2))
        w = np.array([self.gwp[x] for x in GASES])
        return float(g[GASES.index("CO2e")] / (g * w).sum())

    def crf(self, end_year: float = 2100, include_committed: bool = True) -> float:
        sl = slice(None) if include_committed else slice(0, self.n_within)
        gases = {g: self.by_act_gas[:, k, sl].sum(axis=0) for k, g in enumerate(GASES)}
        return dynamic.crf_at(self.years_all[sl], gases, end_year)


def characterize(inv: Inventory, factors: dict, p: dict) -> Impacts:
    years_all = np.concatenate([YEARS, POST_YEARS])
    D = np.stack([np.concatenate([inv.within[a], inv.post[a]]) for a in ACT_LIST])   # (A, Y)
    F = np.stack([np.stack([factors[a][g] for g in GASES]) for a in ACT_LIST])       # (A, G, Y)
    return Impacts(D[:, None, :] * F, years_all, len(YEARS), gwp100_weights(p))


def assess(res: Result, p: dict, dh_allocation: str = "dutch") -> Impacts:
    return characterize(inventory(res, p), unit_factors(p, dh_allocation), p)


def stranded_embodied(res: Result, p: dict) -> float:
    """Embodied emissions (kg CO2e) of systems removed before end of life (attribution metric)."""
    base = equipment_base(p)
    s = p["technologies.embodied_scale"] * (1 + p["technologies.embodied_A4A5_share"])
    a = res.act
    return float(s * (a["stranded_emb_boiler"].sum() * base["boiler_ind"]
                      + a["stranded_emb_hybrid"].sum() * (base["boiler_ind"] + base["hp_hybrid_unit"])
                      + a["stranded_emb_ae"].sum() * base["hp_ae"]
                      + a["stranded_emb_collboiler_kW"].sum() * base["boiler_collective"]))


def summary(res: Result, p: dict, imp: Impacts | None = None) -> dict:
    imp = imp or assess(res, p)
    n = res.n_apt
    cum = imp.cumulative_within()
    com = imp.committed()
    heat = res.act["heat_GJ"].sum()
    heat_post = res.post["heat_GJ"].sum()
    cat = imp.by_category()
    within = cat.iloc[: imp.n_within].sum()
    emb_cats = ["Equipment: production & installation", "Equipment: end-of-life",
                "Heat-network connection & substations", "In-building conversion works"]
    fossil_2050 = res.tech_counts[-1][[0, 1, 5]].sum() / n
    return {
        "scenario": res.scenario, "seed": res.seed,
        "cum_t": cum / 1000, "committed_t": com / 1000, "total_t": (cum + com) / 1000,
        "cum_t_per_apt": cum / 1000 / n, "total_t_per_apt": (cum + com) / 1000 / n,
        "kg_per_GJ": cum / heat, "heat_GJ": float(heat), "heat_post_GJ": float(heat_post),
        "kg_per_GJ_committed": com / heat_post if heat_post > 0 else 0.0,
        "dwelling_years": float(res.act["dwellings"].sum()),
        "co2e_aggregate_share": imp.co2e_aggregate_share(),
        "annual_2026_t": imp.annual[0] / 1000, "annual_2050_t": imp.annual[imp.n_within - 1] / 1000,
        "embodied_share": float(within[emb_cats].sum() / cum),
        "refrigerant_share": float(within["Refrigerant emissions"] / cum),
        "crf_2100": imp.crf(2100), "crf_2100_within": imp.crf(2100, include_committed=False),
        "gasfree_2050": float(res.tech_counts[-1][[2, 3, 4]].sum() / n),
        "fossil_2050": float(fossil_2050),
        "share_dh_2050": float(res.tech_counts[-1][4] / n),
        "share_collhp_2050": float(res.tech_counts[-1][3] / n),
        "share_hybrid_2050": float(res.tech_counts[-1][1] / n),
        "share_ae_2050": float(res.tech_counts[-1][2] / n),
        "boilers_installed": res.metrics.get("boilers_installed", 0),
        "votes": res.metrics.get("votes", 0), "passed": res.metrics.get("passed", 0),
        "sunk_blocked": res.metrics.get("sunk_blocked", 0),
        "stranded_units": res.metrics.get("stranded_units", 0),
        "stranded_emb_t": stranded_embodied(res, p) / 1000,
        "coll_boiler_relock": res.metrics.get("coll_boiler_relock", 0),
        "mean_switch_year": float(np.mean(np.where(res.switch_year > 0, res.switch_year, 2051))),
    }
