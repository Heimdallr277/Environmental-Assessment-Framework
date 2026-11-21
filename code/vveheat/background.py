"""Time-explicit background emission factors (electricity, gas, district heat).

All factors are returned per simulation year as numpy arrays and are split by
greenhouse gas so that both static GWP100 and dynamic radiative-forcing metrics
can be computed from the same inventory.
"""
from __future__ import annotations

import numpy as np

YEARS = np.arange(2026, 2051)
TECHS_MIX = ["gas", "coal", "other_fossil", "wind", "solar", "nuclear", "biomass", "hydro"]
PATHWAYS = ("slow", "moderate", "fast")


def _lifecycle_vector(p: dict) -> np.ndarray:
    return np.array([p[f"electricity.lifecycle_factor.{t}"] for t in TECHS_MIX], dtype=float)


def mix_based_intensity_2025(p: dict) -> float:
    mix = np.array([p[f"electricity.mix_2025.{t}"] for t in TECHS_MIX])
    return float(mix @ _lifecycle_vector(p))


def grid_intensity_pathway(p: dict, pathway: str, years: np.ndarray = YEARS) -> np.ndarray:
    """Lifecycle grid intensity (kg CO2e/kWh, at generation) for one named pathway.

    Mix shares are interpolated linearly between 2025 (observed, Ember) and the scenario
    anchor years; the mix-based intensity is scaled so that 2025 reproduces the Ember value.
    """
    lc = _lifecycle_vector(p)
    anchors = {2025: np.array([p[f"electricity.mix_2025.{t}"] for t in TECHS_MIX])}
    for y, shares in p["_tables"]["pathway_mix"][pathway].items():
        anchors[int(y)] = np.asarray(shares, dtype=float)
    ay = np.array(sorted(anchors))
    ints = np.array([anchors[y] / anchors[y].sum() @ lc for y in ay])
    scale = p["electricity.ember_intensity_2025"] / ints[0]
    return np.interp(years, ay, ints * scale) / 1000.0


def grid_intensity(p: dict, years: np.ndarray = YEARS) -> np.ndarray:
    """Grid intensity for the continuous pathway index (0 slow, 0.5 moderate, 1 fast)."""
    x = float(p["electricity.pathway_index"])
    s, m, f = (grid_intensity_pathway(p, k, years) for k in PATHWAYS)
    if x <= 0.5:
        w = x / 0.5
        return (1 - w) * s + w * m
    w = (x - 0.5) / 0.5
    return (1 - w) * m + w * f


def greengas_share(p: dict, years: np.ndarray = YEARS) -> np.ndarray:
    tab = p["_tables"]["greengas_share"]
    def curve(d):
        ys = np.array(sorted(int(k) for k in d))
        return np.interp(years, ys, [d[y] for y in ys])
    x = float(p["gas.greengas_index"])
    return (1 - x) * curve(tab["low"]) + x * curve(tab["high"])


def dh_intensity(p: dict, years: np.ndarray = YEARS, allocation: str = "dutch") -> np.ndarray:
    """District-heat intensity at the end user, kg CO2e/GJ."""
    ay = np.array([2026, 2030, 2040, 2050])
    av = np.array([p["district_heat.ef_2026"], p["district_heat.ef_2030"],
                   p["district_heat.ef_2040"], p["district_heat.ef_2050"]])
    base = np.interp(years, ay, av)
    # scale the post-2026 level, never exceeding the 2026 anchor
    out = np.where(years <= 2026, base, np.minimum(base * p["district_heat.ef_scale"], av[0]))
    if allocation == "burdened":
        avi = np.interp(years, [2022, 2030, 2050],
                        [p["district_heat.avi_share_2022"], p["district_heat.avi_share_2030"],
                         p["district_heat.avi_share_2030"]])
        out = out + avi * p["district_heat.avi_fossil_alloc"]
    return out


def gas_factors(p: dict, years: np.ndarray = YEARS) -> dict[str, np.ndarray]:
    """Per m3 of delivered gas: kg CO2 (fossil), kg CH4 (fossil), kg CO2e (green-gas chain)."""
    gg = greengas_share(p, years)
    wtt = p["gas.upstream_CO2e_ar4"]
    ch4_share = p["gas.upstream_ch4_share"]
    ch4_kg = wtt * ch4_share / p["gas.upstream_gwp_ch4_source"]
    co2_up = wtt * (1 - ch4_share)
    return {
        "CO2": (1 - gg) * (p["gas.combustion_CO2"] + co2_up),
        "CH4": (1 - gg) * ch4_kg,
        "CO2e_chain": gg * p["gas.greengas_chain"],
        "greengas_share": gg,
    }


def background_series(p: dict, dh_allocation: str = "dutch") -> dict[str, np.ndarray]:
    """All background factors for YEARS. Units: kg/kWh, kg/m3, kg/GJ."""
    grid = grid_intensity(p) * p["electricity.grid_loss_factor"]
    g = gas_factors(p)
    return {
        "elec": grid,                                       # kg CO2e per kWh delivered (annual avg)
        "elec_SH": grid * p["electricity.winter_weight_SH"],  # heating-season weighted
        "gas_CO2": g["CO2"], "gas_CH4": g["CH4"], "gas_chain": g["CO2e_chain"],
        "greengas_share": g["greengas_share"],
        "dh": dh_intensity(p, allocation=dh_allocation),    # kg CO2e per GJ delivered
    }
