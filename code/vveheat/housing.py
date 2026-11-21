"""Synthetic neighborhood of multi-apartment (VvE) buildings.

The neighborhood is SYNTHETIC: building sizes, tenure, demand and technical feasibility are
sampled from the distributions in the parameter registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gamma

import networkx as nx
import numpy as np

from . import technologies as tech

OWNER, LANDLORD, CORPORATION = 0, 1, 2


@dataclass
class Neighborhood:
    # buildings
    n_apt: np.ndarray            # (B,)
    period: np.ndarray           # (B,) construction-period index
    collective_boiler: np.ndarray
    dh_zone: np.ndarray
    feasible_coll_hp: np.ndarray
    mjop_offset: np.ndarray
    neighbors: list             # list of arrays of neighbor building indices
    # apartments
    bld: np.ndarray              # (N,) building index of each apartment
    tenure: np.ndarray
    gas_m3: np.ndarray           # reference boiler gas use (SH + DHW + cooking share)
    q_sh: np.ndarray
    q_dhw: np.ndarray
    env: np.ndarray
    disc: np.ndarray
    feas_hybrid: np.ndarray
    feas_ae: np.ndarray
    init_tech: np.ndarray
    init_end: np.ndarray         # year in which the initial individual system is replaced
    coll_init_end: np.ndarray    # (B,) for collective-boiler buildings
    life_draw_u: np.ndarray      # (N, G) uniforms for successive equipment lives
    coll_life_draw_u: np.ndarray  # (B, G)

    @property
    def B(self) -> int:
        return len(self.n_apt)

    @property
    def N(self) -> int:
        return len(self.bld)


def weibull_life(mean, shape: float, u: np.ndarray) -> np.ndarray:
    """Integer service life (>= 1 yr) by inverse-CDF sampling; ``mean`` may be an array."""
    scale = np.asarray(mean, dtype=float) / gamma(1 + 1 / shape)
    return np.maximum(1, np.round(scale * (-np.log(1 - u)) ** (1 / shape))).astype(int)


def build(p: dict, seed: int, start_year: int = 2026, generations: int = 8) -> Neighborhood:
    rng = np.random.default_rng([seed, 1001])
    h = "housing."
    B = int(p[h + "n_buildings"])
    med, sig = p[h + "apt_per_building_median"], p[h + "apt_per_building_sigma"]
    n_apt = np.clip(np.round(rng.lognormal(np.log(med), sig, B)),
                    p[h + "apt_per_building_min"], p[h + "apt_per_building_max"]).astype(int)
    ps = p["_tables"]["period_shares"]
    shares = np.asarray(ps["share"], dtype=float)
    mult = np.asarray(ps["demand_multiplier"], dtype=float)
    mult = mult / (shares @ mult)                    # keeps the population mean at the CBS value
    period = rng.choice(len(shares), size=B, p=shares / shares.sum())
    coll = rng.random(B) < p[h + "share_collective_boiler"]
    zone = rng.random(B) < p[h + "dh_zone_share"]
    feas_coll = rng.random(B) < p[h + "feasible_collective_hp"]
    mjop = rng.integers(0, int(p["institutional.mjop_interval"]), B)

    G = nx.connected_watts_strogatz_graph(B, int(p["behavior.network_k"]),
                                          p["behavior.network_p"], seed=int(rng.integers(1 << 31)))
    neighbors = [np.array(sorted(G.neighbors(b)), dtype=int) for b in range(B)]

    bld = np.repeat(np.arange(B), n_apt)
    N = bld.size
    ten_p = np.array([p[h + "tenure_owner_occupier"], p[h + "tenure_private_landlord"],
                      p[h + "tenure_corporation"]])
    tenure = rng.choice(3, size=N, p=ten_p / ten_p.sum())
    cv = p[h + "gas_use_cv"]
    s2 = np.log(1 + cv ** 2)
    noise = rng.lognormal(-s2 / 2, np.sqrt(s2), N)
    gas = p[h + "gas_use_mean_m3"] * mult[period[bld]] * noise
    q_sh, q_dhw = tech.useful_heat(p, gas)
    env = rng.beta(p["behavior.env_beta_a"], p["behavior.env_beta_b"], N)
    disc = p["behavior.discount_rate_median"] * rng.lognormal(0, p["behavior.discount_sigma"], N)
    u_feas = rng.random(N)
    feas_h = u_feas < p[h + "feasible_hybrid"]
    feas_ae = u_feas < p[h + "feasible_allelectric"]      # nested: AE-feasible => hybrid-feasible

    # initial systems: individual boilers with a small share of early heat-pump adopters
    init = np.where(coll[bld], tech.COLL_BOILER, tech.BOILER)
    u0 = rng.random(N)
    init = np.where((init == tech.BOILER) & feas_ae & (u0 < 0.03), tech.AE, init)
    init = np.where((init == tech.BOILER) & feas_h & (u0 > 0.95), tech.HYBRID, init)

    life_u = rng.random((N, generations))
    coll_life_u = rng.random((B, generations))
    shape = p["technologies.life_weibull_shape"]
    life_mean = tech.mean_life(p)
    first_life = weibull_life(life_mean[init], shape, life_u[:, 0])
    # stationary age structure: remaining life uniform on {0, ..., life-1} years after start
    age = np.floor(rng.random(N) * first_life).astype(int)
    init_end = start_year + (first_life - 1 - age)
    cl = weibull_life(life_mean[tech.COLL_BOILER], shape, coll_life_u[:, 0])
    cage = np.floor(rng.random(B) * cl).astype(int)
    coll_init_end = np.where(coll, start_year + cl - 1 - cage, 10 ** 6)
    return Neighborhood(n_apt, period, coll, zone, feas_coll, mjop, neighbors, bld, tenure, gas,
                         q_sh, q_dhw, env, disc, feas_h, feas_ae, init, init_end, coll_init_end,
                         life_u, coll_life_u)
