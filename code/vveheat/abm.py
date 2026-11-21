"""Agent-based model of heating-system transitions in multi-apartment (VvE) buildings.

Agents
  * apartment owners (owner-occupiers, private landlords, housing corporations) who replace
    individual systems at end-of-life (multinomial logit) and vote in VvE meetings;
  * VvEs (one per building) that place collective options on the agenda, vote under a
    quorum + majority rule counted on integer votes, and implement approved projects after a lag.

Interaction
  * descriptive social norm (heat-pump share in own and network-neighbor buildings);
  * Bayesian (beta-binomial) learning about heat-pump reliability from observed systems;
  * collective decisions bind all apartments (individual systems become stranded).

Coupling to the LCA
  The model returns physical activity quantities by year (energy, installations,
  end-of-life, refrigerant stock); ``lca.inventory`` turns these into an activity-year
  demand table. Households perceive only operational emissions using current-year
  background factors (a light feedback from the LCA background to behavior).

Common random numbers: all stochastic draws are pre-generated from the seed with fixed
shapes, so different scenarios run with the same seed share the same random environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import technologies as T
from .background import YEARS, background_series
from .calibration import calibrate_asc
from .housing import CORPORATION, LANDLORD, OWNER, Neighborhood, build, weibull_life
from .scenarios import Scenario, dh_decision_window

# committed (post-2050) accounting runs until the last retirement of the 2050 stock
POST_YEARS = np.arange(2051, 2131)
R290, R32 = 1, 2

ACT_KEYS = [
    "gas_m3", "elec_SH_kWh", "elec_DHW_kWh", "dh_GJ", "heat_GJ",
    "inst_boiler", "inst_hybrid", "inst_ae", "inst_collhp_kW", "inst_collboiler_kW",
    "dh_connect", "dh_substation", "retrofit_collhp", "retrofit_dh",
    "eol_boiler", "eol_hybrid", "eol_ae", "eol_collhp_kW", "eol_collboiler_kW", "eol_dh_substation",
    "repair_ind", "repair_coll_kW",
    "ref_stock_R290", "ref_stock_R32", "ref_eol_R290", "ref_eol_R32",
    "stranded_emb_boiler", "stranded_emb_hybrid", "stranded_emb_ae", "stranded_emb_collboiler_kW",
    "subsidy_eur", "dwellings",
]

# vote rules
STRICT_MAJORITY = ("strict_majority",)


def vote_passes(yes: int, cast: int, rule: tuple) -> bool:
    """Exact integer vote counting."""
    if cast <= 0:
        return False
    if rule[0] == "frac":
        return rule[2] * yes >= rule[1] * cast
    if rule[0] == "strict_majority":
        return 2 * yes > cast
    raise ValueError(rule)


@dataclass
class Result:
    scenario: str
    seed: int
    years: np.ndarray
    act: dict                      # activity quantities, (25,) arrays
    post: dict                     # committed activity quantities after 2050
    tech_counts: np.ndarray        # (25, 6) apartments by technology at end of year
    n_apt: int
    metrics: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    final_tech: np.ndarray | None = None
    switch_year: np.ndarray | None = None   # first year each apartment left fossil heating
    belief_mean: np.ndarray | None = None   # (25,) mean perceived HP reliability
    conversions: dict | None = None         # per-unit conversion records (for replays)
    group_act: dict | None = None           # {group: act} when groups were given
    group_post: dict | None = None


class Model:
    def __init__(self, p: dict, scenario: Scenario, seed: int, nb: Neighborhood | None = None,
                 asc: np.ndarray | None = None, schedule: dict | None = None,
                 dh_allocation: str = "dutch", groups: np.ndarray | None = None):
        self.p, self.sc, self.seed = p, scenario, seed
        self.nb = nb if nb is not None else build(p, seed)
        self.asc = asc if asc is not None else calibrate_asc(p)["asc"]
        self.schedule = schedule
        self.bg = background_series(p, dh_allocation=dh_allocation)
        nb = self.nb
        N, B, Y = nb.N, nb.B, len(YEARS)
        rng = np.random.default_rng([seed, 3003])
        # ---- pre-drawn random numbers (common random numbers across scenarios)
        self.gumbel = -np.log(-np.log(rng.random((Y, N, 3))))
        self.u_vote = rng.random((Y, N))
        self.u_turn = rng.random((Y, N))
        self.u_turn2 = rng.random((Y, N))
        self.u_agenda = rng.random((Y, B))
        self.u_lag = rng.random((Y, B))
        self.u_ref = rng.random((Y, N))
        # ---- static per-apartment quantities
        self.flows6 = T.energy_flows(p, nb.q_sh, nb.q_dhw)
        self.op6 = T.operational_cost(p, self.flows6)
        h = p["behavior.horizon_years"]
        self.crf = T.crf(nb.disc, h)
        self.crf_corp = float(T.crf(0.04, h))
        self.life_mean = T.mean_life(p)
        self.shape = p["technologies.life_weibull_shape"]
        self.apts_of = [np.flatnonzero(nb.bld == b) for b in range(B)]
        self.kw_hp = p["technologies.hp_collective.kw_per_apt"]
        self.kw_cb = p["technologies.boiler_collective.kw_per_apt"]
        # ---- unit groups for split accounting (apartment-level; building = group of its apartments)
        self.groups = None if groups is None else np.asarray(groups, dtype=int)
        if self.groups is not None:
            self.n_groups = int(self.groups.max()) + 1
            self.bgroup = np.array([np.bincount(self.groups[a], minlength=self.n_groups).argmax()
                                    for a in self.apts_of])
        self._init_state()

    # ------------------------------------------------------------------ state
    def _init_state(self):
        nb, p = self.nb, self.p
        N, B = nb.N, nb.B
        self.tech = nb.init_tech.copy()
        self.tech0 = self.tech.copy()
        self.end = nb.init_end.copy()
        self.life = self.life_mean[self.tech].astype(float)   # nominal life of the installed system
        self.inst_year = np.full(N, 2025)
        self.gen = np.ones(N, dtype=int)
        self.bridged = np.zeros(N, dtype=bool)
        self.ref_kind = np.where(np.isin(self.tech, (T.HYBRID, T.AE)), R32, 0)
        self.ref_charge = np.where(self.tech == T.HYBRID, p["technologies.hybrid_ind.charge_R32"],
                                   np.where(self.tech == T.AE, p["technologies.allelectric_ind.charge_R32"], 0.0))
        self.btech = np.where(nb.collective_boiler, T.COLL_BOILER, -1)
        self.bend = nb.coll_init_end.copy()
        self.blife = np.full(B, self.life_mean[T.COLL_BOILER])
        self.bgen = np.ones(B, dtype=int)
        self.bref = np.zeros(B)
        self.pending = np.full(B, -1)
        self.pending_tech = np.full(B, -1)
        self.pending_from = np.full(B, -10 ** 6)
        self.decided = np.full(B, -1)
        self.conv_year = np.full(B, -1)
        self.conv_tech = np.full(B, -1)
        self.first_hp_year = np.full(N, -1)
        self.spell_start = np.full(N, 2025)     # year the current technology of each apartment started
        self.cooldown = np.full(B, -1)
        if self.schedule is not None:
            sch = self.schedule
            sb = (sch["bld_year"] <= 2050) & (sch["bld_tech"] >= 0)
            self.pending = np.where(sb, np.maximum(sch["bld_year"], 2026), -1)
            self.pending_tech = np.where(sb, sch["bld_tech"], -1)
            bridge = np.asarray(sch.get("bld_bridge", np.full(B, 100)))
            self.pending_from = np.where(sb, self.pending - bridge, -10 ** 6)
            self.apt_bridge = np.asarray(sch.get("apt_bridge", np.zeros(N, dtype=int)))
        a0 = p["behavior.prior_rel_mean"] * p["behavior.prior_rel_strength"]
        self.ba = np.full(N, a0)
        self.bb = np.full(N, p["behavior.prior_rel_strength"] - a0)
        self.switch_year = np.where(np.isin(self.tech, T.GAS_FREE), 2025, 0)
        self.act = {k: np.zeros(len(YEARS)) for k in ACT_KEYS}
        if self.groups is not None:
            self.gact = {g: {k: np.zeros(len(YEARS)) for k in ACT_KEYS} for g in range(self.n_groups)}
        self.tech_counts = np.zeros((len(YEARS), 6), dtype=int)
        self.belief_mean = np.zeros(len(YEARS))
        self.events: list[dict] = []
        self.metrics = {"votes": 0, "passed": 0, "passed_dh": 0, "passed_hp": 0,
                        "sunk_blocked": 0, "quorum_fail": 0, "bridging": 0,
                        "boilers_installed": 0, "stranded_units": 0,
                        "stranded_remaining_years": 0.0, "coll_boiler_relock": 0,
                        "votes_exact_threshold": 0}

    # ------------------------------------------------------------------ accounting
    def _acc(self, key: str, yi: int, amount, apts: np.ndarray | None = None, bld: int | None = None):
        """Add an activity quantity; split it between unit groups when groups are defined.

        ``apts``: apartment indices with per-apartment amounts (array) or one amount each (scalar).
        ``bld``: a building index; the whole amount goes to the building's group.
        """
        if apts is not None:
            vals = np.broadcast_to(np.asarray(amount, dtype=float), apts.shape)
            self.act[key][yi] += vals.sum()
            if self.groups is not None and apts.size:
                gs = np.bincount(self.groups[apts], weights=vals, minlength=self.n_groups)
                for g in range(self.n_groups):
                    self.gact[g][key][yi] += gs[g]
        else:
            self.act[key][yi] += float(amount)
            if self.groups is not None:
                self.gact[self.bgroup[bld]][key][yi] += float(amount)

    # ------------------------------------------------------------------ helpers
    def _draw_life(self, tech_code: int, u: np.ndarray) -> np.ndarray:
        return weibull_life(self.life_mean[tech_code], self.shape, u)

    def _apt_life_u(self, idx: np.ndarray) -> np.ndarray:
        g = np.minimum(self.gen[idx], self.nb.life_draw_u.shape[1] - 1)
        return self.nb.life_draw_u[idx, g]

    def _bld_life_u(self, b: int) -> float:
        g = min(self.bgen[b], self.nb.coll_life_draw_u.shape[1] - 1)
        return self.nb.coll_life_draw_u[b, g]

    def _zone_active(self, t: int) -> bool:
        """Heat-network zoning is in force (announced or built) in year t."""
        sc = self.sc
        if sc.dh_arrival is None:
            return False
        if sc.dh_cancel and t >= self.p["policy.dh_cancel_year"]:
            return False
        return True

    def _policy(self, t: int) -> dict:
        p, sc = self.p, self.sc
        sub = p["prices.subsidy_share_hp"]
        extra = p["policy.collective_subsidy_extra"] if (sc.collective_subsidy and t >= 2027) else 0.0
        deed = ("frac", int(p["institutional.deed_num"]), int(p["institutional.deed_den"]))
        rule_hp = STRICT_MAJORITY if (sc.majority_reform and t >= p["policy.majority_reform_year"]) else deed
        rule_dh = rule_hp
        if sc.dh_majority and t >= p["policy.dh_majority_year"]:
            rule_dh = STRICT_MAJORITY
        w = dh_decision_window(sc, p)
        dh_av = w is not None and w[0] <= t <= w[1]
        return {"cap": T.capex(p, subsidy_ind=sub, subsidy_coll=sub, extra_coll=extra),
                "grant": T.subsidy_per_apartment(p, subsidy_ind=sub, subsidy_coll=sub, extra_coll=extra),
                "rule_hp": rule_hp, "rule_dh": rule_dh, "dh_available": dh_av,
                "standard": sc.hybrid_standard and t >= p["policy.hybrid_standard_year"]}

    def _remove_individual(self, idx: np.ndarray, yi: int, t: int, stranded: bool):
        """End-of-life accounting for individual systems in ``idx``."""
        for code, key in ((T.BOILER, "boiler"), (T.HYBRID, "hybrid"), (T.AE, "ae")):
            m = idx[self.tech[idx] == code]
            if m.size == 0:
                continue
            self._acc(f"eol_{key}", yi, 1.0, apts=m)
            if stranded:
                rem = np.clip(self.end[m] - t, 0, None)
                frac = np.clip(rem / np.maximum(self.life[m], 1), 0, 1)
                self._acc(f"stranded_emb_{key}", yi, frac, apts=m)
                self.metrics["stranded_units"] += int((rem > 0).sum())
                self.metrics["stranded_remaining_years"] += float(rem.sum())
        r = idx[self.ref_kind[idx] == R290]
        self._acc("ref_eol_R290", yi, self.ref_charge[r], apts=r)
        r = idx[self.ref_kind[idx] == R32]
        self._acc("ref_eol_R32", yi, self.ref_charge[r], apts=r)
        self.ref_kind[idx] = 0
        self.ref_charge[idx] = 0.0

    def _install_individual(self, idx: np.ndarray, new: np.ndarray, yi: int, t: int, pol: dict):
        p = self.p
        for code, key in ((T.BOILER, "boiler"), (T.HYBRID, "hybrid"), (T.AE, "ae")):
            m = idx[new == code]
            if m.size == 0:
                continue
            self._acc(f"inst_{key}", yi, 1.0, apts=m)
            self._acc("subsidy_eur", yi, pol["grant"][code], apts=m)
            if code == T.BOILER:
                self.metrics["boilers_installed"] += m.size
            if code in (T.HYBRID, T.AE):
                share = p["technologies.refrigerant.r32_share_2026"] if t <= 2026 \
                    else p["technologies.refrigerant.r32_share_post2026"]
                r32 = self.u_ref[yi, m] < share
                kind = np.where(r32, R32, R290)
                tk = "hybrid_ind" if code == T.HYBRID else "allelectric_ind"
                ch = np.where(r32, p[f"technologies.{tk}.charge_R32"], p[f"technologies.{tk}.charge_R290"])
                self.ref_kind[m] = kind
                self.ref_charge[m] = ch
                first = m[self.first_hp_year[m] < 0]
                self.first_hp_year[first] = t
        self.gen[idx] += 1
        u = self._apt_life_u(idx)
        lives = weibull_life(self.life_mean[new], self.shape, u)
        self.spell_start[idx[self.tech[idx] != new]] = t
        self.tech[idx] = new
        self.inst_year[idx] = t
        self.end[idx] = t + lives
        self.life[idx] = lives
        self.bridged[idx] = False
        gf = idx[np.isin(new, T.GAS_FREE) & (self.switch_year[idx] == 0)]
        self.switch_year[gf] = t

    # ------------------------------------------------------------------ social
    def _social(self):
        nb = self.nb
        is_hp = np.isin(self.tech, T.HEAT_PUMPS)
        hp_b = np.bincount(nb.bld, weights=is_hp, minlength=nb.B)
        share_b = hp_b / nb.n_apt
        coll_b = np.isin(self.btech, (T.COLL_HP, T.DH)).astype(float)
        w = self.p["behavior.soc_own_weight"]
        nshare = np.array([share_b[n].mean() if n.size else 0.0 for n in nb.neighbors])
        ncoll = np.array([coll_b[n].mean() if n.size else 0.0 for n in nb.neighbors])
        soc_hp = w * share_b + (1 - w) * nshare
        return hp_b, share_b, soc_hp, ncoll

    def _learn(self, yi: int, hp_b: np.ndarray):
        nb, p = self.nb, self.p
        rng = np.random.default_rng([self.seed, 7, yi])
        nsum = np.array([hp_b[n].sum() for n in nb.neighbors])
        lam = hp_b + p["behavior.obs_neighbor_rate"] * nsum
        n_obs = np.minimum(rng.poisson(lam[nb.bld]), int(p["behavior.obs_cap"]))
        succ = rng.binomial(n_obs, p["behavior.true_rel"])
        self.ba += succ
        self.bb += n_obs - succ

    # ------------------------------------------------------------------ collective
    def _implement(self, b: int, ctech: int, yi: int, t: int, pol: dict):
        p = self.p
        idx = self.apts_of[b]
        n = idx.size
        ind = idx[np.isin(self.tech[idx], T.INDIVIDUAL)]
        if ind.size:
            self._remove_individual(ind, yi, t, stranded=True)
        if self.btech[b] == T.COLL_BOILER:
            kw = self.kw_cb * n
            self._acc("eol_collboiler_kW", yi, kw, bld=b)
            rem = max(self.bend[b] - t, 0)
            self._acc("stranded_emb_collboiler_kW", yi, kw * min(rem / self.blife[b], 1.0), bld=b)
        if ctech == T.COLL_HP:
            kw = self.kw_hp * n
            self._acc("inst_collhp_kW", yi, kw, bld=b)
            self._acc("retrofit_collhp", yi, float(ind.size), bld=b)
            self.bref[b] = p["technologies.hp_collective.charge_R290_per_kw"] * kw
        elif ctech == T.DH:
            self._acc("dh_connect", yi, float(n), bld=b)
            self._acc("retrofit_dh", yi, float(ind.size), bld=b)
            self.bref[b] = 0.0
        self._acc("subsidy_eur", yi, pol["grant"][ctech] * n, bld=b)
        self.btech[b] = ctech
        self.spell_start[idx] = t
        self.tech[idx] = ctech
        self.bgen[b] += 1
        life = self._draw_life(ctech, np.array([self._bld_life_u(b)]))[0]
        self.bend[b] = t + life
        self.blife[b] = life
        self.pending[b] = -1
        self.conv_year[b] = t
        self.conv_tech[b] = ctech
        new_sw = idx[self.switch_year[idx] == 0]
        self.switch_year[new_sw] = t
        self.events.append({"year": t, "building": int(b), "type": "implement",
                            "tech": T.SHORT[ctech], "n_apt": int(n), "decided": int(self.decided[b])})

    def _vote(self, b: int, prop: int, yi: int, t: int, pol: dict, soc_coll: float) -> bool:
        p, nb = self.p, self.nb
        idx = self.apts_of[b]
        n = idx.size
        cap = pol["cap"]
        cur = self.tech[idx]
        crf = self.crf[idx]
        op = self.op6
        cap_cur = cap[cur]   # replacement capex of the current system (what owners would otherwise pay)
        if self.btech[b] == T.COLL_BOILER:
            f_rem = np.full(n, np.clip((self.bend[b] - t) / self.blife[b], 0, 1))
        else:
            f_rem = np.clip((self.end[idx] - t) / np.maximum(self.life[idx], 1), 0, 1)
        ac_sq = cap_cur * crf + op[cur, idx]
        ac_c = cap[prop] * crf + op[prop, idx]
        sunk = p["institutional.sunk_cost_weight"] * f_rem * cap_cur * crf
        ghg = self._ghg
        u_base = (p["behavior.beta_env"] * nb.env[idx] * (ghg[cur, idx] - ghg[prop, idx])
                  + p["behavior.beta_soc"] * soc_coll - p["institutional.inertia"])
        if prop == T.COLL_HP:
            r = self.ba[idx] / (self.ba[idx] + self.bb[idx])
            u_base = u_base + p["behavior.beta_rel"] * (r - p["behavior.prior_rel_mean"])
        bc = p["behavior.beta_cost"]
        U = u_base + bc * (ac_c - ac_sq + sunk) / 100.0
        U_ns = u_base + bc * (ac_c - ac_sq) / 100.0
        ten = nb.tenure[idx]
        P = np.where(ten == OWNER, 1 / (1 + np.exp(-U)), 0.0)
        P_ns = np.where(ten == OWNER, 1 / (1 + np.exp(-U_ns)), 0.0)
        P = np.where(ten == LANDLORD, p["institutional.landlord_yes_prob"], P)
        P_ns = np.where(ten == LANDLORD, p["institutional.landlord_yes_prob"], P_ns)
        corp_yes = float(cap[prop] <= p["institutional.corporation_budget"])
        P = np.where(ten == CORPORATION, corp_yes, P)
        P_ns = np.where(ten == CORPORATION, corp_yes, P_ns)
        to = p["institutional.turnout"]
        turn_p = np.where(ten == OWNER, to, np.where(ten == LANDLORD, 0.5 * to, 1.0))
        part = self.u_turn[yi, idx] < turn_p
        if part.sum() / n < p["institutional.quorum"]:
            self.metrics["quorum_fail"] += 1
            part = self.u_turn2[yi, idx] < turn_p          # second meeting, no quorum requirement
        cast = int(part.sum())
        if cast == 0:
            return False
        yes = int(((self.u_vote[yi, idx] < P) & part).sum())
        yes_ns = int(((self.u_vote[yi, idx] < P_ns) & part).sum())
        rule = pol["rule_dh"] if prop == T.DH else pol["rule_hp"]
        passed = vote_passes(yes, cast, rule)
        passed_ns = vote_passes(yes_ns, cast, rule)
        if rule[0] == "frac" and rule[2] * yes == rule[1] * cast:
            self.metrics["votes_exact_threshold"] += 1
        self.metrics["votes"] += 1
        if passed_ns and not passed:
            self.metrics["sunk_blocked"] += 1
        self.events.append({"year": t, "building": int(b), "type": "vote", "tech": T.SHORT[prop],
                            "yes": yes, "cast": cast, "yes_share": yes / cast,
                            "rule": "strict_majority" if rule[0] == "strict_majority" else f"{rule[1]}/{rule[2]}",
                            "turnout": float(part.mean()), "passed": passed, "n_apt": int(n)})
        return passed

    def _meetings(self, yi: int, t: int, pol: dict, share_b: np.ndarray, ncoll: np.ndarray):
        p, nb, sc = self.p, self.nb, self.sc
        I = "institutional."
        zone_on = self._zone_active(t)
        for b in range(nb.B):
            if self.pending[b] >= 0 or self.btech[b] in (T.COLL_HP, T.DH) or self.cooldown[b] > t:
                continue
            in_zone = nb.dh_zone[b] and zone_on
            if in_zone and pol["dh_available"]:
                prop = T.DH
            elif nb.feasible_coll_hp[b]:
                prop = T.COLL_HP
            else:
                continue
            mjop = (t - 2026 - nb.mjop_offset[b]) % int(p[I + "mjop_interval"]) == 0
            pa = p[I + "agenda_base"] + p[I + "agenda_mjop_boost"] * mjop \
                + p[I + "agenda_interest_boost"] * share_b[b]
            if prop == T.DH and t - sc.dh_arrival <= 2:
                pa += p[I + "agenda_dh_offer_boost"]
            if nb.n_apt[b] < p[I + "small_vve_size"]:
                pa *= p[I + "small_vve_factor"]
            if in_zone and not pol["dh_available"] and prop == T.COLL_HP:
                pa *= 0.25                                    # waiting for the announced heat network
            if sc.alignment and mjop:
                pa = 1.0
            forced = self.btech[b] == T.COLL_BOILER and self.bend[b] - t <= 1
            if forced:
                pa = 1.0
            if self.u_agenda[yi, b] >= min(pa, 1.0):
                continue
            if self._vote(b, prop, yi, t, pol, ncoll[b]):
                self.metrics["passed"] += 1
                self.metrics["passed_dh" if prop == T.DH else "passed_hp"] += 1
                lmin, lmax = int(p[I + "impl_lag_min"]), int(p[I + "impl_lag_max"])
                lag = lmin + int(self.u_lag[yi, b] * (lmax - lmin + 1))
                self.pending[b] = t + lag
                self.pending_tech[b] = prop
                self.pending_from[b] = t
                self.decided[b] = t
            else:
                self.cooldown[b] = t + int(p[I + "cooldown"])

    # ------------------------------------------------------------------ individual choice
    def _choose(self, idx: np.ndarray, yi: int, t: int, pol: dict, soc_hp: np.ndarray) -> np.ndarray:
        p, nb = self.p, self.nb
        if self.sc.individual == "frozen":
            out = self.tech[idx].copy()
            if pol["standard"]:
                zone = nb.dh_zone[nb.bld[idx]] & self._zone_active(t)
                must = (out == T.BOILER) & nb.feas_hybrid[idx] & ~zone
                if self.schedule is not None and self.schedule.get("hold_ind", True):
                    held = np.isin(self.schedule["apt_tech"][idx], T.INDIVIDUAL) & (self.schedule["apt_year"][idx] > t)
                    must &= ~held
                out[must] = T.HYBRID
            return out
        opts = (T.BOILER, T.HYBRID, T.AE)
        cap, op, ghg = pol["cap"], self.op6, self._ghg
        ten = nb.tenure[idx]
        crf = np.where(ten == CORPORATION, self.crf_corp, self.crf[idx])
        env = np.where(ten == OWNER, nb.env[idx], np.where(ten == CORPORATION, 0.7, 0.0))
        r = self.ba[idx] / (self.ba[idx] + self.bb[idx])
        soc = soc_hp[nb.bld[idx]]
        V = np.empty((idx.size, 3))
        m = "technologies.maintenance."
        maint = {T.BOILER: p["technologies.boiler_ind.maintenance"], T.HYBRID: p[m + "hybrid_ind"],
                 T.AE: p[m + "allelectric_ind"]}
        for k, j in enumerate(opts):
            energy_cost = np.where(ten == LANDLORD, maint[j], op[j, idx])   # split incentive
            cost = cap[j] * crf + energy_cost
            V[:, k] = (self.asc[k] + p["behavior.beta_cost"] * cost / 100.0
                       + p["behavior.beta_env"] * env * (ghg[T.BOILER, idx] - ghg[j, idx]))
            if j != T.BOILER:
                V[:, k] += p["behavior.beta_soc"] * soc + p["behavior.beta_rel"] * (r - p["behavior.prior_rel_mean"])
        feas = np.ones((idx.size, 3), dtype=bool)
        feas[:, 1] = nb.feas_hybrid[idx]
        feas[:, 2] = nb.feas_ae[idx]
        if pol["standard"]:
            zone = nb.dh_zone[nb.bld[idx]] & self._zone_active(t)
            feas[:, 0] = ~(nb.feas_hybrid[idx] & ~zone)
        U = np.where(feas, V + self.gumbel[yi, idx, :], -np.inf)
        return np.array(opts)[np.argmax(U, axis=1)]

    # ------------------------------------------------------------------ main loop
    def step(self, yi: int, t: int):
        p, nb = self.p, self.nb
        pol = self._policy(t)
        self._ghg = T.operational_ghg(p, self.flows6, self.bg, yi)
        hp_b, share_b, soc_hp, ncoll = self._social()

        # A. implement approved / scheduled collective projects (start of year)
        for b in np.flatnonzero(self.pending == t):
            self._implement(b, int(self.pending_tech[b]), yi, t, pol)
        # A'. exogenous early individual switches (replays)
        if self.schedule is not None:
            self._apply_schedule(yi, t, pol)

        # B. collective system end-of-life
        for b in np.flatnonzero((self.bend == t) & (self.btech >= 0)):
            n = nb.n_apt[b]
            bt = self.btech[b]
            if bt == T.COLL_BOILER and self.pending[b] > t and t >= self.pending_from[b]:
                self._acc("repair_coll_kW", yi, self.kw_cb * n, bld=b)
                self.bend[b] = self.pending[b]
                continue
            if bt == T.COLL_BOILER:
                self._acc("eol_collboiler_kW", yi, self.kw_cb * n, bld=b)
                self._acc("inst_collboiler_kW", yi, self.kw_cb * n, bld=b)
                self._acc("subsidy_eur", yi, pol["grant"][T.COLL_BOILER] * n, bld=b)
                self.metrics["coll_boiler_relock"] += 1
            elif bt == T.COLL_HP:
                kw = self.kw_hp * n
                self._acc("eol_collhp_kW", yi, kw, bld=b)
                self._acc("inst_collhp_kW", yi, kw, bld=b)
                self._acc("ref_eol_R290", yi, self.bref[b], bld=b)
                self._acc("subsidy_eur", yi, pol["grant"][T.COLL_HP] * n, bld=b)
            elif bt == T.DH:
                self._acc("eol_dh_substation", yi, float(n), bld=b)
                self._acc("dh_substation", yi, float(n), bld=b)
            self.bgen[b] += 1
            life = self._draw_life(bt, np.array([self._bld_life_u(b)]))[0]
            self.bend[b] = t + life
            self.blife[b] = life

        # C. individual end-of-life
        fail = np.flatnonzero(np.isin(self.tech, T.INDIVIDUAL) & (self.end <= t))
        if fail.size:
            bfail = nb.bld[fail]
            pend = self.pending[bfail]
            known = (pend > t) & (t >= self.pending_from[bfail])
            rep = fail[known]
            if rep.size:
                self._acc("repair_ind", yi, 1.0, apts=rep)
                self.end[rep] = self.pending[nb.bld[rep]]
            fail = fail[~known]
            if self.schedule is not None and fail.size:
                fail = self._schedule_bridge(fail, yi, t)
            if self.sc.alignment and fail.size:
                if self.schedule is not None:      # replays: a due scheduled conversion is never bridged
                    due = (self.schedule["apt_year"][fail] <= t) & np.isin(self.schedule["apt_tech"][fail], T.INDIVIDUAL)
                    fail = np.sort(np.concatenate([self._bridge(fail[~due], yi, t), fail[due]]))
                else:
                    fail = self._bridge(fail, yi, t)
            if fail.size:
                new = self._choose(fail, yi, t, pol, soc_hp)
                if self.schedule is not None:
                    tgt = self.schedule["apt_tech"][fail]
                    due = (self.schedule["apt_year"][fail] <= t) & np.isin(tgt, T.INDIVIDUAL)
                    new = np.where(due, tgt, new)
                self._remove_individual(fail, yi, t, stranded=False)
                self._install_individual(fail, new, yi, t, pol)

        # D. operation in year t
        f = self.flows6
        cols = np.arange(nb.N)
        heat = nb.q_sh + nb.q_dhw
        self._acc("gas_m3", yi, f["gas"][self.tech, cols], apts=cols)
        self._acc("elec_SH_kWh", yi, f["elec_SH"][self.tech, cols], apts=cols)
        self._acc("elec_DHW_kWh", yi, f["elec_DHW"][self.tech, cols], apts=cols)
        self._acc("dh_GJ", yi, f["dh"][self.tech, cols], apts=cols)
        self._acc("heat_GJ", yi, heat, apts=cols)
        self._acc("dwellings", yi, 1.0, apts=cols)
        r290 = np.flatnonzero(self.ref_kind == R290)
        self._acc("ref_stock_R290", yi, self.ref_charge[r290], apts=r290)
        for b in np.flatnonzero(self.bref > 0):
            self._acc("ref_stock_R290", yi, self.bref[b], bld=b)
        r32 = np.flatnonzero(self.ref_kind == R32)
        self._acc("ref_stock_R32", yi, self.ref_charge[r32], apts=r32)
        self.tech_counts[yi] = np.bincount(self.tech, minlength=6)

        # E. VvE meetings (decisions take effect after the implementation lag)
        if self.sc.collective:
            self._meetings(yi, t, pol, share_b, ncoll)

        # F. learning from observed heat pumps
        self._learn(yi, np.bincount(nb.bld, weights=np.isin(self.tech, T.HEAT_PUMPS), minlength=nb.B))
        self.belief_mean[yi] = float(np.mean(self.ba / (self.ba + self.bb)))

    def _bridge(self, fail: np.ndarray, yi: int, t: int) -> np.ndarray:
        """S5: repair instead of replace if a collective decision is due within the window."""
        p, nb, sc = self.p, self.nb, self.sc
        w = int(p["policy.bridging_window"])
        b = nb.bld[fail]
        interval = int(p["institutional.mjop_interval"])
        to_mjop = (nb.mjop_offset[b] - (t - 2026)) % interval
        to_dh = np.full(fail.size, 10 ** 6)
        if self._zone_active(t):
            zone = nb.dh_zone[b]
            to_dh = np.where(zone & (sc.dh_arrival > t), sc.dh_arrival - t, 10 ** 6)
        gap = np.minimum(np.where(to_mjop == 0, interval, to_mjop), to_dh)
        ok = (gap <= w) & ~self.bridged[fail] & (self.cooldown[b] <= t) \
            & np.isin(self.btech[b], (-1,))
        br = fail[ok]
        if br.size:
            self._acc("repair_ind", yi, 1.0, apts=br)
            self.end[br] = t + gap[ok] + 1
            self.bridged[br] = True
            self.metrics["bridging"] += int(br.size)
        return fail[~ok]

    def _schedule_bridge(self, fail: np.ndarray, yi: int, t: int) -> np.ndarray:
        sch = self.schedule
        ay = sch["apt_year"][fail]
        ok = np.isin(sch["apt_tech"][fail], T.INDIVIDUAL) & (ay > t) & (ay - t <= self.apt_bridge[fail])
        br = fail[ok]
        if br.size:
            self._acc("repair_ind", yi, 1.0, apts=br)
            self.end[br] = ay[ok]
        return fail[~ok]

    def _apply_schedule(self, yi: int, t: int, pol: dict):
        sch = self.schedule
        idx = np.flatnonzero((sch["apt_year"] == t) & np.isin(self.tech, T.INDIVIDUAL)
                             & np.isin(sch["apt_tech"], T.INDIVIDUAL) & (self.end > t)
                             & (self.tech != sch["apt_tech"]))
        if idx.size:
            self._remove_individual(idx, yi, t, stranded=True)
            self._install_individual(idx, sch["apt_tech"][idx], yi, t, pol)

    def run(self) -> Result:
        for yi, t in enumerate(YEARS):
            self.step(yi, int(t))
        post, gpost = self._committed()
        conv = {"bld_year": self.conv_year.copy(), "bld_tech": self.conv_tech.copy(),
                "bld_decided": self.decided.copy(), "first_hp_year": self.first_hp_year.copy(),
                "spell_start": self.spell_start.copy(), "initial_tech": self.tech0.copy(),
                "final_tech": self.tech.copy(), "btech": self.btech.copy()}
        return Result(self.sc.name, self.seed, YEARS, self.act, post,
                      self.tech_counts, self.nb.N, self.metrics, self.events,
                      final_tech=self.tech.copy(), switch_year=self.switch_year.copy(),
                      belief_mean=self.belief_mean.copy(), conversions=conv,
                      group_act=self.gact if self.groups is not None else None, group_post=gpost)

    # ------------------------------------------------------------------ committed (post-2050)
    def _committed(self):
        """Operation of the 2050 stock until its scheduled retirement, with background factors frozen at 2050. 
        Projects decided but not implemented by 2050 are outside this stock-only convention."""
        nb, f = self.nb, self.flows6
        P = len(POST_YEARS)
        G = self.n_groups if self.groups is not None else 1
        grp = self.groups if self.groups is not None else np.zeros(nb.N, dtype=int)
        bgrp = self.bgroup if self.groups is not None else np.zeros(nb.B, dtype=int)
        posts = [{k: np.zeros(P) for k in ACT_KEYS} for _ in range(G)]
        end = np.where(np.isin(self.tech, T.INDIVIDUAL), self.end, self.bend[nb.bld])
        assert end.max() <= POST_YEARS[-1], "extend POST_YEARS: equipment outlives the accounting window"
        cols = np.arange(nb.N)
        tt = self.tech
        heat = nb.q_sh + nb.q_dhw

        def gsum(vals, mask):
            return np.bincount(grp[mask], weights=vals[mask], minlength=G)

        def bsum(vals, mask):
            return np.bincount(bgrp[mask], weights=vals[mask], minlength=G)

        for k, y in enumerate(POST_YEARS):
            alive = end > y
            dying = end == y
            ind_alive = alive & np.isin(tt, T.INDIVIDUAL)
            bd = self.bend == y
            sums = {
                "gas_m3": gsum(f["gas"][tt, cols], alive),
                "elec_SH_kWh": gsum(f["elec_SH"][tt, cols], alive),
                "elec_DHW_kWh": gsum(f["elec_DHW"][tt, cols], alive),
                "dh_GJ": gsum(f["dh"][tt, cols], alive),
                "heat_GJ": gsum(heat, alive),
                "dwellings": gsum(np.ones(nb.N), alive),
                "ref_stock_R290": gsum(self.ref_charge, ind_alive & (self.ref_kind == R290))
                + bsum(self.bref, (self.bend > y) & (self.btech == T.COLL_HP)),
                "ref_stock_R32": gsum(self.ref_charge, ind_alive & (self.ref_kind == R32)),
                "eol_boiler": gsum(np.ones(nb.N), dying & (tt == T.BOILER)),
                "eol_hybrid": gsum(np.ones(nb.N), dying & (tt == T.HYBRID)),
                "eol_ae": gsum(np.ones(nb.N), dying & (tt == T.AE)),
                "ref_eol_R290": gsum(self.ref_charge, dying & (self.ref_kind == R290))
                + bsum(self.bref, bd & (self.btech == T.COLL_HP)),
                "ref_eol_R32": gsum(self.ref_charge, dying & (self.ref_kind == R32)),
                "eol_collhp_kW": bsum(self.kw_hp * nb.n_apt.astype(float), bd & (self.btech == T.COLL_HP)),
                "eol_collboiler_kW": bsum(self.kw_cb * nb.n_apt.astype(float), bd & (self.btech == T.COLL_BOILER)),
                "eol_dh_substation": bsum(nb.n_apt.astype(float), bd & (self.btech == T.DH)),
            }
            for key, arr in sums.items():
                for g in range(G):
                    posts[g][key][k] = arr[g]
        total = {k: sum(posts[g][k] for g in range(G)) for k in ACT_KEYS}
        return total, (dict(enumerate(posts)) if self.groups is not None else None)


def run(p: dict, scenario: Scenario, seed: int, **kw) -> Result:
    return Model(p, scenario, seed, **kw).run()
