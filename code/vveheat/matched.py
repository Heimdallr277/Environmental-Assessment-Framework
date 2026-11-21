"""How much does conversion timing change emissions when the adopters, their feasible technologies 
and the final uptake are held fixed?

Procedure for one neighborhood (seed) and one reference scenario (S0 or S6):
  1. Run the behavioral ABM. Record the *adopter units*: buildings that implemented a collective
     system (implementation year, technology, decision-to-implementation lag) and apartments outside
     them that installed an individual heat pump still in place in 2050 (year of the first heat pump,
     final technology). All other apartments are non-adopters.
  2. Replay the same neighborhood with frozen behavior (like-for-like replacement, no meetings) and
     convert exactly the adopter units, to the same technologies, on SCHEDULED dates. The replay with
     the ABM's own dates is the *reference replay*; every contrast is computed against it, never against
     the behavioral run. Every unit converts by 2050 in every variant, so final uptake is identical.
  3. Hold the reference dates and vary technology within each unit's feasible set.
"""
from __future__ import annotations

import numpy as np

from . import lca
from . import technologies as T
from .abm import Model, Result
from .housing import build
from .scenarios import BASE, SCENARIOS, dh_connection_allowed, dh_decision_window

TIMINGS = ("endogenous", "earliest", "earliest_p3", "earliest_p5", "natural", "minus5", "plus5", "all2050")
REF_VARIANT = "endogenous"
REFERENCE_LABEL = "Reference replay (ABM dates)"
TIMING_LABELS = {"endogenous": REFERENCE_LABEL, "earliest": "Earliest feasible",
                 "earliest_p3": "Earliest + 3 preparation years", "earliest_p5": "Earliest + 5 preparation years",
                 "natural": "At end-of-life", "minus5": "5 years earlier", "plus5": "5 years later",
                 "all2050": "All in 2050"}
PREP = {"earliest": 0, "earliest_p3": 3, "earliest_p5": 5}   # extra preparation years, collective units only
COLL_ASSIGN = {"ref": "reference", "hp": "collective HP where feasible", "dh": "DH where eligible"}
IND_ASSIGN = {"ref": "reference", "ae": "all-electric where feasible", "hy": "hybrid"}
TECH_VARIANTS = tuple(f"c{c}_i{i}" for c in COLL_ASSIGN for i in IND_ASSIGN)      # cref_iref = reference
TECH_LABELS = {f"c{c}_i{i}": f"Collective: {cl}; individual: {il}" for c, cl in COLL_ASSIGN.items()
               for i, il in IND_ASSIGN.items()}
REPLAY = BASE.with_(name="REPLAY", label="Frozen-behavior replay", collective=False, individual="frozen")
NON_ADOPTER, COLL_ADOPTER, IND_ADOPTER = 0, 1, 2
_P_DEFAULT = {"policy.dh_cancel_year": 2032, "institutional.impl_lag_min": 1}


def replay_scenario(sc):
    return REPLAY.with_(dh_arrival=sc.dh_arrival, dh_cancel=sc.dh_cancel, hybrid_standard=sc.hybrid_standard,
                        alignment=sc.alignment)


def adopter_units(res: Result, nb, p: dict) -> dict:
    """Adopter units of a behavioral run."""
    c = res.conversions
    coll = c["bld_year"] >= 0
    lag = np.where(coll, c["bld_year"] - c["bld_decided"], 0)
    lag = np.where(coll & (c["bld_decided"] < 0), 1, lag)
    gas_start = np.isin(c["initial_tech"], (T.BOILER, T.COLL_BOILER))
    hp_end = np.isin(c["final_tech"], (T.HYBRID, T.AE)) & (c["first_hp_year"] >= 2026)
    ind = hp_end & ~coll[nb.bld] & gas_start
    pre_hp = hp_end & ~coll[nb.bld] & ~gas_start
    groups = np.full(nb.N, NON_ADOPTER)
    groups[coll[nb.bld]] = COLL_ADOPTER
    groups[ind] = IND_ADOPTER
    # behavioral diagnostics: first heat pump vs. attainment of the 2050 technology
    att = c["spell_start"]
    return {"coll": coll, "bld_year": c["bld_year"].copy(), "bld_tech": c["bld_tech"].copy(),
            "lag": lag.astype(int), "ind": ind, "apt_year": np.where(ind, c["first_hp_year"], -1),
            "apt_tech": np.where(ind, c["final_tech"], -1), "groups": groups,
            "n_pre2026_hp_flagged": int(pre_hp.sum()),
            "n_ind_first_hp_before_final": int((ind & (c["first_hp_year"] < att)).sum())}


def _earliest(u: dict, nb, sc, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Earliest feasible conversion year per unit (shared DH rule): collective heat pumps 2026 + the
    unit's decision-to-implementation lag; DH first decision year of the network + lag; individual
    heat pumps 2026."""
    w = dh_decision_window(sc, p)
    dh_first = w[0] if w is not None else 10 ** 6
    avail = np.where(u["bld_tech"] == T.DH, dh_first, 2026)
    bld_e = np.where(u["coll"], avail + u["lag"], 10 ** 6)
    apt_e = np.where(u["ind"], 2026, 10 ** 6)
    return bld_e, apt_e


def timing_schedule(u: dict, nb, sc, timing: str, rule: str = "decision", p: dict | None = None,
                    hold_ind: bool = True) -> dict:
    """Scheduled conversion dates of the adopter units for one timing variant."""
    p = p if p is not None else _P_DEFAULT
    bld_e, apt_e = _earliest(u, nb, sc, p)
    by, ay = u["bld_year"].astype(int), u["apt_year"].astype(int)
    if timing == "endogenous":
        b, a = by, ay
    elif timing in PREP:
        b, a = np.minimum(bld_e + PREP[timing], np.where(u["coll"], by, 10 ** 6)), apt_e
    elif timing == "natural":
        med = np.array([np.ceil(np.median(nb.init_end[nb.bld == k])) for k in range(nb.B)]).astype(int)
        occ = np.where(nb.collective_boiler, nb.coll_init_end, med)
        b = np.maximum(bld_e, occ)
        a = np.maximum(apt_e, nb.init_end)
    elif timing == "minus5":
        b, a = np.maximum(bld_e, by - 5), np.maximum(apt_e, ay - 5)
    elif timing == "plus5":
        b, a = np.minimum(2050, by + 5), np.minimum(2050, ay + 5)
    elif timing == "all2050":
        b, a = np.full(nb.B, 2050), np.full(nb.N, 2050)
    else:
        raise ValueError(timing)
    b = np.where(u["coll"], np.minimum(b, 2050), 10 ** 6)
    a = np.where(u["ind"], np.minimum(a, 2050), 10 ** 6)
    if rule == "decision":
        bld_bridge, apt_bridge = u["lag"], np.zeros(nb.N, dtype=int)
    elif rule == "anticipate":
        bld_bridge, apt_bridge = np.full(nb.B, 100), np.full(nb.N, 100)
    else:
        raise ValueError(rule)
    return {"bld_year": b, "bld_tech": np.where(u["coll"], u["bld_tech"], -1), "bld_bridge": bld_bridge,
            "apt_year": a, "apt_tech": np.where(u["ind"], u["apt_tech"], -1), "apt_bridge": apt_bridge,
            "hold_ind": hold_ind}


def feasibility_ok(sch: dict, u: dict, nb, sc, p: dict) -> bool:
    """Every scheduled conversion is feasible at its date (shared rules)."""
    c = u["coll"]
    t = sch["bld_tech"]
    ok_dh = np.all(dh_connection_allowed(sch["bld_year"][c & (t == T.DH)], u["lag"][c & (t == T.DH)], sc, p)) \
        and np.all(nb.dh_zone[c & (t == T.DH)])
    ok_hp = np.all(nb.feasible_coll_hp[c & (t == T.COLL_HP)] | (u["bld_tech"][c & (t == T.COLL_HP)] == T.COLL_HP))
    ok_hp &= np.all(sch["bld_year"][c] >= 2026 + np.minimum(u["lag"][c], 1))
    i = u["ind"]
    ok_ind = np.all(nb.feas_ae[i & (sch["apt_tech"] == T.AE)] | (u["apt_tech"][i & (sch["apt_tech"] == T.AE)] == T.AE))
    ok_ind &= np.all(nb.feas_hybrid[i & (sch["apt_tech"] == T.HYBRID)] | (u["apt_tech"][i & (sch["apt_tech"] == T.HYBRID)] == T.HYBRID))
    return bool(ok_dh and ok_hp and ok_ind)


def tech_schedule(u: dict, nb, sc, variant: str, p: dict | None = None) -> dict:
    """Reference dates."""
    p = p if p is not None else _P_DEFAULT
    sch = timing_schedule(u, nb, sc, "endogenous", p=p)
    bt, at = sch["bld_tech"].copy(), sch["apt_tech"].copy()
    ca, ia = variant[1:].split("_i")
    if ca == "hp":
        bt = np.where(u["coll"] & nb.feasible_coll_hp, T.COLL_HP, bt)
    elif ca == "dh":
        dh_ok = nb.dh_zone & dh_connection_allowed(sch["bld_year"], u["lag"], sc, p)
        bt = np.where(u["coll"] & dh_ok, T.DH, bt)
    elif ca != "ref":
        raise ValueError(variant)
    if ia == "ae":
        at = np.where(u["ind"] & nb.feas_ae, T.AE, at)
    elif ia == "hy":
        at = np.where(u["ind"] & nb.feas_hybrid, T.HYBRID, at)
    elif ia != "ref":
        raise ValueError(variant)
    sch["bld_tech"], sch["apt_tech"] = bt, at
    return sch


def conversion_years(res: Result, u: dict, nb, sch: dict | None = None) -> dict:
    """Apartment-weighted scheduled and REALIZED dates of the adopter units in one run."""
    c = res.conversions
    ca = u["coll"][nb.bld]
    ia = u["ind"]
    conv = c["bld_year"][nb.bld].astype(float)
    att = np.where(ca, conv, np.where(ia, c["spell_start"], np.nan)).astype(float)
    fhp = c["first_hp_year"].astype(float)
    fhp = np.where(fhp >= 2026, fhp, np.inf)
    first = np.where(ca, np.minimum(fhp, conv), np.where(ia, fhp, np.nan))
    mask = ca | ia

    def mean(x, m):
        return float(np.mean(x[m])) if m.any() else float("nan")

    out = {"realized_year": mean(att, mask), "realized_year_coll": mean(att, ca), "realized_year_ind": mean(att, ia),
           "first_hp_year": mean(first, mask)}
    if sch is not None:
        sched = np.where(ca, sch["bld_year"][nb.bld], sch["apt_year"]).astype(float)
        out["scheduled_year"] = mean(sched, mask)
        out["n_realized_ne_sched"] = int((att[mask] != sched[mask]).sum())
        out["n_ind_realized_ne_sched"] = int((att[ia] != sched[ia]).sum())
        out["n_first_hp_before_sched"] = int((first[mask] < sched[mask]).sum())
        out["n_ind_first_hp_before_sched"] = int((first[ia] < sched[ia]).sum())
        out["n_coll_first_hp_before_sched"] = int((first[ca] < sched[ca]).sum())
    return out


def group_impacts(res: Result, p: dict) -> dict:
    """In-horizon and committed GWP100 (t) and useful heat (GJ) per unit group."""
    out = {}
    f = lca.unit_factors(p)
    for g, act in res.group_act.items():
        r = Result(res.scenario, res.seed, res.years, act, res.group_post[g], res.tech_counts, res.n_apt)
        imp = lca.characterize(lca.inventory(r, p), f, p)
        out[g] = {"cum_t": imp.cumulative_within() / 1e3, "committed_t": imp.committed() / 1e3,
                  "heat_GJ": float(act["heat_GJ"].sum()), "heat_post_GJ": float(res.group_post[g]["heat_GJ"].sum()),
                  "dwellings": float(act["dwellings"][0])}
    return out


def run_matched(p: dict, seed: int, ref: str = "S0", rules=("decision",), asc=None,
                hold_ind_variants: bool = True) -> list[dict]:
    """All timing and technology variants for one neighborhood and reference scenario."""
    sc = SCENARIOS[ref]
    nb = build(p, seed)
    m = Model(p, sc, seed, nb=nb, asc=asc)
    base = m.run()
    asc = m.asc
    base_imp = lca.assess(base, p)
    u = adopter_units(base, nb, p)
    abm_years = conversion_years(base, u, nb)          # realized dates in the behavioral run
    jobs = [("timing", t, r, True) for r in rules for t in TIMINGS]
    if hold_ind_variants:                              # sensitivity: standard applies to individual adopters
        jobs += [("timing_std", t, rules[0], False) for t in TIMINGS]
    jobs += [("tech", v, rules[0], True) for v in TECH_VARIANTS if v != "cref_iref"]
    runs = []
    for kind, variant, rule, hold in jobs:
        if kind.startswith("timing"):
            sch = timing_schedule(u, nb, sc, variant, rule, p, hold_ind=hold)
        else:
            sch = tech_schedule(u, nb, sc, variant, p)
        rep = Model(p, replay_scenario(sc), seed, nb=nb, asc=asc, schedule=sch, groups=u["groups"]).run()
        runs.append((kind, variant, rule, hold, sch, rep))
    # reference replay of each (kind, rule) family for the regression checks
    refs = {(k, r): rep for k, v, r, h, sch, rep in runs if v == REF_VARIANT}
    rows = []
    for kind, variant, rule, hold, sch, rep in runs:
        imp = lca.assess(rep, p)
        gi = group_impacts(rep, p)
        heat = rep.act["heat_GJ"].sum()
        yrs = conversion_years(rep, u, nb, sch)
        rref = refs.get((kind, rule), refs[("timing", rules[0])])
        adopters = u["groups"] > 0
        row = {"seed": seed, "ref": ref, "kind": kind, "variant": variant, "rule": rule, "hold_ind": hold,
               "cum_t": imp.cumulative_within() / 1e3, "committed_t": imp.committed() / 1e3,
               "kg_per_GJ": imp.cumulative_within() / heat, "heat_GJ": heat,
               "heat_post_GJ": rep.post["heat_GJ"].sum(),
               "abm_cum_t": base_imp.cumulative_within() / 1e3, "abm_committed_t": base_imp.committed() / 1e3,
               "abm_gasfree_2050": float(np.isin(base.final_tech, T.GAS_FREE).mean()),
               "n_final_ne_abm": int((rep.final_tech != base.final_tech).sum()),
               "n_final_ne_abm_adopters": int(((rep.final_tech != base.final_tech) & adopters).sum()),
               "n_apt": nb.N, "n_coll_bld": int(u["coll"].sum()), "n_coll_apt": int((u["groups"] == COLL_ADOPTER).sum()),
               "n_ind_apt": int(u["ind"].sum()), "n_pre2026_hp_flagged": u["n_pre2026_hp_flagged"],
               "n_ind_first_hp_before_final_abm": u["n_ind_first_hp_before_final"],
               "gasfree_2050": float(np.isin(rep.final_tech, T.GAS_FREE).mean()),
               "adopters_2050": float(adopters.mean()), **yrs,
               **{f"abm_{k}": v for k, v in abm_years.items()}}
        for g, name in ((NON_ADOPTER, "non"), (COLL_ADOPTER, "coll"), (IND_ADOPTER, "ind")):
            d = gi.get(g, {"cum_t": 0.0, "committed_t": 0.0, "heat_GJ": 0.0, "heat_post_GJ": 0.0})
            row[f"{name}_cum_t"] = d["cum_t"]
            row[f"{name}_committed_t"] = d["committed_t"]
            row[f"{name}_heat_GJ"] = d["heat_GJ"]
            row[f"{name}_kg_per_GJ"] = d["cum_t"] * 1e3 / d["heat_GJ"] if d["heat_GJ"] > 0 else np.nan
        ad_heat = row["coll_heat_GJ"] + row["ind_heat_GJ"]
        row["adopter_cum_t"] = row["coll_cum_t"] + row["ind_cum_t"]
        row["adopter_kg_per_GJ"] = row["adopter_cum_t"] * 1e3 / ad_heat if ad_heat > 0 else np.nan
        # regression checks against the reference replay of the same family
        gref = group_impacts(rref, p)
        row["chk_heat_equal"] = bool(np.isclose(heat, rref.act["heat_GJ"].sum(), rtol=1e-12))
        row["chk_non_adopters_unchanged"] = bool(np.isclose(gi[NON_ADOPTER]["cum_t"], gref[NON_ADOPTER]["cum_t"],
                                                            rtol=1e-9, atol=1e-9))
        if kind.startswith("timing"):
            row["chk_final_tech_by_dwelling"] = bool(np.array_equal(rep.final_tech, rref.final_tech))
        else:
            row["chk_final_tech_by_dwelling"] = bool(np.array_equal(rep.final_tech[~adopters], rref.final_tech[~adopters]))
        row["chk_feasible_at_date"] = feasibility_ok(sch, u, nb, sc, p)
        row["chk_collective_dates_realized"] = bool(np.array_equal(
            rep.conversions["bld_year"][u["coll"]], sch["bld_year"][u["coll"]]))
        row["chk_individual_dates_realized"] = (yrs["n_ind_realized_ne_sched"] == 0) if hold else True
        rows.append(row)
    return rows


def main():
    """Run the paired timing comparison for the selected neighborhood seeds."""
    import argparse

    import pandas as pd

    from .params import Registry

    pathways = {"slow": 0.0, "moderate": 0.5, "fast": 1.0}
    parser = argparse.ArgumentParser(description="Compare heating conversion dates in matched neighborhoods.")
    parser.add_argument("--seeds", type=int, default=40, help="number of neighborhoods, using seeds 1 through N")
    parser.add_argument("--scenario", choices=("S0", "S6"), default="S0", help="adopter population")
    parser.add_argument("--grid", choices=tuple(pathways), default="moderate")
    parser.add_argument("--repairs", choices=("decision", "anticipate"), default="decision")
    parser.add_argument("--csv", action="store_true", help="print the summary as a full-precision CSV")
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")

    parameters = Registry.load().central()
    parameters["electricity.pathway_index"] = pathways[args.grid]
    records = []
    for seed in range(1, args.seeds + 1):
        rows = run_matched(parameters, seed, ref=args.scenario, rules=(args.repairs,), hold_ind_variants=False)
        for row in rows:
            failed = [name for name, passed in row.items() if name.startswith("chk_") and not passed]
            if failed:
                raise RuntimeError(f"Seed {seed}, schedule {row['variant']} failed {', '.join(failed)}")
            if row["kind"] == "timing":
                records.append(row)

    totals = pd.DataFrame(records).pivot(
        index="seed", columns="variant", values=["cum_t", "committed_t", "coll_cum_t", "ind_cum_t", "heat_GJ"]
    )

    def paired_change(metric):
        values = totals[metric]
        return values.sub(values[REF_VARIANT], axis="index")

    change = paired_change("cum_t")
    percent = change.div(totals["cum_t"][REF_VARIANT], axis="index") * 100
    intensity = (change * 1000).div(totals["heat_GJ"][REF_VARIANT], axis="index")
    summary = pd.DataFrame({
        "mean_pct": percent.mean(),
        "p05_pct": percent.quantile(0.05),
        "p95_pct": percent.quantile(0.95),
        "mean_t": change.mean(),
        "kg_per_GJ": intensity.mean(),
        "collective_t": paired_change("coll_cum_t").mean(),
        "individual_t": paired_change("ind_cum_t").mean(),
        "post2050_t": paired_change("committed_t").mean(),
        "n_lower": change.lt(0).sum(),
    }).reindex(TIMINGS)
    summary.index.name = "schedule"
    if args.csv:
        print(summary.to_csv(lineterminator="\n"), end="")
    else:
        print(f"Synthetic neighborhoods (n={args.seeds}), {args.scenario} adopters, {args.grid} grid, {args.repairs} repairs")
        print(summary.to_string(float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
