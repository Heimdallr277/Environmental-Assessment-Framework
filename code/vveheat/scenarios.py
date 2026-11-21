"""Scenario and policy definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    hybrid_standard: bool = False      # S1: hybrid-or-better at boiler end-of-life (incl. apartments where feasible)
    majority_reform: bool = False      # S2: simple majority for collective sustainability decisions
    collective_subsidy: bool = False   # S3: extra subsidy on collective HP and DH connections
    alignment: bool = False            # S5: mandatory heating plan in MJOP years + bridging repairs
    dh_majority: bool = False          # S2d (hypothetical): strict simple majority for DH offers only
    dh_cancel: bool = False            # S4c: announced heat network canceled in policy.dh_cancel_year
    collective: bool = True            # VvE decision-making enabled (False = individual-only, E4)
    individual: str = "choice"         # 'choice' | 'frozen' (like-for-like replacement, E3)
    dh_arrival: int | None = 2035      # year the heat network reaches zone buildings (None: never)
    overrides: dict = field(default_factory=dict)

    def with_(self, **kw) -> "Scenario":
        return replace(self, **kw)


BASE = Scenario("S0", "Current policy (common 2/3 deed rule)")
SCENARIOS: dict[str, Scenario] = {
    "S0": BASE,
    "S1": BASE.with_(name="S1", label="Hybrid standard 2029", hybrid_standard=True),
    "S2": BASE.with_(name="S2", label="Simple-majority reform", majority_reform=True),
    "S2d": BASE.with_(name="S2d", label="Simple majority for DH offers (hypothetical)", dh_majority=True),
    "S3": BASE.with_(name="S3", label="Collective subsidy +20 pp", collective_subsidy=True),
    "S23": BASE.with_(name="S23", label="Simple majority + collective subsidy", majority_reform=True,
                      collective_subsidy=True),
    "S4e": BASE.with_(name="S4e", label="Heat network 2030", dh_arrival=2030),
    "S4l": BASE.with_(name="S4l", label="Heat network 2040", dh_arrival=2040),
    "S4n": BASE.with_(name="S4n", label="No heat network (known)", dh_arrival=None),
    "S4c": BASE.with_(name="S4c", label="Network announced, canceled 2032", dh_cancel=True),
    "S5": BASE.with_(name="S5", label="Replacement alignment", alignment=True),
    "S6": BASE.with_(name="S6", label="Package (S1+S2+S3+S5)", hybrid_standard=True,
                     majority_reform=True, collective_subsidy=True, alignment=True),
    "E3": BASE.with_(name="E3", label="Frozen technology (like-for-like)", collective=False,
                     individual="frozen"),
    "E4": BASE.with_(name="E4", label="Individual decisions only", collective=False),
}

def dh_decision_window(sc: Scenario, p: dict):
    if sc.dh_arrival is None:
        return None
    last = 10 ** 6
    if sc.dh_cancel:
        last = int(p["policy.dh_cancel_year"]) - 1
    if last < sc.dh_arrival:
        return None
    return int(sc.dh_arrival), last


def dh_connection_allowed(year, prep, sc: Scenario, p: dict):
    w = dh_decision_window(sc, p)
    year = np.asarray(year)
    if w is None:
        return np.zeros(year.shape, dtype=bool)
    dec = year - np.asarray(prep)
    return (dec >= w[0]) & (dec <= w[1])


def dh_first_connection_year(sc: Scenario, p: dict, prep: int | None = None):
    w = dh_decision_window(sc, p)
    if w is None:
        return None
    return w[0] + int(p["institutional.impl_lag_min"] if prep is None else prep)


POLICY_SET = ["S0", "S1", "S2", "S2d", "S3", "S23", "S4e", "S4l", "S4n", "S4c", "S5", "S6"]
