"""Parameter registry: loading, flattening, provenance and sampling.

All model inputs live in ``code/vveheat/data/parameters.yaml``. 
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
# canonical registry ships as package data
DEFAULT_REGISTRY = Path(__file__).resolve().parent / "data" / "parameters.yaml"

GROUPS = ("behavior", "institutional", "technical", "background")


def _walk(node: Any, prefix: str = "") -> Iterator[tuple[str, dict]]:
    if isinstance(node, dict):
        if "value" in node and not isinstance(node["value"], dict):
            yield prefix, node
            return
        for k, v in node.items():
            yield from _walk(v, f"{prefix}.{k}" if prefix else str(k))


@dataclass
class Registry:
    raw: dict
    leaves: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_REGISTRY) -> "Registry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        reg = cls(raw=raw)
        for key, leaf in _walk({k: v for k, v in raw.items() if k != "sources"}):
            reg.leaves[key] = leaf
        return reg

    # ------------------------------------------------------------------ access
    def central(self) -> dict[str, Any]:
        """Flat dict of central values plus non-leaf tables (pathways, shares)."""
        flat = {k: leaf["value"] for k, leaf in self.leaves.items()}
        flat["_tables"] = {
            "pathway_mix": self.raw["electricity"]["pathway_mix"],
            "greengas_share": self.raw["gas"]["greengas_share"],
            "period_shares": self.raw["housing"]["period_shares"],
        }
        return flat

    def uncertain(self, gsa_only: bool = False) -> dict[str, dict]:
        out = {}
        for k, leaf in self.leaves.items():
            if "dist" not in leaf:
                continue
            if gsa_only and not leaf.get("gsa", False):
                continue
            out[k] = leaf
        return out

    def group_of(self, key: str) -> str:
        g = self.leaves[key].get("group", "technical")
        # map registry groups onto the four variance-decomposition groups
        return {"behavior": "behavior", "institutional": "institutional",
                "housing": "technical", "technical": "technical", "cost": "behavior",
                "background": "background", "method": "background"}[g]

    def provenance_table(self) -> list[dict]:
        rows = []
        for k, leaf in self.leaves.items():
            rows.append({
                "key": k, "value": leaf["value"], "unit": leaf.get("unit", ""),
                "type": leaf.get("type", ""), "source": leaf.get("source", ""),
                "group": leaf.get("group", ""), "dist": _fmt_dist(leaf.get("dist")),
                "gsa": bool(leaf.get("gsa", False)), "note": leaf.get("note", ""),
            })
        return rows

    def sources(self) -> dict[str, str]:
        return dict(self.raw["sources"])


def _fmt_dist(d: dict | None) -> str:
    if not d:
        return ""
    t = d["type"]
    if t == "uniform":
        return f"U({d['low']}, {d['high']})"
    if t == "triangular":
        return f"Tri({d['low']}, {d['mode']}, {d['high']})"
    if t == "loguniform":
        return f"logU({d['low']}, {d['high']})"
    return str(d)


def ppf(dist: dict, u: np.ndarray) -> np.ndarray:
    """Inverse CDF for the supported distributions (u in [0,1])."""
    t = dist["type"]
    if t == "uniform":
        return dist["low"] + u * (dist["high"] - dist["low"])
    if t == "loguniform":
        lo, hi = np.log(dist["low"]), np.log(dist["high"])
        return np.exp(lo + u * (hi - lo))
    if t == "triangular":
        a, c, b = dist["low"], dist["mode"], dist["high"]
        fc = (c - a) / (b - a)
        return np.where(u < fc, a + np.sqrt(u * (b - a) * (c - a)),
                        b - np.sqrt((1 - u) * (b - a) * (b - c)))
    raise ValueError(t)


def with_overrides(base: dict, overrides: dict | None) -> dict:
    p = copy.copy(base)
    if overrides:
        for k, v in overrides.items():
            if k not in p and not k.startswith("_"):
                raise KeyError(f"unknown parameter {k}")
            p[k] = v
    return p


def sample_lhs(reg: Registry, n: int, seed: int, keys: list[str] | None = None) -> list[dict]:
    """Latin-hypercube Monte Carlo sample of all uncertain parameters (independent)."""
    rng = np.random.default_rng(seed)
    unc = reg.uncertain()
    keys = keys or list(unc)
    base = reg.central()
    U = np.empty((n, len(keys)))
    for j in range(len(keys)):
        U[:, j] = (rng.permutation(n) + rng.random(n)) / n
    out = []
    for i in range(n):
        ov = {k: float(ppf(unc[k]["dist"], np.array(U[i, j]))) for j, k in enumerate(keys)}
        out.append(with_overrides(base, ov))
    return out
