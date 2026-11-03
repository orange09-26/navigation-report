# -*- coding: utf-8 -*-
"""Offline policy probes using the exact benchmark and executor."""

import os
import sys
import tempfile

sys.path.insert(0, "/home/orange")
import mapgpt_v5 as v
import mapgpt_vm_demo_v4 as d

maps = v.load_maps("/home/orange/benchmark_maps_v5.json")[:8]
ORIGINAL_CHOOSE = v.choose_deterministic


def utility(candidates, risk_aware):
    return ORIGINAL_CHOOSE(candidates, risk_aware) if candidates else None


def nav_cost(candidates, risk_aware):
    if not candidates:
        return None
    goals = [c for c in candidates if c["kind"] == "GOAL"]
    if goals:
        return min(goals, key=lambda c: c["path_len"])
    return min(
        candidates,
        key=lambda c: (
            c["path_len"] + c["goal_h"],
            0.25 * c["risk"] - 0.10 * c["gain"],
            c["cell"],
        ),
    )


def goal_plus_info(candidates, risk_aware):
    if not candidates:
        return None
    goals = [c for c in candidates if c["kind"] == "GOAL"]
    if goals:
        return min(goals, key=lambda c: c["path_len"])
    return min(
        candidates,
        key=lambda c: (
            c["path_len"] + 0.82 * c["goal_h"] + 1.8 * c["risk"] - 1.0 * c["gain"],
            c["cell"],
        ),
    )


def step_efficient(candidates, risk_aware):
    if not candidates:
        return None
    goals = [c for c in candidates if c["kind"] == "GOAL"]
    if goals:
        return min(goals, key=lambda c: c["path_len"])
    return min(
        candidates,
        key=lambda c: (
            c["path_len"] + 2.0 * c["risk"],
            0.55 * c["goal_h"] - c["gain"],
            c["cell"],
        ),
    )


for name, fn in [
    ("utility", utility),
    ("nav_cost", nav_cost),
    ("goal_info", goal_plus_info),
    ("step_efficient", step_efficient),
]:
    v.choose_deterministic = fn
    v.apply_risk_veto = lambda chosen, candidates: (chosen, "")
    recs = []
    fd, path = tempfile.mkstemp()
    os.close(fd)
    logger = d.RunLogger(path)
    for spec in maps:
        env = v.FixedBenchmarkEnv(spec)
        rec, _ = v.run_hierarchical_condition(
            "risk_waypoint", env, logger, 80, "probe", True
        )
        recs.append(rec)
    logger.close()
    m = v.aggregate(recs)
    print(name, {
        k: round(m[k], 4) for k in ("SR", "SPL", "AP", "COV", "STEPS")
    })
    print(" ", [(r["map_id"], r["success"], r["steps"], r["actual"]) for r in recs])
