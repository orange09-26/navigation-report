# -*- coding: utf-8 -*-
"""MapGPT v10: adaptive macro-branch navigation.

Key additions over v6/v9:
1. Macro branches group multiple frontier clusters at a shared decision branch.
2. Large maps support step-level receding-horizon replanning.
3. Observation size adapts to corridor geometry and committed goal mode.
4. Complexity-aware abstraction and predicted/optimistic cost blending.
5. Multi-resolution predictor fusion and alternative waypoints remain as
   explicit ablation options.

The effective v10 profile keeps the tested geometry observation policy and
step-level replanning on large maps. Legacy return-budget, entropy, regret and
LLM tie-break options remain only for reproducible negative ablations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(os.getenv("MAPGPT_ROOT", "/home/orange"))
sys.path.insert(0, str(ROOT))
import mapgpt_v5 as v5
import mapgpt_v6 as v6
from map_prediction_v6 import LogisticMapPredictor

Cell = Tuple[int, int]
DIRS = v5.DIRS
NMS_DIST = int(os.getenv("V7_NMS_DIST", "4"))
BRANCH_MAX = int(os.getenv("V7_BRANCH_MAX", "2"))
TOPK = int(os.getenv("V7_TOPK", "6"))
ROUTE_REPLAN_STEPS = int(os.getenv("V10_ROUTE_REPLAN_STEPS", "0"))
PRED_WEIGHT = float(os.getenv("V10_PRED_WEIGHT", "-1"))
WAYPOINT_MODE = os.getenv("V10_WAYPOINT_MODE", "entry")
OBS_POLICY = os.getenv("V10_OBS_POLICY", "fixed")
FIXED_OBS_SIZE = int(os.getenv("MAPGPT_OBS_SIZE", "5"))
BUDGET_GATE = float(os.getenv("V10_BUDGET_GATE", "0"))
BUDGET_MULT = float(os.getenv("V10_BUDGET_MULT", "1.5"))
BUDGET_MARGIN = int(os.getenv("V10_BUDGET_MARGIN", "10"))
BUDGET_GOAL_W = float(os.getenv("V10_BUDGET_GOAL_W", "2.0"))
SCORE_W = {
    "path": float(os.getenv("V7_W_PATH", "1.20")),
    "goal": float(os.getenv("V7_W_GOAL", "0.80")),
    "risk": float(os.getenv("V7_W_RISK", "2.60")),
    "info": float(os.getenv("V7_W_INFO", "0.30")),
    "unknown": float(os.getenv("V7_W_UNKNOWN", "0.20")),
    "uncertainty": float(os.getenv("V7_W_UNCERTAINTY", "0.60")),
}


def _goal_score_distance(
    condition: str, pred_h: float, optimistic_h: float
) -> float:
    """Continuous macro/micro cost blending used by v10 ablations."""
    if PRED_WEIGHT >= 0:
        weight = min(1.0, max(0.0, PRED_WEIGHT))
        return weight * pred_h + (1.0 - weight) * optimistic_h
    return optimistic_h if condition in (
        "macro_hybrid_common", "macro_hybrid_common_llm"
    ) else pred_h


def _select_waypoint(
    members: Sequence[Dict[str, Any]], fallback: Dict[str, Any]
) -> Dict[str, Any]:
    """Select a branch representative without changing branch membership."""
    if not members or WAYPOINT_MODE == "entry":
        return fallback
    if WAYPOINT_MODE == "info":
        return max(
            members,
            key=lambda item: (item["cluster_info"], -item["path_len"], item["entry"]),
        )
    if WAYPOINT_MODE == "deep":
        return max(
            members,
            key=lambda item: (item["path_len"], item["cluster_info"], item["entry"]),
        )
    if WAYPOINT_MODE == "quick":
        return min(
            members,
            key=lambda item: (item["path_len"], -item["cluster_info"], item["entry"]),
        )
    return fallback

CONDITIONS = (
    "macro",
    "macro_return",
    "macro_uncertainty",
    "macro_return_uncertainty",
    "macro_return_uncertainty_regret",
    "macro_semantic_llm",
    "macro_micro",
    "macro_micro_llm",
    "macro_adaptive",
    "macro_adaptive_llm",
    "macro_hybrid_common",
    "macro_hybrid_common_llm",
    "macro_v10",
)


@dataclass
class MacroMemory:
    attempts: int = 0
    failures: int = 0
    info_gain: int = 0
    progress: int = 0
    regret: float = 0.0
    taboo_until: int = -1

    def penalty(self, step: int) -> float:
        value = 2.0 * self.failures + 0.25 * max(0.0, self.regret)
        if step < self.taboo_until:
            value += 10.0
        return value


def _resolve_condition(env: v5.FixedBenchmarkEnv, condition: str) -> str:
    """Complexity-aware policy chosen from dimensions only."""
    if condition != "macro_v10":
        return condition
    return "macro_hybrid_common" if env.width < 24 else "macro"


def _known_nodes(env: v5.FixedBenchmarkEnv) -> set[Cell]:
    return set(env.topo_nodes)


def _degree(env: v5.FixedBenchmarkEnv, cell: Cell) -> int:
    nodes = _known_nodes(env)
    return sum((cell[0] + dx, cell[1] + dy) in nodes for dx, dy in DIRS)


def _initial_observation(spec: Dict[str, Any], max_steps: int) -> int:
    size = max(int(spec["size"][0]), int(spec["size"][1]))
    if OBS_POLICY == "budget_map":
        if size < 24:
            return 5
        return 9 if max_steps != 100 else 7
    if OBS_POLICY in ("staged", "stall", "geometry"):
        return 7 if size >= 24 else 5
    if OBS_POLICY == "commit_wide":
        return 7 if size >= 24 else FIXED_OBS_SIZE
    if OBS_POLICY == "wide":
        return 9
    return FIXED_OBS_SIZE


def _adaptive_observation(
    spec: Dict[str, Any],
    env: v5.FixedBenchmarkEnv,
    step: int,
    max_steps: int,
    consecutive_low_gain: int,
) -> int:
    """Resolve a v10 observation-window policy for the next transition."""
    size = max(int(spec["size"][0]), int(spec["size"][1]))
    ratio = step / max(1, max_steps)
    if OBS_POLICY == "fixed":
        return FIXED_OBS_SIZE
    if OBS_POLICY == "budget_map":
        if size < 24:
            return 5
        return 9 if max_steps != 100 else 7
    if OBS_POLICY == "staged":
        if size < 24:
            return 5
        return 9 if ratio >= 0.52 or consecutive_low_gain >= 10 else 7
    if OBS_POLICY == "stall":
        if size < 24:
            return 5
        return 9 if consecutive_low_gain >= 8 else 7
    if OBS_POLICY == "geometry":
        if size < 24:
            if ratio >= 0.68 and (
                v5._corridor_width(env, (env.x, env.y)) >= 3
                or _degree(env, (env.x, env.y)) >= 3
            ):
                return FIXED_OBS_SIZE + 2
            return FIXED_OBS_SIZE
        cell = (env.x, env.y)
        width = v5._corridor_width(env, cell)
        return 9 if width >= 3 or _degree(env, cell) >= 3 else 7
    if OBS_POLICY == "commit_wide":
        if env.goal in env.topo_nodes and env.goal not in env.visited_cells:
            return 9
        if size < 24:
            return FIXED_OBS_SIZE
        cell = (env.x, env.y)
        width = v5._corridor_width(env, cell)
        return 9 if width >= 3 or _degree(env, cell) >= 3 else 7
    if OBS_POLICY == "wide":
        return 9
    return 5


def _first_decision(env: v5.FixedBenchmarkEnv, path: Sequence[Cell]) -> Optional[Cell]:
    for cell in path[1:]:
        if _degree(env, cell) >= 3:
            return cell
    return path[min(1, len(path) - 1)] if path else None


def _branch_id(env: v5.FixedBenchmarkEnv, path: Sequence[Cell]) -> Tuple[Cell, Cell]:
    if not path:
        return ((-1, -1), (0, 0))
    for i, cell in enumerate(path[1:], start=1):
        if _degree(env, cell) >= 3:
            nxt = path[i + 1] if i + 1 < len(path) else cell
            direction = (
                0 if nxt[0] == cell[0] else (1 if nxt[0] > cell[0] else -1),
                0 if nxt[1] == cell[1] else (1 if nxt[1] > cell[1] else -1),
            )
            return cell, direction
    decision = path[min(1, len(path) - 1)]
    src = path[0]
    direction = (
        0 if decision[0] == src[0] else (1 if decision[0] > src[0] else -1),
        0 if decision[1] == src[1] else (1 if decision[1] > src[1] else -1),
    )
    return decision, direction


def _nearby_unknown(env: v5.FixedBenchmarkEnv, cell: Cell, radius: int = 3) -> int:
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = (cell[0] + dx, cell[1] + dy)
            if env.in_bounds(*nb) and nb not in env.explored_cells:
                count += 1
    return count


def _entropy(cell_probs: np.ndarray) -> float:
    p = np.clip(cell_probs, 1e-6, 1 - 1e-6)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def _region_uncertainty(
    env: v5.FixedBenchmarkEnv, prob: np.ndarray, cell: Cell, radius: int = 2
) -> float:
    vals = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = (cell[0] + dx, cell[1] + dy)
            if env.in_bounds(*nb) and nb not in env.explored_cells:
                vals.append(prob[nb[1], nb[0]])
    return _entropy(np.asarray(vals)) if vals else 0.0


def build_macro_candidates(
    env: v5.FixedBenchmarkEnv,
    predictor: LogisticMapPredictor,
    condition: str,
    max_steps: int,
    step: int,
    memory: Dict[Any, MacroMemory],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    prob = predictor.probabilities(env)
    pred_dist = v6.predicted_distance_map(env, prob)
    current_optimistic = v5.optimistic_distance(
        env, (env.x, env.y), env.goal
    )
    clusters = v5.cluster_frontiers(env)
    raw = []
    for cluster in clusters:
        for cell in cluster:
            path = v5.known_astar(env, cell)
            if path is None:
                continue
            info = v5._information_gain(env, cell)
            width = v5._corridor_width(env, cell)
            risk = (
                0.46 * v5._unknown_neighbor_count(env, cell) / 4.0
                + 0.32 * (width <= 2)
            )
            pred_h = float(pred_dist.get(
                cell, v5.optimistic_distance(env, cell, env.goal)
            ))
            preliminary = (
                SCORE_W["path"] * (len(path) - 1)
                + SCORE_W["goal"] * pred_h
                + SCORE_W["risk"] * risk
                - SCORE_W["info"] * math.log1p(info)
                - SCORE_W["unknown"] * math.log1p(_nearby_unknown(env, cell))
            )
            raw.append({
                "entry": cell,
                "path": path,
                "path_len": len(path) - 1,
                "cluster_info": info,
                "cluster_risk": risk,
                "pred_h": pred_h,
                "unknown_near": _nearby_unknown(env, cell),
                "uncertainty": _region_uncertainty(env, prob, cell),
                "branch_id": _branch_id(env, path),
                "preliminary": preliminary,
            })

    raw.sort(key=lambda item: (item["preliminary"], item["entry"]))
    selected = []
    per_branch: Counter = Counter()
    for item in raw:
        if per_branch[item["branch_id"]] >= BRANCH_MAX:
            continue
        if any(
            v6.manhattan(item["entry"], other["entry"]) < NMS_DIST
            for other in selected
        ):
            continue
        selected.append(item)
        per_branch[item["branch_id"]] += 1
        if len(selected) >= TOPK:
            break

    initial_lower_bound = max(
        8, v6.manhattan(tuple(env.start), tuple(env.goal))
    )
    micro_fallback = (
        condition in ("macro_adaptive", "macro_adaptive_llm")
        and step >= 45
        and env.forward_steps > 1.35 * initial_lower_bound
    )
    if micro_fallback:
        selected = []

    candidates = []
    for chosen in selected:
        branch_id = chosen["branch_id"]
        members = [
            item for item in raw
            if item["branch_id"] == branch_id
            and v6.manhattan(item["entry"], chosen["entry"]) <= 3
        ]
        members.sort(key=lambda item: (
            v6.manhattan(item["entry"], chosen["entry"]),
            item["entry"],
        ))
        waypoint = _select_waypoint(members, chosen)
        entry = waypoint["entry"]
        info_gain = sum(item["cluster_info"] for item in members)
        unknown_near = sum(item["unknown_near"] for item in members)
        uncertainty = max(item["uncertainty"] for item in members)
        pred_h = waypoint["pred_h"]
        optimistic_h = v5.optimistic_distance(env, entry, env.goal)
        return_cost = waypoint["path_len"]
        remaining = max_steps - step
        budget_commit = (
            BUDGET_GATE > 0
            and remaining
            <= BUDGET_MULT * current_optimistic + BUDGET_MARGIN
        )
        predicted_completion = pred_h
        budget_needed = waypoint["path_len"] + return_cost + predicted_completion
        budget_safe = remaining >= budget_needed + 5
        decision = branch_id[0]
        hist = memory.get(branch_id, MacroMemory())
        history_bonus = min(3.0, hist.info_gain / 5.0 + max(0, hist.progress) / 3.0)

        goal_for_score = _goal_score_distance(
            condition, pred_h, optimistic_h
        )
        base_score = (
            SCORE_W["path"] * waypoint["path_len"]
            + SCORE_W["goal"] * goal_for_score
            + SCORE_W["risk"] * chosen["cluster_risk"]
            - SCORE_W["info"] * math.log1p(info_gain)
            - SCORE_W["unknown"] * math.log1p(unknown_near)
            + SCORE_W["uncertainty"] * uncertainty
            + hist.penalty(step)
            - history_bonus
        )
        if budget_commit:
            base_score += BUDGET_GATE * BUDGET_GOAL_W * optimistic_h
        use_return_budget = condition in (
            "macro_return", "macro_return_uncertainty",
            "macro_return_uncertainty_regret", "macro_semantic_llm",
            "macro_micro", "macro_micro_llm",
            "macro_adaptive", "macro_adaptive_llm",
            "macro_hybrid_common", "macro_hybrid_common_llm",
        )
        use_uncertainty = condition in (
            "macro_uncertainty", "macro_return_uncertainty",
            "macro_return_uncertainty_regret", "macro_semantic_llm",
            "macro_micro", "macro_micro_llm",
            "macro_adaptive", "macro_adaptive_llm",
            "macro_hybrid_common", "macro_hybrid_common_llm",
        )
        if use_return_budget:
            if not budget_safe:
                base_score += 30.0 + max(0, budget_needed - remaining)
        if use_uncertainty:
            base_score += 0.8 * uncertainty
        score = base_score
        candidates.append({
            "kind": "MACRO",
            "branch_id": branch_id,
            "decision": decision,
            "entry": waypoint["entry"],
            "anchor": members[-1]["entry"],
            "path": waypoint["path"],
            "path_len": waypoint["path_len"],
            "return_cost": return_cost,
            "pred_h": pred_h,
            "optimistic_h": optimistic_h,
            "info_gain": info_gain,
            "unknown_near": unknown_near,
            "uncertainty": uncertainty,
            "risk": chosen["cluster_risk"],
            "budget_needed": budget_needed,
            "budget_safe": budget_safe,
            "history_penalty": hist.penalty(step),
            "score": score,
            "members": [item["entry"] for item in members],
        })

    if env.goal in env.topo_nodes and env.goal not in env.visited_cells:
        path = v5.known_astar(env, env.goal)
        if path is not None:
            candidates.append({
                "kind": "GOAL",
                "branch_id": _branch_id(env, path),
                "decision": _first_decision(env, path),
                "entry": env.goal,
                "anchor": env.goal,
                "path": path,
                "path_len": len(path) - 1,
                "return_cost": 0,
                "pred_h": 0.0,
                "info_gain": 0,
                "unknown_near": 0,
                "uncertainty": 0.0,
                "risk": 0.0,
                "budget_needed": len(path) - 1,
                "budget_safe": True,
                "history_penalty": 0.0,
                "score": -1000.0,
                "members": [env.goal],
            })

    if condition in (
        "macro_micro", "macro_micro_llm",
        "macro_adaptive", "macro_adaptive_llm",
        "macro_hybrid_common", "macro_hybrid_common_llm",
    ) and (
        condition in (
            "macro_micro", "macro_micro_llm",
            "macro_hybrid_common", "macro_hybrid_common_llm",
        ) or micro_fallback
    ):
        micro, _ = v6.candidate_frontiers_v6(env, predictor, True)
        for item in micro[:2]:
            if item["kind"] == "GOAL":
                continue
            path = item["path"]
            branch_id = _branch_id(env, path)
            uncertainty = _region_uncertainty(env, prob, item["cell"])
            hist = memory.get(branch_id, MacroMemory())
            score = (
                SCORE_W["path"] * item["path_len"]
                + SCORE_W["goal"] * item["optimistic_h"]
                + SCORE_W["risk"] * item["risk"]
                - SCORE_W["info"] * math.log1p(item["gain"])
                + SCORE_W["uncertainty"] * uncertainty
                + hist.penalty(step)
            )
            candidates.append({
                "kind": "MICRO",
                "branch_id": branch_id,
                "decision": branch_id[0],
                "entry": item["cell"],
                "anchor": item["cell"],
                "path": path,
                "path_len": item["path_len"],
                "return_cost": item["path_len"],
                "pred_h": item["optimistic_h"],
                "optimistic_h": item["optimistic_h"],
                "info_gain": item["gain"],
                "unknown_near": 0,
                "uncertainty": uncertainty,
                "risk": item["risk"],
                "budget_needed": 2 * item["path_len"] + item["optimistic_h"],
                "budget_safe": max_steps - step >= 2 * item["path_len"] + item["optimistic_h"] + 5,
                "history_penalty": hist.penalty(step),
                "score": score,
                "members": [item["cell"]],
            })

    candidates.sort(key=lambda c: (c["kind"] != "GOAL", c["score"], c["entry"]))
    candidates = candidates[:TOPK]
    for i, cand in enumerate(candidates):
        cand["label"] = f"M{i + 1}"

    unknown_mask = np.ones((env.height, env.width), dtype=bool)
    for x, y in env.explored_cells:
        unknown_mask[y, x] = False
    if unknown_mask.any():
        truth = np.asarray([
            [not env.is_free(x, y) for x in range(env.width)]
            for y in range(env.height)
        ])
        pred = prob >= predictor.threshold
        ap = float((truth[unknown_mask] == pred[unknown_mask]).mean())
        conf = float(np.mean(np.maximum(prob[unknown_mask], 1 - prob[unknown_mask])))
    else:
        ap, conf = 1.0, 1.0
    return candidates, {
        "AP_UNSEEN": ap,
        "MAP_CONF": conf,
        "MEAN_ENTROPY": _entropy(prob[unknown_mask]) if unknown_mask.any() else 0.0,
    }


def _llm_tiebreak(
    env: v5.FixedBenchmarkEnv, candidates: Sequence[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], bool]:
    if len(candidates) < 2 or candidates[0]["kind"] == "GOAL":
        return (candidates[0] if candidates else None), False
    gap = candidates[1]["score"] - candidates[0]["score"]
    if gap > 1.2:
        return candidates[0], False
    rows = []
    for c in candidates[:3]:
        rows.append(
            f"{c['label']}: decision={c['decision']} entry={c['entry']} "
            f"forward_cost={c['path_len']} return_cost={c['return_cost']} "
            f"pred_goal_h={c['pred_h']:.1f} info={c['info_gain']} "
            f"uncertainty={c['uncertainty']:.2f} risk={c['risk']:.2f} "
            f"budget_safe={c['budget_safe']}"
        )
    prompt = "\n".join([
        env.instruction,
        f"robot=({env.x},{env.y}) goal={env.goal}",
        "Select one macro branch. Prefer the instruction direction when geometry is nearly tied.",
        "Candidates:",
        *rows,
        "Return exactly <TARGET>M1</TARGET>.",
    ])
    text, _ = v5.d.request_llm(prompt)
    parsed = v5.d.parse_llm_response(text)
    raw = v5.d._strip_tag(text, "TARGET").strip()
    import re
    match = re.search(r"M\s*(\d+)", raw, flags=re.IGNORECASE)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx], True
    return candidates[0], True


def run_episode(
    spec: Dict[str, Any], condition: str, predictor: LogisticMapPredictor,
    max_steps: int, logger: v5.d.RunLogger
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    v5.d.OBS_SIZE = _initial_observation(spec, max_steps)
    env = v5.FixedBenchmarkEnv(spec)
    condition = _resolve_condition(env, condition)
    if PRED_WEIGHT == -2:
        globals()["PRED_WEIGHT"] = 0.0 if env.width < 24 else 1.0
    if ROUTE_REPLAN_STEPS < 0:
        globals()["ROUTE_REPLAN_STEPS"] = 0 if env.width < 24 else 1
    memory: Dict[Any, MacroMemory] = {}
    chosen = None
    route: List[Cell] = []
    route_start_step = 0
    route_start_explored = len(env.explored_cells)
    route_start_pred = v5.optimistic_distance(env, (env.x, env.y), env.goal)
    edge_counts: Counter = Counter()
    rows = []
    calls = recoveries = 0
    low_gain_streak = 0
    route_new_cells = len(env.explored_cells)
    plan_forward_mark = env.forward_steps
    ap_values = []
    uncertainty_values = []
    success = False
    reason = "max_steps"
    t0 = time.time()

    for step in range(max_steps):
        if (env.x, env.y) == env.goal:
            result = env.step(v5.d.ACTION_STOP)
            success = bool(result["success"])
            reason = result.get("reason", "stop_at_goal")
            break

        branch_finished = chosen is not None and (
            not route or (env.x, env.y) == chosen["entry"]
        )
        route_expired = (
            chosen is not None
            and bool(route)
            and ROUTE_REPLAN_STEPS > 0
            and env.forward_steps - plan_forward_mark >= ROUTE_REPLAN_STEPS
        )
        need_replan = (
            chosen is None or not route or branch_finished or route_expired
        )
        if need_replan:
            if chosen is not None and branch_finished:
                info_gain = len(env.explored_cells) - route_start_explored
                mem = memory.setdefault(chosen["branch_id"], MacroMemory())
                mem.attempts += 1
                mem.info_gain += info_gain
                expected_info = max(1.0, chosen["info_gain"])
                actual_progress = route_start_pred - v5.optimistic_distance(env, (env.x, env.y), env.goal)
                mem.progress += actual_progress
                mem.regret = max(0.0, expected_info - info_gain) / expected_info
                if info_gain <= 0 and actual_progress <= 0:
                    mem.failures += 1
                    mem.taboo_until = step + 10
                    recoveries += 1
                if condition == "macro_return_uncertainty_regret" and (
                    mem.regret > 0.75 or actual_progress < -2
                ):
                    mem.taboo_until = step + 12
                    recoveries += 1
            if route_new_cells > 0:
                growth = len(env.explored_cells) - route_new_cells
                low_gain_streak = low_gain_streak + 1 if growth < 2 else 0
            route_new_cells = len(env.explored_cells)
            if OBS_POLICY != "fixed":
                v5.d.OBS_SIZE = _adaptive_observation(
                    spec, env, step, max_steps, low_gain_streak
                )
            candidates, diag = build_macro_candidates(
                env, predictor, condition, max_steps, step, memory
            )
            ap_values.append(diag["AP_UNSEEN"])
            uncertainty_values.append(diag["MEAN_ENTROPY"])
            if condition in (
                "macro_semantic_llm", "macro_micro_llm", "macro_adaptive_llm",
                "macro_hybrid_common_llm",
            ):
                chosen, used_llm = _llm_tiebreak(env, candidates)
                calls += int(used_llm)
            else:
                chosen = candidates[0] if candidates else None
            if chosen:
                route = list(chosen["path"])
                if route and route[0] == (env.x, env.y):
                    route.pop(0)
                route_start_step = step
                route_start_explored = len(env.explored_cells)
                route_start_pred = v5.optimistic_distance(env, (env.x, env.y), env.goal)
                plan_forward_mark = env.forward_steps
            else:
                route = []

        if not route:
            options = []
            for dx, dy in DIRS:
                nb = (env.x + dx, env.y + dy)
                if nb in env.topo_nodes:
                    options.append(nb)
            options.sort(key=lambda c: v6.manhattan(c, env.goal))
            route = [options[0]] if options else []
            if not route:
                env.step(v5.d.ACTION_RIGHT)
                continue

        target = route[0]
        action = v5.turn_toward(env, target)
        if OBS_POLICY != "fixed":
            v5.d.OBS_SIZE = _adaptive_observation(
                spec, env, step, max_steps, low_gain_streak
            )
        prev = (env.x, env.y)
        result = env.step(action)
        if result["moved"] and route and (env.x, env.y) == route[0]:
            route.pop(0)
            edge = tuple(sorted((prev, (env.x, env.y))))
            edge_counts[edge] += 1
        elif result["bumped"]:
            route = []
            chosen = None
        trajectory = env.visited_cells
        oscillating = (
            condition != "macro"
            and len(trajectory) >= 5
            and trajectory[-1] == trajectory[-3] == trajectory[-5]
            and trajectory[-2] == trajectory[-4]
        )
        repeated_edge = route and edge_counts[tuple(sorted((prev, (env.x, env.y))))] >= 5
        if condition != "macro" and (oscillating or repeated_edge):
            if chosen is not None:
                mem = memory.setdefault(chosen["branch_id"], MacroMemory())
                mem.failures += 1
                mem.taboo_until = step + 10
            route = []
            chosen = None
            recoveries += 1
        rows.append({
            "step": step, "x": env.x, "y": env.y, "action": action,
            "moved": int(result["moved"]), "bumped": int(result["bumped"]),
            "macro": f"{chosen['label']}:{chosen['entry']}" if chosen else "",
            "decision": f"{chosen['decision']}" if chosen else "",
        })
        if result["done"]:
            success = bool(result["success"])
            reason = result.get("reason", "")
            break

    rec = {
        "map_id": spec["id"], "condition": condition,
        "success": success, "reason": reason,
        "shortest": env.shortest_len, "actual": env.forward_steps,
        "steps": len(rows), "osr": env.goal in env.visited_cells,
        "ne": v6.manhattan((env.x, env.y), tuple(env.goal)),
        "coverage": len(env.topo_nodes) / max(1, spec["metrics"]["reachable_free"]),
        "bumps": sum(r["bumped"] for r in rows),
        "recoveries": recoveries, "llm_calls": calls,
        "AP_UNSEEN": float(np.mean(ap_values)) if ap_values else 0,
        "ENTROPY": float(np.mean(uncertainty_values)) if uncertainty_values else 0,
        "end_pose": f"{env.x},{env.y},{v5.d.HEADING_NAME[env.heading]}",
    }
    return rec, rows


def metrics(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    return {
        "N": n,
        "SR": sum(r["success"] for r in records) / n,
        "SPL": sum(
            r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0
            for r in records
        ) / n,
        "NE": sum(r["ne"] for r in records) / n,
        "COV": sum(r["coverage"] for r in records) / n,
        "BUMPS": sum(r["bumps"] for r in records),
        "RECOVERIES": sum(r["recoveries"] for r in records),
        "AP_UNSEEN": sum(r["AP_UNSEEN"] for r in records) / n,
        "ENTROPY": sum(r["ENTROPY"] for r in records) / n,
        "LLM_CALLS": sum(r["llm_calls"] for r in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--maps-file", default=str(ROOT / "benchmark_maps_v5.json"))
    parser.add_argument("--predictor-model", default=str(ROOT / "map_predictor_v6.json"))
    parser.add_argument("--start-map", type=int, default=0)
    parser.add_argument("--num-ep", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--obs-size", type=int, default=5)
    parser.add_argument("--predictor-mode", choices=("logistic", "structured"), default="logistic")
    parser.add_argument("--predictor-secondary")
    parser.add_argument("--predictor-fusion-weight", type=float, default=0.0)
    parser.add_argument("--obs-policy", choices=(
        "fixed", "budget_map", "staged", "stall", "geometry",
        "commit_wide", "wide"
    ), default="fixed")
    parser.add_argument("--route-replan-steps", type=int, default=0)
    parser.add_argument("--pred-weight", type=float, default=-1.0)
    parser.add_argument("--waypoint-mode", choices=(
        "entry", "info", "deep", "quick"
    ), default="entry")
    parser.add_argument("--budget-gate", type=float, default=0.0)
    parser.add_argument("--budget-mult", type=float, default=1.5)
    parser.add_argument("--budget-margin", type=int, default=10)
    parser.add_argument("--budget-goal-weight", type=float, default=2.0)
    parser.add_argument("--tag", default="v7")
    args = parser.parse_args()
    v5.d.LLM_URL = os.getenv(
        "MAPGPT_LLM_URL", "http://127.0.0.1:5000/get_llm_response"
    )
    v5.d.SLEEP_AFTER_OK_SEC = float(os.getenv("MAPGPT_SLEEP_OK", "0.1"))
    v5.d.OBS_SIZE = args.obs_size
    globals()["FIXED_OBS_SIZE"] = args.obs_size
    globals()["OBS_POLICY"] = args.obs_policy
    globals()["ROUTE_REPLAN_STEPS"] = args.route_replan_steps
    globals()["PRED_WEIGHT"] = args.pred_weight
    globals()["WAYPOINT_MODE"] = args.waypoint_mode
    globals()["BUDGET_GATE"] = args.budget_gate
    globals()["BUDGET_MULT"] = args.budget_mult
    globals()["BUDGET_MARGIN"] = args.budget_margin
    globals()["BUDGET_GOAL_W"] = args.budget_goal_weight
    maps = json.loads(Path(args.maps_file).read_text(encoding="utf-8"))["maps"]
    maps = maps[args.start_map:args.start_map + args.num_ep]
    base_predictor = LogisticMapPredictor(
        json.loads(Path(args.predictor_model).read_text(encoding="utf-8"))
    )
    if args.predictor_secondary and args.predictor_fusion_weight > 0:
        secondary = LogisticMapPredictor(
            json.loads(Path(args.predictor_secondary).read_text(encoding="utf-8"))
        )
        primary = base_predictor
        weight = min(1.0, max(0.0, args.predictor_fusion_weight))

        class FusionPredictor:
            threshold = (
                weight * primary.threshold
                + (1.0 - weight) * secondary.threshold
            )

            def probabilities(self, env):
                return (
                    weight * primary.probabilities(env)
                    + (1.0 - weight) * secondary.probabilities(env)
                )

        base_predictor = FusionPredictor()
    if args.predictor_mode == "structured":
        from map_prediction_structured_v8 import structured_prob

        class StructuredPredictor:
            threshold = 0.48

            def probabilities(self, env):
                return structured_prob(env, base_predictor.probabilities(env))

        predictor = StructuredPredictor()
    else:
        predictor = base_predictor
    outdir = ROOT / f"results_{args.tag}"
    outdir.mkdir(exist_ok=True)
    logger = v5.d.RunLogger(str(outdir / "log.txt"))
    records, all_rows = [], []
    for spec in maps:
        rec, rows = run_episode(spec, args.condition, predictor, args.max_steps, logger)
        for row in rows:
            row["episode"] = spec["id"]
            row["condition"] = args.condition
        records.append(rec)
        all_rows.extend(rows)
        print(
            f"[{spec['id']}] success={rec['success']} steps={rec['steps']} "
            f"actual={rec['actual']} shortest={rec['shortest']} "
            f"recoveries={rec['recoveries']} calls={rec['llm_calls']}",
            flush=True,
        )
    logger.close()
    (outdir / "episodes.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "steps.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = metrics(records)
    (outdir / "summary.txt").write_text(
        "\n".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in {"tag": args.tag, **summary}.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tag": args.tag, **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
