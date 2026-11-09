# -*- coding: utf-8 -*-
"""Leakage-free deterministic and map-prediction navigation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import secrets
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path("/home/orange")
sys.path.insert(0, str(ROOT))
import mapgpt_v5 as v5
from map_prediction_v6 import LogisticMapPredictor

Cell = Tuple[int, int]
DIRS = v5.DIRS

CONDITIONS = (
    "base_noleak",
    "anti_stuck",
    "map_pred",
    "map_conf_gate",
    "map_pred_anti",
    "ap_oracle_diag",
    "jitter_base",
    "jitter_anti",
    "truth_mix_25_diag",
    "truth_mix_50_diag",
    "truth_mix_75_diag",
    "truth_full_diag",
)


@dataclass
class BranchStat:
    attempts: int = 0
    failures: int = 0
    last_info_gain: int = 0
    last_progress: int = 0
    taboo_until: int = -1


class BranchMemory:
    def __init__(self) -> None:
        self.stats: Dict[Cell, BranchStat] = {}
        self.edge_counts: Counter = Counter()
        self.position_counts: Counter = Counter()
        self.last_position_trigger: int = -10

    def penalty(self, branch: Optional[Cell], step: int) -> float:
        if branch is None:
            return 0.0
        stat = self.stats.get(branch, BranchStat())
        penalty = 1.5 * stat.failures
        if step < stat.taboo_until:
            penalty += 8.0
        return penalty

    def observe_route(
        self,
        branch: Optional[Cell],
        info_gain: int,
        progress: int,
        step: int,
        route_cost: int = 0,
    ) -> bool:
        if branch is None:
            return False
        stat = self.stats.setdefault(branch, BranchStat())
        stat.attempts += 1
        stat.last_info_gain = info_gain
        stat.last_progress = progress
        failed = (
            (info_gain <= 0 and progress < 0)
            or (route_cost >= 6 and progress <= -3)
            or (route_cost >= 10 and info_gain <= 2)
        )
        if failed:
            stat.failures += 1
            stat.taboo_until = step + 8
        elif info_gain > 4 or progress > 2:
            stat.failures = max(0, stat.failures - 1)
        return failed


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def known_degree(env: v5.FixedBenchmarkEnv, cell: Cell) -> int:
    nodes = set(env.topo_nodes)
    return sum((cell[0] + dx, cell[1] + dy) in nodes for dx, dy in DIRS)


def branch_key(env: v5.FixedBenchmarkEnv, path: Sequence[Cell]) -> Optional[Cell]:
    if not path:
        return None
    junctions = [cell for cell in path[1:] if known_degree(env, cell) >= 3]
    if junctions:
        return junctions[0]
    return path[min(1, len(path) - 1)]


def predicted_distance_map(
    env: v5.FixedBenchmarkEnv, prob: np.ndarray
) -> Dict[Cell, float]:
    """Distance to goal on a predicted occupancy map.

    Known walls are impassable. Predicted walls have a large but finite cost,
    so one uncertain wall prediction does not make the whole map unusable.
    """
    import heapq

    src = tuple(env.goal)
    q = [(0.0, src)]
    dist = {src: 0.0}
    while q:
        d, cur = heapq.heappop(q)
        if d != dist.get(cur):
            continue
        for dx, dy in DIRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not env.in_bounds(*nb):
                continue
            if nb in env.explored_cells and not env.is_free(*nb):
                continue
            wall_prob = float(prob[nb[1], nb[0]])
            step_cost = 1.0 + 7.0 * wall_prob
            nd = d + step_cost
            if nd < dist.get(nb, 1e18):
                dist[nb] = nd
                heapq.heappush(q, (nd, nb))
    return dist


def candidate_frontiers_v6(
    env: v5.FixedBenchmarkEnv,
    predictor: LogisticMapPredictor,
    use_map_prediction: bool,
    truth_mix: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    clusters = v5.cluster_frontiers(env)
    reference = max(8.0, float(manhattan(tuple(env.start), tuple(env.goal))))
    candidates: List[Dict[str, Any]] = []
    prob = predictor.probabilities(env)
    if truth_mix > 0:
        truth_wall_prob = np.asarray([
            [0.0 if env.is_free(x, y) else 1.0 for x in range(env.width)]
            for y in range(env.height)
        ])
        prob = (1.0 - truth_mix) * prob + truth_mix * truth_wall_prob
    pred_dist = predicted_distance_map(env, prob) if use_map_prediction else {}
    for comp in clusters:
        scored_cells = []
        for cell in comp:
            path = v5.known_astar(env, cell)
            if path is None:
                continue
            gain = v5._information_gain(env, cell)
            unknown_nb = v5._unknown_neighbor_count(env, cell)
            width = v5._corridor_width(env, cell)
            risk = (
                0.46 * (unknown_nb / 4.0)
                + 0.32 * (1.0 if width <= 2 else 0.0)
                + 0.22 * min(1.0, max(0, len(path) - 1 - manhattan(
                    (env.x, env.y), cell
                )) / reference)
            )
            scored_cells.append((gain, -len(path), -risk, cell, path, risk, width))
        if not scored_cells:
            continue
        scored_cells.sort(reverse=True)
        gain, _, _, cell, path, risk, width = scored_cells[0]
        optimistic = v5.optimistic_distance(env, cell, env.goal)
        goal_term = 1.0 - min(1.0, manhattan(cell, env.goal) / (2.0 * reference + 1.0))
        distance_term = 1.0 - min(1.0, (len(path) - 1) / (2.0 * reference + 1.0))
        utility = 0.44 * goal_term + 0.24 * gain + 0.18 * distance_term - 0.14 * risk
        map_term = 0.0
        pred_h = float(optimistic)
        if use_map_prediction:
            pred_h = pred_dist.get(cell, float(optimistic))
            map_term = 1.0 - min(1.0, pred_h / (2.5 * reference + 1.0))
            utility += 0.13 * map_term
        candidates.append({
            "cell": cell,
            "path": path,
            "path_len": len(path) - 1,
            "goal_h": manhattan(cell, env.goal),
            "optimistic_h": optimistic,
            "pred_h": pred_h,
            "map_term": map_term,
            "gain": gain,
            "risk": risk,
            "width": width,
            "utility": utility,
            "kind": "FRONTIER",
            "branch": branch_key(env, path),
        })
    if env.goal in env.topo_nodes and env.goal not in env.visited_cells:
        path = v5.known_astar(env, env.goal)
        if path is not None:
            candidates.append({
                "cell": env.goal,
                "path": path,
                "path_len": len(path) - 1,
                "goal_h": 0,
                "optimistic_h": 0,
                "pred_h": 0.0,
                "map_term": 1.0 if use_map_prediction else 0.0,
                "gain": 0.0,
                "risk": 0.0,
                "width": v5._corridor_width(env, env.goal),
                "utility": 2.0,
                "kind": "GOAL",
                "branch": branch_key(env, path),
            })
    candidates.sort(key=lambda c: (-c["utility"], c["path_len"], c["risk"]))
    candidates = candidates[:6]
    for i, cand in enumerate(candidates):
        cand["label"] = f"C{i + 1}"

    unknown_mask = np.ones((env.height, env.width), dtype=bool)
    for x, y in env.explored_cells:
        unknown_mask[y, x] = False
    if unknown_mask.any():
        truth_wall = np.asarray([
            [not env.is_free(x, y) for x in range(env.width)]
            for y in range(env.height)
        ])
        pred_wall = prob >= predictor.threshold
        ap_unseen = float((truth_wall[unknown_mask] == pred_wall[unknown_mask]).mean())
        confidence = float(np.mean(np.maximum(prob[unknown_mask], 1 - prob[unknown_mask])))
    else:
        ap_unseen, confidence = 1.0, 1.0
    return candidates, {
        "ap_unseen_pred": ap_unseen,
        "map_confidence": confidence,
        "predicted_mean_wall_prob": float(prob[unknown_mask].mean()) if unknown_mask.any() else 0.0,
    }


def select_candidate(
    candidates: Sequence[Dict[str, Any]],
    memory: BranchMemory,
    step: int,
    antistuck: bool,
    map_diag: Dict[str, float],
    oracle_ap: bool,
    condition: str,
    max_steps: int,
    jitter: bool = False,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    goals = [c for c in candidates if c["kind"] == "GOAL"]
    if goals:
        return goals[0]
    scored = []
    for cand in candidates:
        map_reliability = 1.0
        if condition == "map_conf_gate":
            map_reliability = max(0.0, min(1.0, (
                map_diag["map_confidence"] - 0.62
            ) / 0.28))
        elif oracle_ap:
            # Diagnostic only: actual AP_UNSEEN is unavailable to a real agent.
            map_reliability = 1.0 if map_diag["ap_unseen_pred"] >= 0.62 else 0.0
        score = (
            cand["path_len"] * 0.34
            + cand["optimistic_h"] * 0.36
            + cand["risk"] * 4.0
            - cand["gain"] * 2.0
            - cand["map_term"] * 2.8 * map_reliability
        )
        if antistuck:
            score += memory.penalty(cand["branch"], step)
            if step >= int(0.45 * max_steps):
                # Late-budget commitment: stop rewarding side exploration.
                score = (
                    cand["path_len"] * 0.10
                    + cand["optimistic_h"] * 1.60
                    + cand["risk"] * 0.8
                    - cand["map_term"] * 0.5
                )
        scored.append((score, cand["risk"], cand["cell"], cand))
    scored.sort(key=lambda row: (row[0], row[1], row[2]))
    if jitter and len(scored) >= 2 and abs(scored[1][0] - scored[0][0]) < 1.8:
        # Controlled stress test for unstable rankings at symmetric forks.
        return scored[(step // 6) % 2][3]
    return scored[0][3]


def plan_prompt(
    env: v5.FixedBenchmarkEnv,
    candidates: Sequence[Dict[str, Any]],
    step: int,
    map_diag: Dict[str, float],
) -> str:
    lines = [
        "You select one safe frontier waypoint for a partially observed indoor robot.",
        env.instruction,
        f"robot=({env.x},{env.y}) destination=({env.goal[0]},{env.goal[1]}) "
        f"step={step} forward_steps={env.forward_steps}",
        "## Known occupancy matrix",
        v5.render_fused_matrix(env),
        "## Topology",
        v5.render_topology(env, candidates),
        "## Candidates",
    ]
    for c in candidates:
        lines.append(
            f"{c['label']} {c['kind']} cell={c['cell']} path={c['path_len']} "
            f"goal_h={c['goal_h']} pred_h={c['pred_h']:.1f} info={c['gain']:.2f} "
            f"risk={c['risk']:.2f} utility={c['utility']:.3f}"
        )
    lines.extend([
        f"Prediction confidence={map_diag['map_confidence']:.2f}",
        "Choose GOAL if present. Otherwise choose the candidate with the best "
        "combination of goal progress, new information, and low risk.",
        "Return exactly: <TARGET>C1</TARGET>",
    ])
    return "\n".join(lines)


def run_episode(
    spec: Dict[str, Any],
    condition: str,
    predictor: LogisticMapPredictor,
    max_steps: int,
    nonce: str,
    logger: v5.d.RunLogger,
    use_llm: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    env = v5.FixedBenchmarkEnv(spec)
    memory = BranchMemory()
    memory.position_counts[(env.x, env.y)] = 1
    antistuck = condition in ("anti_stuck", "map_pred_anti", "jitter_anti")
    use_map = condition in (
        "map_pred", "map_conf_gate", "map_pred_anti", "ap_oracle_diag",
        "jitter_base", "jitter_anti",
        "truth_mix_25_diag", "truth_mix_50_diag",
        "truth_mix_75_diag", "truth_full_diag",
    )
    oracle_ap = condition == "ap_oracle_diag"
    jitter = condition in ("jitter_base", "jitter_anti")
    truth_mix = {
        "truth_mix_25_diag": 0.25,
        "truth_mix_50_diag": 0.50,
        "truth_mix_75_diag": 0.75,
        "truth_full_diag": 1.00,
    }.get(condition, 0.0)
    route: List[Cell] = []
    chosen = None
    route_start_step = 0
    route_start_known = len(env.topo_nodes)
    route_start_explored = len(env.explored_cells)
    route_start_opt = v5.optimistic_distance(env, (env.x, env.y), env.goal)
    rows = []
    ap_values = []
    conf_values = []
    calls = 0
    recoveries = 0
    failure_reason = "max_steps"
    success = False
    t0 = time.time()

    for step in range(max_steps):
        if (env.x, env.y) == env.goal:
            result = env.step(v5.d.ACTION_STOP)
            success = bool(result["success"])
            failure_reason = result.get("reason", "stop_at_goal")
            break
        need_replan = (
            chosen is None
            or not route
            or chosen["cell"] not in env.topo_nodes
            or (env.x, env.y) == chosen["cell"]
        )
        if need_replan:
            if chosen is not None:
                info_gain = len(env.explored_cells) - route_start_explored
                progress = route_start_opt - v5.optimistic_distance(
                    env, (env.x, env.y), env.goal
                )
                failed_branch = memory.observe_route(
                    chosen.get("branch"), info_gain, progress, step,
                    route_cost=step - route_start_step,
                )
                if antistuck and failed_branch:
                    recoveries += 1
            candidates, map_diag = candidate_frontiers_v6(
                env, predictor, use_map, truth_mix
            )
            ap_values.append(map_diag["ap_unseen_pred"])
            conf_values.append(map_diag["map_confidence"])
            chosen = select_candidate(
                candidates, memory, step, antistuck, map_diag, oracle_ap,
                condition, max_steps, jitter
            )
            if chosen:
                route = list(chosen["path"])
                if route and route[0] == (env.x, env.y):
                    route.pop(0)
                route_start_step = step
                route_start_known = len(env.topo_nodes)
                route_start_explored = len(env.explored_cells)
                route_start_opt = v5.optimistic_distance(
                    env, (env.x, env.y), env.goal
                )
            else:
                route = []

        pos = (env.x, env.y)
        trajectory = env.visited_cells
        oscillating = (
            len(trajectory) >= 5
            and trajectory[-1] == trajectory[-3] == trajectory[-5]
            and trajectory[-2] == trajectory[-4]
        )
        if antistuck and oscillating and chosen and step - memory.last_position_trigger >= 4:
            chosen["branch"] = chosen.get("branch") or pos
            memory.observe_route(
                chosen["branch"], 0, 0, step,
                route_cost=step - route_start_step,
            )
            memory.last_position_trigger = step
            recoveries += 1
            route = []
            chosen = None
            continue

        if not route:
            # Move to an untried known neighbor, then allow a fresh selection.
            options = []
            for dx, dy in DIRS:
                nb = (env.x + dx, env.y + dy)
                if nb in env.topo_nodes:
                    edge = tuple(sorted(((env.x, env.y), nb)))
                    options.append((memory.edge_counts[edge], nb))
            options.sort()
            route = [options[0][1]] if options else []
            if not route:
                env.step(v5.d.ACTION_RIGHT)
                continue

        target = route[0]
        action = v5.turn_toward(env, target)
        prev = (env.x, env.y)
        result = env.step(action)
        if result["moved"]:
            edge = tuple(sorted((prev, (env.x, env.y))))
            memory.edge_counts[edge] += 1
            if route and (env.x, env.y) == route[0]:
                route.pop(0)
        elif result["bumped"]:
            route = []
            chosen = None
        memory.position_counts[(env.x, env.y)] += 1
        if memory.edge_counts[tuple(sorted((prev, (env.x, env.y))))] >= 5 and antistuck:
            if chosen:
                memory.observe_route(
                    chosen.get("branch"), 0, 0, step,
                    route_cost=step - route_start_step,
                )
            route = []
            chosen = None
            recoveries += 1
        rows.append({
            "step": step,
            "x": env.x,
            "y": env.y,
            "action": action,
            "moved": int(result["moved"]),
            "bumped": int(result["bumped"]),
            "target": f"{chosen['label']}:{chosen['cell']}" if chosen else "",
            "ap_unseen_pred": map_diag["ap_unseen_pred"] if need_replan else np.nan,
            "map_confidence": map_diag["map_confidence"] if need_replan else np.nan,
        })
        if result["done"]:
            success = bool(result["success"])
            failure_reason = result.get("reason", "")
            break

    rec = {
        "map_id": spec["id"],
        "condition": condition,
        "success": success,
        "reason": failure_reason,
        "shortest": int(env.shortest_len),
        "actual": int(env.forward_steps),
        "steps": len(rows),
        "osr": env.goal in env.visited_cells,
        "ne": manhattan((env.x, env.y), tuple(env.goal)),
        "coverage": len(env.topo_nodes) / max(1.0, float(spec["metrics"]["reachable_free"])),
        "bumps": sum(r["bumped"] for r in rows),
        "recoveries": recoveries,
        "branch_failures": sum(stat.failures for stat in memory.stats.values()),
        "llm_calls": calls,
        "ap_unseen_pred": float(np.mean(ap_values)) if ap_values else 0.0,
        "map_confidence": float(np.mean(conf_values)) if conf_values else 0.0,
        "wall_sec": round(time.time() - t0, 3),
        "end_pose": f"{env.x},{env.y},{v5.d.HEADING_NAME[env.heading]}",
    }
    return rec, rows


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    succ = [r for r in records if r["success"]]

    def mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "N": n,
        "SR": len(succ) / max(1, n),
        "SPL": mean(
            r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0.0
            for r in records
        ),
        "OSR": mean(int(r["osr"]) for r in records),
        "NE": mean(r["ne"] for r in records),
        "COV": mean(r["coverage"] for r in records),
        "BUMPS": sum(r["bumps"] for r in records),
        "RECOVERIES": sum(r["recoveries"] for r in records),
        "BRANCH_FAILURES": sum(r.get("branch_failures", 0) for r in records),
        "AP_UNSEEN_PRED": mean(r["ap_unseen_pred"] for r in records),
        "MAP_CONF": mean(r["map_confidence"] for r in records),
        "LLM_CALLS": sum(r["llm_calls"] for r in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--maps-file", default=str(ROOT / "benchmark_maps_v5.json"))
    parser.add_argument("--predictor-model", default=str(ROOT / "map_predictor_v6.json"))
    parser.add_argument("--start-map", type=int, default=0)
    parser.add_argument("--num-ep", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--tag", default="v6")
    parser.add_argument("--use-llm", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(Path(args.maps_file).read_text(encoding="utf-8"))
    maps = manifest["maps"][args.start_map:args.start_map + args.num_ep]
    predictor = LogisticMapPredictor(
        json.loads(Path(args.predictor_model).read_text(encoding="utf-8"))
    )
    outdir = ROOT / f"results_{args.tag}"
    outdir.mkdir(exist_ok=True)
    logger = v5.d.RunLogger(str(outdir / "log.txt"))
    records, all_rows = [], []
    for spec in maps:
        rec, rows = run_episode(
            spec, args.condition, predictor, args.max_steps,
            secrets.token_hex(4), logger, args.use_llm
        )
        records.append(rec)
        for row in rows:
            row["episode"] = spec["id"]
            row["condition"] = args.condition
            # JSON-safe NaN representation
            for key in ("ap_unseen_pred", "map_confidence"):
                if isinstance(row[key], float) and math.isnan(row[key]):
                    row[key] = None
        all_rows.extend(rows)
        print(
            f"[{spec['id']}] success={rec['success']} steps={rec['steps']} "
            f"actual={rec['actual']} shortest={rec['shortest']} NE={rec['ne']} "
            f"recoveries={rec['recoveries']}",
            flush=True,
        )
    logger.close()
    (outdir / "episodes.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "steps.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = aggregate(records)
    (outdir / "summary.txt").write_text(
        "\n".join(
            f"{key}={round(value, 4) if isinstance(value, float) else value}"
            for key, value in {"tag": args.tag, **metrics}.items()
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tag": args.tag, **metrics}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
