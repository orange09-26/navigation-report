# -*- coding: utf-8 -*-
"""MapGPT v5: synchronized topology + occupancy-matrix navigation.

The benchmark is a controlled text simulation.  It does not reproduce the
paper's Matterport3D numbers.  The implementation tests four ideas:

1. Fused spatial memory: a language topology and an explicit occupancy matrix
   are present in every decision context.
2. Hierarchical control: the LLM selects a frontier/waypoint; deterministic A*
   executes an action chunk on the currently known graph.
3. Execution-aware risk: candidate cost, information gain, dead-end risk and
   narrow-corridor risk are validated before execution.
4. Recovery from behavioral inertia: a failed or repeatedly unproductive route
   is temporarily vetoed instead of being retried verbatim.

The legacy AP calculation from mapgpt_vm_demo_v4 is kept unchanged.  The map
prompt asks the LLM to reproduce the fused matrix at decision time, making AP a
measure of representation fidelity rather than asking for an impossible global
map from unseen cells.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import re
import secrets
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path("/home/orange")
sys.path.insert(0, str(ROOT))

import mapgpt_vm_demo_v4 as d

Cell = Tuple[int, int]
DIRS = d.HEADING_DELTA

CONDITIONS = (
    "mapgpt",
    "fused_step",
    "topo_waypoint",
    "matrix_waypoint",
    "fused_waypoint",
    "risk_waypoint",
)

CFG = {
    "anchor_topk": 6,
    "action_chunk": 7,
    "replan_after_new_cells": 8,
    "stall_window": 8,
    "route_risk_veto": 0.72,
    "llm_cost_override": 1.65,
    "revisit_limit": 3,
    "risk_weights": {
        "goal": 0.44,
        "info": 0.24,
        "distance": 0.18,
        "risk": 0.14,
    },
}


class FixedBenchmarkEnv(d.GridNavEnv):
    def __init__(self, spec: Dict[str, Any]) -> None:
        h, w = spec["size"]
        super().__init__(height=h, width=w)
        self.spec = spec
        self.map_id = spec["id"]
        grid = [
            [d.FREE if ch in (".", "S", "G") else d.WALL for ch in row]
            for row in spec["ascii"]
        ]
        self.grid = grid
        self.start = tuple(spec["start"])
        self.goal = tuple(spec["goal"])
        self.shortest_len = int(spec["metrics"]["shortest"])
        self.rooms = [(1, 1, w - 2, h - 2)]
        self.instruction = self._benchmark_instruction(spec)
        self.reset_agent()

    def _benchmark_instruction(self, spec: Dict[str, Any]) -> str:
        sx, sy = spec["start"]
        gx, gy = spec["goal"]
        dx, dy = gx - sx, gy - sy
        ns = "south" if dy > 0 else ("north" if dy < 0 else "same row")
        ew = "east" if dx > 0 else ("west" if dx < 0 else "same column")
        m = spec["metrics"]
        return (
            "You are a wheeled indoor robot in an unfamiliar benchmark building. "
            f"The destination is roughly toward the {ns}-{ew} of the start, with "
            f"destination coordinate ({gx},{gy}). Navigate around obstacles and "
            f"through connected corridors. This scene contains about {m['junctions']} "
            f"turning points and {m['dead_ends']} dead ends; do not assume a direct "
            "straight line exists. Stop only after standing on the destination cell."
        )


def load_maps(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data["maps"])


def known_free(env: d.GridNavEnv) -> Set[Cell]:
    return set(env.topo_nodes)


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def known_astar(env: d.GridNavEnv, target: Cell) -> Optional[List[Cell]]:
    """Shortest route on observed free cells only."""
    nodes = known_free(env)
    src = (env.x, env.y)
    if target not in nodes:
        return None
    open_heap: List[Tuple[int, int, Cell]] = [(manhattan(src, target), 0, src)]
    parent: Dict[Cell, Optional[Cell]] = {src: None}
    gscore: Dict[Cell, int] = {src: 0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == target:
            path: List[Cell] = []
            while cur is not None:
                path.append(cur)
                cur = parent[cur]  # type: ignore[assignment]
            path.reverse()
            return path
        if g != gscore.get(cur):
            continue
        for dx, dy in DIRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb not in nodes:
                continue
            ng = g + 1
            if ng < gscore.get(nb, 10**9):
                gscore[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + manhattan(nb, target), ng, nb))
    return None


def _unknown_neighbor_count(env: d.GridNavEnv, cell: Cell) -> int:
    cnt = 0
    for dx, dy in DIRS:
        nb = (cell[0] + dx, cell[1] + dy)
        if env.in_bounds(*nb) and nb not in env.explored_cells:
            cnt += 1
    return cnt


def frontier_cells(env: d.GridNavEnv) -> List[Cell]:
    visited = set(env.visited_cells)
    return [
        cell for cell in known_free(env)
        if cell not in visited and _unknown_neighbor_count(env, cell) > 0
    ]


def cluster_frontiers(env: d.GridNavEnv) -> List[List[Cell]]:
    fs = set(frontier_cells(env))
    clusters: List[List[Cell]] = []
    while fs:
        seed = fs.pop()
        comp = [seed]
        q = deque([seed])
        while q:
            x, y = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (x + dx, y + dy)
                    if nb in fs:
                        fs.remove(nb)
                        comp.append(nb)
                        q.append(nb)
        clusters.append(comp)
    return clusters


def _corridor_width(env: d.GridNavEnv, cell: Cell) -> int:
    x, y = cell
    horizontal = int(env.is_free(x - 1, y)) + int(env.is_free(x + 1, y))
    vertical = int(env.is_free(x, y - 1)) + int(env.is_free(x, y + 1))
    return max(horizontal, vertical) + 1


def _information_gain(env: d.GridNavEnv, cell: Cell, radius: int = 3) -> float:
    total = 0
    unknown = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nb = (cell[0] + dx, cell[1] + dy)
            if env.in_bounds(*nb):
                total += 1
                if nb not in env.explored_cells:
                    unknown += 1
    return unknown / float(max(total, 1))


def optimistic_distance(env: d.GridNavEnv, src: Cell, target: Cell) -> int:
    """A* distance when unseen cells are optimistically traversable.

    Known walls remain blocked.  This is not ground truth; it resolves the
    systematic error of Manhattan distance around already-observed obstacles.
    """
    open_heap: List[Tuple[int, int, int, Cell]] = [
        (manhattan(src, target), 0, 0, src)
    ]
    best: Dict[Cell, Tuple[int, int]] = {src: (0, 0)}
    while open_heap:
        _, g, unknown_cost, cur = heapq.heappop(open_heap)
        if cur == target:
            return int((g + 2) // 4)
        if best.get(cur) != (g, unknown_cost):
            continue
        for dx, dy in DIRS:
            nb = (cur[0] + dx, cur[1] + dy)
            if not env.in_bounds(*nb):
                continue
            if nb in env.explored_cells and not env.is_free(*nb):
                continue
            is_unknown = nb not in env.explored_cells
            cost = 5 if is_unknown else 4
            state = (g + cost, unknown_cost + int(is_unknown))
            if state < best.get(nb, (10**9, 10**9)):
                best[nb] = state
                heapq.heappush(
                    open_heap,
                    (state[0] + 4 * manhattan(nb, target), state[0], state[1], nb),
                )
    return manhattan(src, target)


def candidate_frontiers(env: d.GridNavEnv, condition: str) -> List[Dict[str, Any]]:
    clusters = cluster_frontiers(env)
    candidates: List[Dict[str, Any]] = []
    shortest = max(1, env.shortest_len)
    for comp in clusters:
        scored_cells = []
        for cell in comp:
            path = known_astar(env, cell)
            if path is None:
                continue
            gain = _information_gain(env, cell)
            unknown_nb = _unknown_neighbor_count(env, cell)
            width = _corridor_width(env, cell)
            risk = (
                0.46 * (unknown_nb / 4.0)
                + 0.32 * (1.0 if width <= 2 else 0.0)
                + 0.22 * min(1.0, max(0, len(path) - shortest) / shortest)
            )
            scored_cells.append((gain, -len(path), -risk, cell, path, risk, width))
        if not scored_cells:
            continue
        scored_cells.sort(reverse=True)
        gain, neg_len, neg_risk, cell, path, risk, width = scored_cells[0]
        goal_h = manhattan(cell, env.goal)
        optimistic_h = optimistic_distance(env, cell, env.goal)
        goal_term = 1.0 - min(1.0, goal_h / float(shortest * 2 + 1))
        dist_term = 1.0 - min(1.0, (len(path) - 1) / float(shortest * 2 + 1))
        info_term = min(1.0, gain + 0.15 * min(4, _unknown_neighbor_count(env, cell)))
        w = CFG["risk_weights"]
        utility = (
            w["goal"] * goal_term
            + w["info"] * info_term
            + w["distance"] * dist_term
            - w["risk"] * risk
        )
        candidates.append({
            "cell": cell,
            "path": path,
            "path_len": len(path) - 1,
            "goal_h": goal_h,
            "gain": gain,
            "risk": risk,
            "width": width,
            "utility": utility,
            "optimistic_h": optimistic_h,
            "nav_cost": len(path) - 1 + optimistic_h,
            "kind": "FRONTIER",
        })

    # The goal becomes a candidate as soon as its cell is observed and connected.
    if (env.goal in known_free(env)) and (env.goal not in env.visited_cells):
        path = known_astar(env, env.goal)
        if path is not None:
            cells = [c for c in path if c not in env.visited_cells]
            risk = 0.08 * sum(1 for c in cells if _corridor_width(env, c) <= 2)
            risk /= max(1, len(cells))
            candidates.append({
                "cell": env.goal,
                "path": path,
                "path_len": len(path) - 1,
                "goal_h": 0,
                "gain": 0.0,
                "risk": risk,
                "width": _corridor_width(env, env.goal),
                "utility": 1.25 - 0.25 * risk,
                "nav_cost": len(path) - 1,
                "optimistic_h": 0,
                "kind": "GOAL",
            })

    # One candidate per spatial cluster, then a global deterministic ranking.
    candidates.sort(key=lambda c: (-c["utility"], c["risk"], c["path_len"], c["cell"]))
    top = candidates[:CFG["anchor_topk"]]
    if condition == "risk_waypoint":
        top.sort(key=lambda c: (
            c["kind"] != "GOAL",
            c["nav_cost"] + 2.2 * c["risk"] - 0.9 * c["gain"],
            c["risk"],
            c["cell"],
        ))
    for i, cand in enumerate(top):
        cand["label"] = f"C{i + 1}"
    return top


def topology_cells(env: d.GridNavEnv, frontier_set: set[Cell]) -> List[Cell]:
    nodes = known_free(env)
    out = {(env.x, env.y), env.goal}
    for cell in nodes:
        deg = sum((cell[0] + dx, cell[1] + dy) in nodes for dx, dy in DIRS)
        if deg >= 3 or deg == 1 or cell in frontier_set:
            out.add(cell)
    return sorted(out)


def render_fused_matrix(env: d.GridNavEnv) -> str:
    """Render full coordinates with unknown cells hidden, not the truth map."""
    lines = [
        "FUSED_OCCUPANCY_MATRIX axis=x_right,y_down; "
        "legend R=robot *=visited .=known-free #=known-wall G=destination ?=unseen"
    ]
    for y in range(env.height):
        row = []
        for x in range(env.width):
            cell = (x, y)
            if cell == (env.x, env.y):
                row.append("R")
            elif cell == env.goal and cell in env.explored_cells:
                row.append("G")
            elif cell not in env.explored_cells:
                row.append("?")
            elif not env.is_free(x, y):
                row.append("#")
            elif cell in env.visited_cells:
                row.append("*")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)


def render_topology(env: d.GridNavEnv, candidates: Sequence[Dict[str, Any]]) -> str:
    nodes = known_free(env)
    frontier_set = {c["cell"] for c in candidates}
    selected = topology_cells(env, frontier_set)
    id_by_cell = env.topo_nodes
    lines = [
        "TOPOLOGICAL_SKELETON (node IDs refer to synchronized cells in the fused matrix)",
        "role edge endpoint; J=junction, D=dead-end, F=frontier, R=robot, G=destination-if-seen",
    ]
    for cell in selected[:28]:
        deg = sum((cell[0] + dx, cell[1] + dy) in nodes for dx, dy in DIRS)
        roles = []
        if cell == (env.x, env.y):
            roles.append("R")
        if cell == env.goal and cell in nodes:
            roles.append("G")
        if deg >= 3:
            roles.append("J")
        if deg == 1:
            roles.append("D")
        if cell in frontier_set:
            roles.append("F")
        pid = id_by_cell.get(cell, "?")
        neighbours = []
        for dx, dy in DIRS:
            nb = (cell[0] + dx, cell[1] + dy)
            if nb in nodes:
                neighbours.append(str(id_by_cell.get(nb, "?")))
        dir_count = len(neighbours)
        lines.append(
            f"Place {pid} cell=({cell[0]},{cell[1]}) roles={''.join(roles) or 'C'} "
            f"degree={dir_count} neighbours={','.join(neighbours[:8]) or '(none)'}"
        )
    return "\n".join(lines)


def render_candidates(candidates: Sequence[Dict[str, Any]]) -> str:
    if not candidates:
        return "CANDIDATES: none (exploration appears complete)"
    lines = [
        "CANDIDATES ranked by execution-aware utility; choose exactly one label",
        "label kind cell path_len manhattan_to_goal optimistic_h info_gain risk "
        "corridor_width nav_cost utility",
    ]
    for c in candidates:
        lines.append(
            f"{c['label']} {c['kind']} ({c['cell'][0]},{c['cell'][1]}) "
            f"path={c['path_len']} goal_h={c['goal_h']} opt_h={c['optimistic_h']} "
            f"info={c['gain']:.2f} "
            f"risk={c['risk']:.2f} width={c['width']} nav_cost={c['nav_cost']} "
            f"utility={c['utility']:.3f}"
        )
    return "\n".join(lines)


def history_summary(env: d.GridNavEnv, history: Sequence[str]) -> str:
    recent = list(history[-6:])
    return "recent actions: " + (" | ".join(recent) if recent else "(none)")


def build_prompt_v5(
    condition: str,
    env: d.GridNavEnv,
    step_idx: int,
    history: Sequence[str],
    candidates: Sequence[Dict[str, Any]],
    risk_note: str,
    request_nonce: str,
) -> str:
    use_matrix = condition != "topo_waypoint"
    use_topo = condition != "matrix_waypoint"
    mode = "primitive_action" if condition in ("mapgpt", "fused_step") else "waypoint"
    blocks = [
        "[fresh_request_id=%s]" % request_nonce,
        "You are a zero-shot MapGPT navigation agent. You control a wheeled robot in a fixed, "
        "partially observed occupancy-grid building. Do not invent unseen geometry.",
        "## TASK MEMORY\n" + env.instruction,
        f"## PROGRESS\nstep={step_idx} forward_steps={env.forward_steps} "
        f"robot=({env.x},{env.y}) heading={d.HEADING_NAME[env.heading]} "
        f"destination=({env.goal[0]},{env.goal[1]}) shortest_known_reference={env.shortest_len}",
        "## LOCAL OBSERVATION\n" + env.last_obs_text,
        "## TEMPORAL MEMORY\n" + history_summary(env, history),
    ]
    if use_matrix:
        blocks.append("## SPATIAL MEMORY: " + render_fused_matrix(env))
    if use_topo:
        blocks.append("## SEMANTIC MEMORY\n" + render_topology(env, candidates))
    blocks.append("## FRONTIER CANDIDATES\n" + render_candidates(candidates))
    if risk_note:
        blocks.append("## EXECUTION-LEVEL FEEDBACK\n" + risk_note)

    if mode == "primitive_action":
        blocks.append(
            "## DECISION\nChoose one primitive action to make progress toward the best candidate. "
            "A deterministic controller is not being used in this condition."
        )
        action_help = (
            "<SUBGOAL>candidate label or cell</SUBGOAL>\n"
            "<ACTION>FORWARD|LEFT|RIGHT|STOP</ACTION>"
        )
    else:
        blocks.append(
            "## DECISION\nChoose one candidate label. The local geometric controller will run A* "
            "on the KNOWN graph and execute a short action chunk. Prefer GOAL when present. "
            "Prefer lower risk only when its goal/information value is comparable."
        )
        action_help = (
            "<TARGET>C1</TARGET>  (exactly one candidate label; STOP only if standing on destination)"
        )
    blocks.append(
        "## OUTPUT\n"
        "Return only these XML tags in this order:\n"
        "<THINK>short decision evidence</THINK>\n"
        "<PLAN>multi-step waypoint plan</PLAN>\n"
        + action_help + "\n"
        "<MAP>\n"
    )
    if use_matrix:
        blocks.append(
            "Copy the provided FUSED_OCCUPANCY_MATRIX exactly, including every row, then add at "
            "most 6 inferred 'Place X is connected with Places Y,Z.' lines. This grid is only "
            "the observed map; '?' remains unknown."
        )
    else:
        blocks.append(
            "Summarize the observed topology as at most 12 'Place X is connected with Places ...' "
            "lines and at most 12 CELL x,y,FREE/WALL lines."
        )
    blocks.append("</MAP>")
    return "\n\n".join(blocks) + "\n"


def parse_target(text: str, candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = d._strip_tag(text, "TARGET").strip()
    m = re.search(r"C\s*(\d+)", raw, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"<TARGET>\s*C\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    m = re.search(r"(-?\d+)\s*,\s*(-?\d+)", raw)
    if m:
        cell = (int(m.group(1)), int(m.group(2)))
        for cand in candidates:
            if cand["cell"] == cell:
                return cand
    return None


def choose_deterministic(candidates: Sequence[Dict[str, Any]], risk_aware: bool) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    if not risk_aware:
        return candidates[0]
    goal = [c for c in candidates if c["kind"] == "GOAL"]
    if goal:
        return min(goal, key=lambda c: (c["risk"], c["path_len"]))
    return min(
        candidates,
        key=lambda c: (
            c["kind"] != "GOAL",
            c["nav_cost"] + 2.2 * c["risk"] - 0.9 * c["gain"],
            c["risk"],
            c["cell"],
        ),
    )


def apply_risk_veto(
    chosen: Optional[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Counterfactual route check for the risk-aware condition."""
    if not chosen or not candidates:
        return chosen, ""
    best = candidates[0]
    notes = []
    chosen_nav = chosen["nav_cost"] + 2.2 * chosen["risk"] - 0.9 * chosen["gain"]
    best_nav = best["nav_cost"] + 2.2 * best["risk"] - 0.9 * best["gain"]
    if chosen["risk"] > CFG["route_risk_veto"] and best["risk"] + 0.12 < chosen["risk"]:
        notes.append(
            f"chosen {chosen['label']} risk={chosen['risk']:.2f} exceeded veto; "
            f"switched to {best['label']} risk={best['risk']:.2f}"
        )
        chosen = best
    elif chosen_nav > best_nav + 4.0 and chosen["kind"] != "GOAL":
        notes.append(
            f"chosen {chosen['label']} execution-cost score={chosen_nav:.1f} exceeded "
            f"{best['label']} score={best_nav:.1f}; switched to the safer fallback"
        )
        chosen = best
    return chosen, "; ".join(notes)


def turn_toward(env: d.GridNavEnv, target: Cell) -> str:
    dx, dy = target[0] - env.x, target[1] - env.y
    if dx == 0 and dy == 0:
        return d.ACTION_STOP
    if abs(dx) >= abs(dy):
        desired = 1 if dx > 0 else 3
    else:
        desired = 2 if dy > 0 else 0
    if env.heading == desired:
        return d.ACTION_FORWARD
    diff = (desired - env.heading) % 4
    return d.ACTION_RIGHT if diff == 1 else d.ACTION_LEFT


def compact_matrix_only(env: d.GridNavEnv) -> str:
    return render_fused_matrix(env)


def step_action(env: d.GridNavEnv, action: str) -> Dict[str, Any]:
    return env.step(action)


def estimate_step_ap(parsed: Dict[str, Any], env: d.GridNavEnv) -> Tuple[float, float]:
    """Return (fused AP, legacy AP).

    Fused AP evaluates whether the structured MAP reconstructs the currently
    observed occupancy and topology.  Missing observed cells are wrong, so a
    sparse map cannot look perfect merely by predicting a few easy cells.
    Legacy AP is retained byte-for-byte through mapgpt_vm_demo_v4.
    """
    legacy = float(d.compute_ap_step(parsed, env))
    pred: Dict[Cell, str] = dict(parsed.get("pred_cells") or {})
    ascii_cells = d.align_ascii_cells(
        parsed.get("ascii_rows"), parsed.get("ascii_robot"), env
    )
    pred.update(ascii_cells)
    observed = {
        (x, y) for y in range(env.height) for x in range(env.width)
        if (x, y) in env.explored_cells
    }
    correct = 0
    for cell in observed:
        if pred.get(cell) == env.true_cell_label(*cell):
            correct += 1
    cell_acc = correct / float(max(1, len(observed)))

    nodes = known_free(env)
    pos_to_pid = env.topo_nodes
    pred_edges: Set[Tuple[int, int]] = set()
    for cell, label in pred.items():
        if label != "FREE" or cell not in nodes:
            continue
        pid = pos_to_pid[cell]
        for dx, dy in ((1, 0), (0, 1)):
            nb = (cell[0] + dx, cell[1] + dy)
            if nb in nodes and pred.get(nb) == "FREE":
                qid = pos_to_pid[nb]
                pred_edges.add((pid, qid) if pid <= qid else (qid, pid))
    true_edges = set(env.topo_edges)
    union = pred_edges | true_edges
    edge_f1 = (
        len(pred_edges & true_edges) / float(len(union)) if union else 1.0
    )
    fused = 0.70 * cell_acc + 0.30 * edge_f1
    return float(fused), legacy


def action_from_text(llm_text: str, env: d.GridNavEnv) -> Tuple[str, str, bool]:
    parsed = d.parse_llm_response(llm_text)
    if (not llm_text) or (not parsed["parse_ok"]) or parsed["action"] not in d.VALID_ACTIONS:
        return d.fallback_action(env), "FALLBACK", parsed["parse_ok"]
    return parsed["action"], "LLM", True


def run_primitive_condition(
    condition: str,
    env: d.GridNavEnv,
    logger: d.RunLogger,
    max_steps: int,
    run_nonce: str,
    mock_llm: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    history: List[str] = []
    step_rows: List[Dict[str, Any]] = []
    aps: List[float] = []
    calls = 0
    stop_rejects = 0
    success = False
    reason = "max_steps"
    t0 = time.time()
    prev_planning = "Navigation has just started, with no planning yet."

    for step_idx in range(max_steps):
        candidates = candidate_frontiers(env, condition)
        if condition == "mapgpt":
            prompt = d.build_mapgpt_prompt(env, step_idx, history, prev_planning)
        else:
            prompt = build_prompt_v5(
                condition, env, step_idx, history, candidates, "", f"{run_nonce}:{step_idx}"
            )
        if mock_llm:
            chosen = choose_deterministic(candidates, False)
            if chosen:
                llm_text = (
                    f"<THINK>mock choose {chosen['label']}</THINK><PLAN>advance</PLAN>"
                    f"<SUBGOAL>{chosen['cell'][0]},{chosen['cell'][1]}</SUBGOAL>"
                    f"<ACTION>FORWARD</ACTION><MAP>{render_fused_matrix(env)}</MAP>"
                )
            else:
                llm_text = "<THINK>no frontier</THINK><PLAN>stop</PLAN><ACTION>STOP</ACTION><MAP></MAP>"
            net = "mock"
        else:
            llm_text, net = d.request_llm(prompt)
            calls += 1
        parsed = d.parse_llm_response(llm_text)
        if parsed.get("plan"):
            prev_planning = parsed["plan"]
        ap, ap_legacy = estimate_step_ap(parsed, env)
        aps.append((ap, ap_legacy))
        action, source, _ = action_from_text(llm_text, env)
        if action == d.ACTION_STOP and (env.x, env.y) != env.goal and stop_rejects < 3:
            stop_rejects += 1
            action = turn_toward(env, candidates[0]["cell"]) if candidates else d.ACTION_LEFT
            source = "STOP_REJECT"
        res = step_action(env, action)
        done = bool(res["done"])
        if done:
            success = bool(res["success"])
            reason = str(res.get("reason", ""))
        history.append(f"{action}->({env.x},{env.y})")
        row = {
            "step": step_idx, "action": action, "source": source, "x": env.x, "y": env.y,
            "moved": int(res["moved"]), "bumped": int(res["bumped"]),
            "ap_fused": round(ap, 4), "ap_legacy": round(ap_legacy, 4),
            "llm_calls": calls, "target": "",
            "risk_veto": "", "done": int(done),
        }
        step_rows.append(row)
        logger.write(
            f"episode step={step_idx} condition={condition} pose=({env.x},{env.y}) "
            f"action={action} source={source} AP_FUSED={ap:.4f} "
            f"AP_LEGACY={ap_legacy:.4f} done={done}\n"
            f"PROMPT_BEGIN\n{prompt}\nPROMPT_END\nLLM_BEGIN\n{llm_text}\nLLM_END"
        )
        if mock_llm and action == d.ACTION_FORWARD:
            # The mock is only a harness check, not a policy.
            continue
        if done:
            break
    else:
        success = env.at_goal()
        reason = "reached_goal" if success else "max_steps"

    rec = make_episode_record(
        env, success, reason, step_rows, aps, calls, 0, stop_rejects, t0
    )
    return rec, step_rows


def _candidate_reached(env: d.GridNavEnv, candidate: Dict[str, Any]) -> bool:
    return (env.x, env.y) == candidate["cell"]


def run_hierarchical_condition(
    condition: str,
    env: d.GridNavEnv,
    logger: d.RunLogger,
    max_steps: int,
    run_nonce: str,
    mock_llm: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    history: List[str] = []
    step_rows: List[Dict[str, Any]] = []
    aps: List[float] = []
    calls = 0
    veto_count = 0
    recovery_count = 0
    success = False
    reason = "max_steps"
    route: List[Cell] = []
    chosen: Optional[Dict[str, Any]] = None
    consecutive_unproductive = 0
    vetoed: Dict[Cell, int] = {}
    last_new_cells = 0
    t0 = time.time()
    risk_aware = condition == "risk_waypoint"

    for step_idx in range(max_steps):
        candidates = candidate_frontiers(env, condition)
        if risk_aware:
            candidates = [c for c in candidates if vetoed.get(c["cell"], 0) < 2]
        at_goal = (env.x, env.y) == env.goal
        need_replan = (
            chosen is None
            or not route
            or _candidate_reached(env, chosen)
            or chosen["cell"] not in known_free(env)
            or (step_idx - last_new_cells >= CFG["replan_after_new_cells"])
        )
        risk_note = ""
        if need_replan:
            prompt = build_prompt_v5(
                condition, env, step_idx, history, candidates, "", f"{run_nonce}:{step_idx}"
            )
            if mock_llm:
                chosen_new = choose_deterministic(candidates, risk_aware)
                llm_text = (
                    f"<THINK>mock choose {chosen_new['label'] if chosen_new else 'STOP'}</THINK>"
                    f"<PLAN>waypoint</PLAN>"
                    f"<TARGET>{chosen_new['label'] if chosen_new else 'STOP'}</TARGET>"
                    f"<MAP>{render_fused_matrix(env)}</MAP>"
                )
                net = "mock"
            else:
                llm_text, net = d.request_llm(prompt)
                calls += 1
            parsed = d.parse_llm_response(llm_text)
            ap, ap_legacy = estimate_step_ap(parsed, env)
            aps.append((ap, ap_legacy))
            chosen_new = parse_target(llm_text, candidates)
            if chosen_new is None:
                chosen_new = choose_deterministic(candidates, risk_aware)
                source = "FALLBACK"
            else:
                source = "LLM"
            if risk_aware:
                chosen_new, veto_note = apply_risk_veto(chosen_new, candidates)
                if veto_note:
                    veto_count += 1
                    risk_note = veto_note
                    history.append(veto_note)
                    source += "+VETO"
            chosen = chosen_new
            if chosen:
                route = list(chosen["path"])
                if route and route[0] == (env.x, env.y):
                    route.pop(0)
            else:
                route = []
            logger.write(
                f"episode step={step_idx} condition={condition} REPLAN cell=({env.x},{env.y}) "
                f"target={chosen['label'] if chosen else 'NONE'} source={source} "
                f"AP_FUSED={ap:.4f} AP_LEGACY={ap_legacy:.4f} "
                f"note={risk_note}\nPROMPT_BEGIN\n{prompt}\nPROMPT_END\nLLM_BEGIN\n{llm_text}\nLLM_END"
            )
        else:
            ap, ap_legacy = 0.0, 0.0

        if at_goal:
            res = env.step(d.ACTION_STOP)
            success = bool(res["success"])
            reason = str(res.get("reason", "stop_at_goal"))
            step_rows.append({
                "step": step_idx, "action": d.ACTION_STOP, "source": "GOAL_GATE",
                "x": env.x, "y": env.y, "moved": 0, "bumped": 0,
                "ap_fused": round(ap, 4), "ap_legacy": round(ap_legacy, 4),
                "llm_calls": calls, "target": "GOAL", "risk_veto": risk_note, "done": 1,
            })
            break

        if not route:
            # No known route. Prefer an unvisited known neighbour before asking again.
            cur = (env.x, env.y)
            options = []
            for dx, dy in DIRS:
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in known_free(env):
                    options.append(nb)
            options.sort(key=lambda c: (vetoed.get(c, 0), c in env.visited_cells, manhattan(c, env.goal)))
            if options:
                route = [options[0]]
            else:
                action = d.ACTION_RIGHT
                res = env.step(action)
                step_rows.append({
                    "step": step_idx, "action": action, "source": "RECOVERY",
                    "x": env.x, "y": env.y, "moved": 0, "bumped": 0,
                    "ap_fused": 0.0, "ap_legacy": 0.0,
                    "llm_calls": calls, "target": "", "risk_veto": "", "done": int(res["done"]),
                })
                continue

        target = route[0]
        action = turn_toward(env, target)
        before_known = len(known_free(env))
        before_visited = len(set(env.visited_cells))
        res = step_action(env, action)
        after_known = len(known_free(env))
        after_visited = len(set(env.visited_cells))
        if res["moved"]:
            if after_known > before_known or after_visited > before_visited:
                last_new_cells = step_idx
                consecutive_unproductive = 0
            else:
                consecutive_unproductive += 1
            if route and (env.x, env.y) == route[0]:
                route.pop(0)
        elif res["bumped"]:
            route = []
            chosen = None
            consecutive_unproductive += 1
        done = bool(res["done"])
        if done:
            success = bool(res["success"])
            reason = str(res.get("reason", ""))

        if risk_aware and consecutive_unproductive >= CFG["revisit_limit"] and chosen:
            vetoed[chosen["cell"]] = vetoed.get(chosen["cell"], 0) + 1
            history.append(
                f"recovery: veto {chosen['label']} at {chosen['cell']} after "
                f"{consecutive_unproductive} unproductive steps"
            )
            route = []
            chosen = None
            recovery_count += 1
            consecutive_unproductive = 0

        history.append(f"{action}->({env.x},{env.y})")
        row = {
            "step": step_idx, "action": action, "source": "A_STAR_CHUNK",
            "x": env.x, "y": env.y, "moved": int(res["moved"]), "bumped": int(res["bumped"]),
            "ap_fused": round(ap, 4), "ap_legacy": round(ap_legacy, 4),
            "llm_calls": calls,
            "target": f"{chosen['label']}:{chosen['cell']}" if chosen else "",
            "risk_veto": risk_note, "done": int(done),
        }
        if res["bumped"]:
            row["source"] = "EXECUTOR_RECOVERY"
        step_rows.append(row)
        if done:
            break
        if step_idx - last_new_cells >= CFG["replan_after_new_cells"]:
            # The next loop performs a fresh memory-grounded decision.
            chosen = None
            route = []
    else:
        success = env.at_goal()
        reason = "reached_goal" if success else "max_steps"

    rec = make_episode_record(
        env, success, reason, step_rows, aps, calls, recovery_count, 0, t0
    )
    rec["veto_count"] = veto_count
    return rec, step_rows


def make_episode_record(
    env: d.GridNavEnv,
    success: bool,
    reason: str,
    rows: Sequence[Dict[str, Any]],
    aps: Sequence[Tuple[float, float]],
    calls: int,
    recoveries: int,
    stop_rejects: int,
    t0: float,
) -> Dict[str, Any]:
    free_total = sum(
        1 for y in range(env.height) for x in range(env.width) if env.grid[y][x] == d.FREE
    )
    known_correct = 0
    known_total = 0
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in env.explored_cells:
                continue
            known_total += 1
            known_correct += int(
                env.is_free(x, y) == (env.true_cell_label(x, y) == "FREE")
            )
    map_fidelity = known_correct / float(max(1, known_total))
    return {
        "map_id": env.map_id,
        "success": bool(success),
        "reason": reason,
        "shortest": int(env.shortest_len),
        "actual": int(env.forward_steps),
        "steps": len(rows),
        "osr": bool(env.goal in env.visited_cells),
        "ne": manhattan((env.x, env.y), env.goal),
        "ap": sum(x[0] for x in aps) / len(aps) if aps else 0.0,
        "ap_legacy": sum(x[1] for x in aps) / len(aps) if aps else 0.0,
        "map_fidelity": map_fidelity,
        "coverage": len([c for c in known_free(env) if env.is_free(*c)]) / float(max(1, free_total)),
        "llm_calls": calls,
        "recoveries": recoveries,
        "stop_rejects": stop_rejects,
        "fallback_steps": sum(r["source"].startswith("FALLBACK") for r in rows),
        "bumps": sum(r["bumped"] for r in rows),
        "end_pose": f"{env.x},{env.y},{d.HEADING_NAME[env.heading]}",
        "wall_sec": round(time.time() - t0, 1),
        "start": f"{env.start[0]},{env.start[1]}",
        "goal": f"{env.goal[0]},{env.goal[1]}",
    }


def aggregate(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    succ = [r for r in records if r["success"]]

    def mean(vals: Iterable[float]) -> float:
        vals = list(vals)
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "N": n,
        "SR": len(succ) / float(max(1, n)),
        "SPL": mean(
            r["shortest"] / float(max(r["actual"], r["shortest"], 1)) if r["success"] else 0.0
            for r in records
        ),
        "SPL_succ_only": mean(
            r["shortest"] / float(max(r["actual"], r["shortest"], 1)) for r in succ
        ),
        "OSR": mean(1.0 if r["osr"] else 0.0 for r in records),
        "NE": mean(r["ne"] for r in records),
        "AP": mean(r["ap"] for r in records),
        "AP_LLM_LEGACY": mean(r["ap_legacy"] for r in records),
        "MAP_FID": mean(r["map_fidelity"] for r in records),
        "COV": mean(r["coverage"] for r in records),
        "LLM_CALLS": sum(r["llm_calls"] for r in records),
        "STEPS": sum(r["steps"] for r in records),
        "RECOVERIES": sum(r["recoveries"] for r in records),
        "VETOS": sum(r.get("veto_count", 0) for r in records),
        "WALL_SEC": sum(r["wall_sec"] for r in records),
    }


def write_outputs(outdir: Path, tag: str, records: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "episodes.json").write_text(
        json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "steps.json").write_text(
        json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if records:
        with (outdir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            wr.writeheader()
            wr.writerows(records)
    if rows:
        with (outdir / "steps.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    m = aggregate(records)
    lines = [f"tag={tag} N={len(records)}"]
    lines.extend(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in m.items())
    (outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-file", default=str(ROOT / "benchmark_maps_v5.json"))
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--start-map", type=int, default=0)
    parser.add_argument("--num-ep", type=int, default=12)
    parser.add_argument("--map-id", default="")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--obs-size", type=int, default=5)
    parser.add_argument("--tag", default="v5")
    parser.add_argument("--nonce", default="")
    parser.add_argument("--mock-llm", action="store_true")
    args = parser.parse_args()

    maps = load_maps(args.maps_file)
    if args.map_id:
        maps = [m for m in maps if m["id"] == args.map_id]
    else:
        maps = maps[args.start_map:args.start_map + args.num_ep]
    if not maps:
        raise SystemExit("no maps selected")

    d.LLM_URL = os.getenv("MAPGPT_LLM_URL", "http://127.0.0.1:5000/get_llm_response")
    d.OBS_SIZE = args.obs_size
    d.MAX_STEPS = args.max_steps
    d.SLEEP_AFTER_OK_SEC = float(os.getenv("MAPGPT_SLEEP_OK", "0.25"))
    run_nonce = args.nonce or secrets.token_hex(8)
    outdir = ROOT / f"results_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    logger = d.RunLogger(str(outdir / "log.txt"))
    config = {
        "tag": args.tag,
        "condition": args.condition,
        "maps": [m["id"] for m in maps],
        "maps_file": args.maps_file,
        "max_steps": args.max_steps,
        "obs_size": args.obs_size,
        "mock_llm": args.mock_llm,
        "nonce": run_nonce,
        "llm_url": d.LLM_URL,
        "model": os.getenv("MAPGPT_MODEL", "qwen-plus"),
        "cfg": CFG,
    }
    (outdir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for spec in maps:
        env = FixedBenchmarkEnv(spec)
        if args.condition in ("mapgpt", "fused_step"):
            rec, rows = run_primitive_condition(
                args.condition, env, logger, args.max_steps, run_nonce, args.mock_llm
            )
        else:
            rec, rows = run_hierarchical_condition(
                args.condition, env, logger, args.max_steps, run_nonce, args.mock_llm
            )
        rec["condition"] = args.condition
        for i, row in enumerate(rows):
            row["episode"] = spec["id"]
            row["condition"] = args.condition
            row["index"] = i
        records.append(rec)
        all_rows.extend(rows)
        print(
            f"[MAP {spec['id']}] success={rec['success']} steps={rec['steps']} "
            f"shortest={rec['shortest']} actual={rec['actual']} AP={rec['ap']:.4f} "
            f"calls={rec['llm_calls']} recoveries={rec['recoveries']}",
            flush=True,
        )
        # Incremental summaries protect long runs from a late interruption.
        (outdir / "episodes.partial.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    logger.close()
    m = write_outputs(outdir, args.tag, records, all_rows)
    print("\n".join(f"{k}={v}" for k, v in m.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
