# -*- coding: utf-8 -*-
"""Lightweight occupancy-map completion for MapGPT v6 experiments.

The predictor is trained only on the four reserve maps in the fixed manifest.
The eight evaluation maps are never used to fit its parameters.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

ROOT = Path("/home/orange")
sys.path.insert(0, str(ROOT))
import mapgpt_v5 as v5
import benchmark_maps_v5 as maps_v5

Cell = Tuple[int, int]
CHECKPOINTS = (5, 15, 30)


def _sign(env: v5.FixedBenchmarkEnv, cell: Cell) -> float:
    if cell not in env.explored_cells:
        return 0.0
    return -1.0 if not env.is_free(*cell) else 1.0


def _nearest_known_distance(
    env: v5.FixedBenchmarkEnv, cell: Cell, want_wall: bool, limit: int = 7
) -> float:
    x, y = cell
    best = limit
    for radius in range(1, limit + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                nb = (x + dx, y + dy)
                if nb not in env.explored_cells:
                    continue
                is_wall = not env.is_free(*nb)
                if is_wall == want_wall:
                    best = min(best, max(abs(dx), abs(dy)))
        if best < limit:
            break
    return best / float(limit)


def cell_features(env: v5.FixedBenchmarkEnv, cell: Cell) -> List[float]:
    x, y = cell
    left = _sign(env, (x - 1, y))
    right = _sign(env, (x + 1, y))
    up = _sign(env, (x, y - 1))
    down = _sign(env, (x, y + 1))
    free1 = wall1 = free2 = wall2 = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dx == 0 and dy == 0:
                continue
            nb = (x + dx, y + dy)
            if nb not in env.explored_cells:
                continue
            if env.is_free(*nb):
                if max(abs(dx), abs(dy)) == 1:
                    free1 += 1
                free2 += 1
            else:
                if max(abs(dx), abs(dy)) == 1:
                    wall1 += 1
                wall2 += 1

    opposite_h = 1.0 if left != 0 and left == right else 0.0
    opposite_v = 1.0 if up != 0 and up == down else 0.0
    opposite_wall_h = 1.0 if left == right == -1 else 0.0
    opposite_wall_v = 1.0 if up == down == -1 else 0.0
    opposite_free_h = 1.0 if left == right == 1 else 0.0
    opposite_free_v = 1.0 if up == down == 1 else 0.0

    row_known = []
    col_known = []
    for xx in range(env.width):
        if (xx, y) in env.explored_cells:
            row_known.append(1.0 if env.is_free(xx, y) else 0.0)
    for yy in range(env.height):
        if (x, yy) in env.explored_cells:
            col_known.append(1.0 if env.is_free(x, yy) else 0.0)
    row_free = sum(row_known) / max(1, len(row_known))
    col_free = sum(col_known) / max(1, len(col_known))

    return [
        1.0,
        left, right, up, down,
        opposite_h, opposite_v,
        opposite_wall_h, opposite_wall_v,
        opposite_free_h, opposite_free_v,
        free1 / 8.0, wall1 / 8.0, free2 / 24.0, wall2 / 24.0,
        _nearest_known_distance(env, cell, False),
        _nearest_known_distance(env, cell, True),
        row_free, col_free,
        x / max(1.0, env.width - 1.0),
        y / max(1.0, env.height - 1.0),
        (x == 0 or y == 0 or x == env.width - 1 or y == env.height - 1),
    ]


def _random_snapshots(
    spec: Dict[str, Any], seed: int, count: int
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    snapshots = []
    for episode in range(count):
        env = v5.FixedBenchmarkEnv(spec)
        env.goal = (-10, -10)  # delay goal termination during training walks
        for step in range(36):
            free_dirs = []
            for heading, (dx, dy) in enumerate(v5.DIRS):
                nb = (env.x + dx, env.y + dy)
                if env.is_free(*nb):
                    free_dirs.append(heading)
            target_heading = rng.choice(free_dirs)
            if env.heading != target_heading:
                diff = (target_heading - env.heading) % 4
                env.step(v5.d.ACTION_RIGHT if diff == 1 else v5.d.ACTION_LEFT)
            else:
                env.step(v5.d.ACTION_FORWARD)
            if step + 1 in (6, 12, 20, 30):
                snapshots.append({
                    "map_id": spec["id"] + f"_train{episode}",
                    "step": step + 1,
                    "explored": sorted(env.explored_cells),
                    "grid": [row[:] for row in env.grid],
                    "height": env.height,
                    "width": env.width,
                    "pose": [env.x, env.y, env.heading],
                    "start": list(env.start),
                    "goal": list(env.goal),
                })
        # Goal is restored for reproducibility metadata only.
        snapshots[-1]["split"] = "train"
    return snapshots


def _snapshots_from_trajectory(
    specs: Sequence[Dict[str, Any]], steps_path: Path
) -> List[Dict[str, Any]]:
    steps = json.loads(steps_path.read_text(encoding="utf-8"))
    by_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in steps:
        by_map.setdefault(row["episode"], []).append(row)
    snapshots = []
    for spec in specs:
        env = v5.FixedBenchmarkEnv(spec)
        for row in sorted(by_map.get(spec["id"], []), key=lambda r: r["step"]):
            env.step(row["action"])
            if row["step"] + 1 in CHECKPOINTS:
                snapshots.append({
                    "map_id": spec["id"],
                    "step": row["step"] + 1,
                    "explored": sorted(env.explored_cells),
                    "grid": [grid_row[:] for grid_row in env.grid],
                    "height": env.height,
                    "width": env.width,
                    "pose": [env.x, env.y, env.heading],
                    "start": list(env.start),
                    "goal": list(env.goal),
                    "split": "test",
                })
    return snapshots


def _load_snapshot(data: Dict[str, Any]) -> v5.FixedBenchmarkEnv:
    spec = {
        "id": data["map_id"],
        "size": [data["height"], data["width"]],
        "start": [0, 0],
        "goal": [0, 0],
        "metrics": {"shortest": 1, "junctions": 0, "dead_ends": 0},
        "ascii": ["#" * data["width"] for _ in range(data["height"])],
    }
    env = v5.FixedBenchmarkEnv(spec)
    env.grid = [list(row) for row in data["grid"]]
    env.explored_cells = {tuple(cell) for cell in data["explored"]}
    return env


def _training_matrix(snapshots: Sequence[Dict[str, Any]]):
    xs = []
    ys = []
    for snapshot in snapshots:
        env = _load_snapshot(snapshot)
        for y in range(env.height):
            for x in range(env.width):
                if (x, y) in env.explored_cells:
                    continue
                xs.append(cell_features(env, (x, y)))
                ys.append(0.0 if env.is_free(x, y) else 1.0)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def train_predictor(manifest: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    train_specs = manifest["maps"][8:11]
    supplied = 0
    for family_index, family in enumerate(("office", "loop", "maze", "trap")):
        for sample_index in range(3):
            try:
                spec = maps_v5._make_spec(
                    family,
                    "medium",
                    100 + sample_index,
                    20000 + family_index * 1000 + sample_index * 101,
                    21,
                )
                spec["id"] = f"{family}_train{sample_index}"
                train_specs.append(spec)
                supplied += 1
            except RuntimeError:
                continue
    validation_specs = manifest["maps"][11:12]
    snapshots = []
    for i, spec in enumerate(train_specs):
        snapshots.extend(_random_snapshots(spec, 9100 + i, 5))
    validation_snapshots = []
    for i, spec in enumerate(validation_specs):
        validation_snapshots.extend(_random_snapshots(spec, 9300 + i, 4))
    x, y = _training_matrix(snapshots)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    weights = np.zeros(z.shape[1], dtype=np.float64)
    weights[0] = -0.1
    lr = 0.08
    l2 = 1e-4
    wall_weight = 0.5 * len(y) / max(1.0, (y == 1).sum())
    free_weight = 0.5 * len(y) / max(1.0, (y == 0).sum())
    sample_weight = np.where(y == 1, wall_weight, free_weight)
    for _ in range(250):
        logits = np.clip(z @ weights, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = z.T @ ((pred - y) * sample_weight) / len(y) + l2 * weights
        weights -= lr * grad
    provisional = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
    }
    predictor = LogisticMapPredictor(provisional)
    validation_scores = [_score_snapshot(snapshot, predictor) for snapshot in validation_snapshots]
    validation = [
        _metrics_from_scores(scored_snapshot, 0.5) for scored_snapshot in validation_scores
    ]
    thresholds = np.arange(0.25, 0.76, 0.02)
    scored = [(
        float(np.mean([
            _metrics_from_scores(
                scored_snapshot, float(threshold)
            )["balanced_accuracy"]
            for scored_snapshot in validation_scores
        ])),
        float(threshold),
    ) for threshold in thresholds]
    scored.sort(reverse=True)
    model = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": weights.tolist(),
        "threshold": scored[0][1],
        "train_samples": int(len(y)),
        "train_maps": [spec["id"] for spec in train_specs],
        "validation_maps": [spec["id"] for spec in validation_specs],
        "validation_ap_unseen_at_0.5": float(np.mean(
            [r["ap_unseen"] for r in validation]
        )),
        "validation_bal_acc_best": scored[0][0],
        "validation_ap_unseen_at_best_bal_acc": float(np.mean([
            _metrics_from_scores(scored_snapshot, scored[0][1])["ap_unseen"]
            for scored_snapshot in validation_scores
        ])),
    }
    out_path.write_text(json.dumps(model), encoding="utf-8")
    return model


class LogisticMapPredictor:
    def __init__(self, model: Dict[str, Any]) -> None:
        self.mean = np.asarray(model["mean"], dtype=np.float64)
        self.std = np.asarray(model["std"], dtype=np.float64)
        self.weights = np.asarray(model["weights"], dtype=np.float64)
        self.threshold = float(model.get("threshold", 0.5))

    def probabilities(self, env: v5.FixedBenchmarkEnv) -> np.ndarray:
        p = np.full((env.height, env.width), 0.5, dtype=np.float64)
        for y in range(env.height):
            for x in range(env.width):
                if (x, y) in env.explored_cells:
                    p[y, x] = 0.0 if env.is_free(x, y) else 1.0
                    continue
                features = np.asarray(cell_features(env, (x, y)))
                z = (features - self.mean) / self.std
                logit = float(np.clip(z @ self.weights, -30, 30))
                p[y, x] = 1.0 / (1.0 + np.exp(-logit))
        return p


def evaluate_snapshots(
    snapshots: Sequence[Dict[str, Any]],
    predictor: LogisticMapPredictor,
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    threshold = predictor.threshold if threshold is None else float(threshold)
    return [
        _metrics_from_scores(_score_snapshot(snapshot, predictor), threshold)
        for snapshot in snapshots
    ]


def _score_snapshot(
    snapshot: Dict[str, Any], predictor: LogisticMapPredictor
) -> Dict[str, Any]:
    env = _load_snapshot(snapshot)
    prob = predictor.probabilities(env)
    truths = []
    scores = []
    confidences = []
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.explored_cells:
                continue
            truths.append(1 if not env.is_free(x, y) else 0)
            scores.append(float(prob[y, x]))
            confidences.append(max(float(prob[y, x]), 1.0 - float(prob[y, x])))
    return {
        "map_id": snapshot["map_id"],
        "step": snapshot["step"],
        "truth": np.asarray(truths, dtype=int),
        "scores": np.asarray(scores, dtype=float),
        "mean_confidence": float(np.mean(confidences)) if confidences else 1.0,
    }


def _metrics_from_scores(scored_snapshot: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    truth_arr = scored_snapshot["truth"]
    pred_arr = (scored_snapshot["scores"] >= threshold).astype(int)
    acc = float((truth_arr == pred_arr).mean()) if len(truth_arr) else 1.0
    wall_tp = int(((truth_arr == 1) & (pred_arr == 1)).sum())
    wall_fp = int(((truth_arr == 0) & (pred_arr == 1)).sum())
    wall_fn = int(((truth_arr == 1) & (pred_arr == 0)).sum())
    free_tp = int(((truth_arr == 0) & (pred_arr == 0)).sum())
    free_fp = int(((truth_arr == 1) & (pred_arr == 0)).sum())
    free_fn = int(((truth_arr == 0) & (pred_arr == 1)).sum())
    wall_recall = wall_tp / max(1, wall_tp + wall_fn)
    free_recall = free_tp / max(1, free_tp + free_fp)

    def f1(tp, fp, fn):
        return 2 * tp / max(1, 2 * tp + fp + fn)

    return {
        "map_id": scored_snapshot["map_id"],
        "step": scored_snapshot["step"],
        "unknown_cells": int(len(truth_arr)),
        "ap_unseen": acc,
        "wall_f1": f1(wall_tp, wall_fp, wall_fn),
        "free_f1": f1(free_tp, free_fp, free_fn),
        "balanced_accuracy": 0.5 * (wall_recall + free_recall),
        "mean_confidence": scored_snapshot["mean_confidence"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-file", default=str(ROOT / "benchmark_maps_v5.json"))
    parser.add_argument(
        "--steps-file", default=str(ROOT / "results_v5_risk_waypoint" / "steps.json")
    )
    parser.add_argument("--model-out", default=str(ROOT / "map_predictor_v6.json"))
    parser.add_argument("--snapshots-out", default=str(ROOT / "map_prediction_snapshots_v6.json"))
    parser.add_argument("--results-out", default=str(ROOT / "map_prediction_heuristic_v6.json"))
    args = parser.parse_args()
    manifest = json.loads(Path(args.maps_file).read_text(encoding="utf-8"))
    model = train_predictor(manifest, Path(args.model_out))
    predictor = LogisticMapPredictor(model)
    test_snapshots = _snapshots_from_trajectory(
        manifest["maps"][:8], Path(args.steps_file)
    )
    Path(args.snapshots_out).write_text(
        json.dumps(test_snapshots, ensure_ascii=False), encoding="utf-8"
    )
    results = evaluate_snapshots(test_snapshots, predictor)
    Path(args.results_out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "train_samples": model["train_samples"],
        "train_maps": model["train_maps"],
        "test_snapshots": len(results),
        "AP_UNSEEN": np.mean([r["ap_unseen"] for r in results]),
        "WALL_F1": np.mean([r["wall_f1"] for r in results]),
        "FREE_F1": np.mean([r["free_f1"] for r in results]),
        "MEAN_CONFIDENCE": np.mean([r["mean_confidence"] for r in results]),
        "THRESHOLD": model["threshold"],
        "VALIDATION_BAL_ACC": model["validation_bal_acc_best"],
        "VALIDATION_AP_UNSEEN": model["validation_ap_unseen_at_best_bal_acc"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
