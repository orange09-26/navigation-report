# -*- coding: utf-8 -*-
"""Evaluate zero-shot LLM occupancy prediction on identical partial maps."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests

API = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
KEY_PATH = Path("/home/orange/.dashscope_key")
LOG_PATH = Path("/home/orange/llm_map_prediction_v6_calls.jsonl")
_log_lock = threading.Lock()


def partial_matrix(snapshot: Dict[str, Any]) -> str:
    h, w = snapshot["height"], snapshot["width"]
    explored = {tuple(cell) for cell in snapshot["explored"]}
    robot = tuple(snapshot["pose"][:2])
    rows = []
    for y in range(h):
        row = []
        for x in range(w):
            cell = (x, y)
            if cell == robot:
                row.append("R")
            elif cell not in explored:
                row.append("?")
            elif snapshot["grid"][y][x] != 0:
                row.append("#")
            else:
                row.append(".")
        rows.append("".join(row))
    return "\n".join(rows)


def prediction_prompt(snapshot: Dict[str, Any]) -> str:
    return f"""You are predicting an unseen 2D indoor occupancy map.

Grid size: {snapshot['width']} columns by {snapshot['height']} rows.
Coordinates: x increases to the right; y increases downward.
Robot: {tuple(snapshot['pose'][:2])} heading={snapshot['pose'][2]}.
Start: {tuple(snapshot['start'])}. Goal: {tuple(snapshot['goal'])}.

The partial map uses:
R = robot
. = observed free
# = observed wall/obstacle
? = unseen

{partial_matrix(snapshot)}

Complete the unseen cells. Infer likely free/corridor/open-space cells and wall boundaries
from visible corridors, room boundaries, and structural continuity. Do not move the robot.
Return exactly {snapshot['height']} numbered rows.
Each row must start with the two-digit y index, a colon, and no space, for example:
00:#####################
01:#...................#
Use only . for free and # for wall after the colon.
No markdown, explanation, blank lines, or spaces.
"""


def call_model(model: str, prompt: str, max_tokens: int = 1400) -> Tuple[str, Dict[str, Any]]:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(4):
        try:
            response = requests.post(
                API,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body,
                timeout=180,
            )
            if response.status_code != 200:
                last_error = f"http_{response.status_code}:{response.text[:200]}"
                time.sleep(3 + attempt * 2)
                continue
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            meta = {
                "model": payload.get("model", model),
                "usage": payload.get("usage", {}),
                "request_id": response.headers.get("X-DashScope-Request-Id", ""),
            }
            return content, meta
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            time.sleep(3 + attempt * 2)
    return "", {"error": str(last_error)}


def parse_prediction(text: str, width: int, height: int):
    labelled = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})\s*[:=|]\s*([#.RG*r]+)\s*$", line)
        if match:
            y = int(match.group(1))
            compact = match.group(2)
            if 0 <= y < height and len(compact) == width:
                compact = compact.replace("R", ".").replace("G", ".").replace("*", ".")
                labelled[y] = compact
    if len(labelled) == height:
        return [[1 if ch == "#" else 0 for ch in labelled[y]] for y in range(height)]
    rows = []
    for line in text.splitlines():
        compact = re.sub(r"[^#.RG*]", "", line.upper())
        if len(compact) == width:
            compact = compact.replace("R", ".").replace("G", ".").replace("*", ".")
            rows.append(compact)
        if len(rows) == height:
            break
    if len(rows) != height:
        return None
    return [[1 if ch == "#" else 0 for ch in row] for row in rows]


def f1(truth: np.ndarray, pred: np.ndarray, value: int) -> float:
    tp = int(((truth == value) & (pred == value)).sum())
    fp = int(((truth != value) & (pred == value)).sum())
    fn = int(((truth == value) & (pred != value)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def recall(truth: np.ndarray, pred: np.ndarray, value: int) -> float:
    tp = int(((truth == value) & (pred == value)).sum())
    fn = int(((truth == value) & (pred != value)).sum())
    return tp / max(1, tp + fn)


def evaluate(snapshot: Dict[str, Any], predicted) -> Dict[str, Any]:
    truth = np.asarray(snapshot["grid"], dtype=int)
    explored = {tuple(cell) for cell in snapshot["explored"]}
    unknown_mask = np.asarray([
        [(x, y) not in explored for x in range(snapshot["width"])]
        for y in range(snapshot["height"])
    ])
    if predicted is None:
        return {
            "parse_ok": False,
            "unknown_cells": int(unknown_mask.sum()),
            "ap_unseen": 0.0,
            "wall_f1": 0.0,
            "free_f1": 0.0,
            "known_accuracy": 0.0,
        }
    pred = np.asarray(predicted, dtype=int)
    truth_u = truth[unknown_mask]
    pred_u = pred[unknown_mask]
    known_mask = ~unknown_mask
    return {
        "parse_ok": True,
        "unknown_cells": int(unknown_mask.sum()),
        "ap_unseen": float((truth_u == pred_u).mean()),
        "wall_f1": f1(truth_u, pred_u, 1),
        "free_f1": f1(truth_u, pred_u, 0),
        "balanced_accuracy": 0.5 * (
            recall(truth_u, pred_u, 1) + recall(truth_u, pred_u, 0)
        ),
        "known_accuracy": float((truth[known_mask] == pred[known_mask]).mean()),
    }


def run_model(model: str, snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for i, snapshot in enumerate(snapshots):
        prompt = prediction_prompt(snapshot)
        text, meta = call_model(model, prompt)
        predicted = parse_prediction(text, snapshot["width"], snapshot["height"])
        retries = 0
        if predicted is None:
            retries = 1
            correction = (
                prompt
                + "\nYour previous response was invalid. Return ONLY the exact two-symbol grid."
            )
            text, meta = call_model(model, correction)
            predicted = parse_prediction(text, snapshot["width"], snapshot["height"])
        metrics = evaluate(snapshot, predicted)
        result = {
            "model_requested": model,
            "model_returned": meta.get("model", model),
            "map_id": snapshot["map_id"],
            "step": snapshot["step"],
            "retries": retries,
            "usage": meta.get("usage", {}),
            **metrics,
            "output_text": text[:3000],
        }
        results.append(result)
        with _log_lock:
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "requested": model,
                    "map_id": snapshot["map_id"],
                    "step": snapshot["step"],
                    "usage": meta.get("usage", {}),
                    "error": meta.get("error"),
                }, ensure_ascii=False) + "\n")
        print(
            f"[{model}] {i + 1}/{len(snapshots)} {snapshot['map_id']} "
            f"step={snapshot['step']} parse={metrics['parse_ok']} "
            f"AP={metrics['ap_unseen']:.4f}",
            flush=True,
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshots", default="/home/orange/map_prediction_snapshots_v6.json"
    )
    parser.add_argument(
        "--models", default="qwen-plus,qwen3.8-flash,qwen3.8-max"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--out", default="/home/orange/map_prediction_models_v6.json"
    )
    args = parser.parse_args()
    snapshots = json.loads(Path(args.snapshots).read_text(encoding="utf-8"))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_model, model, snapshots): model for model in models}
        for future in as_completed(futures):
            all_results.extend(future.result())
    all_results.sort(key=lambda r: (r["model_requested"], r["map_id"], r["step"]))
    Path(args.out).write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSUMMARY")
    for model in models:
        rows = [r for r in all_results if r["model_requested"] == model]
        print(json.dumps({
            "model": model,
            "calls": len(rows),
            "parse_rate": float(np.mean([r["parse_ok"] for r in rows])),
            "AP_UNSEEN": float(np.mean([r["ap_unseen"] for r in rows])),
            "WALL_F1": float(np.mean([r["wall_f1"] for r in rows])),
            "FREE_F1": float(np.mean([r["free_f1"] for r in rows])),
            "KNOWN_ACC": float(np.mean([r["known_accuracy"] for r in rows])),
            "TOTAL_TOKENS": sum(
                int(r.get("usage", {}).get("total_tokens", 0) or 0) for r in rows
            ),
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
