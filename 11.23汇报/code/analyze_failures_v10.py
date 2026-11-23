import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("work/v10_results")
OUT = Path("work/v10_analysis/failure_cases.csv")
CASES = (
    ("25x25_80", "results_v10_r3_large80_geom_re1"),
    ("27x27_80", "results_v10_val2_geom_re1_80"),
    ("29x29_80", "results_v10b_final_80"),
    ("29x29_100", "results_v10b_final_100"),
    ("25x25_replan4_100", "results_v10_r2_large_geom_re4_fuse"),
)


def movement_metrics(rows):
    moves = []
    for row in rows:
        cell = (row["x"], row["y"])
        if not moves or cell != moves[-1]:
            moves.append(cell)
    edges = Counter(
        tuple(sorted((a, b)))
        for a, b in zip(moves, moves[1:])
    )
    backtracks = sum(
        1 for a, b, c in zip(moves, moves[1:], moves[2:])
        if a == c
    )
    return {
        "unique_cells": len(set(moves)),
        "moved_cells": max(0, len(moves) - 1),
        "immediate_backtracks": backtracks,
        "max_edge_repeat": max(edges.values()) if edges else 0,
    }


records = []
for suite, folder in CASES:
    directory = ROOT / folder
    episodes = json.loads(
        (directory / "episodes.json").read_text(encoding="utf-8")
    )
    steps_path = directory / "steps.json"
    if not steps_path.exists():
        continue
    steps = json.loads(steps_path.read_text(encoding="utf-8"))
    by_episode = defaultdict(list)
    for row in steps:
        by_episode[row["episode"]].append(row)
    for episode in episodes:
        if episode["success"]:
            continue
        details = movement_metrics(by_episode[episode["map_id"]])
        records.append({
            "suite": suite,
            "condition": folder,
            "map_id": episode["map_id"],
            "reason": episode["reason"],
            "shortest": episode["shortest"],
            "actual": episode["actual"],
            "steps": episode["steps"],
            "remaining_to_goal": episode["ne"],
            "coverage": episode["coverage"],
            "bumps": episode["bumps"],
            **details,
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(OUT, len(records))
