# -*- coding: utf-8 -*-
"""Build paired per-map and aggregate comparisons for v5 results."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

CONDITIONS = [
    "mapgpt",
    "fused_step",
    "topo_waypoint",
    "matrix_waypoint",
    "fused_waypoint",
    "risk_waypoint",
]


def bootstrap_ci(values: Sequence[float], reps: int = 5000, seed: int = 17) -> List[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    n = len(values)
    boots = [mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(reps)]
    boots.sort()
    return [boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]]


def load_records(root: Path, condition: str) -> List[Dict[str, Any]]:
    path = root / f"results_v5_{condition}" / "episodes.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    succ = [r for r in records if r["success"]]

    def avg(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "N": n,
        "SR": len(succ) / max(1, n),
        "SPL": avg(
            r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0.0
            for r in records
        ),
        "SPL_succ_only": avg(
            r["shortest"] / max(r["actual"], r["shortest"], 1) for r in succ
        ),
        "NE": avg(r["ne"] for r in records),
        "AP": avg(r["ap"] for r in records),
        "AP_LLM_LEGACY": avg(r["ap_legacy"] for r in records),
        "COV": avg(r["coverage"] for r in records),
        "LLM_CALLS": sum(r["llm_calls"] for r in records),
        "CALLS_PER_MAP": avg(r["llm_calls"] for r in records),
        "STEPS": sum(r["steps"] for r in records),
        "RECOVERIES": sum(r["recoveries"] for r in records),
        "VETOS": sum(r.get("veto_count", 0) for r in records),
        "WALL_SEC": sum(r["wall_sec"] for r in records),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/orange")
    parser.add_argument("--out", default="/home/orange/results_v5_analysis")
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = {c: load_records(root, c) for c in CONDITIONS}
    records = {c: r for c, r in records.items() if r}
    if not records:
        raise SystemExit("no result files found")
    summaries = {c: summarize(r) for c, r in records.items()}

    aggregate_path = out / "aggregate.csv"
    keys = list(next(iter(summaries.values())).keys())
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["condition"] + keys)
        wr.writeheader()
        for condition in CONDITIONS:
            if condition in summaries:
                wr.writerow({"condition": condition, **summaries[condition]})

    baseline = {r["map_id"]: r for r in records.get("mapgpt", [])}
    per_map_rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition not in records:
            continue
        for rec in records[condition]:
            base = baseline.get(rec["map_id"])
            row = {
                "condition": condition,
                "map_id": rec["map_id"],
                "success": rec["success"],
                "shortest": rec["shortest"],
                "actual": rec["actual"],
                "spl": (
                    rec["shortest"] / max(rec["actual"], rec["shortest"], 1)
                    if rec["success"] else 0.0
                ),
                "ap": rec["ap"],
                "ap_legacy": rec["ap_legacy"],
                "coverage": rec["coverage"],
                "llm_calls": rec["llm_calls"],
                "recoveries": rec["recoveries"],
            }
            if base is not None:
                base_spl = (
                    base["shortest"] / max(base["actual"], base["shortest"], 1)
                    if base["success"] else 0.0
                )
                row["delta_spl_vs_baseline"] = row["spl"] - base_spl
                row["delta_ap_vs_baseline"] = rec["ap"] - base["ap"]
                row["delta_calls_vs_baseline"] = rec["llm_calls"] - base["llm_calls"]
            per_map_rows.append(row)

    per_map_path = out / "per_map.csv"
    with per_map_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_map_rows[0].keys()))
        wr.writeheader()
        wr.writerows(per_map_rows)

    paired: Dict[str, Any] = {}
    for condition in conditions_with_pairs(records):
        d_spl = []
        d_ap = []
        d_ap_legacy = []
        d_sr = []
        d_calls = []
        wins_spl = ties_spl = losses_spl = 0
        wins_ap = ties_ap = losses_ap = 0
        wins_sr = ties_sr = losses_sr = 0
        for map_id, base in baseline.items():
            rec = next((r for r in records[condition] if r["map_id"] == map_id), None)
            if rec is None:
                continue
            base_spl = base["shortest"] / max(base["actual"], base["shortest"], 1) if base["success"] else 0.0
            cur_spl = rec["shortest"] / max(rec["actual"], rec["shortest"], 1) if rec["success"] else 0.0
            a = cur_spl - base_spl
            b = rec["ap"] - base["ap"]
            b_legacy = rec["ap_legacy"] - base["ap_legacy"]
            sr_delta = int(rec["success"]) - int(base["success"])
            c = rec["llm_calls"] - base["llm_calls"]
            d_spl.append(a)
            d_ap.append(b)
            d_ap_legacy.append(b_legacy)
            d_sr.append(sr_delta)
            d_calls.append(c)
            wins_spl += a > 1e-9
            losses_spl += a < -1e-9
            ties_spl += abs(a) <= 1e-9
            wins_ap += b > 1e-9
            losses_ap += b < -1e-9
            ties_ap += abs(b) <= 1e-9
            wins_sr += sr_delta > 0
            losses_sr += sr_delta < 0
            ties_sr += sr_delta == 0
        paired[condition] = {
            "n_pairs": len(d_spl),
            "delta_spl": mean(d_spl),
            "delta_spl_ci95": bootstrap_ci(d_spl),
            "delta_ap": mean(d_ap),
            "delta_ap_ci95": bootstrap_ci(d_ap),
            "delta_ap_legacy": mean(d_ap_legacy),
            "delta_ap_legacy_ci95": bootstrap_ci(d_ap_legacy),
            "delta_sr": mean(d_sr),
            "delta_sr_ci95": bootstrap_ci(d_sr),
            "delta_calls": mean(d_calls),
            "sr_wins_ties_losses": [wins_sr, ties_sr, losses_sr],
            "spl_wins_ties_losses": [wins_spl, ties_spl, losses_spl],
            "ap_wins_ties_losses": [wins_ap, ties_ap, losses_ap],
        }

    analysis = {"aggregate": summaries, "paired_vs_mapgpt": paired}
    (out / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))
    return 0


def conditions_with_pairs(records: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    if "mapgpt" not in records:
        return []
    return [c for c in CONDITIONS if c != "mapgpt" and c in records]


if __name__ == "__main__":
    raise SystemExit(main())
