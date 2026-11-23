from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/orange")
SCRIPT = ROOT / "mapgpt_v7.py"
LOG_DIR = ROOT / "matrix_v10_logs"
WORKERS = int(os.getenv("MATRIX_WORKERS", "4"))


def job(
    tag: str,
    maps: str,
    max_steps: int = 100,
    obs: int = 5,
    condition: str = "macro_v10",
    predictor: str = "map_predictor_v8_multires.json",
    secondary: str | None = None,
    fusion: float = 0.0,
    obs_policy: str = "fixed",
    pred_weight: float = -2.0,
    replan: int = 0,
    waypoint: str = "entry",
    budget_gate: float = 0.0,
    budget_mult: float = 1.5,
    budget_goal_weight: float = 2.0,
) -> dict:
    return {
        "tag": tag,
        "maps": maps,
        "max_steps": max_steps,
        "obs": obs,
        "condition": condition,
        "predictor": predictor,
        "secondary": secondary,
        "fusion": fusion,
        "obs_policy": obs_policy,
        "pred_weight": pred_weight,
        "replan": replan,
        "waypoint": waypoint,
        "budget_gate": budget_gate,
        "budget_mult": budget_mult,
        "budget_goal_weight": budget_goal_weight,
    }


SMALL = "benchmark_maps_v5.json"
LARGE = "benchmark_maps_v8.json"

JOBS = [
    job("v10_r1_small_base", SMALL),
    job("v10_r1_small_pred05", SMALL, pred_weight=0.5),
    job("v10_r1_small_pred10", SMALL, pred_weight=1.0),
    job("v10_r1_small_replan4", SMALL, replan=4),
    job("v10_r1_small_replan8", SMALL, replan=8),
    job("v10_r1_small_replan12", SMALL, replan=12),
    job("v10_r1_small_way_info", SMALL, waypoint="info"),
    job("v10_r1_small_way_deep", SMALL, waypoint="deep"),
    job("v10_r1_small_way_quick", SMALL, waypoint="quick"),
    job("v10_r1_large_base", LARGE, obs=7),
    job("v10_r1_large_pred05", LARGE, obs=7, pred_weight=0.5),
    job("v10_r1_large_pred00", LARGE, obs=7, pred_weight=0.0),
    job("v10_r1_large_replan4", LARGE, obs=7, replan=4),
    job("v10_r1_large_replan8", LARGE, obs=7, replan=8),
    job("v10_r1_large_replan12", LARGE, obs=7, replan=12),
    job("v10_r1_large_way_info", LARGE, obs=7, waypoint="info"),
    job("v10_r1_large_way_deep", LARGE, obs=7, waypoint="deep"),
    job("v10_r1_large_way_quick", LARGE, obs=7, waypoint="quick"),
    job(
        "v10_r1_large_fuse25", LARGE, obs=7,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.25,
    ),
    job(
        "v10_r1_large_fuse50", LARGE, obs=7,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.5,
    ),
    job(
        "v10_r1_large_fuse75", LARGE, obs=7,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.75,
    ),
    job("v10_r1_large_staged", LARGE, obs=7, obs_policy="staged"),
    job("v10_r1_large_stall", LARGE, obs=7, obs_policy="stall"),
    job("v10_r1_large_geometry", LARGE, obs=7, obs_policy="geometry"),
    job("v10_r1_large_obs9", LARGE, obs=9),
    job("v10_r2_small_geometry", SMALL, obs=5, obs_policy="geometry"),
    job("v10_r2_large_replan2", LARGE, obs=7, replan=2),
    job("v10_r2_large_replan3", LARGE, obs=7, replan=3),
    job("v10_r2_large_replan6", LARGE, obs=7, replan=6),
    job("v10_r2_large_geom_re2", LARGE, obs=7, obs_policy="geometry", replan=2),
    job("v10_r2_large_geom_re4", LARGE, obs=7, obs_policy="geometry", replan=4),
    job("v10_r2_large_obs9_re4", LARGE, obs=9, replan=4),
    job(
        "v10_r2_large_geom_fuse", LARGE, obs=7, obs_policy="geometry",
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.5,
    ),
    job(
        "v10_r2_large_geom_re4_fuse", LARGE, obs=7,
        obs_policy="geometry", replan=4,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.5,
    ),
    job("v10_r2_large80_obs9", LARGE, max_steps=80, obs=9),
    job(
        "v10_r2_large80_geom", LARGE, max_steps=80, obs=7,
        obs_policy="geometry",
    ),
    job(
        "v10_r2_large80_geom_re4", LARGE, max_steps=80, obs=7,
        obs_policy="geometry", replan=4,
    ),
    job("v10_r2_large120_obs9", LARGE, max_steps=120, obs=9),
    job(
        "v10_r2_large120_geom", LARGE, max_steps=120, obs=7,
        obs_policy="geometry",
    ),
    job(
        "v10_r2_large120_geom_re4", LARGE, max_steps=120, obs=7,
        obs_policy="geometry", replan=4,
    ),
    job("v10_r3_small_commit", SMALL, obs=5, obs_policy="commit_wide"),
    job(
        "v10_r3_large_commit_re4", LARGE, obs=7,
        obs_policy="commit_wide", replan=4,
    ),
    job(
        "v10_r3_large_commit_re4_fuse", LARGE, obs=7,
        obs_policy="commit_wide", replan=4,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.5,
    ),
    job("v10_r3_large_geom_re1", LARGE, obs=7, obs_policy="geometry", replan=1),
    job("v10_r3_large_geom_re2", LARGE, obs=7, obs_policy="geometry", replan=2),
    job("v10_r3_large_geom_re3", LARGE, obs=7, obs_policy="geometry", replan=3),
    job(
        "v10_r3_large80_geom_re1", LARGE, max_steps=80, obs=7,
        obs_policy="geometry", replan=1,
    ),
    job(
        "v10_r3_large80_geom_re2", LARGE, max_steps=80, obs=7,
        obs_policy="geometry", replan=2,
    ),
    job(
        "v10_r3_large80_commit_re2", LARGE, max_steps=80, obs=7,
        obs_policy="commit_wide", replan=2,
    ),
    job(
        "v10_r3_large80_commit_re2_fuse", LARGE, max_steps=80, obs=7,
        obs_policy="commit_wide", replan=2,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.5,
    ),
    job(
        "v10_r3_large120_commit_re4_fuse", LARGE, max_steps=120, obs=7,
        obs_policy="commit_wide", replan=4,
        predictor="map_predictor_v6.json",
        secondary="map_predictor_v8_multires.json", fusion=0.5,
    ),
    job(
        "v10_r4_large_gate1", LARGE, obs=7,
        obs_policy="geometry", replan=1,
        budget_gate=1.0, budget_mult=1.5, budget_goal_weight=2.0,
    ),
    job(
        "v10_r4_large_gate05", LARGE, obs=7,
        obs_policy="geometry", replan=1,
        budget_gate=0.5, budget_mult=1.5, budget_goal_weight=2.0,
    ),
    job(
        "v10_r4_large_gate1_m20", LARGE, obs=7,
        obs_policy="geometry", replan=1,
        budget_gate=1.0, budget_mult=2.0, budget_goal_weight=2.0,
    ),
    job(
        "v10_r4_large80_gate1", LARGE, max_steps=80, obs=7,
        obs_policy="geometry", replan=1,
        budget_gate=1.0, budget_mult=1.5, budget_goal_weight=2.0,
    ),
    job(
        "v10_r4_large80_gate05", LARGE, max_steps=80, obs=7,
        obs_policy="geometry", replan=1,
        budget_gate=0.5, budget_mult=1.5, budget_goal_weight=2.0,
    ),
]


def done(item: dict) -> bool:
    return (ROOT / f"results_{item['tag']}" / "summary.txt").exists()


def command(item: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition", item["condition"],
        "--maps-file", str(ROOT / item["maps"]),
        "--predictor-model", str(ROOT / item["predictor"]),
        "--num-ep", "8",
        "--max-steps", str(item["max_steps"]),
        "--obs-size", str(item["obs"]),
        "--obs-policy", item["obs_policy"],
        "--pred-weight", str(item["pred_weight"]),
        "--route-replan-steps", str(item["replan"]),
        "--waypoint-mode", item["waypoint"],
        "--budget-gate", str(item["budget_gate"]),
        "--budget-mult", str(item["budget_mult"]),
        "--budget-goal-weight", str(item["budget_goal_weight"]),
        "--tag", item["tag"],
    ]
    if item["secondary"]:
        cmd.extend([
            "--predictor-secondary", str(ROOT / item["secondary"]),
            "--predictor-fusion-weight", str(item["fusion"]),
        ])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="*")
    args = parser.parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    wanted = set(args.tags or [])
    pending = [
        item for item in JOBS
        if (not wanted or item["tag"] in wanted) and not done(item)
    ]
    running = {}
    while pending or running:
        while pending and len(running) < WORKERS:
            item = pending.pop(0)
            env = dict(os.environ)
            env["MAPGPT_ROOT"] = str(ROOT)
            env["MAPGPT_SLEEP_OK"] = "0"
            out = (LOG_DIR / f"{item['tag']}.out").open("w", encoding="utf-8")
            err = (LOG_DIR / f"{item['tag']}.err").open("w", encoding="utf-8")
            proc = subprocess.Popen(
                command(item), cwd=ROOT, env=env, stdout=out, stderr=err
            )
            running[item["tag"]] = (proc, out, err)
            print("START", item["tag"], proc.pid, flush=True)
        finished = []
        for tag, (proc, out, err) in running.items():
            code = proc.poll()
            if code is None:
                continue
            out.close()
            err.close()
            print("END", tag, code, flush=True)
            finished.append(tag)
        for tag in finished:
            running.pop(tag)
        time.sleep(1)
    print("MATRIX_DONE", flush=True)
    failed = []
    for item in JOBS:
        if (not wanted or item["tag"] in wanted) and not done(item):
            failed.append(item["tag"])
    print(json.dumps({"failed": failed}, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
