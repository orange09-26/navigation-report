from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/orange")
LOG_DIR = ROOT / "matrix_v10_validation_logs"
WORKERS = int(os.getenv("MATRIX_WORKERS", "4"))
MAPS = ROOT / "benchmark_maps_v10.json"


def v6_job(tag: str, steps: int) -> dict:
    return {
        "tag": tag,
        "script": "mapgpt_v6.py",
        "condition": "map_pred_anti",
        "predictor": "map_predictor_v6.json",
        "steps": steps,
        "obs": 5,
        "obs_policy": "fixed",
        "pred_weight": -2.0,
        "replan": 0,
        "fusion": False,
    }


def v10_job(
    tag: str,
    steps: int,
    obs: int,
    obs_policy: str,
    replan: int,
    fusion: bool = False,
) -> dict:
    return {
        "tag": tag,
        "script": "mapgpt_v7.py",
        "condition": "macro_v10",
        "predictor": (
            "map_predictor_v6.json"
            if fusion else "map_predictor_v8_multires.json"
        ),
        "steps": steps,
        "obs": obs,
        "obs_policy": obs_policy,
        "pred_weight": -2.0,
        "replan": replan,
        "fusion": fusion,
    }


JOBS = []
for steps in (80, 100, 120):
    JOBS.extend([
        v6_job(f"v10_val_v6_{steps}", steps),
        v10_job(f"v10_val_base_{steps}", steps, 9, "fixed", 0),
        v10_job(f"v10_val_geom_{steps}", steps, 7, "geometry", 0),
        v10_job(f"v10_val_re4_{steps}", steps, 9, "fixed", 4),
        v10_job(f"v10_val_geom_re4_{steps}", steps, 7, "geometry", 4),
        v10_job(
            f"v10_val_geom_re4_fuse_{steps}",
            steps, 7, "geometry", 4, fusion=True,
        ),
    ])

for steps in (80, 100, 120):
    JOBS.append(
        v10_job(
            f"v10_val2_geom_re1_{steps}",
            steps, 7, "geometry", 1,
        )
    )

for steps in (100, 120):
    JOBS.extend([
        v10_job(
            f"v10_val2_geom_re1_fuse_{steps}",
            steps, 7, "geometry", 1, fusion=True,
        ),
        v10_job(
            f"v10_val2_commit_re4_fuse_{steps}",
            steps, 7, "commit_wide", 4, fusion=True,
        ),
        v10_job(
            f"v10_val2_commit_re1_{steps}",
            steps, 7, "commit_wide", 1,
        ),
    ])


def command(item: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / item["script"]),
        "--condition", item["condition"],
        "--maps-file", str(MAPS),
        "--predictor-model", str(ROOT / item["predictor"]),
        "--num-ep", "8",
        "--max-steps", str(item["steps"]),
        "--tag", item["tag"],
    ]
    if item["script"] == "mapgpt_v7.py":
        cmd.extend([
            "--obs-size", str(item["obs"]),
            "--obs-policy", item["obs_policy"],
            "--pred-weight", str(item["pred_weight"]),
            "--route-replan-steps", str(item["replan"]),
        ])
        if item["fusion"]:
            cmd.extend([
                "--predictor-secondary",
                str(ROOT / "map_predictor_v8_multires.json"),
                "--predictor-fusion-weight", "0.5",
            ])
    return cmd


def done(item: dict) -> bool:
    return (ROOT / f"results_{item['tag']}" / "summary.txt").exists()


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    pending = [item for item in JOBS if not done(item)]
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
    failed = [item["tag"] for item in JOBS if not done(item)]
    print("VALIDATION_DONE", failed, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
