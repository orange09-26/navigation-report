from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/orange")
MAPS = ROOT / "benchmark_maps_v10b.json"
LOG_DIR = ROOT / "v10b_logs"
WORKERS = int(os.getenv("MATRIX_WORKERS", "4"))


def item(script: str, condition: str, tag: str, steps: int,
         obs: int = 5, policy: str = "fixed", replan: int = 0) -> dict:
    return {
        "script": script,
        "condition": condition,
        "tag": tag,
        "steps": steps,
        "obs": obs,
        "policy": policy,
        "replan": replan,
    }


JOBS = []
for steps in (80, 100, 120):
    JOBS.extend([
        item("mapgpt_v6.py", "map_pred_anti", f"v10b_v6_{steps}", steps),
        item("mapgpt_v7.py", "macro_v10", f"v10b_base_{steps}", steps, 9),
        item(
            "mapgpt_v7.py", "macro_v10",
            f"v10b_final_{steps}", steps, 7, "geometry", 1,
        ),
    ])
for steps in (100, 120):
    JOBS.append(item(
        "mapgpt_v7.py", "macro_v10",
        f"v10b_re4_{steps}", steps, 7, "geometry", 4,
    ))


def command(job: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / job["script"]),
        "--condition", job["condition"],
        "--maps-file", str(MAPS),
        "--predictor-model", str(ROOT / (
            "map_predictor_v6.json"
            if job["script"] == "mapgpt_v6.py"
            else "map_predictor_v8_multires.json"
        )),
        "--num-ep", "8",
        "--max-steps", str(job["steps"]),
        "--tag", job["tag"],
    ]
    if job["script"] == "mapgpt_v7.py":
        cmd.extend([
            "--obs-size", str(job["obs"]),
            "--obs-policy", job["policy"],
            "--route-replan-steps", str(job["replan"]),
            "--pred-weight", "-2",
        ])
    return cmd


def done(job: dict) -> bool:
    return (ROOT / f"results_{job['tag']}" / "summary.txt").exists()


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    pending = [job for job in JOBS if not done(job)]
    running = {}
    while pending or running:
        while pending and len(running) < WORKERS:
            job = pending.pop(0)
            env = dict(os.environ)
            env["MAPGPT_ROOT"] = str(ROOT)
            env["MAPGPT_SLEEP_OK"] = "0"
            out = (LOG_DIR / f"{job['tag']}.out").open("w", encoding="utf-8")
            err = (LOG_DIR / f"{job['tag']}.err").open("w", encoding="utf-8")
            proc = subprocess.Popen(
                command(job), cwd=ROOT, env=env, stdout=out, stderr=err
            )
            running[job["tag"]] = (proc, out, err)
            print("START", job["tag"], proc.pid, flush=True)
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
    failed = [job["tag"] for job in JOBS if not done(job)]
    print("DONE", failed, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
