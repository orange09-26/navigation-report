# -*- coding: utf-8 -*-
"""Run fixed-map v5 conditions with bounded LLM concurrency."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/home/orange")
SCRIPT = ROOT / "mapgpt_v5.py"
LOG_DIR = ROOT / "matrix_v5_logs"
MAX_WORKERS = int(os.getenv("MATRIX_WORKERS", "4"))
MAX_STEPS = int(os.getenv("MAPGPT_MAX_STEPS", "60"))
NUM_MAPS = int(os.getenv("MAPGPT_NUM_MAPS", "8"))
SLEEP_OK = os.getenv("MAPGPT_SLEEP_OK", "0.15")
STAMP = time.strftime("%Y%m%d_%H%M%S")

CONDITIONS = [
    # Hierarchical conditions start first because they give fast, high-value
    # evidence while the two primitive-action baselines run for much longer.
    "risk_waypoint",
    "fused_waypoint",
    "matrix_waypoint",
    "topo_waypoint",
    "fused_step",
    "mapgpt",
]


def tag_for(condition: str) -> str:
    return f"v5_{condition}"


def done(condition: str) -> bool:
    return (ROOT / f"results_{tag_for(condition)}" / "summary.txt").exists()


def launch(condition: str):
    tag = tag_for(condition)
    out_path = LOG_DIR / f"{tag}.out"
    err_path = LOG_DIR / f"{tag}.err"
    out = out_path.open("w", encoding="utf-8")
    err = err_path.open("w", encoding="utf-8")
    env = dict(os.environ)
    env["MAPGPT_LLM_URL"] = "http://127.0.0.1:5000/get_llm_response"
    env["MAPGPT_SLEEP_OK"] = SLEEP_OK
    env["MAPGPT_MODEL"] = os.getenv("MAPGPT_MODEL", "qwen-plus")
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--condition",
        condition,
        "--maps-file",
        str(ROOT / "benchmark_maps_v5.json"),
        "--start-map",
        "0",
        "--num-ep",
        str(NUM_MAPS),
        "--max-steps",
        str(MAX_STEPS),
        "--tag",
        tag,
        "--nonce",
        f"{tag}_{STAMP}",
    ]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=err)
    return proc, out, err, cmd


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    pending = [c for c in CONDITIONS if not done(c)]
    completed = [c for c in CONDITIONS if done(c)]
    manifest = {
        "stamp": STAMP,
        "conditions": CONDITIONS,
        "already_done": completed,
        "pending": pending,
        "workers": MAX_WORKERS,
        "max_steps": MAX_STEPS,
        "num_maps": NUM_MAPS,
        "maps_file": str(ROOT / "benchmark_maps_v5.json"),
    }
    (LOG_DIR / f"matrix_{STAMP}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    running = {}
    while pending or running:
        while pending and len(running) < MAX_WORKERS:
            condition = pending.pop(0)
            proc, out, err, cmd = launch(condition)
            running[condition] = (proc, out, err)
            print(f"START {condition} pid={proc.pid} cmd={' '.join(cmd)}", flush=True)
        finished = []
        for condition, (proc, out, err) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            out.close()
            err.close()
            print(f"END {condition} rc={rc}", flush=True)
            finished.append(condition)
        for condition in finished:
            running.pop(condition)
        time.sleep(3)
    print(f"MATRIX_DONE stamp={STAMP}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
