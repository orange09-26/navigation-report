import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(os.getenv("V10_REMOTE_ROOT", "/home/orange"))
OUT = Path(os.getenv("V10_FIGURES_DIR", "/home/orange/figures_v10"))
OUT.mkdir(parents=True, exist_ok=True)


def metrics(path):
    rows = json.load(open(ROOT / path / "episodes.json", encoding="utf-8"))
    n = len(rows)
    sr = sum(r["success"] for r in rows) / n
    spl = sum(
        r["shortest"] / max(r["actual"], r["shortest"], 1)
        if r["success"] else 0
        for r in rows
    ) / n
    ap = sum(r.get("AP_UNSEEN", 0) for r in rows) / n
    return {"SR": sr, "SPL": spl, "AP": ap}


manifest = json.load(
    open(ROOT / "benchmark_maps_v10.json", encoding="utf-8")
)
fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
for ax, spec in zip(axes.ravel(), manifest["maps"]):
    grid = np.array([
        [0 if ch in ".SG" else 1 for ch in row]
        for row in spec["ascii"]
    ])
    ax.imshow(grid, cmap="Greys", interpolation="nearest")
    ax.scatter(*spec["start"], c="#2ca02c", s=35, label="start", zorder=3)
    ax.scatter(*spec["goal"], c="#d62728", marker="*", s=90, label="goal", zorder=3)
    m = spec["metrics"]
    ax.set_title(
        f"{spec['id']}\nshortest={m['shortest']} dead={m['dead_ends']} cycles={m['cycles']}",
        fontsize=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])
axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
fig.suptitle("Held-out 27x27 validation maps (never used for v9 tuning)")
fig.savefig(OUT / "fig1_maps_v10.png", dpi=180, bbox_inches="tight")
plt.close(fig)


suites = [
    (
        "25x25",
        "results_v8_v6final100",
        "results_v10_r1_large_base",
        "results_v10_r3_large_geom_re1",
    ),
    (
        "27x27 held-out",
        "results_v10_val_v6_100",
        "results_v10_val_base_100",
        "results_v10_val2_geom_re1_100",
    ),
    (
        "29x29 held-out",
        "results_v10b_v6_100",
        "results_v10b_base_100",
        "results_v10b_final_100",
    ),
]
labels = ["v6 final", "v10 base", "v10 final"]
colors = ["#8c8c8c", "#4c78a8", "#e45756"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
x = np.arange(len(suites))
width = 0.24
for idx, label in enumerate(labels):
    vals = [metrics(row[1 + idx]) for row in suites]
    axes[0].bar(x + (idx - 1) * width, [v["SR"] for v in vals], width,
                color=colors[idx], label=label)
    axes[1].bar(x + (idx - 1) * width, [v["SPL"] for v in vals], width,
                color=colors[idx], label=label)
for idx, label in enumerate(("v10 base", "v10 final")):
    vals = [metrics(row[2 + idx]) for row in suites]
    axes[2].bar(x + (idx - 0.5) * 0.3, [v["AP"] for v in vals], 0.3,
                color=colors[idx + 1], label=label)
for ax, title in zip(
    axes,
    ("Success rate", "SPL", "Unseen-cell map accuracy (AP_UNSEEN)"),
):
    ax.set_xticks(x)
    ax.set_xticklabels([row[0] for row in suites])
    ax.set_ylim(0, 1.04)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
axes[0].legend(frameon=False, fontsize=8)
fig.suptitle("100-step comparison on identical maps")
fig.savefig(OUT / "fig2_main_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)


ablation = [
    ("info waypoint", "results_v10_r1_large_way_info"),
    ("quick waypoint", "results_v10_r1_large_way_quick"),
    ("macro base", "results_v10_r1_large_base"),
    ("9x9 fixed view", "results_v10_r1_large_obs9"),
    ("geometry view", "results_v10_r1_large_geometry"),
    ("replan / 4 steps", "results_v10_r1_large_replan4"),
    ("geometry + replan / 1", "results_v10_r3_large_geom_re1"),
    ("+ predictor fusion", "results_v10_r3_large_commit_re4_fuse"),
]
vals = [(name, metrics(path)) for name, path in ablation]
fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
y = np.arange(len(vals))
ax.barh(y, [v["SPL"] for _, v in vals], color="#4c78a8")
ax.set_yticks(y)
ax.set_yticklabels([name for name, _ in vals])
ax.set_xlim(0.74, 0.97)
ax.set_xlabel("SPL on 25x25 maps, 100 steps")
ax.set_title("V10 mechanism ablation")
ax.grid(axis="x", alpha=0.2)
for i, (_, value) in enumerate(vals):
    ax.text(value["SPL"] + 0.002, i, f'{value["SPL"]:.3f}',
            va="center", fontsize=8)
fig.savefig(OUT / "fig3_ablation.png", dpi=180, bbox_inches="tight")
plt.close(fig)


budget_suites = [
    (
        "25x25",
        {
            80: ("results_v10_r2_large80_obs9", "results_v10_r3_large80_geom_re1"),
            100: ("results_v10_r1_large_base", "results_v10_r3_large_geom_re1"),
            120: ("results_v10_r2_large120_obs9", "results_v10_final25_120"),
        },
    ),
    (
        "27x27 held-out",
        {
            80: ("results_v10_val_base_80", "results_v10_val2_geom_re1_80"),
            100: ("results_v10_val_base_100", "results_v10_val2_geom_re1_100"),
            120: ("results_v10_val_base_120", "results_v10_val2_geom_re1_120"),
        },
    ),
    (
        "29x29 held-out",
        {
            80: ("results_v10b_base_80", "results_v10b_final_80"),
            100: ("results_v10b_base_100", "results_v10b_final_100"),
            120: ("results_v10b_base_120", "results_v10b_final_120"),
        },
    ),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), constrained_layout=True)
for col, (name, paths) in enumerate(budget_suites):
    budgets = sorted(paths)
    b = [metrics(paths[step][0]) for step in budgets]
    f = [metrics(paths[step][1]) for step in budgets]
    for row, key, title in (
        (0, "SPL", "SPL"),
        (1, "SR", "Success rate"),
    ):
        ax = axes[row, col]
        ax.plot(budgets, [v[key] for v in b], marker="o", label="v10 base")
        ax.plot(budgets, [v[key] for v in f], marker="s", label="v10 final")
        ax.set_xticks(budgets)
        ax.set_ylim(0.55, 1.02)
        ax.set_title(f"{name}: {title}")
        ax.grid(alpha=0.2)
        if col == 0:
            ax.set_ylabel(title)
axes[0, 0].legend(frameon=False)
fig.suptitle("Budget sensitivity on identical maps")
fig.savefig(OUT / "fig4_budget.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print("figures written", OUT)
