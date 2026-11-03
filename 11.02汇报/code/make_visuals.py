# -*- coding: utf-8 -*-
"""Create map-suite and performance figures from v5 result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ORDER = [
    "mapgpt",
    "fused_step",
    "topo_waypoint",
    "matrix_waypoint",
    "fused_waypoint",
    "risk_waypoint",
]
LABELS = {
    "mapgpt": "MapGPT baseline",
    "fused_step": "Topology+matrix\nper-step",
    "topo_waypoint": "Topology only\nwaypoint",
    "matrix_waypoint": "Matrix only\nwaypoint",
    "fused_waypoint": "Fused\nwaypoint",
    "risk_waypoint": "Risk-aware\nfusion",
}


def draw_maps(manifest_path: Path, output: Path, num_maps: int) -> None:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = data["maps"][:num_maps]
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(13, 7.0), constrained_layout=True)
    for ax, spec in zip(axes.flat, specs):
        arr = np.zeros((len(spec["ascii"]), len(spec["ascii"][0])), dtype=int)
        for y, row in enumerate(spec["ascii"]):
            for x, ch in enumerate(row):
                arr[y, x] = 1 if ch == "#" else 0
        ax.imshow(arr, cmap=ListedColormap(["#f4f1ea", "#31343a"]), origin="upper")
        sx, sy = spec["start"]
        gx, gy = spec["goal"]
        ax.scatter([sx], [sy], s=55, c="#1b9e77", marker="o", edgecolor="white", linewidth=1.2)
        ax.scatter([gx], [gy], s=68, c="#d95f02", marker="*", edgecolor="white", linewidth=0.8)
        m = spec["metrics"]
        ax.set_title(
            f"{spec['id']} ({spec['difficulty']})\n"
            f"L={m['shortest']} J={m['junctions']} D={m['dead_ends']} C={m['cycles']}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Patch(facecolor="#31343a", label="wall / obstacle"),
        Patch(facecolor="#f4f1ea", edgecolor="#888", label="free"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1b9e77",
                   markeredgecolor="white", markersize=8, label="start"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#d95f02",
                   markeredgecolor="white", markersize=10, label="goal"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.015),
        ncol=4, frameon=False
    )
    fig.suptitle(
        f"MapGPT v5 fixed benchmark: {len(specs)} evaluated scenes, identical for every policy",
        fontsize=14,
        y=1.06,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_performance(analysis_path: Path, output_dir: Path) -> None:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    conditions = [c for c in ORDER if c in agg]
    colors = ["#777777", "#9c755f", "#8c6bb1", "#00a6a6", "#4c78a8", "#d1495b"]
    color_by = dict(zip(ORDER, colors))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    metrics = [("SR", "Success rate"), ("SPL", "SPL"), ("AP", "Fused-map AP")]
    x = np.arange(len(conditions))
    for ax, (key, title) in zip(axes, metrics):
        vals = [agg[c][key] for c in conditions]
        bars = ax.bar(x, vals, color=[color_by[c] for c in conditions], width=0.72)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[c] for c in conditions], rotation=24, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    n_maps = agg[conditions[0]]["N"] if conditions else 0
    fig.suptitle(f"Navigation quality on the same {int(n_maps)} maps", fontsize=14)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "performance_overview_v5.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for c in conditions:
        xv = agg[c]["SPL"]
        yv = agg[c]["AP"]
        ax.scatter(xv, yv, s=150 + agg[c]["LLM_CALLS"] / 30,
                   color=color_by[c], edgecolor="white", linewidth=1.4, zorder=3,
                   label=LABELS[c].replace("\n", " "))
    ax.set_xlabel("SPL")
    ax.set_ylabel("AP_FUSED")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.22)
    ax.set_title("SPL–AP trade-off (marker size ∝ total LLM calls)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(output_dir / "spl_ap_tradeoff_v5.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    baseline = agg.get("mapgpt")
    if baseline:
        deltas = []
        for c in conditions:
            if c == "mapgpt":
                continue
            deltas.append((c, agg[c]["SPL"] - baseline["SPL"], agg[c]["AP"] - baseline["AP"]))
        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        x = np.arange(len(deltas))
        ax.bar(x - 0.18, [d[1] for d in deltas], 0.36, label="ΔSPL", color="#4c78a8")
        ax.bar(x + 0.18, [d[2] for d in deltas], 0.36, label="ΔAP", color="#d1495b")
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[d[0]] for d in deltas], rotation=24, ha="right", fontsize=8)
        ax.set_ylabel("Absolute change vs. MapGPT baseline")
        ax.set_title("Paired improvements on the fixed map suite")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output_dir / "paired_improvements_v5.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def draw_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333",
                             linewidth=1.2, joinstyle="round")
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="top",
                fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h - 0.65, body, ha="center", va="top",
                fontsize=8.5, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.7, color="#555"))

    box(0.3, 2.2, 2.0, 2.1, "Observation",
        "5x5 local view\npose + walls\nfrontier evidence", "#e9f3f0")
    box(2.8, 4.1, 3.0, 1.65, "Task memory",
        "language goal\nprogress + recent actions", "#fff2d8")
    box(2.8, 2.1, 3.0, 1.65, "Spatial memory",
        "known occupancy matrix\n+ synchronized skeleton topology", "#dceaf7")
    box(2.8, 0.25, 3.0, 1.45, "Temporal memory",
        "short action history\nstale-route recovery", "#f1e5f5")
    box(6.35, 2.0, 2.25, 2.2, "Candidate generator",
        "known-graph A*\ninformation gain\nrisk + dead-end veto", "#e8e8e8")
    box(9.15, 2.65, 2.0, 1.55, "Zero-shot LLM",
        "select C1/C2/...\nnot every motor action", "#fbdedd")
    box(11.7, 2.2, 2.0, 2.2, "Executor",
        "A* action chunk\ncollision checks\nmap update", "#e4f0df")

    arrow(2.3, 3.2, 2.8, 4.7)
    arrow(2.3, 3.1, 2.8, 2.9)
    arrow(2.3, 2.8, 2.8, 1.0)
    arrow(5.8, 4.55, 6.35, 3.65)
    arrow(5.8, 2.85, 6.35, 2.95)
    arrow(5.8, 1.25, 6.35, 2.25)
    arrow(8.6, 3.1, 9.15, 3.35)
    arrow(11.15, 3.35, 11.7, 3.3)
    arrow(12.7, 2.2, 12.7, 0.05)
    arrow(12.7, 0.05, 1.3, 0.05)
    arrow(1.3, 0.05, 1.3, 2.2)
    ax.set_title("MapGPT v5: topology–matrix fusion with hierarchical execution",
                 fontsize=15, pad=12)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "method_architecture_v5.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--analysis", default="")
    parser.add_argument("--num-maps", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    draw_maps(Path(args.manifest), out / "map_suite_v5.png", args.num_maps)
    draw_architecture(out)
    if args.analysis:
        draw_performance(Path(args.analysis), out)
    print("figures written to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
