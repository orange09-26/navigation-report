import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("work")
OUT = Path("outputs/figures_v6")
OUT.mkdir(parents=True, exist_ok=True)
analysis = json.load(open(ROOT / "v6_analysis" / "analysis.json", encoding="utf-8"))
agg = analysis["aggregate"]


def bar_metrics(labels, metric, title, filename, colors=None):
    vals = [agg[label][metric] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(labels))
    colors = colors or ["#777777", "#9c755f", "#4c78a8", "#d1495b"]
    bars = ax.bar(x, vals, color=colors[:len(labels)], width=0.65)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


main_labels = ["base100", "anti_v3", "map_pred", "final"]
display = ["No-map baseline", "Anti-stuck", "Map prediction", "Prediction+anti-stuck"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
for ax, metric in zip(axes, ("SR", "SPL")):
    vals = [agg[label][metric] for label in main_labels]
    x = np.arange(4)
    bars = ax.bar(x, vals, color=["#777777", "#9c755f", "#4c78a8", "#d1495b"], width=0.65)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=18, ha="right", fontsize=8)
    ax.set_title(metric)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
fig.suptitle("Map prediction is the main performance gain; branch control adds efficiency")
fig.savefig(OUT / "fig1_navigation_main.png", dpi=180, bbox_inches="tight")
plt.close(fig)

activation_labels = ["map_pred", "truth_mix25_diag", "truth_mix50_diag", "truth_full_diag"]
activation_names = ["Predicted map", "25% oracle mix", "50% oracle mix", "Full oracle map"]
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for i, (label, name) in enumerate(zip(activation_labels, activation_names)):
    x = agg[label]["AP_UNSEEN_PRED"]
    y = agg[label]["SPL"]
    ax.scatter(x, y, s=120, label=f"{name}: AP {x:.3f}, SPL {y:.3f}", zorder=3)
ax.set_xlabel("AP_UNSEEN")
ax.set_ylabel("SPL")
ax.set_xlim(0.55, 1.03)
ax.set_ylim(0.695, 0.715)
ax.grid(alpha=0.2)
ax.set_title("Oracle map-accuracy sweep (diagnostic, not deployable)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(OUT / "fig2_ap_activation.png", dpi=180, bbox_inches="tight")
plt.close(fig)

models = json.load(open(ROOT / "map_prediction_models_v6.json", encoding="utf-8"))
heur = json.load(open(ROOT / "map_prediction_heuristic_v6.json", encoding="utf-8"))
model_rows = []
for model in ("qwen-plus", "qwen-max"):
    group = [r for r in models if r["model_requested"] == model]
    valid = [r for r in group if r["parse_ok"]]
    model_rows.append({
        "name": model,
        "parse": np.mean([r["parse_ok"] for r in group]),
        "ap_all": np.mean([r["ap_unseen"] for r in group]),
        "ap_valid": np.mean([r["ap_unseen"] for r in valid]),
        "balanced": np.mean([
            # Balanced accuracy is supplied by the offline analysis.
            0.0
        ]) if not valid else None,
    })

# Balanced accuracy for LLM rows is computed by analyze_map_prediction_v6.py.
from analyze_map_prediction_v6 import rows as analyzed_rows  # noqa: E402

for row in model_rows:
    group = [r for r in analyzed_rows if r["model_requested"] == row["name"] and r["parse_ok"]]
    row["balanced"] = np.mean([r["balanced_accuracy"] for r in group])

fig, ax = plt.subplots(figsize=(8.0, 4.4))
names = ["Logistic", "qwen-plus", "qwen-max"]
parse_rates = [1.0, model_rows[0]["parse"], model_rows[1]["parse"]]
aps = [
    np.mean([r["ap_unseen"] for r in heur]),
    model_rows[0]["ap_all"],
    model_rows[1]["ap_all"],
]
bals = [
    np.mean([r["balanced_accuracy"] for r in heur]),
    model_rows[0]["balanced"],
    model_rows[1]["balanced"],
]
x = np.arange(3)
ax.bar(x - 0.22, parse_rates, 0.22, label="format parse rate")
ax.bar(x, aps, 0.22, label="AP_UNSEEN (all calls)")
ax.bar(x + 0.22, bals, 0.22, label="balanced accuracy")
ax.set_ylim(0, 1.05)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_title("Map prediction: larger model does not imply better spatial prediction")
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.2)
fig.savefig(OUT / "fig3_map_models.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.2))
x = np.arange(2)
vals_sr = [agg["jitter_base"]["SR"], agg["jitter_anti"]["SR"]]
vals_spl = [agg["jitter_base"]["SPL"], agg["jitter_anti"]["SPL"]]
ax.bar(x - 0.18, vals_sr, 0.36, label="SR")
ax.bar(x + 0.18, vals_spl, 0.36, label="SPL")
ax.set_xticks(x)
ax.set_xticklabels(["Noisy fork baseline", "Branch control"])
ax.set_ylim(0, 0.85)
ax.set_title("Stress test under unstable candidate rankings (80 steps)")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
fig.savefig(OUT / "fig4_jitter_stress.png", dpi=180, bbox_inches="tight")
plt.close(fig)

with (ROOT / "map_prediction_summary.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["model", "parse_rate", "AP_all", "AP_valid", "balanced_accuracy_valid"])
    writer.writeheader()
    writer.writerow({
        "model": "Logistic",
        "parse_rate": 1.0,
        "AP_all": float(np.mean([r["ap_unseen"] for r in heur])),
        "AP_valid": float(np.mean([r["ap_unseen"] for r in heur])),
        "balanced_accuracy_valid": float(np.mean([r["balanced_accuracy"] for r in heur])),
    })
    for model in model_rows:
        writer.writerow({
            "model": model["name"],
            "parse_rate": float(model["parse"]),
            "AP_all": float(model["ap_all"]),
            "AP_valid": float(model["ap_valid"]),
            "balanced_accuracy_valid": float(model["balanced"]),
        })

print("figures written", OUT)
