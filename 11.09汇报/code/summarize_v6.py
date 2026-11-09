import csv
import json
import random
from pathlib import Path
from statistics import mean

ROOT = Path("work/v6_final")
OUT = Path("work/v6_analysis")
OUT.mkdir(exist_ok=True)

CONDITIONS = {
    "base100": "results_v6_base100",
    "anti_v3": "results_v6_anti100_v3",
    "map_pred": "results_v6_map_pred100_model2",
    "final": "results_v6_map_pred_anti100_model2",
    "confidence_gate": "results_v6_map_conf_gate100_model2",
    "AP_oracle_diag": "results_v6_ap_oracle100_model2",
    "jitter_base": "results_v6_jitter_base_model2",
    "jitter_anti": "results_v6_jitter_anti_model2",
    "truth_mix25_diag": "results_v6_truth_mix25_model2",
    "truth_mix50_diag": "results_v6_truth_mix50_model2",
    "truth_full_diag": "results_v6_truth_full_model2",
}


def load(folder):
    return json.load(open(ROOT / folder / "episodes.json", encoding="utf-8"))


def metrics(rows):
    n = len(rows)
    return {
        "N": n,
        "SR": sum(r["success"] for r in rows) / n,
        "SPL": sum(
            r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0
            for r in rows
        ) / n,
        "OSR": sum(r["osr"] for r in rows) / n,
        "NE": mean(r["ne"] for r in rows),
        "COV": mean(r["coverage"] for r in rows),
        "BUMPS": sum(r["bumps"] for r in rows),
        "RECOVERIES": sum(r["recoveries"] for r in rows),
        "BRANCH_FAILURES": sum(r.get("branch_failures", 0) for r in rows),
        "AP_UNSEEN_PRED": mean(r["ap_unseen_pred"] for r in rows),
        "MAP_CONF": mean(r["map_confidence"] for r in rows),
        "LLM_CALLS": sum(r["llm_calls"] for r in rows),
    }


data = {label: load(folder) for label, folder in CONDITIONS.items()}
summary = {label: metrics(rows) for label, rows in data.items()}

with (OUT / "aggregate.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=["condition"] + list(next(iter(summary.values()))))
    writer.writeheader()
    for label, row in summary.items():
        writer.writerow({"condition": label, **row})

base = {r["map_id"]: r for r in data["base100"]}
main_conditions = ["base100", "anti_v3", "map_pred", "final", "confidence_gate", "AP_oracle_diag"]
per_map = []
for label in main_conditions:
    for r in data[label]:
        b = base[r["map_id"]]
        spl = r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0
        bspl = b["shortest"] / max(b["actual"], b["shortest"], 1) if b["success"] else 0
        per_map.append({
            "condition": label,
            "map_id": r["map_id"],
            "success": r["success"],
            "steps": r["steps"],
            "shortest": r["shortest"],
            "actual": r["actual"],
            "SPL": spl,
            "NE": r["ne"],
            "AP_UNSEEN_PRED": r["ap_unseen_pred"],
            "MAP_CONF": r["map_confidence"],
            "RECOVERIES": r["recoveries"],
            "BRANCH_FAILURES": r.get("branch_failures", 0),
            "delta_SPL_vs_base": spl - bspl,
        })
with (OUT / "per_map.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(per_map[0]))
    writer.writeheader()
    writer.writerows(per_map)


def bootstrap(values):
    rng = random.Random(17)
    n = len(values)
    if not n:
        return [0, 0]
    samples = []
    for _ in range(5000):
        samples.append(mean(values[rng.randrange(n)] for _ in range(n)))
    samples.sort()
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples)) - 1]]


paired = {}
for label in main_conditions:
    if label == "base100":
        continue
    current = {r["map_id"]: r for r in data[label]}
    dspl = []
    dap = []
    for map_id, b in base.items():
        r = current[map_id]
        cspl = r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0
        bspl = b["shortest"] / max(b["actual"], b["shortest"], 1) if b["success"] else 0
        dspl.append(cspl - bspl)
        dap.append(r["ap_unseen_pred"] - b["ap_unseen_pred"])
    paired[label] = {
        "delta_SPL": mean(dspl),
        "delta_SPL_ci95": bootstrap(dspl),
        "delta_AP": mean(dap),
        "delta_AP_ci95": bootstrap(dap),
    }

report = {
    "aggregate": summary,
    "paired_vs_base": paired,
    "jitter_delta": {
        "SR": summary["jitter_anti"]["SR"] - summary["jitter_base"]["SR"],
        "SPL": summary["jitter_anti"]["SPL"] - summary["jitter_base"]["SPL"],
    },
    "map_prediction_activation": {
        label: {
            "SR": summary[label]["SR"],
            "SPL": summary[label]["SPL"],
            "AP": summary[label]["AP_UNSEEN_PRED"],
        }
        for label in ["map_pred", "truth_mix25_diag", "truth_mix50_diag", "truth_full_diag"]
    },
}
(OUT / "analysis.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2))
