import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict


ROOT = Path("work/v10_results")
OUT = Path("work/v10_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def metrics(path: Path) -> Dict[str, Any]:
    rows = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
    n = len(rows)
    sr = sum(bool(row["success"]) for row in rows) / n
    spl = sum(
        row["shortest"] / max(row["actual"], row["shortest"], 1)
        if row["success"] else 0.0
        for row in rows
    ) / n
    ap = mean(row.get("AP_UNSEEN", 0.0) for row in rows)
    return {
        "N": n,
        "SR": sr,
        "SPL": spl,
        "NE": mean(row["ne"] for row in rows),
        "COV": mean(row["coverage"] for row in rows),
        "AP_UNSEEN": ap,
        "JOINT": math.sqrt(max(0.0, spl * ap)),
        "BUMPS": sum(row["bumps"] for row in rows),
        "RECOVERIES": sum(row["recoveries"] for row in rows),
        "LLM_CALLS": sum(row["llm_calls"] for row in rows),
        "OSR": sum(bool(row["osr"]) for row in rows) / n,
    }


results = {
    path.name.replace("results_", ""): metrics(path)
    for path in sorted(ROOT.glob("results_*"))
    if (path / "episodes.json").exists()
}

final = {
    "21x21_100": {
        "baseline": "v9_obs5_hycommon",
        "final": "v10_r1_small_base",
    },
    "25x25_80": {
        "baseline": "v10_r2_large80_obs9",
        "final": "v10_r3_large80_geom_re1",
    },
    "25x25_100": {
        "baseline": "v10_r1_large_base",
        "final": "v10_r3_large_geom_re1",
    },
    "25x25_120": {
        "baseline": "v10_r2_large120_obs9",
        "final": "v10_final25_120",
    },
    "27x27_80": {
        "baseline": "v10_val_base_80",
        "final": "v10_val2_geom_re1_80",
    },
    "27x27_100": {
        "baseline": "v10_val_base_100",
        "final": "v10_val2_geom_re1_100",
    },
    "27x27_120": {
        "baseline": "v10_val_base_120",
        "final": "v10_val2_geom_re1_120",
    },
    "29x29_80": {
        "baseline": "v10b_base_80",
        "final": "v10b_final_80",
    },
    "29x29_100": {
        "baseline": "v10b_base_100",
        "final": "v10b_final_100",
    },
    "29x29_120": {
        "baseline": "v10b_base_120",
        "final": "v10b_final_120",
    },
}

comparison = {}
for suite, pair in final.items():
    base = results[pair["baseline"]]
    new = results[pair["final"]]
    comparison[suite] = {
        "baseline_condition": pair["baseline"],
        "final_condition": pair["final"],
        "baseline": base,
        "final": new,
        "delta": {
            key: new[key] - base[key]
            for key in ("SR", "SPL", "AP_UNSEEN", "JOINT", "NE")
        },
    }

payload = {"results": results, "final_comparison": comparison}
(OUT / "aggregate.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
fields = [
    "condition", "N", "SR", "SPL", "NE", "COV", "AP_UNSEEN",
    "JOINT", "BUMPS", "RECOVERIES", "LLM_CALLS", "OSR",
]
with (OUT / "aggregate.csv").open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for name, row in results.items():
        writer.writerow({"condition": name, **row})

per_map_rows = []
for suite, pair in final.items():
    base_rows = {
        row["map_id"]: row
        for row in json.loads(
            (
                ROOT / f"results_{pair['baseline']}" / "episodes.json"
            ).read_text(
                encoding="utf-8"
            )
        )
    }
    final_rows = {
        row["map_id"]: row
        for row in json.loads(
            (
                ROOT / f"results_{pair['final']}" / "episodes.json"
            ).read_text(
                encoding="utf-8"
            )
        )
    }
    for map_id in sorted(base_rows):
        base_row = base_rows[map_id]
        final_row = final_rows[map_id]
        base_spl = (
            base_row["shortest"]
            / max(base_row["actual"], base_row["shortest"], 1)
            if base_row["success"] else 0
        )
        final_spl = (
            final_row["shortest"]
            / max(final_row["actual"], final_row["shortest"], 1)
            if final_row["success"] else 0
        )
        per_map_rows.append({
            "suite": suite,
            "map_id": map_id,
            "baseline": pair["baseline"],
            "final": pair["final"],
            "baseline_success": int(base_row["success"]),
            "final_success": int(final_row["success"]),
            "baseline_spl": base_spl,
            "final_spl": final_spl,
            "delta_spl": final_spl - base_spl,
            "baseline_ap": base_row.get("AP_UNSEEN", 0),
            "final_ap": final_row.get("AP_UNSEEN", 0),
            "delta_ap": (
                final_row.get("AP_UNSEEN", 0)
                - base_row.get("AP_UNSEEN", 0)
            ),
        })
with (OUT / "per_map_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as fh:
    writer = csv.DictWriter(fh, fieldnames=list(per_map_rows[0]))
    writer.writeheader()
    writer.writerows(per_map_rows)

print(json.dumps(comparison, ensure_ascii=False, indent=2))
