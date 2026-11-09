import json
from pathlib import Path

import numpy as np

root = Path("work/v6_final")
conditions = [
    "v6_base100",
    "v6_anti100_v3",
    "v6_map_pred100_model2",
    "v6_map_pred_anti100_model2",
    "v6_map_conf_gate100_model2",
    "v6_ap_oracle100_model2",
    "v6_jitter_base_model2",
    "v6_jitter_anti_model2",
]
all_rows = []
for condition in conditions:
    rows = json.load(open(root / ("results_" + condition) / "episodes.json", encoding="utf-8"))
    all_rows.extend(rows)
    successes = [r for r in rows if r["success"]]
    failures = [r for r in rows if not r["success"]]
    print("\n", condition)
    print(
        "SR", round(len(successes) / len(rows), 4),
        "SPL", round(sum(
            r["shortest"] / max(r["actual"], r["shortest"], 1) if r["success"] else 0
            for r in rows
        ) / len(rows), 4),
        "AP", round(float(np.mean([r["ap_unseen_pred"] for r in rows])), 4),
        "BRANCH_FAIL", sum(r.get("branch_failures", 0) for r in rows),
        "RECOVERY", sum(r.get("recoveries", 0) for r in rows),
    )
    print("success AP", round(float(np.mean([r["ap_unseen_pred"] for r in successes])), 4) if successes else None)
    print("failure AP", round(float(np.mean([r["ap_unseen_pred"] for r in failures])), 4) if failures else None)
    for r in failures:
        print(" FAIL", r["map_id"], "NE", r["ne"], "steps", r["steps"], "AP", round(r["ap_unseen_pred"], 4))

x = np.asarray([r["ap_unseen_pred"] for r in all_rows])
y = np.asarray([1 if r["success"] else 0 for r in all_rows])
print("\nCORR AP-success", float(np.corrcoef(x, y)[0, 1]))
