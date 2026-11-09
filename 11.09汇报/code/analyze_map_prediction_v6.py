import json
import sys

import numpy as np

sys.path.insert(0, "work")
import llm_map_prediction_v6 as llmp


def recall(truth, pred, value):
    tp = int(((truth == value) & (pred == value)).sum())
    fn = int(((truth == value) & (pred != value)).sum())
    return tp / max(1, tp + fn)


rows = json.load(open("work/map_prediction_models_v6.json", encoding="utf-8"))
snapshots = {
    (s["map_id"], s["step"]): s
    for s in json.load(open("work/map_prediction_snapshots_v6.json", encoding="utf-8"))
}
for row in rows:
    snapshot = snapshots[(row["map_id"], row["step"])]
    pred = llmp.parse_prediction(row["output_text"], snapshot["width"], snapshot["height"])
    row["balanced_accuracy"] = 0.0
    if pred is None:
        continue
    truth = np.asarray(snapshot["grid"], dtype=int)
    explored = {tuple(c) for c in snapshot["explored"]}
    unknown = np.asarray([
        [(x, y) not in explored for x in range(snapshot["width"])]
        for y in range(snapshot["height"])
    ])
    t = truth[unknown]
    p = np.asarray(pred, dtype=int)[unknown]
    row["balanced_accuracy"] = 0.5 * (recall(t, p, 1) + recall(t, p, 0))
print("MODEL SUMMARY")
for model in sorted(set(r["model_requested"] for r in rows)):
    group = [r for r in rows if r["model_requested"] == model]
    valid = [r for r in group if r["parse_ok"]]
    print({
        "model": model,
        "calls": len(group),
        "parse_rate": float(np.mean([r["parse_ok"] for r in group])),
        "AP_all": float(np.mean([r["ap_unseen"] for r in group])),
        "AP_valid": float(np.mean([r["ap_unseen"] for r in valid])) if valid else 0,
        "wall_f1_valid": float(np.mean([r["wall_f1"] for r in valid])) if valid else 0,
        "free_f1_valid": float(np.mean([r["free_f1"] for r in valid])) if valid else 0,
        "known_accuracy_valid": float(np.mean([r["known_accuracy"] for r in valid])) if valid else 0,
        "balanced_accuracy_valid": float(np.mean([r["balanced_accuracy"] for r in valid])) if valid else 0,
        "total_tokens": sum(int(r.get("usage", {}).get("total_tokens", 0) or 0) for r in group),
    })

heur = json.load(open("work/map_prediction_heuristic_v6.json", encoding="utf-8"))
print("\nHEURISTIC", {
    "n": len(heur),
    "AP": float(np.mean([r["ap_unseen"] for r in heur])),
    "wall_f1": float(np.mean([r["wall_f1"] for r in heur])),
    "free_f1": float(np.mean([r["free_f1"] for r in heur])),
    "confidence": float(np.mean([r["mean_confidence"] for r in heur])),
    "balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in heur])),
})

print("\nPER MAP")
for map_id in sorted(set(r["map_id"] for r in rows)):
    print(map_id)
    h = [r for r in heur if r["map_id"] == map_id]
    print("  heuristic", round(float(np.mean([r["ap_unseen"] for r in h])), 4))
    for model in sorted(set(r["model_requested"] for r in rows)):
        group = [
            r for r in rows
            if r["map_id"] == map_id and r["model_requested"] == model
        ]
        print(
            " ", model,
            "parse", sum(bool(r["parse_ok"]) for r in group),
            "AP", round(float(np.mean([r["ap_unseen"] for r in group])), 4),
        )
