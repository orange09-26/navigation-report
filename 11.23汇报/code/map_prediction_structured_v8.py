"""Structural ensemble on top of the trained v6 logistic map predictor."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import map_prediction_v6 as mp6


ROOT = Path("/home/orange")


def structured_prob(env, logistic_p: np.ndarray) -> np.ndarray:
    known = np.zeros_like(logistic_p, dtype=bool)
    free = np.zeros_like(known)
    wall = np.zeros_like(known)
    for x, y in env.explored_cells:
        known[y, x] = True
        free[y, x] = env.is_free(x, y)
        wall[y, x] = not env.is_free(x, y)
    free_dist = distance_transform_edt(~free)
    wall_dist = distance_transform_edt(~wall)
    nearest = free_dist / np.maximum(1e-6, free_dist + wall_dist)
    p = 0.55 * logistic_p + 0.45 * nearest
    h, w = p.shape
    for y in range(h):
        for x in range(w):
            if known[y, x]:
                p[y, x] = 1.0 if wall[y, x] else 0.0
                continue
            left = (x - 1, y)
            right = (x + 1, y)
            up = (x, y - 1)
            down = (x, y + 1)
            pair_adjust = 0.0
            for a, b in ((left, right), (up, down)):
                if a in env.explored_cells and b in env.explored_cells:
                    if not env.is_free(*a) and not env.is_free(*b):
                        pair_adjust += 0.10
                    elif env.is_free(*a) and env.is_free(*b):
                        pair_adjust -= 0.10
            y0, y1 = max(0, y - 2), min(h, y + 3)
            x0, x1 = max(0, x - 2), min(w, x + 3)
            fc = int(free[y0:y1, x0:x1].sum())
            wc = int(wall[y0:y1, x0:x1].sum())
            p[y, x] += pair_adjust + 0.08 * (wc - fc) / max(1, wc + fc)
            p[y, x] = np.clip(p[y, x], 0.01, 0.99)
    return p


def scores(snapshot, predictor):
    env = mp6._load_snapshot(snapshot)
    logistic = predictor.probabilities(env)
    p = structured_prob(env, logistic)
    truth = []
    values = []
    conf = []
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.explored_cells:
                continue
            truth.append(0 if env.is_free(x, y) else 1)
            values.append(float(p[y, x]))
            conf.append(max(float(p[y, x]), 1.0 - float(p[y, x])))
    return {
        "map_id": snapshot["map_id"],
        "step": snapshot["step"],
        "truth": np.asarray(truth),
        "p": np.asarray(values),
        "conf": float(np.mean(conf)),
    }


def metrics(scored, threshold):
    truth = scored["truth"]
    pred = (scored["p"] >= threshold).astype(int)
    acc = float((truth == pred).mean())
    wtp = int(((truth == 1) & (pred == 1)).sum())
    wfn = int(((truth == 1) & (pred == 0)).sum())
    ftp = int(((truth == 0) & (pred == 0)).sum())
    ffp = int(((truth == 0) & (pred == 1)).sum())
    wall_recall = wtp / max(1, wtp + wfn)
    free_recall = ftp / max(1, ftp + ffp)
    wall_f1 = 2 * wtp / max(1, 2 * wtp +
                             int(((truth == 0) & (pred == 1)).sum()) + wfn)
    free_f1 = 2 * ftp / max(1, 2 * ftp +
                           int(((truth == 1) & (pred == 0)).sum()) + ffp)
    return {
        "ap_unseen": acc,
        "balanced_accuracy": 0.5 * (wall_recall + free_recall),
        "wall_f1": wall_f1,
        "free_f1": free_f1,
        "mean_confidence": scored["conf"],
    }


def main():
    model = json.load(open(ROOT / "map_predictor_v6.json", encoding="utf-8"))
    predictor = mp6.LogisticMapPredictor(model)
    test = json.load(open(ROOT / "map_prediction_snapshots_v6.json", encoding="utf-8"))
    manifest = json.load(open(ROOT / "benchmark_maps_v5.json", encoding="utf-8"))
    validation = []
    for i, spec in enumerate(manifest["maps"][8:]):
        validation.extend(mp6._random_snapshots(spec, 12300 + i, 4))
    val_scored = [scores(s, predictor) for s in validation]
    threshold_grid = np.arange(0.30, 0.76, 0.02)
    calibration = []
    for threshold in threshold_grid:
        vals = [metrics(s, threshold)["balanced_accuracy"] for s in val_scored]
        calibration.append((float(np.mean(vals)), float(threshold)))
    calibration.sort(reverse=True)
    threshold = calibration[0][1]
    test_scored = [scores(s, predictor) for s in test]
    results = []
    for s in test_scored:
        results.append({"map_id": s["map_id"], "step": s["step"], **metrics(s, threshold)})
    out = ROOT / "map_prediction_structured_v8.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "threshold": threshold,
        "validation_balanced_accuracy": calibration[0][0],
        "AP_UNSEEN": float(np.mean([r["ap_unseen"] for r in results])),
        "BALANCED_ACC": float(np.mean([r["balanced_accuracy"] for r in results])),
        "WALL_F1": float(np.mean([r["wall_f1"] for r in results])),
        "FREE_F1": float(np.mean([r["free_f1"] for r in results])),
        "CONF": float(np.mean([r["mean_confidence"] for r in results])),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
