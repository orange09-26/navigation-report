"""Create a harder held-out 25x25 benchmark for v8."""

import json
from pathlib import Path

import benchmark_maps_v5 as base


def build():
    specs = []
    families = [
        ("office", 21001), ("office", 22001),
        ("loop", 23001), ("loop", 24001),
        ("maze", 25001), ("maze", 26001),
        ("trap", 27001), ("trap", 28001),
    ]
    for i, (family, seed) in enumerate(families):
        for attempt in range(20):
            try:
                spec = base._make_spec(
                    family, "hard", 1, seed + attempt * 7919, 25
                )
            except RuntimeError:
                continue
            metrics = spec["metrics"]
            if (
                metrics["shortest"] >= 28
                and metrics["junctions"] >= 18
                and metrics["dead_ends"] >= 5
                and metrics["cycles"] >= 15
            ):
                spec["id"] = f"v8_{family}_{i + 1:02d}"
                specs.append(spec)
                break
        else:
            raise RuntimeError(f"unable to generate hard {family} map")
    return {
        "version": "v8-hard-25x25",
        "description": "Held-out 25x25 maps; never used for v7 parameter tuning.",
        "maps": specs,
    }


if __name__ == "__main__":
    manifest = build()
    out = Path(__file__).with_name("benchmark_maps_v8.json")
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for spec in manifest["maps"]:
        print(spec["id"], spec["metrics"])
