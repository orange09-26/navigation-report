"""Create a second held-out 29x29 validation set for robustness checks."""

import json
from pathlib import Path

import benchmark_maps_v5 as base


def build():
    specs = []
    for family in ("office", "loop", "maze", "trap"):
        for copy_index in range(2):
            for attempt in range(120):
                try:
                    spec = base._make_spec(
                        family,
                        "hard",
                        1,
                        90000 + copy_index * 24000 + attempt * 7919,
                        29,
                    )
                except RuntimeError:
                    continue
                metrics = spec["metrics"]
                if (
                    metrics["shortest"] >= 40
                    and metrics["junctions"] >= 28
                    and metrics["dead_ends"] >= 6
                    and metrics["cycles"] >= 22
                ):
                    spec["id"] = f"v10b_{family}_{copy_index + 1:02d}"
                    specs.append(spec)
                    break
            else:
                raise RuntimeError(
                    f"unable to generate hard 29x29 {family} {copy_index}"
                )
    return {
        "version": "v10b-hard-29x29",
        "description": "Second held-out 29x29 robustness suite.",
        "maps": specs,
    }


if __name__ == "__main__":
    manifest = build()
    out = Path(__file__).with_name("benchmark_maps_v10b.json")
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for spec in manifest["maps"]:
        print(spec["id"], spec["metrics"])
