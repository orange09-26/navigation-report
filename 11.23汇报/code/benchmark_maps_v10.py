"""Create a held-out 27x27 validation set for v10."""

import json
from pathlib import Path

import benchmark_maps_v5 as base


def build():
    specs = []
    families = ("office", "loop", "maze", "trap")
    for family_index, family in enumerate(families):
        for copy_index in range(2):
            base_seed = 70000 + copy_index * 20000
            for attempt in range(80):
                try:
                    spec = base._make_spec(
                        family,
                        "hard",
                        1,
                        base_seed + attempt * 7919,
                        27,
                    )
                except RuntimeError:
                    continue
                metrics = spec["metrics"]
                if (
                    metrics["shortest"] >= 35
                    and metrics["junctions"] >= 25
                    and metrics["dead_ends"] >= 6
                    and metrics["cycles"] >= 20
                ):
                    spec["id"] = (
                        f"v10_{family}_{copy_index + 1:02d}"
                    )
                    specs.append(spec)
                    break
            else:
                raise RuntimeError(
                    f"unable to generate hard {family} map {copy_index}"
                )
    return {
        "version": "v10-hard-27x27",
        "description": (
            "Eight held-out 27x27 maps generated after v10 methods were fixed."
        ),
        "maps": specs,
    }


if __name__ == "__main__":
    manifest = build()
    out = Path(__file__).with_name("benchmark_maps_v10.json")
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for spec in manifest["maps"]:
        print(spec["id"], spec["metrics"])
