# -*- coding: utf-8 -*-
"""Merge partitioned v5 runs into one condition directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--parts", type=int, required=True)
    parser.add_argument("--out-tag", required=True)
    args = parser.parse_args()
    root = Path("/home/orange")
    episodes = []
    steps = []
    for part in range(args.parts):
        d = root / f"results_v5_{args.condition}_p{part}"
        ep_path = d / "episodes.json"
        st_path = d / "steps.json"
        if not ep_path.exists() or not st_path.exists():
            raise SystemExit(f"missing completed part: {d}")
        episodes.extend(json.loads(ep_path.read_text(encoding="utf-8")))
        steps.extend(json.loads(st_path.read_text(encoding="utf-8")))
    episodes.sort(key=lambda r: r["map_id"])
    out = root / f"results_{args.out_tag}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.json").open("w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    with (out / "steps.json").open("w", encoding="utf-8") as f:
        json.dump(steps, f, ensure_ascii=False, indent=2)
    # Keep metric aggregation in one place.
    import sys
    sys.path.insert(0, str(root))
    from mapgpt_v5 import write_outputs

    write_outputs(out, args.out_tag, episodes, steps)
    print(f"combined {len(episodes)} episodes into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
