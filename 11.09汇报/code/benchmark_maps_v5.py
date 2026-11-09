# -*- coding: utf-8 -*-
"""Fixed, reproducible benchmark scenes for the MapGPT v5 experiments.

The maps are deliberately generated ahead of all policy runs.  Every policy
receives exactly the same JSON manifest, so differences cannot come from map
regeneration or changing start/goal pairs.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FREE = "."
WALL = "#"
START = "S"
GOAL = "G"
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
Cell = Tuple[int, int]


def _inside(size: int, x: int, y: int) -> bool:
    return 1 <= x < size - 1 and 1 <= y < size - 1


def _carve_rect(grid: List[List[str]], x0: int, y0: int, x1: int, y1: int) -> None:
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if _inside(len(grid), x, y):
                grid[y][x] = FREE


def _carve_corridor(
    grid: List[List[str]], a: Cell, b: Cell, rng: random.Random, width: int = 1
) -> None:
    x, y = a
    tx, ty = b
    first_horizontal = bool(rng.getrandbits(1))
    axes = ("h", "v") if first_horizontal else ("v", "h")
    for axis in axes:
        if axis == "h":
            step = 1 if tx >= x else -1
            while x != tx:
                if _inside(len(grid), x, y):
                    for yy in range(y, min(len(grid) - 1, y + width)):
                        grid[yy][x] = FREE
                x += step
        else:
            step = 1 if ty >= y else -1
            while y != ty:
                if _inside(len(grid), x, y):
                    for xx in range(x, min(len(grid[0]) - 1, x + width)):
                        grid[y][xx] = FREE
                y += step


def _bfs_dist(grid: List[List[str]], src: Cell) -> Dict[Cell, int]:
    h, w = len(grid), len(grid[0])
    dist = {src: 0}
    q = deque([src])
    while q:
        x, y = q.popleft()
        for dx, dy in DIRS:
            n = (x + dx, y + dy)
            if 0 <= n[0] < w and 0 <= n[1] < h and grid[n[1]][n[0]] == FREE and n not in dist:
                dist[n] = dist[(x, y)] + 1
                q.append(n)
    return dist


def _reachable_from(grid: List[List[str]], src: Cell) -> set[Cell]:
    return set(_bfs_dist(grid, src))


def _farthest_pair(grid: List[List[str]], rng: random.Random) -> Optional[Tuple[Cell, Cell, int]]:
    free = [(x, y) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch == FREE]
    if not free:
        return None
    probe = rng.choice(free)
    d0 = _bfs_dist(grid, probe)
    if not d0:
        return None
    a = max(d0, key=d0.get)
    da = _bfs_dist(grid, a)
    ordered = sorted(da.items(), key=lambda item: (-item[1], item[0]))
    for b, dist in ordered:
        if 15 <= dist <= 54:
            # Avoid always picking the least robust extreme endpoint.
            pool = [item for item in ordered[:12] if item[1] >= max(15, dist - 5)]
            if pool:
                b, dist = rng.choice(pool)
            return a, b, dist
    b, dist = ordered[-1]
    return a, b, dist


def _room_map(size: int, seed: int, hard_obstacles: bool) -> List[List[str]]:
    grid = [[WALL for _ in range(size)] for _ in range(size)]
    rng = random.Random(seed)
    rooms: List[Tuple[int, int, int, int]] = []
    for _ in range(6):
        for _attempt in range(80):
            rw = rng.randint(3, 6)
            rh = rng.randint(3, 6)
            x0 = rng.randint(1, size - rw - 2)
            y0 = rng.randint(1, size - rh - 2)
            rect = (x0, y0, x0 + rw - 1, y0 + rh - 1)
            overlaps_core = False
            for ox0, oy0, ox1, oy1 in rooms:
                if x0 - 1 <= ox1 and x0 + rw >= ox0 and y0 - 1 <= oy1 and y0 + rh >= oy0:
                    overlaps_core = True
                    break
            if not overlaps_core:
                rooms.append(rect)
                _carve_rect(grid, *rect)
                break
    centers = [((x0 + x1) // 2, (y0 + y1) // 2) for x0, y0, x1, y1 in rooms]
    if len(centers) < 4:
        return grid
    for i in range(len(centers) - 1):
        _carve_corridor(grid, centers[i], centers[i + 1], rng, width=1)
    _carve_corridor(grid, centers[0], centers[-1], rng, width=1)
    _carve_corridor(grid, centers[1], centers[-2], rng, width=1)

    # Add short side branches.  Some become decoys and some form loops.
    for _ in range(8 if hard_obstacles else 5):
        base = rng.choice(centers)
        direction = rng.choice(DIRS)
        length = rng.randint(3, 7)
        end = (base[0] + direction[0] * length, base[1] + direction[1] * length)
        _carve_corridor(grid, base, end, rng, width=1)

    pair = _farthest_pair(grid, rng)
    if pair is None:
        return grid
    start, goal, _ = pair
    protected = set(_bfs_dist(grid, start))
    # Protect a shortest-path tube while inserting room-level obstacles.
    dist_start = _bfs_dist(grid, start)
    dist_goal = _bfs_dist(grid, goal)
    shortest = dist_start.get(goal, 0)
    path_tube = {
        c for c in protected
        if dist_start.get(c, 10**9) + dist_goal.get(c, 10**9) <= shortest + 2
    }
    candidates = [
        c for c in protected
        if c not in path_tube and c not in (start, goal)
        and 2 < c[0] < size - 2 and 2 < c[1] < size - 2
    ]
    rng.shuffle(candidates)
    target_blocks = 12 if hard_obstacles else 7
    for i in range(0, min(len(candidates), target_blocks * 3), 3):
        block = candidates[i:i + 3]
        old = [grid[y][x] for x, y in block]
        for x, y in block:
            grid[y][x] = WALL
        if (start not in _reachable_from(grid, goal)):
            for (x, y), val in zip(block, old):
                grid[y][x] = val
    return grid


def _maze_map(size: int, seed: int, loops: bool) -> List[List[str]]:
    grid = [[WALL for _ in range(size)] for _ in range(size)]
    rng = random.Random(seed)
    sx, sy = 1, 1
    grid[sy][sx] = FREE
    stack = [(sx, sy)]
    visited = {(sx, sy)}
    while stack:
        x, y = stack[-1]
        choices = []
        for dx, dy in DIRS:
            nx, ny = x + 2 * dx, y + 2 * dy
            if _inside(size, nx, ny) and (nx, ny) not in visited:
                choices.append((nx, ny, x + dx, y + dy))
        if not choices:
            stack.pop()
            continue
        nx, ny, mx, my = rng.choice(choices)
        grid[my][mx] = FREE
        grid[ny][nx] = FREE
        visited.add((nx, ny))
        stack.append((nx, ny))

    # Open some walls to create cycles and widen a few key passages.
    wall_edges = []
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if grid[y][x] != WALL:
                continue
            if ((grid[y][x - 1] == FREE and grid[y][x + 1] == FREE)
                    or (grid[y - 1][x] == FREE and grid[y + 1][x] == FREE)):
                wall_edges.append((x, y))
    rng.shuffle(wall_edges)
    open_count = max(8, len(wall_edges) // (5 if loops else 10))
    for x, y in wall_edges[:open_count]:
        grid[y][x] = FREE
    for _ in range(8):
        x = rng.randrange(2, size - 3)
        y = rng.randrange(2, size - 3)
        _carve_rect(grid, x, y, min(size - 2, x + rng.randint(1, 2)),
                    min(size - 2, y + rng.randint(1, 2)))
    return grid


def _metrics(grid: List[List[str]], start: Cell, goal: Cell) -> Dict[str, int]:
    dist = _bfs_dist(grid, start)
    reachable_free = {
        (x, y) for y, row in enumerate(grid) for x, ch in enumerate(row) if ch == FREE
    }
    main_free = set(dist)
    junctions = 0
    dead_ends = 0
    edges2 = 0
    for x, y in main_free:
        deg = sum((x + dx, y + dy) in main_free for dx, dy in DIRS)
        edges2 += deg
        junctions += int(deg >= 3)
        dead_ends += int(deg == 1)
    edges2 //= 2
    return {
        "shortest": int(dist.get(goal, 0)),
        "reachable_free": len(main_free),
        "island_free": len(reachable_free - main_free),
        "junctions": junctions,
        "dead_ends": dead_ends,
        "cycles": edges2 - len(main_free) + 1,
    }


def _make_spec(family: str, difficulty: str, serial: int, seed: int, size: int) -> Dict[str, Any]:
    for attempt in range(300):
        actual_seed = seed + attempt * 7919
        if family in ("office", "loop"):
            grid = _room_map(size, actual_seed, hard_obstacles=(family == "loop"))
        else:
            grid = _maze_map(size, actual_seed, loops=(family == "maze"))
        rng = random.Random(actual_seed ^ 0x5EED)
        pair = _farthest_pair(grid, rng)
        if pair is None:
            continue
        start, goal, shortest = pair
        m = _metrics(grid, start, goal)
        min_j = 9 if difficulty == "hard" else 6
        min_dead = 6 if difficulty == "hard" else 3
        if (
            18 <= m["shortest"] <= 55
            and m["junctions"] >= min_j
            and m["dead_ends"] >= min_dead
            and m["island_free"] <= 5
        ):
            rows = ["".join(row) for row in grid]
            rows[start[1]] = rows[start[1]][:start[0]] + START + rows[start[1]][start[0] + 1:]
            rows[goal[1]] = rows[goal[1]][:goal[0]] + GOAL + rows[goal[1]][goal[0] + 1:]
            return {
                "id": f"{family}_{serial:02d}",
                "family": family,
                "difficulty": difficulty,
                "seed": actual_seed,
                "size": [size, size],
                "start": list(start),
                "goal": list(goal),
                "metrics": m,
                "ascii": rows,
            }
    raise RuntimeError(f"could not generate {family} map {serial}")


def build_manifest() -> Dict[str, Any]:
    specs: List[Dict[str, Any]] = []
    specs.append(_make_spec("office", "medium", 1, 5001, 21))
    specs.append(_make_spec("office", "hard", 2, 6001, 21))
    specs.append(_make_spec("loop", "medium", 1, 7001, 21))
    specs.append(_make_spec("loop", "hard", 2, 8001, 21))
    specs.append(_make_spec("maze", "medium", 1, 9001, 21))
    specs.append(_make_spec("maze", "hard", 2, 10001, 21))
    specs.append(_make_spec("trap", "medium", 1, 11001, 21))
    specs.append(_make_spec("trap", "hard", 2, 12001, 21))
    specs.append(_make_spec("office", "hard", 3, 13001, 21))
    specs.append(_make_spec("loop", "hard", 3, 14001, 21))
    specs.append(_make_spec("maze", "hard", 3, 15001, 21))
    specs.append(_make_spec("trap", "hard", 3, 16001, 21))
    return {
        "version": "v5-fixed-2026-11-02",
        "description": "Twelve fixed 21x21 scenes with rooms, loops, bottlenecks and decoy branches.",
        "maps": specs,
    }


def load_manifest(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        path = str(Path(__file__).with_name("benchmark_maps_v5.json"))
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    manifest = build_manifest()
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).with_name("benchmark_maps_v5.json")))
    args = parser.parse_args()
    manifest = build_manifest()
    Path(args.out).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.out, "maps", len(manifest["maps"]))
    for spec in manifest["maps"]:
        print(spec["id"], spec["difficulty"], spec["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
