# -*- coding: utf-8 -*-
"""
================================================================================
项目名称：MapGPT 原理验证 Demo（Ubuntu 虚拟机 · 2D 网格文本仿真）
文件建议名：mapgpt_vm_demo.py
================================================================================

【项目说明 / 免责声明 —— 必须保留】
本工程为原理验证 Demo，不训练、不微调大模型。
自制 2D 网格房间场景，并非官方 Matterport3D / Habitat / R2R 测试集。
无图像输入、不安装 Habitat、不克隆 MapGPT 原始仓库、不下载 Matterport3D。
所计算的 SR / SPL / AP 仅用于验证「Map-Guided Prompting + 拓扑地图 + 逐步导航」
这一范式能否在文本仿真中跑通，数值不与 MapGPT 论文原版结果对标。

【算法对照（论文逻辑，非官方实现）】
论文：MapGPT: Map-Guided Prompting with Adaptive Path Planning for
Vision-and-Language Navigation (ACL 2024 / arXiv:2401.07314)
任务形态对齐 R2R：给定语言指令，在未知室内环境逐步导航，可执行停止。
本 Demo 将 Matterport 视点图替换为 2D 网格：
  - 在线构建拓扑图 G_t={V_t, E_t}（已探索节点 / 当前可达节点 / 未探索可见节点）
  - Prompt 按 MapGPT 结构组装：Task Description、Instruction、History、
    Observation、Action Space、Trajectory、Map Connectivity、Previous Planning
  - 要求 LLM 输出 Thought + 更新后的全局拓扑地图 + 多步规划 + 下一步动作
  - 动作空间约束为四选一：前进 / 左转 / 右转 / 停止（对应网格差动底盘）

【硬性接口约束】
虚拟机绝不 import openai，绝不调用 dashscope。
全部 LLM 请求使用 requests POST：
    URL  : http://192.168.43.38:5000/get_llm_response
    Body : {"prompt": "<组装好的 MapGPT 风格 prompt>"}
代码中不出现任何 API 密钥（密钥由 Windows 中转服务持有）。

【指标】
  SR  = 成功 episode 数 / 20
  SPL = 仅对成功 episode：最短路径 / max(实际行走路径, 最短路径)，再取平均
  AP  = LLM 预测拓扑 与 仿真真实地图 的匹配精度（逐步平均后再对 episode 平均）
================================================================================
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# ---------------------------------------------------------------------------
# 全局常量
# ---------------------------------------------------------------------------
# Windows Flask 中转（密钥只在 Windows，虚拟机不保存任何 API Key）
LLM_URL = "http://100.79.62.8:5000/get_llm_response"
NUM_EPISODES = 20  # 完整评测 20 局
SLEEP_AFTER_OK_SEC = 2.5  # 成功调用后降温，降低 HTTP 500 限流风险
SLEEP_ON_HTTP500_SEC = 8.0  # 碰到 500 多等一会再重试，不整场停掉
MAX_STEPS = 60
GRID_H = 15
GRID_W = 15
OBS_RADIUS = 2  # 局部邻域 Chebyshev 半径，5x5
REQUEST_TIMEOUT_SEC = 120.0
MAX_RETRY = 4  # 超时/500/失败后额外重试 4 次，尽量把 20 局跑完
SLEEP_BETWEEN_RETRY = 1.5
EPISODE_SEED_BASE = 20240928

# 栅格语义
FREE = 0
WALL = 1

# 朝向：0 北(y-1) / 1 东(x+1) / 2 南(y+1) / 3 西(x-1)
HEADING_DELTA = ((0, -1), (1, 0), (0, 1), (-1, 0))
HEADING_NAME = ("NORTH", "EAST", "SOUTH", "WEST")
HEADING_CN = ("北", "东", "南", "西")
HEADING_GLYPH = ("^", ">", "v", "<")

ACTION_FORWARD = "FORWARD"
ACTION_LEFT = "LEFT"
ACTION_RIGHT = "RIGHT"
ACTION_STOP = "STOP"
VALID_ACTIONS = (ACTION_FORWARD, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP)

# MapGPT 风格动作选项标签（论文输出形如 "Action: B"）
ACTION_BY_LABEL = {
    "A": ACTION_STOP,
    "B": ACTION_FORWARD,
    "C": ACTION_LEFT,
    "D": ACTION_RIGHT,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "log.txt")


class ExperimentAbort(RuntimeError):
    """中转 500 / AP=0 / LLM 调用失败时立即终止整场实验，禁止继续用兜底凑局。"""


# ===========================================================================
# 1. 仿真环境类：2D 网格房间 + 局部观测 + 在线拓扑
# ===========================================================================
class GridNavEnv:
    """未知室内网格导航环境。机器人看不到全局地图，只能看见局部邻域。"""

    def __init__(self, height: int = GRID_H, width: int = GRID_W) -> None:
        self.height = height
        self.width = width
        self.grid: List[List[int]] = []
        self.start: Tuple[int, int] = (1, 1)
        self.goal: Tuple[int, int] = (1, 1)
        self.x = 1
        self.y = 1
        self.heading = 1  # 初始朝东
        self.explored_cells: Set[Tuple[int, int]] = set()  # 曾经进入局部视野的格子
        self.visited_cells: List[Tuple[int, int]] = []  # 实际站过的格子（轨迹）
        self.topo_nodes: Dict[Tuple[int, int], int] = {}  # 观察到的可走格 -> Place ID
        self.topo_edges: Set[Tuple[int, int]] = set()  # Place ID 无向边
        self.next_place_id = 0
        self.instruction = ""
        self.shortest_len = 1
        self.forward_steps = 0  # 实际成功前进的格数，作为行走路径长度
        self.last_obs_text = ""
        self.rooms: List[Tuple[int, int, int, int]] = []

    # ---------- 地图生成 ----------
    def generate(self, rng_seed: int) -> None:
        """随机生成连通的多房间+走廊场景，保证起终点可达且距离足够。"""
        import random as _random

        rng = _random.Random(rng_seed)
        for _attempt in range(80):
            grid = [[WALL for _ in range(self.width)] for _ in range(self.height)]
            rooms: List[Tuple[int, int, int, int]] = []
            n_rooms = rng.randint(3, 5)
            for _ in range(n_rooms):
                rw = rng.randint(3, 6)
                rh = rng.randint(3, 6)
                x0 = rng.randint(1, max(1, self.width - rw - 1))
                y0 = rng.randint(1, max(1, self.height - rh - 1))
                x1 = min(self.width - 2, x0 + rw - 1)
                y1 = min(self.height - 2, y0 + rh - 1)
                self._carve_rect(grid, x0, y0, x1, y1)
                rooms.append((x0, y0, x1, y1))
            # 用走廊把房间依次连通，避免孤岛
            centers = [self._room_center(r) for r in rooms]
            for i in range(len(centers) - 1):
                self._carve_corridor(grid, centers[i], centers[i + 1], rng)
            if len(centers) >= 3:
                self._carve_corridor(grid, centers[0], centers[-1], rng)

            free = [(x, y) for y in range(self.height) for x in range(self.width) if grid[y][x] == FREE]
            if len(free) < 20:
                continue
            start = rng.choice(free)
            # 终点尽量选距离较远的可走格
            far = [p for p in free if self._manhattan(p, start) >= 6]
            if not far:
                continue
            far.sort(key=lambda p: -self._manhattan(p, start))
            goal = far[rng.randint(0, min(8, len(far) - 1))]
            dist = self.bfs_shortest(grid, start, goal)
            if dist is None or dist < 5:
                continue
            self.grid = grid
            self.rooms = rooms
            self.start = start
            self.goal = goal
            self.shortest_len = dist
            self.instruction = self._build_instruction(start, goal, rooms)
            self.reset_agent()
            return
        # 极端兜底：一条可走的矩形环廊
        self._build_fallback_map()

    def _build_fallback_map(self) -> None:
        grid = [[WALL for _ in range(self.width)] for _ in range(self.height)]
        self._carve_rect(grid, 2, 2, 12, 5)
        self._carve_rect(grid, 2, 9, 12, 12)
        self._carve_corridor(grid, (7, 5), (7, 9), None)
        self.grid = grid
        self.rooms = [(2, 2, 12, 5), (2, 9, 12, 12)]
        self.start = (3, 3)
        self.goal = (11, 11)
        self.shortest_len = self.bfs_shortest(grid, self.start, self.goal) or 16
        self.instruction = self._build_instruction(self.start, self.goal, self.rooms)
        self.reset_agent()

    @staticmethod
    def _carve_rect(grid: List[List[int]], x0: int, y0: int, x1: int, y1: int) -> None:
        h, w = len(grid), len(grid[0])
        for y in range(max(1, y0), min(h - 1, y1 + 1)):
            for x in range(max(1, x0), min(w - 1, x1 + 1)):
                grid[y][x] = FREE

    @staticmethod
    def _carve_corridor(
        grid: List[List[int]],
        a: Tuple[int, int],
        b: Tuple[int, int],
        rng: Any,
    ) -> None:
        x, y = a
        tx, ty = b
        # 随机先横后竖或先竖后横，形成 L 型走廊
        horizontal_first = True if rng is None else bool(rng.randint(0, 1))
        if horizontal_first:
            step = 1 if tx >= x else -1
            while x != tx:
                x += step
                if 1 <= x < len(grid[0]) - 1 and 1 <= y < len(grid) - 1:
                    grid[y][x] = FREE
            step = 1 if ty >= y else -1
            while y != ty:
                y += step
                if 1 <= x < len(grid[0]) - 1 and 1 <= y < len(grid) - 1:
                    grid[y][x] = FREE
        else:
            step = 1 if ty >= y else -1
            while y != ty:
                y += step
                if 1 <= x < len(grid[0]) - 1 and 1 <= y < len(grid) - 1:
                    grid[y][x] = FREE
            step = 1 if tx >= x else -1
            while x != tx:
                x += step
                if 1 <= x < len(grid[0]) - 1 and 1 <= y < len(grid) - 1:
                    grid[y][x] = FREE

    @staticmethod
    def _room_center(room: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x0, y0, x1, y1 = room
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def bfs_shortest(
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[int]:
        """四连通最短路径长度（格子边数）。不可达返回 None。"""
        h, w = len(grid), len(grid[0])
        q: deque = deque([(start[0], start[1], 0)])
        seen = {start}
        while q:
            x, y, d = q.popleft()
            if (x, y) == goal:
                return d
            for dx, dy in HEADING_DELTA:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == FREE and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    q.append((nx, ny, d + 1))
        return None

    def _build_instruction(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        rooms: List[Tuple[int, int, int, int]],
    ) -> str:
        """R2R 风格自然语言指令：描述从起点去终点，不泄露完整地图。"""
        dx, dy = goal[0] - start[0], goal[1] - start[1]
        horiz = "east" if dx >= 0 else "west"
        vert = "south" if dy >= 0 else "north"
        return (
            f"You are a wheeled indoor robot. Start in the current room and navigate to the destination. "
            f"The destination cell is approximately toward the {vert}-{horiz} of the starting place. "
            f"Avoid walls, do not go through obstacles, and stop when you reach the destination. "
            f"Destination coordinate on the grid is ({goal[0]}, {goal[1]}). "
            f"There are about {max(2, len(rooms))} connected rooms linked by corridors."
        )

    # ---------- 智能体状态 ----------
    def reset_agent(self) -> None:
        self.x, self.y = self.start
        self.heading = 1
        self.explored_cells = set()
        self.visited_cells = [(self.x, self.y)]
        self.topo_nodes = {}
        self.topo_edges = set()
        self.next_place_id = 0
        self.forward_steps = 0
        self._update_observation_and_graph()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[y][x] == FREE

    def front_cell(self) -> Tuple[int, int]:
        dx, dy = HEADING_DELTA[self.heading]
        return self.x + dx, self.y + dy

    def at_goal(self) -> bool:
        return (self.x, self.y) == self.goal

    def _visible_cells(self) -> Set[Tuple[int, int]]:
        """以机器人为中心、墙体遮挡的局部可见格子（墙本身若相邻/视线可达也可看见）。"""
        vis: Set[Tuple[int, int]] = {(self.x, self.y)}
        # 在局部窗口内，从机器人出发沿自由格 BFS，墙格若与可见自由格 8 邻接则可见
        q = deque([(self.x, self.y)])
        reached_free = {(self.x, self.y)}
        while q:
            cx, cy = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if not self.in_bounds(nx, ny):
                        continue
                    if max(abs(nx - self.x), abs(ny - self.y)) > OBS_RADIUS:
                        continue
                    if self.grid[ny][nx] == FREE:
                        if (nx, ny) not in reached_free:
                            reached_free.add((nx, ny))
                            vis.add((nx, ny))
                            q.append((nx, ny))
                    else:
                        vis.add((nx, ny))  # 看见这面墙，但不能穿过
        return vis

    def _register_place(self, cell: Tuple[int, int]) -> int:
        if cell not in self.topo_nodes:
            self.topo_nodes[cell] = self.next_place_id
            self.next_place_id += 1
        return self.topo_nodes[cell]

    def _update_observation_and_graph(self) -> None:
        vis = self._visible_cells()
        self.explored_cells |= vis
        # 为所有可见的可走格分配 Place ID，并按四连通写入拓扑边（仅限双方均已观察到）
        visible_free = [c for c in vis if self.is_free(*c)]
        for c in visible_free:
            self._register_place(c)
        for x, y in visible_free:
            pid = self.topo_nodes[(x, y)]
            for dx, dy in HEADING_DELTA:
                n = (x + dx, y + dy)
                if n in self.topo_nodes:
                    a, b = (pid, self.topo_nodes[n]) if pid <= self.topo_nodes[n] else (self.topo_nodes[n], pid)
                    if a != b:
                        self.topo_edges.add((a, b))
        self.last_obs_text = self.render_local_observation()

    def render_local_observation(self) -> str:
        """以机器人为中心的自我中心局部符号图 + 自然语言描述。机器人看不到全局。"""
        vis = self._visible_cells()
        lines: List[str] = []
        heading = HEADING_NAME[self.heading]
        lines.append(
            f"egocentric_local_view radius={OBS_RADIUS} heading={heading}({HEADING_CN[self.heading]}) "
            f"robot_cell=({self.x},{self.y}) glyph={HEADING_GLYPH[self.heading]}"
        )
        lines.append("symbols: R=robot  .=free  #=wall/obstacle  G=goal_if_visible  ?=out_of_view")
        for dy in range(-OBS_RADIUS, OBS_RADIUS + 1):
            row = []
            for dx in range(-OBS_RADIUS, OBS_RADIUS + 1):
                wx, wy = self.x + dx, self.y + dy
                if not self.in_bounds(wx, wy) or (wx, wy) not in vis:
                    row.append("?")
                elif (wx, wy) == (self.x, self.y):
                    row.append("R")
                elif (wx, wy) == self.goal:
                    row.append("G")
                elif self.grid[wy][wx] == WALL:
                    row.append("#")
                else:
                    row.append(".")
            lines.append("".join(row))
        # 四个相对方向（相对当前朝向：前/右/后/左）
        rel_names = ("front", "right", "back", "left")
        for i, name in enumerate(rel_names):
            abs_h = (self.heading + i) % 4
            dx, dy = HEADING_DELTA[abs_h]
            nx, ny = self.x + dx, self.y + dy
            if not self.in_bounds(nx, ny):
                desc = "out_of_bounds"
            elif self.grid[ny][nx] == WALL:
                desc = "WALL"
            else:
                extra = " GOAL" if (nx, ny) == self.goal else ""
                desc = f"FREE{extra} cell=({nx},{ny})"
            lines.append(f"{name}: {desc}")
        fx, fy = self.front_cell()
        lines.append(f"forward_blocked={not self.is_free(fx, fy)}")
        return "\n".join(lines)

    def render_explored_partial_map(self) -> str:
        """历史已探索区域（仍非全局完整地图）：只打印曾经见过的格子。"""
        if not self.explored_cells:
            return "(empty)"
        xs = [c[0] for c in self.explored_cells]
        ys = [c[1] for c in self.explored_cells]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        lines = [f"explored_bbox x=[{x0},{x1}] y=[{y0},{y1}] count={len(self.explored_cells)}"]
        for y in range(y0, y1 + 1):
            row = []
            for x in range(x0, x1 + 1):
                if (x, y) not in self.explored_cells:
                    row.append(" ")
                elif (x, y) == (self.x, self.y):
                    row.append("R")
                elif (x, y) == self.goal and (x, y) in self.explored_cells:
                    row.append("G")
                elif self.grid[y][x] == WALL:
                    row.append("#")
                elif (x, y) in set(self.visited_cells):
                    row.append("*")  # 走过
                else:
                    row.append(".")
            lines.append("".join(row))
        lines.append("legend: R=robot *=visited .=seen_free #=seen_wall G=goal space=unseen")
        return "\n".join(lines)

    def system_topological_map_text(self) -> str:
        """
        MapGPT 论文中由系统在线构建、再写入 prompt 的拓扑连通描述。
        模板：Place {id} is connected with Places ...
        当前节点排在最前。
        """
        if not self.topo_nodes:
            return "Map: (empty, navigation just started)"
        id_to_cell = {pid: cell for cell, pid in self.topo_nodes.items()}
        adj: Dict[int, Set[int]] = {pid: set() for pid in id_to_cell}
        for a, b in self.topo_edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        cur_id = self.topo_nodes.get((self.x, self.y), 0)
        order = [cur_id] + [pid for pid in sorted(id_to_cell) if pid != cur_id]
        chunks = []
        for pid in order:
            nbs = sorted(adj.get(pid, set()))
            if nbs:
                nb_txt = ", ".join(f"{n}" for n in nbs)
                chunks.append(f"Place {pid} is connected with Places {nb_txt}.")
            else:
                chunks.append(f"Place {pid} is connected with Places (none observed yet).")
        # 节点标注：explored / accessible / unexplored-visible
        visited_set = set(self.visited_cells)
        fx, fy = self.front_cell()
        accessible = []
        if self.is_free(fx, fy) and (fx, fy) in self.topo_nodes:
            accessible.append(self.topo_nodes[(fx, fy)])
        unexplored = []
        for cell, pid in self.topo_nodes.items():
            if cell not in visited_set:
                unexplored.append(pid)
        annot = [
            f"current_place={cur_id} at cell=({self.x},{self.y})",
            "explored_places=" + ",".join(str(self.topo_nodes[c]) for c in self.visited_cells if c in self.topo_nodes),
            "currently_accessible_places=" + (",".join(str(i) for i in accessible) if accessible else "(blocked or none)"),
            "observed_but_unvisited_places=" + (",".join(str(i) for i in sorted(set(unexplored))) if unexplored else "(none)"),
        ]
        return "Map:\n" + " ".join(chunks) + "\n" + "\n".join(annot)

    def trajectory_place_text(self) -> str:
        ids = []
        for c in self.visited_cells:
            if c in self.topo_nodes:
                ids.append(str(self.topo_nodes[c]))
        if not ids:
            return "Trajectory: (empty)"
        return "Trajectory: Place " + " ".join(ids)

    def true_global_ascii(self) -> str:
        """仅用于 AP 评测，禁止写入发给 LLM 的 prompt。"""
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                if (x, y) == (self.x, self.y):
                    row.append("R")
                elif (x, y) == self.goal:
                    row.append("G")
                elif self.grid[y][x] == WALL:
                    row.append("#")
                else:
                    row.append(".")
            lines.append("".join(row))
        return "\n".join(lines)

    def true_cell_label(self, x: int, y: int) -> str:
        if not self.in_bounds(x, y):
            return "OOB"
        if self.grid[y][x] == WALL:
            return "WALL"
        return "FREE"

    def step(self, action: str) -> Dict[str, Any]:
        """执行四选一动作。返回 done / success / bumped 等信息。"""
        action = action.upper()
        bumped = False
        moved = False
        if action == ACTION_LEFT:
            self.heading = (self.heading - 1) % 4
        elif action == ACTION_RIGHT:
            self.heading = (self.heading + 1) % 4
        elif action == ACTION_FORWARD:
            nx, ny = self.front_cell()
            if self.is_free(nx, ny):
                self.x, self.y = nx, ny
                self.forward_steps += 1
                self.visited_cells.append((self.x, self.y))
                moved = True
            else:
                bumped = True
        elif action == ACTION_STOP:
            self._update_observation_and_graph()
            success = self.at_goal()
            return {
                "done": True,
                "success": success,
                "bumped": False,
                "moved": False,
                "reason": "stop_at_goal" if success else "stop_not_at_goal",
            }
        else:
            # 未知动作视为原地不动
            pass
        self._update_observation_and_graph()
        if self.at_goal():
            return {
                "done": True,
                "success": True,
                "bumped": bumped,
                "moved": moved,
                "reason": "reached_goal",
            }
        return {
            "done": False,
            "success": False,
            "bumped": bumped,
            "moved": moved,
            "reason": "continue",
        }


# ===========================================================================
# 2. Prompt 构造（严格仿 MapGPT 单专家 + 地图引导 + 自适应规划）
# ===========================================================================
def build_mapgpt_prompt(
    env: GridNavEnv,
    step_idx: int,
    history_lines: List[str],
    prev_planning: str,
) -> str:
    """
    组装 MapGPT 风格 prompt。
    论文输入：D, I, H_t, O_t, A_t, Trajectory, Map Connectivity, P_{t-1}
    论文输出：Thought + Action；本 Demo 额外强制输出全局拓扑地图文本，便于计算 AP。
    注意：不把全局真实地图塞进 prompt，只给局部观测与已探索区域。
    """
    history_block = (
        "The navigation has just begun, with no history."
        if not history_lines
        else "step records: " + " | ".join(history_lines)
    )
    fx, fy = env.front_cell()
    front_free = env.is_free(fx, fy)
    action_space = (
        "Action space (you MUST pick exactly one):\n"
        "A. stop\n"
        "B. forward   # move 1 cell along current heading"
        + ("  [FRONT IS WALL, forward will fail]" if not front_free else "  [front is FREE]")
        + "\n"
        "C. left      # turn 90 degrees counter-clockwise, stay in place\n"
        "D. right     # turn 90 degrees clockwise, stay in place"
    )
    if not prev_planning:
        prev_planning = "Navigation has just started, with no planning yet."

    prompt = f"""You are the single navigation expert in MapGPT (Map-Guided Prompting with Adaptive Path Planning) for a Room-to-Room (R2R) style indoor navigation task.

## Task Description
You control a wheeled robot on an unknown 2D occupancy grid (text-only local vision, no RGB image).
The robot cannot see the complete global map. You must maintain a linguistic topological map online,
perform adaptive multi-step path planning, and choose one primitive action per step.
Do not walk through walls. Stop only when you believe you are at the destination.

Definitions:
- Explored node: a Place the robot has already stood on.
- Accessible node: a neighboring Place the robot can enter by FORWARD from the current heading.
- Unexplored visible node: a free cell seen in the local window but not yet visited.
- Topological map uses Place IDs and connectivity sentences, not a secret full GPS dump of unseen cells.

## Instruction (language goal, R2R-style)
{env.instruction}

## Current Pose (assignment requirement; original MapGPT prefers topology over raw GPS)
robot_xy=({env.x}, {env.y})  heading={HEADING_NAME[env.heading]} ({HEADING_CN[env.heading]})  glyph={HEADING_GLYPH[env.heading]}
start_xy=({env.start[0]}, {env.start[1]})  destination_xy=({env.goal[0]}, {env.goal[1]})
step_index={step_idx}  max_steps={MAX_STEPS}  cells_walked={env.forward_steps}

## History H_t
{history_block}

## Observation O_t (LOCAL neighborhood only)
{env.last_obs_text}

## Explored region so far (still NOT the full map)
{env.render_explored_partial_map()}

## Trajectory of explored Places
{env.trajectory_place_text()}

## Map Connectivity (online graph built from what has been observed)
{env.system_topological_map_text()}

## Previous Planning P_{{t-1}}
{prev_planning}

## {action_space}

## Output requirements (STRICT, do not output any extra prose outside tags)
You MUST fill all four tags. The <MAP> block is your predicted GLOBAL topological map
(guess unseen rooms if needed, but mark guesses clearly). The <ACTION> value MUST be one of:
FORWARD, LEFT, RIGHT, STOP
(equivalently you may put A/B/C/D matching the action space).

<THINK>
Write a short navigation thought: where you are, whether to backtrack, which sub-goal is next.
</THINK>
<MAP>
First copy/update connectivity in MapGPT English template, one sentence per Place:
Place <id> is connected with Places <id>, <id>.
Then list cells you believe you know, one per line:
CELL <x>,<y>,FREE
CELL <x>,<y>,WALL
You may also draw an ASCII global sketch using # . R G ? (optional).
</MAP>
<PLAN>
Give an updated multi-step plan over Place IDs / actions, e.g. "go to Place 3, then Place 7, then STOP".
</PLAN>
<ACTION>
FORWARD
</ACTION>
"""
    return prompt.strip() + "\n"


# ===========================================================================
# 3. 网络请求：仅 requests POST，禁止 openai / dashscope
# ===========================================================================
def _extract_text_from_relay_payload(payload: Any) -> str:
    """兼容 Windows 中转可能返回的多种 JSON 包一层层剥到纯文本。"""
    if payload is None:
        return ""
    if isinstance(payload, str):
        s = payload.strip()
        if not s:
            return ""
        if s[:1] in "{[":
            try:
                return _extract_text_from_relay_payload(json.loads(s))
            except json.JSONDecodeError:
                return s
        return s
    if isinstance(payload, (int, float, bool)):
        return str(payload)
    if isinstance(payload, list):
        parts = [_extract_text_from_relay_payload(x) for x in payload]
        return "\n".join(p for p in parts if p)
    if isinstance(payload, dict):
        for key in (
            "response",
            "llm_response",
            "text",
            "content",
            "result",
            "message",
            "output",
            "answer",
            "data",
            "choices",
        ):
            if key in payload and payload[key] is not None:
                val = payload[key]
                if key == "choices" and isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, dict):
                        msg = first.get("message") or first
                        if isinstance(msg, dict) and "content" in msg:
                            return _extract_text_from_relay_payload(msg["content"])
                        if "text" in first:
                            return _extract_text_from_relay_payload(first["text"])
                return _extract_text_from_relay_payload(val)
        # 找不到常见字段则把字符串值拼起来
        bits = [str(v) for v in payload.values() if isinstance(v, str) and v.strip()]
        return "\n".join(bits)
    return str(payload)


def request_llm(prompt: str) -> Tuple[str, str]:
    """
    POST 到 Windows 中转接口。
    成功后 sleep SLEEP_AFTER_OK_SEC。
    HTTP 500 时加长等待再重试，耗尽重试后返回空文本，由主循环走单步兜底，不中断整场 20 局。
    """
    body = {"prompt": prompt}
    last_err = "unknown"
    attempts = MAX_RETRY + 1
    for i in range(attempts):
        try:
            resp = requests.post(
                LLM_URL,
                json=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=REQUEST_TIMEOUT_SEC,
            )
            if resp.status_code == 500:
                last_err = f"http_500:{resp.text[:200]}"
                print(f"  [warn] HTTP 500，{SLEEP_ON_HTTP500_SEC}s 后重试 ({i + 1}/{attempts})", flush=True)
                time.sleep(SLEEP_ON_HTTP500_SEC)
                continue
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}:{resp.text[:300]}"
                time.sleep(SLEEP_BETWEEN_RETRY)
                continue
            try:
                payload = resp.json()
            except ValueError:
                payload = resp.text
            text = _extract_text_from_relay_payload(payload).strip()
            if not text:
                last_err = "empty_body"
                time.sleep(SLEEP_BETWEEN_RETRY)
                continue
            time.sleep(SLEEP_AFTER_OK_SEC)  # 成功调用后降温
            return text, f"ok_attempt_{i + 1}"
        except requests.Timeout:
            last_err = "timeout"
            time.sleep(SLEEP_BETWEEN_RETRY)
        except requests.RequestException as exc:
            last_err = f"request_exception:{exc}"
            time.sleep(SLEEP_BETWEEN_RETRY)
        except Exception as exc:  # noqa: BLE001
            last_err = f"other:{exc}"
            time.sleep(SLEEP_BETWEEN_RETRY)
    return "", f"failed_after_retries:{last_err}"


# ===========================================================================
# 4. 健壮解析：拓扑块 + 动作；失败不崩溃
# ===========================================================================
_ACTION_SYNONYM = [
    (ACTION_FORWARD, r"\b(FORWARD|GO_FORWARD|MOVE_FORWARD|STRAIGHT|前进|直行|向前|往前)\b"),
    (ACTION_LEFT, r"\b(LEFT|TURN_LEFT|TURN-LEFT|左转|向左转|往左)\b"),
    (ACTION_RIGHT, r"\b(RIGHT|TURN_RIGHT|TURN-RIGHT|右转|向右转|往右)\b"),
    (ACTION_STOP, r"\b(STOP|WAIT|HALT|停止|停下|到达|结束)\b"),
]


def _strip_tag(text: str, tag: str) -> str:
    pat = re.compile(rf"<{tag}>(.*?)</{tag}>", re.IGNORECASE | re.DOTALL)
    m = pat.search(text or "")
    return m.group(1).strip() if m else ""


def parse_action(llm_text: str) -> Tuple[str, bool]:
    """
    从自由文本中抽出四选一动作。
    返回 (动作, 是否解析成功)。失败由调用方改用兜底动作。
    """
    if not llm_text:
        return ACTION_STOP, False
    tagged = _strip_tag(llm_text, "ACTION")
    candidates = [tagged, llm_text]

    for blob in candidates:
        if not blob:
            continue
        # 1) 标准单词
        up = blob.strip().upper()
        for act in VALID_ACTIONS:
            if re.fullmatch(rf"\s*{act}\s*", blob, flags=re.IGNORECASE):
                return act, True
        # 2) Action: B / 选项 A
        m = re.search(r"(?:ACTION|动作|选项)\s*[:：]?\s*([A-D])\b", blob, flags=re.IGNORECASE)
        if m:
            return ACTION_BY_LABEL[m.group(1).upper()], True
        m = re.search(r"^\s*([A-D])[\).:\s]", blob, flags=re.IGNORECASE | re.MULTILINE)
        if m and tagged:
            return ACTION_BY_LABEL[m.group(1).upper()], True
        # 3) 同义词
        for act, pat in _ACTION_SYNONYM:
            if re.search(pat, blob, flags=re.IGNORECASE):
                return act, True
        if up in ACTION_BY_LABEL:
            return ACTION_BY_LABEL[up], True

    # 全文再扫一次同义词（取最后一次出现，避免 THINK 里提前提到 STOP）
    last_act = None
    last_pos = -1
    for act, pat in _ACTION_SYNONYM:
        for m in re.finditer(pat, llm_text, flags=re.IGNORECASE):
            if m.start() >= last_pos:
                last_pos = m.start()
                last_act = act
    if last_act:
        return last_act, True
    return ACTION_STOP, False


def parse_predicted_cells(map_text: str) -> Dict[Tuple[int, int], str]:
    """从 <MAP> 中提取 CELL x,y,LABEL；并尝试把 ASCII 草图对齐。"""
    pred: Dict[Tuple[int, int], str] = {}
    if not map_text:
        return pred
    for m in re.finditer(
        r"CELL\s+(-?\d+)\s*,\s*(-?\d+)\s*,\s*(FREE|WALL|OBSTACLE|#|\.|G|R)",
        map_text,
        flags=re.IGNORECASE,
    ):
        x, y = int(m.group(1)), int(m.group(2))
        lab = m.group(3).upper()
        if lab in ("WALL", "OBSTACLE", "#"):
            pred[(x, y)] = "WALL"
        else:
            pred[(x, y)] = "FREE"
    # ASCII：寻找全由 #.?RG 空格组成的连续行
    ascii_rows: List[str] = []
    for line in map_text.splitlines():
        s = line.rstrip("\n")
        compact = s.replace(" ", "")
        if compact and re.fullmatch(r"[#.RGWFrg?\*]+", compact):
            ascii_rows.append(compact.replace("W", "#").replace("F", ".").replace("*", "."))
        elif ascii_rows and not compact:
            break
    if ascii_rows:
        robot_off = None
        for j, row in enumerate(ascii_rows):
            if "R" in row:
                robot_off = (row.index("R"), j)
                break
        # 若找到 R，则把该图对齐到当前机器人坐标；否则假设左上为 (0,0)
        origin_x, origin_y = 0, 0
        # 对齐需要调用处传入机器人位置；此处先用 (0,0)，主函数再二次对齐
        pred["_ascii_rows"] = ascii_rows  # type: ignore[assignment]
        pred["_ascii_robot"] = robot_off  # type: ignore[assignment]
    return pred


def parse_predicted_edges(map_text: str) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    if not map_text:
        return edges
    for m in re.finditer(
        r"Place\s+(\d+)\s+is connected with Places?\s*([0-9, \tandnone\(\)]+)",
        map_text,
        flags=re.IGNORECASE,
    ):
        src = int(m.group(1))
        tail = m.group(2)
        ids = [int(x) for x in re.findall(r"\d+", tail)]
        for dst in ids:
            if dst == src:
                continue
            a, b = (src, dst) if src <= dst else (dst, src)
            edges.add((a, b))
    return edges


def parse_llm_response(llm_text: str) -> Dict[str, Any]:
    """统一解析入口。任何异常都吞掉，保证主循环不崩。"""
    try:
        think = _strip_tag(llm_text, "THINK")
        plan = _strip_tag(llm_text, "PLAN")
        map_text = _strip_tag(llm_text, "MAP")
        if not map_text:
            # 没有标签时，尽量截取 Map: 到 Action 之间
            m = re.search(r"(Map\s*[:：].*?)(?:<ACTION>|Action\s*[:：]|\Z)", llm_text or "", re.IGNORECASE | re.DOTALL)
            map_text = m.group(1).strip() if m else (llm_text or "")
        action, ok = parse_action(llm_text)
        cells = parse_predicted_cells(map_text)
        ascii_rows = cells.pop("_ascii_rows", None)  # type: ignore[arg-type]
        ascii_robot = cells.pop("_ascii_robot", None)  # type: ignore[arg-type]
        edges = parse_predicted_edges(map_text)
        return {
            "think": think,
            "plan": plan,
            "map_text": map_text,
            "action": action,
            "parse_ok": ok,
            "pred_cells": cells,
            "ascii_rows": ascii_rows,
            "ascii_robot": ascii_robot,
            "pred_edges": edges,
        }
    except Exception:  # noqa: BLE001
        return {
            "think": "",
            "plan": "",
            "map_text": llm_text or "",
            "action": ACTION_STOP,
            "parse_ok": False,
            "pred_cells": {},
            "ascii_rows": None,
            "ascii_robot": None,
            "pred_edges": set(),
        }


def fallback_action(env: GridNavEnv) -> str:
    """
    解析失败 / 请求失败时的兜底策略：面向目标的贪心差动，不直接崩溃。
    已在终点则 STOP；否则转向目标主方向，前方可走就 FORWARD。
    """
    if env.at_goal():
        return ACTION_STOP
    dx = env.goal[0] - env.x
    dy = env.goal[1] - env.y
    # 选择需要减小的主轴
    if abs(dx) >= abs(dy):
        desired = 1 if dx > 0 else 3  # 东 or 西
        if dx == 0:
            desired = 2 if dy > 0 else 0
    else:
        desired = 2 if dy > 0 else 0  # 南 or 北
        if dy == 0:
            desired = 1 if dx > 0 else 3
    if env.heading == desired:
        fx, fy = env.front_cell()
        if env.is_free(fx, fy):
            return ACTION_FORWARD
        return ACTION_LEFT  # 前方堵住，尝试转向绕行
    # 选择更近的转向
    diff = (desired - env.heading) % 4
    return ACTION_RIGHT if diff == 1 else ACTION_LEFT


# ===========================================================================
# 5. 指标计算
# ===========================================================================
def align_ascii_cells(
    ascii_rows: Optional[List[str]],
    ascii_robot: Any,
    env: GridNavEnv,
) -> Dict[Tuple[int, int], str]:
    """把 LLM 画的 ASCII 草图对齐到世界坐标。"""
    out: Dict[Tuple[int, int], str] = {}
    if not ascii_rows:
        return out
    if ascii_robot:
        rx, ry = ascii_robot
        origin_x = env.x - rx
        origin_y = env.y - ry
    else:
        origin_x, origin_y = 0, 0
    for j, row in enumerate(ascii_rows):
        for i, ch in enumerate(row):
            wx, wy = origin_x + i, origin_y + j
            if ch in "#":
                out[(wx, wy)] = "WALL"
            elif ch in ".RG*":
                out[(wx, wy)] = "FREE"
    return out


def compute_ap_step(parsed: Dict[str, Any], env: GridNavEnv) -> float:
    """
    单步 AP：预测拓扑 vs 仿真真实地图。
    综合 (1) 预测格子标签与真实占用  (2) 预测 Place 边与系统观测图的边。
    两边都缺则本步 AP=0。
    """
    pred_cells: Dict[Tuple[int, int], str] = dict(parsed.get("pred_cells") or {})
    pred_cells.update(align_ascii_cells(parsed.get("ascii_rows"), parsed.get("ascii_robot"), env))
    cell_scores: List[int] = []
    for (x, y), lab in pred_cells.items():
        if not isinstance(x, int):
            continue
        if not env.in_bounds(x, y):
            continue
        truth = env.true_cell_label(x, y)
        cell_scores.append(1 if lab == truth else 0)
    cell_ap = (sum(cell_scores) / len(cell_scores)) if cell_scores else None

    pred_edges: Set[Tuple[int, int]] = set(parsed.get("pred_edges") or set())
    true_edges = set(env.topo_edges)
    known = set(env.topo_nodes.values())
    pred_edges = {e for e in pred_edges if e[0] in known and e[1] in known}
    edge_ap = None
    if pred_edges or true_edges:
        inter = pred_edges & true_edges
        union = pred_edges | true_edges
        edge_ap = (len(inter) / len(union)) if union else 0.0  # Jaccard

    parts = [p for p in (cell_ap, edge_ap) if p is not None]
    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))


def compute_spl(shortest: int, actual: int) -> float:
    """单条成功轨迹 SPL。路径长度单位：格子前进次数。"""
    if shortest <= 0:
        shortest = 1
    if actual <= 0:
        actual = shortest
    return float(shortest) / float(max(actual, shortest))


def aggregate_metrics(episode_records: List[Dict[str, Any]]) -> Dict[str, float]:
    n = len(episode_records) if episode_records else 1
    succ = [e for e in episode_records if e.get("success")]
    sr = len(succ) / float(n)
    if succ:
        spl = sum(compute_spl(e["shortest"], e["actual"]) for e in succ) / float(len(succ))
    else:
        spl = 0.0
    ap = sum(float(e.get("ap", 0.0)) for e in episode_records) / float(n)
    return {"SR": sr, "SPL": spl, "AP": ap, "success_count": float(len(succ)), "total": float(n)}


# ===========================================================================
# 6. 日志
# ===========================================================================
class RunLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write("# MapGPT VM Demo log\n")
        self._fh.write(f"# LLM_URL={LLM_URL}\n")
        self._fh.write("# 本文件逐行记录 episode / 坐标 / 局部观测 / 完整 prompt / LLM 原文 / 动作 / 成败\n\n")
        self._fh.flush()

    def write(self, text: str) -> None:
        self._fh.write(text)
        if not text.endswith("\n"):
            self._fh.write("\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _clip(s: str, n: int = 8000) -> str:
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n...[truncated {len(s) - n} chars]..."


# ===========================================================================
# 7. Episode 主循环
# ===========================================================================
def run_one_episode(ep_id: int, logger: RunLogger) -> Dict[str, Any]:
    env = GridNavEnv()
    env.generate(EPISODE_SEED_BASE + ep_id * 17)
    history_lines: List[str] = []
    prev_planning = "Navigation has just started, with no planning yet."
    step_aps: List[float] = []
    success = False
    fail_reason = "max_steps"
    logger.write("=" * 88)
    logger.write(
        f"EPISODE {ep_id:02d}  start={env.start} goal={env.goal} shortest={env.shortest_len} "
        f"instruction={env.instruction}"
    )

    for step_idx in range(MAX_STEPS):
        obs = env.last_obs_text
        pose = f"({env.x},{env.y},{HEADING_NAME[env.heading]})"
        prompt = build_mapgpt_prompt(env, step_idx, history_lines, prev_planning)
        llm_text, net_status = request_llm(prompt)
        parsed = parse_llm_response(llm_text)
        used_fallback = False
        if (not llm_text) or (not parsed["parse_ok"]):
            action = fallback_action(env)
            used_fallback = True
        else:
            action = parsed["action"]
            if action not in VALID_ACTIONS:
                action = fallback_action(env)
                used_fallback = True

        ap_step = compute_ap_step(parsed, env)
        step_aps.append(ap_step)
        if parsed.get("plan"):
            prev_planning = parsed["plan"]

        result = env.step(action)
        flag = "RUNNING"
        if result["done"]:
            success = bool(result["success"])
            fail_reason = result.get("reason", "")
            flag = "SUCCESS" if success else "FAIL"

        logger.write("-" * 72)
        logger.write(f"episode={ep_id:02d} step={step_idx:02d} pose={pose} action={action} flag={flag}")
        logger.write(f"local_observation:\n{obs}")
        logger.write(f"net_status={net_status} parse_ok={parsed['parse_ok']} fallback={used_fallback} ap_step={ap_step:.4f}")
        logger.write("PROMPT_BEGIN")
        logger.write(prompt)
        logger.write("PROMPT_END")
        logger.write("LLM_BEGIN")
        logger.write(llm_text if llm_text else f"<EMPTY llm_text status={net_status}>")
        logger.write("LLM_END")
        logger.write(
            f"parsed_map_head:\n{_clip(parsed.get('map_text') or '', 2000)}\n"
            f"think={_clip(parsed.get('think') or '', 500)}\n"
            f"plan={_clip(parsed.get('plan') or '', 500)}"
        )

        history_lines.append(
            f"step {step_idx}: {action}"
            + (" (bumped)" if result.get("bumped") else "")
            + f" -> ({env.x},{env.y},{HEADING_NAME[env.heading]})"
        )
        # 控制台进度（短信息，完整内容在 log.txt）
        print(
            f"[EP {ep_id:02d} STEP {step_idx:02d}] pose={pose} act={action} "
            f"fb={int(used_fallback)} AP={ap_step:.4f} {flag} net={net_status}",
            flush=True,
        )
        if result["done"]:
            break
    else:
        success = env.at_goal()
        fail_reason = "reached_goal" if success else "max_steps"

    ap_ep = sum(step_aps) / float(len(step_aps)) if step_aps else 0.0
    rec = {
        "episode": ep_id,
        "success": success,
        "reason": fail_reason,
        "shortest": int(env.shortest_len),
        "actual": int(env.forward_steps),
        "ap": ap_ep,
        "end_pose": (env.x, env.y, env.heading),
    }
    logger.write(
        f"EPISODE {ep_id:02d} END success={success} reason={fail_reason} "
        f"shortest={env.shortest_len} actual_forward={env.forward_steps} AP={ap_ep:.4f} "
        f"end={rec['end_pose']}"
    )
    logger.write("")
    return rec


def run_all_episodes() -> Dict[str, Any]:
    logger = RunLogger(LOG_PATH)
    records: List[Dict[str, Any]] = []
    try:
        logger.write(f"run_start time={time.strftime('%Y-%m-%d %H:%M:%S')} episodes={NUM_EPISODES} max_steps={MAX_STEPS}")
        for ep in range(NUM_EPISODES):
            try:
                rec = run_one_episode(ep, logger)
            except Exception as exc:  # 单局异常不中断后续局
                traceback.print_exc()
                rec = {
                    "episode": ep,
                    "success": False,
                    "reason": f"episode_exception:{exc}",
                    "shortest": 1,
                    "actual": 0,
                    "ap": 0.0,
                    "end_pose": None,
                }
                logger.write(f"EPISODE {ep:02d} EXCEPTION {exc}\n{traceback.format_exc()}")
            records.append(rec)
            print(
                f"--> episode {ep:02d} done success={rec['success']} "
                f"SR_so_far={sum(1 for r in records if r['success'])}/{len(records)}",
                flush=True,
            )
        metrics = aggregate_metrics(records)
        logger.write("=" * 88)
        logger.write(
            "SUMMARY SR={SR:.4f} SPL={SPL:.4f} AP={AP:.4f} success={success_count:.0f}/{total:.0f}".format(**metrics)
        )
        return {"records": records, "metrics": metrics}
    finally:
        logger.close()


def print_summary(bundle: Dict[str, Any]) -> None:
    metrics = bundle["metrics"]
    records: List[Dict[str, Any]] = bundle["records"]
    print("\n" + "=" * 64)
    print("MapGPT VM Demo 汇总（原理验证，不与论文原版数值对标）")
    print("=" * 64)
    print(f"LLM 中转: {LLM_URL}")
    print(f"日志文件: {LOG_PATH}")
    print(f"总 episode: {int(metrics['total'])}   成功: {int(metrics['success_count'])}")
    print(f"SR  = {metrics['SR']:.4f}    # 成功数 / 20")
    print(f"SPL = {metrics['SPL']:.4f}    # 仅成功局: shortest / max(actual, shortest) 的平均")
    print(f"AP  = {metrics['AP']:.4f}    # 预测拓扑 vs 真实地图，先逐步再对局平均")
    print("-" * 64)
    print(f"{'ep':>4} {'succ':>5} {'short':>6} {'walk':>6} {'spl_i':>7} {'ap':>7} reason")
    for r in records:
        spl_i = compute_spl(r["shortest"], r["actual"]) if r["success"] else 0.0
        print(
            f"{r['episode']:4d} {str(r['success']):>5} {r['shortest']:6d} {r['actual']:6d} "
            f"{spl_i:7.3f} {r['ap']:7.3f} {r.get('reason','')}"
        )
    print("=" * 64)
    print("说明：自制网格非 Matterport；无训练微调；指标只验证范式。")


def main() -> int:
    # 启动自检：禁止误把 openai 写进本文件依赖
    assert "openai" not in sys.modules, "本 Demo 禁止使用 openai 模块"
    print("MapGPT VM Demo 启动")
    print(f"  POST {LLM_URL}")
    print(f"  日志 {LOG_PATH}")
    print(f"  {NUM_EPISODES} episodes × 最多 {MAX_STEPS} 步")
    print("  Windows Flask 中转: 见 LLM_URL；成功调用后 sleep 2.5s")
    bundle = run_all_episodes()
    print_summary(bundle)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断。已写部分 log.txt。", file=sys.stderr)
        sys.exit(130)
