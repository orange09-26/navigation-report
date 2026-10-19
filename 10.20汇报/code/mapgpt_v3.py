# -*- coding: utf-8 -*-
"""
MapGPT v3 —— 在原版 2D 复现基础上做三项升级，并输出完整的逐步评分明细。

升级点：
  U1 锚点系统(Anchor)     : 在"已观测拓扑图"上做 A* 式排序，给出 Top-3 子目标锚点，
                            并把终点坐标翻译成自然语言方位(bearing)，规避 LLM 不擅长坐标运算的问题。
  U2 未探索路径入决策(Frontier): 显式列出所有 frontier(已观测但未走过的可走格)、死胡同标记、
                            每个方向的未探索格数量，并写入硬规则"此路不通立即切换次优候选"。
  U3 防卡死(Anti-Stuck)   : 三重检测器(震荡/重复访问/停滞) -> 提示层警告 -> 连续 3 次无效后
                            触发确定性逃逸(guard override)，全部记录在案，不隐藏。

  U4 停止自检门(Stop Gate) : STOP 只有在 "robot 坐标==指令公开的 destination 坐标" 或
                            "局部观测里看得见 G 标记" 时才放行；否则驳回并要求继续导航，
                            每局最多驳回 3 次(超过后放行以免死锁)。全部记录在案。

评分：所有公式见 metrics 部分注释与 SCORING.md。逐步明细写入 steps.json / steps.csv。
原版 mapgpt_vm_demo.py 一行未改，旧结果完全保留。
"""
import os, sys, json, csv, time, argparse
from collections import deque, Counter

sys.path.insert(0, "/home/orange")
import mapgpt_vm_demo as d

GridNavEnv = d.GridNavEnv
HEADING_DELTA = d.HEADING_DELTA
HEADING_NAME = d.HEADING_NAME

# ---------------------------------------------------------------- 可调参数
CFG = {
    "anchor_topk": 3,          # 锚点候选数量
    "stall_window": 6,         # 连续多少步没有缩短到终点距离 => 停滞
    "revisit_limit": 3,        # 同一格访问次数达到该值 => 重复访问告警
    "osc_window": 4,           # 震荡检测窗口
    "guard_after": 3,          # 连续 N 步处于卡死状态且无新格 => 强制逃逸
    "guard_budget": 5,         # 每局最多强制逃逸次数(防止 guard 替模型跑完全程)
    "stop_reject_cap": 3,      # 每局最多驳回多少次非法 STOP(超过后放行，避免死锁)
    "weights": {"P": 0.30, "V": 0.20, "M": 0.25, "C": 0.15, "G": 0.10},
}


# =========================================================================
# A. 代理侧计算(只用"已观测"信息，不偷看真值地图)
# =========================================================================
def observed_free_cells(env):
    """系统已观测到的可走格(= 拓扑图节点)。"""
    return set(env.topo_nodes.keys())


def known_graph_bfs(env, src):
    """在已观测可走格上做四连通 BFS，返回 {cell: dist}。不可达的格不出现。"""
    nodes = observed_free_cells(env)
    if src not in nodes:
        # 机器人当前格一定被观测过；保险处理
        nodes = nodes | {src}
    dist = {src: 0}
    q = deque([src])
    while q:
        c = q.popleft()
        for dx, dy in HEADING_DELTA:
            n = (c[0] + dx, c[1] + dy)
            if n in nodes and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return dist


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def frontier_cells(env, visited_set):
    """frontier = 已观测、未走过、且在已知图上从当前位置可达的格。"""
    cur = (env.x, env.y)
    dist = known_graph_bfs(env, cur)
    out = []
    for c in observed_free_cells(env):
        if c in visited_set or c == cur:
            continue
        if c not in dist:
            continue
        out.append((c, dist[c]))
    return out, dist


def rank_anchors(env, visited_set, topk=None):
    """
    锚点排序：score(f) = d_known(robot -> f) + manhattan(f -> goal)
    这是 A* 式的代价估计，d_known 只用已观测图，manhattan 用指令中公开的终点坐标。
    返回 [(cell, d_known, h_manhattan, score)] 升序。
    """
    topk = topk or CFG["anchor_topk"]
    fr, _ = frontier_cells(env, visited_set)
    scored = []
    for c, dk in fr:
        h = manhattan(c, env.goal)
        scored.append((c, dk, h, dk + h))
    scored.sort(key=lambda t: (t[3], t[2], t[0]))
    return scored[:topk]


def dead_end_cells(env, visited_set):
    """已走过、且四邻全是墙或已走过的格 => 确认死胡同，不要再规划进去。"""
    out = []
    for c in visited_set:
        if c == (env.x, env.y):
            continue
        free_nb = [n for n in _neighbors(env, c) if env.is_free(*n)]
        unvisited = [n for n in free_nb if n not in visited_set]
        if not unvisited:
            out.append(c)
    return sorted(out)


def _neighbors(env, c):
    for dx, dy in HEADING_DELTA:
        n = (c[0] + dx, c[1] + dy)
        if env.in_bounds(*n):
            yield n


REL_BEARING = ("front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left")


def bearing_text(env):
    """把终点坐标翻译成自然语言方位。论文的核心教训之一：LLM 读关系比读坐标强。"""
    dx = env.goal[0] - env.x
    dy = env.goal[1] - env.y
    dist = abs(dx) + abs(dy)
    if dist == 0:
        return "You are exactly on the destination cell. STOP now."
    # 世界坐标 -> 8 方位角(y 向下为正)
    import math
    ang = (math.degrees(math.atan2(dy, dx))) % 360.0     # 0=E, 90=S, 180=W, 270=N
    world_idx = int(((ang + 22.5) % 360) // 45)          # 0=E,1=SE,2=S,3=SW,4=W,5=NW,6=N,7=NE
    # heading: 0=N,1=E,2=S,3=W  => 世界方位序号(以 E 为 0)
    heading_world = {0: 6, 1: 0, 2: 2, 3: 4}[env.heading]
    rel = REL_BEARING[(world_idx - heading_world) % 8]
    ns = "south" if dy > 0 else ("north" if dy < 0 else "same row")
    ew = "east" if dx > 0 else ("west" if dx < 0 else "same column")
    return ("The destination is %d cells away in a straight-line grid path, "
            "on your %s (it is %d cells %s and %d cells %s of you)."
            % (dist, rel, abs(dy), ns, abs(dx), ew))


def unexplored_direction_counts(env):
    """四个绝对方向上：紧邻格状态 + 该方向已观测未走过的格数量。"""
    visited = set(env.visited_cells)
    obs = observed_free_cells(env)
    lines = []
    for h in range(4):
        dx, dy = HEADING_DELTA[h]
        n = (env.x + dx, env.y + dy)
        if not env.in_bounds(*n):
            st = "OUT_OF_BOUNDS"
        elif not env.is_free(*n):
            st = "WALL"
        elif n in visited:
            st = "FREE_ALREADY_VISITED"
        else:
            st = "FREE_NEVER_VISITED"
        # 该方向半平面内已观测未走过的格数
        cnt = 0
        for c in obs:
            if c in visited:
                continue
            if dx != 0 and (c[0] - env.x) * dx > 0:
                cnt += 1
            elif dy != 0 and (c[1] - env.y) * dy > 0:
                cnt += 1
        rel = {0: "front", 1: "right", 2: "back", 3: "left"}[(h - env.heading) % 4]
        lines.append("%-5s (%s): adjacent=%s, unexplored_observed_cells_in_that_direction=%d"
                     % (HEADING_NAME[h], rel, st, cnt))
    return "\n".join(lines)


# =========================================================================
# B. 防卡死检测
# =========================================================================
def detect_stuck(env, visited_counter, dist_hist):
    """返回触发中的检测器名列表。全部基于代理自身轨迹，不用真值。"""
    flags = []
    traj = env.visited_cells
    # D1 震荡: 最近 4 个位置构成 A-B-A-B
    w = CFG["osc_window"]
    if len(traj) >= w:
        last = traj[-w:]
        if last[0] == last[2] and last[1] == last[3] and last[0] != last[1]:
            flags.append("OSCILLATION(%s<->%s)" % (last[0], last[1]))
    # D2 重复访问
    vc = visited_counter.get((env.x, env.y), 0)
    if vc >= CFG["revisit_limit"]:
        flags.append("REVISIT(cell=%s,count=%d)" % ((env.x, env.y), vc))
    # D3 停滞: 连续 stall_window 步没有缩短到终点的曼哈顿距离
    if len(dist_hist) > CFG["stall_window"]:
        recent = dist_hist[-CFG["stall_window"]:]
        if min(recent) >= dist_hist[-CFG["stall_window"] - 1]:
            flags.append("STALL(no_closer_for_%d_steps)" % CFG["stall_window"])
    return flags


def guard_escape_action(env, visited_counter):
    """
    确定性逃逸：在四个自由邻格中选 (访问次数最少, 其次离终点曼哈顿最近) 的一格，
    转向它然后前进。只输出一个动作，逐步收敛，不会瞬移。
    """
    cur = (env.x, env.y)
    cands = []
    for h in range(4):
        dx, dy = HEADING_DELTA[h]
        n = (cur[0] + dx, cur[1] + dy)
        if env.in_bounds(*n) and env.is_free(*n):
            cands.append((visited_counter.get(n, 0), manhattan(n, env.goal), h, n))
    if not cands:
        return d.ACTION_LEFT, None
    cands.sort(key=lambda t: (t[0], t[1], t[2]))
    want_h = cands[0][2]
    target = cands[0][3]
    if env.heading == want_h:
        return d.ACTION_FORWARD, target
    diff = (want_h - env.heading) % 4
    return (d.ACTION_RIGHT if diff == 1 else d.ACTION_LEFT), target


# =========================================================================
# C. 升级后的 Prompt(在原版 prompt 上注入新块，原版信息一字不删)
# =========================================================================
def stop_allowed(env):
    """
    U4 停止自检门（2026-10-19 修正）。只用代理自己已有的信息，不偷看真值：
    唯一放行条件：指令里公开写明的 destination 坐标 == 当前 robot 坐标。

    修正说明：旧版还允许「局部观测里看得见 G 标记」就停，但 G 在 5x5 窗口里
    出现只代表「终点在附近」，不代表「站在终点上」。旧判断用字符串 "GOAL"
    匹配，而该词只在终点位于正交四邻格时才出现；结果斜对角的 G 会被误判为
    「没看见」，驳回理由与事实相反。现在统一改为坐标判定，避免口径不一致。
    返回 (是否允许 STOP, 理由)。
    """
    if (env.x, env.y) == tuple(env.goal):
        return True, "standing_on_destination_cell"
    gap = manhattan((env.x, env.y), tuple(env.goal))
    return False, ("robot_xy=(%d,%d) != destination_xy=(%d,%d); you are still %d cells away "
                   "(Manhattan grid distance). A G marker in the local view only means the goal is nearby, "
                   "not that you are standing on it."
                   % (env.x, env.y, env.goal[0], env.goal[1], gap))


def build_prompt_v3(env, step_idx, history_lines, prev_planning, visited_counter,
                    stuck_flags, guard_used, subgoal_prev, stop_warn=""):
    base = d.build_mapgpt_prompt(env, step_idx, history_lines, prev_planning)
    visited_set = set(env.visited_cells)
    anchors = rank_anchors(env, visited_set)
    fr, _ = frontier_cells(env, visited_set)
    dead = dead_end_cells(env, visited_set)
    pid = env.topo_nodes

    L = []
    L.append("## [v3-U1] Anchor sub-goals (ranked on the OBSERVED graph only)")
    L.append("ranking formula: score(f) = d_known(robot->f) + manhattan(f->destination); smaller is better")
    if anchors:
        for i, (c, dk, h, sc) in enumerate(anchors):
            L.append("  ANCHOR_%d: Place %s at cell (%d,%d) | d_known=%d | manhattan_to_goal=%d | score=%d"
                     % (i + 1, pid.get(c, "?"), c[0], c[1], dk, h, sc))
    else:
        L.append("  (no frontier left: every observed free cell has been visited)")
    L.append("bearing to destination in plain words: " + bearing_text(env))
    L.append("")

    L.append("## [v3-U2] Unexplored routes you must take into account")
    L.append("total frontier cells (observed but NEVER visited) = %d" % len(fr))
    L.append(unexplored_direction_counts(env))
    if dead:
        L.append("confirmed dead ends (all neighbours are wall or already visited, do NOT plan through them): "
                 + ", ".join("(%d,%d)" % c for c in dead[:12]))
    rep = [(c, n) for c, n in visited_counter.items() if n >= 2]
    if rep:
        rep.sort(key=lambda t: -t[1])
        L.append("cells you already stepped on more than once: "
                 + ", ".join("(%d,%d)x%d" % (c[0], c[1], n) for c, n in rep[:12]))
    L.append("")

    L.append("## [v3-U3] Decision rules (mandatory)")
    L.append("R1. Prefer moving toward ANCHOR_1. If ANCHOR_1 is unreachable or turns out to be a dead end,")
    L.append("    switch to ANCHOR_2, then ANCHOR_3 -- take the second-best route instead of retrying.")
    L.append("R2. NEVER repeat the immediately previous move if it did not put you on a new cell.")
    L.append("R3. If forward is blocked by a wall, do not spam FORWARD: turn, and re-plan from the ranked anchors.")
    L.append("R4. Keep exploring while frontier cells remain. STOP is allowed ONLY when")
    L.append("    robot_xy == destination_xy (standing on the G cell). Seeing G in the 5x5 window")
    L.append("    is NOT enough: walk onto that cell first, then STOP.")
    L.append("R5. Do not enter a cell marked as a confirmed dead end unless every other option is exhausted.")
    L.append("R6. STOP is verified by a self-check on coordinates. If you STOP while")
    L.append("    robot_xy != destination_xy, the STOP is REJECTED and you lose one step.")
    L.append("")

    if stop_warn:
        L.append("## [v3-U4] STOP REJECTED")
        L.append(stop_warn)
        L.append("You are NOT at the destination. Pick an anchor and keep moving. Do not output STOP again")
        L.append("until the self-check passes.")
        L.append("")

    if stuck_flags:
        L.append("## [v3-U3] STUCK WARNING  (detectors fired: %s)" % "; ".join(stuck_flags))
        L.append("You are looping. guard_overrides_used_so_far=%d/%d." % (guard_used, CFG["guard_budget"]))
        L.append("You MUST pick a DIFFERENT sub-goal from the ranked anchor list and move to a cell you have")
        L.append("never stood on. Repeating the same two cells will be penalised in your per-step score.")
        L.append("")

    if subgoal_prev:
        L.append("## Your committed sub-goal last step: cell (%d,%d)" % (subgoal_prev[0], subgoal_prev[1]))
        L.append("If it is still valid, keep it. Change it only when it is reached or proven unreachable.")
        L.append("")

    L.append("## [v3] Extra output requirement")
    L.append("Add ONE more tag before <ACTION>, containing exactly the cell you commit to as sub-goal:")
    L.append("<SUBGOAL>x,y</SUBGOAL>   (use one of the ANCHOR cells; write 'NONE' only if you are stopping)")
    L.append("")

    extra = "\n".join(L)
    marker = "## Output requirements"
    if marker in base:
        return base.replace(marker, extra + marker, 1)
    return base + "\n\n" + extra


# =========================================================================
# D. 评分（全部公式在此，逐步计算）
# =========================================================================
def true_bfs_dist_map(env):
    """评测侧真值：从每个可走格到终点的最短步数。只用于打分，绝不进 prompt。"""
    g = env.grid
    dist = {env.goal: 0}
    q = deque([env.goal])
    while q:
        x, y = q.popleft()
        for dx, dy in HEADING_DELTA:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.width and 0 <= ny < env.height and g[ny][nx] == d.FREE:
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))
    return dist


def score_step(action, source, bumped, moved, d_before, d_after, visit_count_after,
               ap_step, stuck_flags, guard_override, subgoal_now, subgoal_prev,
               subgoal_reached, subgoal_gone, ended, ended_at_goal):
    """
    单步评分，五个分项，全部落在 [0,1]。

    P 进度项  : 用真值 BFS 距离 d(到终点的最短步数)。
                P = clip((d_before - d_after + 1) / 2, 0, 1)
                即靠近一步=1.0，原地不变=0.5，走远一步=0.0。
                本步导致整局结束时: 结束在终点=1.0，结束在非终点=0.0。
                (注意: FORWARD 踩上终点也会结束整局，同样记 1.0，不能只看动作是不是 STOP。)
    V 新颖项  : 首次踏入新格=1.0；踏入走过的格=0.5；原地转向(未撞墙)=0.25；撞墙=0.0。
    M 建图项  : = AP_step，模型本步自建地图与真值地图的一致度(见 compute_ap_step)。
    C 干净项  : 本步无卡死告警且无强制逃逸=1.0；有告警但未被接管=0.5；被 guard 接管=0.0。
    G 承诺项  : 子目标与上一步相同=1.0；上一步子目标已到达=1.0；上一步子目标已失效(不再是
                frontier)=0.75；无理由改换子目标(反复横跳)=0.0；未解析出=0.5(中性)。

    DS = 0.30*P + 0.20*V + 0.25*M + 0.15*C + 0.10*G
    """
    w = CFG["weights"]
    if ended:
        P = 1.0 if ended_at_goal else 0.0
    elif d_before is None or d_after is None:
        P = 0.5
    else:
        P = max(0.0, min(1.0, (d_before - d_after + 1) / 2.0))

    if bumped:
        V = 0.0
    elif moved and visit_count_after <= 1:
        V = 1.0
    elif moved:
        V = 0.5
    else:
        V = 0.25   # 原地转向

    M = float(ap_step)

    if guard_override:
        C = 0.0
    elif stuck_flags:
        C = 0.5
    else:
        C = 1.0

    if subgoal_now is None or subgoal_prev is None:
        G = 0.5
    elif subgoal_now == subgoal_prev:
        G = 1.0
    elif subgoal_reached:
        G = 1.0
    elif subgoal_gone:
        G = 0.75
    else:
        G = 0.0

    DS = w["P"] * P + w["V"] * V + w["M"] * M + w["C"] * C + w["G"] * G
    return {"P": round(P, 4), "V": round(V, 4), "M": round(M, 4), "C": round(C, 4),
            "G": round(G, 4), "DS": round(DS, 4)}


def episode_metrics(recs, n_total):
    """
    局级指标公式：
      SR   = (1/N) * sum(S_i)                       S_i=1 若成功到达并停止
      SPL  = (1/N) * sum( S_i * l_i / max(p_i,l_i) )   <- Anderson 2018 标准定义，失败局记 0
      SPLs = (1/N_succ) * sum_{S_i=1} l_i/max(p_i,l_i) <- 仅成功局均值(旧版口径，一并给出便于对比)
      OSR  = (1/N) * sum( 1[终点出现在轨迹中] )
      NE   = (1/N) * sum( d_BFS(最终位置, 终点) )      不可达时用曼哈顿距离代替
      AP   = (1/N) * sum( (1/T_i) * sum_t AP_{i,t} )
      DS   = (1/N) * sum( (1/T_i) * sum_t DS_{i,t} )   逐步综合得分的局均
      ER   = (1/N) * sum( l_i / max(p_i,l_i) )         路径效率比(不看成败)
      COV  = (1/N) * sum( |已观测可走格| / |全部可走格| )
      STK  = sum(触发卡死告警的步数) / sum(总步数)
      GRD  = sum(guard 强制逃逸次数)
    l_i = BFS 最短路径长度(格边数)，p_i = 实际 FORWARD 成功次数。
    """
    N = float(n_total) if n_total else 1.0
    succ = [r for r in recs if r["success"]]
    sr = len(succ) / N
    spl = sum(r["shortest"] / float(max(r["actual"], r["shortest"], 1)) for r in succ) / N
    spls = (sum(r["shortest"] / float(max(r["actual"], r["shortest"], 1)) for r in succ) / len(succ)) if succ else 0.0
    osr = sum(1 for r in recs if r["osr"]) / N
    ne = sum(r["ne"] for r in recs) / N
    ap = sum(r["ap"] for r in recs) / N
    ds = sum(r["ds_mean"] for r in recs) / N
    er = sum(r["shortest"] / float(max(r["actual"], r["shortest"], 1)) for r in recs) / N
    cov = sum(r["coverage"] for r in recs) / N
    steps = sum(r["steps"] for r in recs)
    stk = (sum(r["stuck_steps"] for r in recs) / float(steps)) if steps else 0.0
    grd = sum(r["guard_count"] for r in recs)
    return {"SR": sr, "SPL": spl, "SPL_succ_only": spls, "OSR": osr, "NE": ne, "AP": ap,
            "DS": ds, "ER": er, "COV": cov, "STK": stk, "GRD": grd,
            "STOPREJ": sum(r.get("stop_rejected", 0) for r in recs),
            "LLMCALLS": sum(r.get("llm_calls", 0) for r in recs),
            "steps_total": steps, "N": n_total}


# =========================================================================
# E. 单局主循环
# =========================================================================
def parse_subgoal(text):
    raw = d._strip_tag(text, "SUBGOAL") if text else ""
    raw = (raw or "").strip()
    if not raw or raw.upper().startswith("NONE"):
        return None
    import re
    m = re.search(r"(-?\d+)\s*,\s*(-?\d+)", raw)
    return (int(m.group(1)), int(m.group(2))) if m else None


def run_episode_v3(ep_id, logger, grid_h, grid_w, seed_base):
    env = GridNavEnv(height=grid_h, width=grid_w)
    env.generate(seed_base + ep_id * 17)
    truth_dist = true_bfs_dist_map(env)
    free_total = sum(1 for y in range(env.height) for x in range(env.width) if env.grid[y][x] == d.FREE)

    history_lines, prev_planning = [], "Navigation has just started, with no planning yet."
    visited_counter = Counter(env.visited_cells)
    dist_hist = [manhattan(env.start, env.goal)]
    step_rows, aps = [], []
    subgoal_prev = None
    stuck_streak = 0
    guard_used = 0
    stop_rejects = 0
    llm_calls_total = 0
    success, reason = False, "max_steps"
    t0 = time.time()

    logger.write("=" * 92)
    logger.write("EPISODE %02d [v3] start=%s goal=%s shortest=%d grid=%dx%d seed=%d"
                 % (ep_id, env.start, env.goal, env.shortest_len, grid_h, grid_w, seed_base + ep_id * 17))
    logger.write("instruction=%s" % env.instruction)

    for step_idx in range(d.MAX_STEPS):
        pose_before = (env.x, env.y, env.heading)
        d_before = truth_dist.get((env.x, env.y))
        stuck_flags = detect_stuck(env, visited_counter, dist_hist)

        # U4: STOP 自检门。非法 STOP 被驳回，并在同一步内带着驳回理由重新询问模型。
        stop_warn = ""
        while True:
            prompt = build_prompt_v3(env, step_idx, history_lines, prev_planning,
                                     visited_counter, stuck_flags, guard_used, subgoal_prev, stop_warn)
            llm_text, net_status = d.request_llm(prompt)
            llm_calls_total += 1
            parsed = d.parse_llm_response(llm_text)
            source, guard_override = "LLM", False
            if (not llm_text) or (not parsed["parse_ok"]) or (parsed["action"] not in d.VALID_ACTIONS):
                action = d.fallback_action(env)
                source = "FALLBACK" if llm_text else "NETFAIL"
                break
            action = parsed["action"]
            if action == d.ACTION_STOP and stop_rejects < CFG["stop_reject_cap"]:
                ok, why = stop_allowed(env)
                if not ok:
                    stop_rejects += 1
                    stop_warn = ("Attempt %d: you output STOP but the self-check failed -> %s"
                                 % (stop_rejects, why))
                    logger.write("[STOP_REJECTED] ep%02d step%02d %s" % (ep_id, step_idx, why))
                    history_lines.append("step %d: STOP REJECTED by self-check (%s), keep navigating"
                                         % (step_idx, why))
                    continue
            break

        # U3 Tier-B: 连续卡死且没有踏上新格 -> 确定性逃逸
        if stuck_flags:
            stuck_streak += 1
        else:
            stuck_streak = 0
        if stuck_streak >= CFG["guard_after"] and guard_used < CFG["guard_budget"]:
            gact, gtarget = guard_escape_action(env, visited_counter)
            action, source, guard_override = gact, "GUARD", True
            guard_used += 1
            stuck_streak = 0
            logger.write("[GUARD] ep%02d step%02d override action=%s target=%s flags=%s"
                         % (ep_id, step_idx, gact, gtarget, stuck_flags))

        ap_step = d.compute_ap_step(parsed, env)
        aps.append(ap_step)
        if parsed.get("plan"):
            prev_planning = parsed["plan"]

        subgoal_now = parse_subgoal(llm_text)
        if source == "GUARD":
            subgoal_now = subgoal_prev   # 接管步不改承诺，避免污染 G

        res = env.step(action)
        visited_counter.update([(env.x, env.y)] if res["moved"] else [])
        d_after = truth_dist.get((env.x, env.y))
        dist_hist.append(manhattan((env.x, env.y), env.goal))

        subgoal_reached = bool(subgoal_prev) and (env.x, env.y) == subgoal_prev
        subgoal_gone = False
        if subgoal_prev and not subgoal_reached and subgoal_now != subgoal_prev:
            vset = set(env.visited_cells)
            fr, _ = frontier_cells(env, vset)
            subgoal_gone = subgoal_prev not in [c for c, _ in fr]

        ended = bool(res["done"])
        ended_at_goal = bool(res["done"] and res["success"])
        sc = score_step(action, source, res["bumped"], res["moved"], d_before, d_after,
                        visited_counter.get((env.x, env.y), 0), ap_step, stuck_flags,
                        guard_override, subgoal_now, subgoal_prev, subgoal_reached,
                        subgoal_gone, ended, ended_at_goal)
        subgoal_prev = subgoal_now

        flag = "RUNNING"
        if res["done"]:
            success = bool(res["success"])
            reason = res.get("reason", "")
            flag = "SUCCESS" if success else "FAIL"

        row = {
            "episode": ep_id, "step": step_idx,
            "pose_before": "%d,%d,%s" % pose_before,
            "pose_after": "%d,%d,%s" % (env.x, env.y, HEADING_NAME[env.heading]),
            "cell": "%d,%d" % (env.x, env.y),
            "action": action, "source": source,
            "bumped": int(res["bumped"]), "moved": int(res["moved"]),
            "d_before": d_before, "d_after": d_after,
            "visit_count": visited_counter.get((env.x, env.y), 0),
            "frontier_left": len(frontier_cells(env, set(env.visited_cells))[0]),
            "anchor1": ("%d,%d" % anchors1[0][0]) if (anchors1 := rank_anchors(env, set(env.visited_cells), 1)) else "",
            "subgoal": ("%d,%d" % subgoal_now) if subgoal_now else "",
            "stuck_flags": ";".join(stuck_flags), "guard_override": int(guard_override),
            "stop_rejected_so_far": stop_rejects,
            "ap_step": round(ap_step, 4),
            "P": sc["P"], "V": sc["V"], "M": sc["M"], "C": sc["C"], "G": sc["G"], "DS": sc["DS"],
            "flag": flag, "net": net_status,
            "think": (parsed.get("think") or "")[:300].replace("\n", " "),
            "plan": (parsed.get("plan") or "")[:300].replace("\n", " "),
        }
        step_rows.append(row)

        logger.write("-" * 76)
        logger.write("ep=%02d step=%02d pose=%s action=%s src=%s flag=%s d:%s->%s DS=%.4f "
                     "(P=%.2f V=%.2f M=%.4f C=%.2f G=%.2f) stuck=%s guard=%d"
                     % (ep_id, step_idx, row["pose_before"], action, source, flag,
                        d_before, d_after, sc["DS"], sc["P"], sc["V"], sc["M"], sc["C"], sc["G"],
                        row["stuck_flags"] or "-", guard_override))
        logger.write("PROMPT_BEGIN\n%s\nPROMPT_END" % prompt)
        logger.write("LLM_BEGIN\n%s\nLLM_END" % (llm_text or "<EMPTY %s>" % net_status))

        history_lines.append("step %d: %s%s -> (%d,%d,%s)"
                             % (step_idx, action, " (bumped)" if res["bumped"] else "",
                                env.x, env.y, HEADING_NAME[env.heading]))
        print("[EP %02d ST %02d] %s act=%s src=%s DS=%.3f d=%s->%s %s"
              % (ep_id, step_idx, row["pose_before"], action, source, sc["DS"],
                 d_before, d_after, flag), flush=True)
        if res["done"]:
            break
    else:
        success = env.at_goal()
        reason = "reached_goal" if success else "max_steps"

    traj = env.visited_cells
    fd = truth_dist.get((env.x, env.y))
    ne = fd if fd is not None else manhattan((env.x, env.y), env.goal)
    observed_free = len([c for c in observed_free_cells(env) if env.is_free(*c)])
    ds_list = [r["DS"] for r in step_rows]
    rec = {
        "episode": ep_id, "success": bool(success), "reason": reason,
        "shortest": int(env.shortest_len), "actual": int(env.forward_steps),
        "steps": len(step_rows),
        "osr": bool(env.goal in traj),
        "ne": int(ne),
        "ap": (sum(aps) / len(aps)) if aps else 0.0,
        "ds_mean": (sum(ds_list) / len(ds_list)) if ds_list else 0.0,
        "coverage": observed_free / float(free_total) if free_total else 0.0,
        "stuck_steps": sum(1 for r in step_rows if r["stuck_flags"]),
        "guard_count": sum(r["guard_override"] for r in step_rows),
        "stop_rejected": stop_rejects,
        "llm_calls": llm_calls_total,
        "fallback_steps": sum(1 for r in step_rows if r["source"] in ("FALLBACK", "NETFAIL")),
        "revisit_max": max(visited_counter.values()) if visited_counter else 0,
        "end_pose": "%d,%d,%s" % (env.x, env.y, HEADING_NAME[env.heading]),
        "seed": seed_base + ep_id * 17, "grid": "%dx%d" % (grid_h, grid_w),
        "wall_sec": round(time.time() - t0, 1),
        "start": "%d,%d" % env.start, "goal": "%d,%d" % env.goal,
        "instruction": env.instruction,
    }
    logger.write("EPISODE %02d END success=%s reason=%s shortest=%d actual=%d steps=%d NE=%d AP=%.4f DS=%.4f COV=%.3f guard=%d"
                 % (ep_id, success, reason, rec["shortest"], rec["actual"], rec["steps"],
                    ne, rec["ap"], rec["ds_mean"], rec["coverage"], rec["guard_count"]))
    return rec, step_rows


# =========================================================================
# F. 入口
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-ep", type=int, default=0)
    ap.add_argument("--num-ep", type=int, default=10)
    ap.add_argument("--grid", type=int, nargs=2, default=[15, 15])
    ap.add_argument("--seed-base", type=int, default=d.EPISODE_SEED_BASE)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--tag", default="v3")
    a = ap.parse_args()

    d.LLM_URL = os.getenv("MAPGPT_LLM_URL", "http://127.0.0.1:5000/get_llm_response")
    d.MAX_STEPS = a.max_steps
    d.SLEEP_AFTER_OK_SEC = float(os.getenv("MAPGPT_SLEEP_OK", "1.0"))
    outdir = "/home/orange/results_%s" % a.tag
    os.makedirs(outdir, exist_ok=True)
    logger = d.RunLogger(os.path.join(outdir, "log.txt"))
    logger.write("RUN v3 tag=%s eps=%d..%d grid=%dx%d max_steps=%d llm=%s cfg=%s"
                 % (a.tag, a.start_ep, a.start_ep + a.num_ep - 1, a.grid[0], a.grid[1],
                    a.max_steps, d.LLM_URL, json.dumps(CFG, ensure_ascii=False)))

    recs, all_steps = [], []
    for ep in range(a.start_ep, a.start_ep + a.num_ep):
        try:
            rec, rows = run_episode_v3(ep, logger, a.grid[0], a.grid[1], a.seed_base)
        except Exception as exc:
            import traceback; traceback.print_exc()
            logger.write("EPISODE %02d EXCEPTION %s\n%s" % (ep, exc, traceback.format_exc()))
            rec = {"episode": ep, "success": False, "reason": "exception:%s" % exc, "shortest": 1,
                   "actual": 0, "steps": 0, "osr": False, "ne": 0, "ap": 0.0, "ds_mean": 0.0,
                   "coverage": 0.0, "stuck_steps": 0, "guard_count": 0, "fallback_steps": 0,
                   "stop_rejected": 0, "llm_calls": 0, "revisit_max": 0, "end_pose": "",
                   "seed": a.seed_base + ep * 17,
                   "grid": "%dx%d" % (a.grid[0], a.grid[1]), "wall_sec": 0, "start": "", "goal": "",
                   "instruction": ""}
            rows = []
        recs.append(rec); all_steps.extend(rows)
        with open(os.path.join(outdir, "episodes.json"), "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=1)
        with open(os.path.join(outdir, "steps.json"), "w", encoding="utf-8") as f:
            json.dump(all_steps, f, ensure_ascii=False, indent=1)
        line = ("[EP %02d] success=%s steps=%d shortest=%d actual=%d NE=%d AP=%.4f DS=%.4f "
                "COV=%.3f guard=%d stoprej=%d SR=%d/%d %.1fs"
                % (ep, rec["success"], rec["steps"], rec["shortest"], rec["actual"], rec["ne"],
                   rec["ap"], rec["ds_mean"], rec["coverage"], rec["guard_count"],
                   rec.get("stop_rejected", 0),
                   sum(r["success"] for r in recs), len(recs), rec["wall_sec"]))
        print(line, flush=True)
        with open(os.path.join(outdir, "console.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    m = episode_metrics(recs, len(recs))
    keys = ["episode", "success", "reason", "shortest", "actual", "steps", "osr", "ne", "ap",
            "ds_mean", "coverage", "stuck_steps", "guard_count", "stop_rejected", "llm_calls",
            "fallback_steps", "revisit_max", "seed", "grid", "start", "goal", "end_pose", "wall_sec"]
    with open(os.path.join(outdir, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wr.writeheader()
        for r in recs: wr.writerow(r)
    if all_steps:
        with open(os.path.join(outdir, "steps.csv"), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_steps[0].keys())); wr.writeheader()
            for r in all_steps: wr.writerow(r)
    summary = "tag=%s N=%d\n" % (a.tag, len(recs)) + "\n".join(
        "%s=%s" % (k, (round(v, 4) if isinstance(v, float) else v)) for k, v in m.items())
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\n" + summary, flush=True)
    logger.write("SUMMARY " + json.dumps(m, ensure_ascii=False))
    logger.close()


if __name__ == "__main__":
    main()
