import fs from "node:fs";

const W = 1600;
const H = 2460;
const font = "Microsoft YaHei, Noto Sans CJK SC, sans-serif";
const parts = [];

function esc(text) {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function text(x, y, value, size = 20, weight = 400, color = "#172033", anchor = "start") {
  parts.push(
    `<text x="${x}" y="${y}" font-family="${font}" font-size="${size}" `
    + `font-weight="${weight}" fill="${color}" text-anchor="${anchor}">`
    + `${esc(value)}</text>`,
  );
}

function box(x, y, w, h, title, lines, fill = "#ffffff", stroke = "#94a3b8") {
  parts.push(
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="16" `
    + `fill="${fill}" stroke="${stroke}" stroke-width="2"/>`,
  );
  text(x + 20, y + 32, title, 25, 500, "#0f766e");
  lines.forEach((line, index) => {
    text(x + 20, y + 64 + index * 29, line, 18, 400, "#172033");
  });
}

function centerBox(x, y, w, h, title, lines, fill = "#ffffff", stroke = "#94a3b8") {
  parts.push(
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="16" `
    + `fill="${fill}" stroke="${stroke}" stroke-width="2"/>`,
  );
  text(x + w / 2, y + 34, title, 25, 500, "#0f766e", "middle");
  lines.forEach((line, index) => {
    text(x + w / 2, y + 67 + index * 29, line, 18, 400, "#172033", "middle");
  });
}

function arrow(x1, y1, x2, y2) {
  parts.push(
    `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" `
    + `stroke="#64748b" stroke-width="4" marker-end="url(#arrow)"/>`,
  );
}

parts.push(
  `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
  `<defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="4"
      orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,8 L9,4 z" fill="#64748b"/>
    </marker>
  </defs>`,
  `<rect width="${W}" height="${H}" fill="#f6f8fb"/>`,
);
text(800, 52, "MapGPT v10 完整运行流程图", 40, 500, "#172033", "middle");
text(
  800, 86,
  "局部观测 → 矩阵/拓扑/历史 → 地图预测 → 宏观分支 → 打分 → A* → 执行 → 重新判断",
  20, 400, "#5b6577", "middle",
);

let y = 118;
centerBox(
  300, y, 1000, 90, "① 接任务与观测窗口",
  [
    "机器人知道：当前位置、终点坐标、地图尺寸、局部观测。",
    "21×21默认5×5；大图默认7×7；岔路口或局部开阔处下一帧用9×9。",
  ],
);
arrow(800, y + 90, 800, y + 120);
y += 120;

centerBox(
  300, y, 1000, 105, "② 有遮挡的局部观测",
  [
    "以机器人为中心，在窗口内做BFS；看到自由格和相邻墙体。",
    "墙后的格子仍不可见；9×9只是最大窗口，不是完整真值。",
  ],
);
arrow(800, y + 105, 800, y + 135);
y += 135;

box(
  70, y, 460, 165, "稠密占用矩阵",
  [
    "已见自由/墙写入矩阵。",
    "未见格保持未知，并输出墙概率p。",
    "每格软代价 = 1 + 7p。",
  ],
  "#eef7ff", "#3b82f6",
);
box(
  570, y, 460, 165, "稀疏拓扑",
  [
    "可见自由格 → Place ID。",
    "四连通自由格建立拓扑边。",
    "用于frontier、岔路判断和A*。",
  ],
  "#f3eeff", "#8b5cf6",
);
box(
  1070, y, 460, 165, "轨迹和分支历史",
  [
    "访问轨迹、边访问次数、目标进展。",
    "分支尝试、失败、信息收益、禁用时间。",
    "结构化数值历史参与航点打分。",
  ],
  "#fff4e8", "#f59e0b",
);
arrow(300, y + 165, 300, y + 195);
arrow(800, y + 165, 800, y + 195);
arrow(1300, y + 165, 1300, y + 195);
y += 195;

box(
  170, y, 400, 150, "③ 地图预测",
  [
    "多分辨率logistic预测器",
    "为每个未知格输出墙概率p。",
    "训练地图与评测地图分离。",
  ],
);
box(
  600, y, 400, 150, "④ 预测软距离",
  [
    "Dijkstra把p转换成到终点难度：",
    "step_cost = 1 + 7p。",
    "只作为启发式，不是可执行道路。",
  ],
);
box(
  1030, y, 400, 150, "⑤ Frontier",
  [
    "已知可走且旁边有未知格的格子。",
    "表示从已知地图进入未知区域的口。",
    "先按连通性聚类，再生成候选。",
  ],
);
arrow(800, y + 150, 800, y + 180);
y += 180;

box(
  170, y, 400, 145, "⑥ 宏观分支",
  [
    "沿A*路线找首个决策路口。",
    "用“路口 + 出口方向”作为分支ID。",
    "把附近frontier合并为一个宏观方向。",
  ],
);
box(
  600, y, 400, 145, "⑦ NMS=4",
  [
    "按候选分数从高到低保留。",
    "与已保留入口距离<4格则抑制。",
    "每分支最多2入口，全局最多6候选。",
  ],
);
box(
  1030, y, 400, 145, "⑧ 具体航点",
  [
    "航点不是模糊方向，而是入口坐标。",
    "例如 (12,7)，可能距离当前位置多格。",
    "到达入口后才进入更深的未知区域。",
  ],
);
arrow(800, y + 145, 800, y + 175);
y += 175;

parts.push(
  `<rect x="190" y="${y}" width="1220" height="205" rx="16" `
  + `fill="#fff8e7" stroke="#d97706" stroke-width="2"/>`,
);
text(220, y + 34, "⑨ 航点打分：分数越低越优先", 25, 500, "#9a3412");
text(
  220, y + 68,
  "score = 1.20×A*去程长度 + 0.80×预计终点距离 + 2.60×风险",
  19, 400, "#172033",
);
text(
  220, y + 98,
  "− 0.30×log(1+信息增益) − 0.20×log(1+附近未知量)",
  19, 400, "#172033",
);
text(
  220, y + 128,
  "+ 0.60×不确定性 + 历史失败惩罚 − 历史收益奖励",
  19, 400, "#172033",
);
text(
  220, y + 164,
  "大图使用预测距离；小图更信任已观察地图上的乐观A*距离。",
  18, 400, "#5b4160",
);
text(
  220, y + 190,
  "历史不是原始文本，而是岔路尝试次数、失败次数、累计信息收益和目标进展。",
  18, 400, "#5b4160",
);
arrow(800, y + 205, 800, y + 235);
y += 235;

centerBox(
  300, y, 1000, 95, "⑩ 终点是否已经进入已知拓扑？",
  [
    "是：GOAL候选直接排到最前。",
    "否：选择当前得分最高的宏观分支入口。",
  ],
  "#eef6ff", "#2563eb",
);
arrow(800, y + 95, 800, y + 125);
y += 125;

centerBox(
  300, y, 1000, 105, "⑪ A*规划实际路线",
  [
    "只在已经确认自由格的拓扑上搜索。",
    "输出“当前位置→具体航点入口”的多格路线，并提供路径长度用于打分。",
  ],
);
arrow(800, y + 105, 800, y + 135);
y += 135;

centerBox(
  300, y, 1000, 100, "⑫ 执行低层动作并更新观测",
  [
    "朝向不对先转向；朝向正确则前进一格。",
    "env.step()更新矩阵、拓扑、轨迹和边计数。",
  ],
);
arrow(800, y + 100, 800, y + 130);
y += 130;

centerBox(
  300, y, 1000, 95, "⑬ 决定何时重新做高层规划",
  [
    "地图宽度≥24：每成功前进一格重新打分。",
    "21×21：沿A*执行到当前航点，到达后再规划。",
  ],
  "#eef6ff", "#2563eb",
);
y += 115;

box(
  240, y, 530, 130, "大图：逐步滚动规划",
  [
    "重新生成frontier、预测距离、",
    "宏观分支、候选分数和A*路线。",
    "及时放弃已过时的旧路线。",
  ],
);
box(
  830, y, 530, 130, "小图：航点式执行",
  [
    "执行当前A*路线直到航点。",
    "地图短、路径已接近最优，",
    "减少不必要的重复规划。",
  ],
);
arrow(505, y + 130, 505, y + 160);
arrow(1095, y + 130, 1095, y + 160);
y += 160;

parts.push(
  `<rect x="170" y="${y}" width="1260" height="180" rx="16" `
  + `fill="#fff1f2" stroke="#e11d48" stroke-width="2"/>`,
);
text(200, y + 34, "⑭ 防卡死与路线恢复", 25, 500, "#be123c");
text(
  200, y + 68,
  "撞墙：清空当前路线并重新规划。",
  18, 400, "#172033",
);
text(
  200, y + 96,
  "振荡：连续5帧重复A-B-A-B-A模式，则重规划。",
  18, 400, "#172033",
);
text(
  200, y + 124,
  "重复边：同一条无向边累计走过5次，则重规划；失败分支无信息且无进展时禁用10步。",
  18, 400, "#172033",
);
text(
  200, y + 152,
  "逐步重规划：大图每前进一格清除过时路线，是最主要的主动防绕路机制。",
  18, 400, "#172033",
);
arrow(800, y + 180, 800, y + 210);
y += 210;

centerBox(
  330, y, 940, 85, "⑮ 停止",
  [
    "机器人坐标严格等于终点后执行STOP；离终点很近也不会提前停止。",
  ],
  "#e9f9ef", "#16a34a",
);
arrow(800, y + 85, 800, y + 115);
y += 115;

centerBox(
  250, y, 1100, 105, "⑯ 输出与评测",
  [
    "记录成功率SR、路径效率SPL、未见格占用预测正确率AP_UNSEEN。",
    "完整真值和最短路径只在episode结束后用于评分，不进入在线决策。",
  ],
  "#eef2f7", "#64748b",
);

text(
  800, H - 42,
  "一句话：用局部地图预测未知区域，把入口合并为宏观方向并打分；A*负责在已知道路上去航点，走一格后用新观测与历史重新判断。",
  18, 400, "#475569", "middle",
);
parts.push("</svg>");

fs.writeFileSync(
  "outputs/figures_v10/fig0_pipeline.svg",
  parts.join("\n"),
  "utf8",
);
console.log("SVG written");
