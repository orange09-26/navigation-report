import fs from "node:fs";
import path from "node:path";

const outputs = path.join(process.cwd(), "outputs", "figures_v10");
const suites = [
  {
    name: "21×21 测试集",
    source: "work/benchmark_maps_v5.json",
    output: "fig_maps_21.svg",
    count: 8,
  },
  {
    name: "25×25 复杂测试集",
    source: "work/benchmark_maps_v8.json",
    output: "fig_maps_25.svg",
    count: 8,
  },
  {
    name: "27×27 留出测试集",
    source: "work/benchmark_maps_v10.json",
    output: "fig_maps_27.svg",
    count: 8,
  },
  {
    name: "29×29 第二留出测试集",
    source: "work/benchmark_maps_v10b.json",
    output: "fig_maps_29.svg",
    count: 8,
  },
];
const font = "Microsoft YaHei, Noto Sans CJK SC, sans-serif";

function escapeXml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderSuite(suite) {
  const manifest = JSON.parse(fs.readFileSync(suite.source, "utf8"));
  const maps = manifest.maps.slice(0, suite.count);
  const rows = 2;
  const cols = 4;
  const cell = 12;
  const mapW = Math.max(...maps.map((map) => map.size[1])) * cell;
  const mapH = Math.max(...maps.map((map) => map.size[0])) * cell;
  const panelW = mapW + 40;
  const panelH = mapH + 95;
  const W = cols * panelW + 80;
  const H = rows * panelH + 105;
  const parts = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" `
      + `viewBox="0 0 ${W} ${H}">`,
    `<rect width="${W}" height="${H}" fill="#f6f8fb"/>`,
    `<text x="${W / 2}" y="46" text-anchor="middle" `
      + `font-family="${font}" font-size="32" font-weight="500" fill="#172033">`
      + `${escapeXml(suite.name)}：${maps.length}张地图</text>`,
    `<circle cx="48" cy="78" r="6" fill="#16a34a"/>`,
    `<text x="62" y="84" font-family="${font}" font-size="16" fill="#334155">起点</text>`,
    `<circle cx="138" cy="78" r="6" fill="#dc2626"/>`,
    `<text x="152" y="84" font-family="${font}" font-size="16" fill="#334155">终点</text>`,
  ];
  maps.forEach((map, index) => {
    const col = index % cols;
    const row = Math.floor(index / cols);
    const ox = 40 + col * panelW + 20;
    const oy = 92 + row * panelH + 42;
    const gridX = ox;
    const gridY = oy + 39;
    parts.push(
      `<text x="${ox}" y="${oy + 22}" font-family="${font}" `
      + `font-size="18" font-weight="500" fill="#172033">`
      + `${escapeXml(map.id)}</text>`,
    );
    parts.push(
      `<rect x="${gridX}" y="${gridY}" width="${mapW}" height="${mapH}" `
      + `fill="#111827"/>`,
    );
    map.ascii.forEach((line, y) => {
      [...line].forEach((ch, x) => {
        if (ch === "." || ch === "S" || ch === "G") {
          parts.push(
            `<rect x="${gridX + x * cell}" y="${gridY + y * cell}" `
            + `width="${cell}" height="${cell}" fill="#f8fafc"/>`,
          );
        }
      });
    });
    const [sx, sy] = map.start;
    const [gx, gy] = map.goal;
    parts.push(
      `<circle cx="${gridX + sx * cell + cell / 2}" `
      + `cy="${gridY + sy * cell + cell / 2}" r="${cell * 0.38}" fill="#16a34a"/>`,
    );
    parts.push(
      `<circle cx="${gridX + gx * cell + cell / 2}" `
      + `cy="${gridY + gy * cell + cell / 2}" r="${cell * 0.42}" fill="#dc2626"/>`,
    );
    const metrics = map.metrics;
    parts.push(
      `<text x="${gridX}" y="${gridY + mapH + 24}" font-family="${font}" `
      + `font-size="14" fill="#475569">`
      + `最短路径 ${metrics.shortest}｜死胡同 ${metrics.dead_ends}｜回路 ${metrics.cycles ?? "-"}`
      + `</text>`,
    );
  });
  parts.push("</svg>");
  fs.mkdirSync(outputs, { recursive: true });
  fs.writeFileSync(path.join(outputs, suite.output), parts.join("\n"), "utf8");
}

suites.forEach(renderSuite);
console.log("map SVG suites written");
