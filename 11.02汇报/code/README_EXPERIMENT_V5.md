# MapGPT v5 运行说明

## 1. 生成固定地图

```bash
cd /home/orange
python3 benchmark_maps_v5.py
```

生成 `benchmark_maps_v5.json`。一旦开始对比，不要再重新生成或修改该文件。

## 2. 启动本地 LLM 中转

```bash
cd /home/orange
MAPGPT_MAX_OUTPUT_TOKENS=500 setsid -f python3 relay_v5.py 5000 \
  > relay_v5_console.log 2>&1
```

中转从 `/home/orange/.dashscope_key` 读取密钥；密钥不会进入实验代码或日志。

## 3. 单条件运行

```bash
MAPGPT_SLEEP_OK=0.15 python3 mapgpt_v5.py \
  --condition risk_waypoint \
  --maps-file benchmark_maps_v5.json \
  --num-ep 8 \
  --max-steps 80 \
  --tag v5_risk_waypoint
```

可选条件：

```
mapgpt fused_step topo_waypoint matrix_waypoint fused_waypoint risk_waypoint
```

## 4. 全矩阵运行

```bash
MATRIX_WORKERS=6 MAPGPT_MAX_STEPS=80 MAPGPT_NUM_MAPS=8 python3 run_matrix_v5.py
```

## 5. 汇总

```bash
python3 analyze_v5.py \
  --root /home/orange \
  --out /home/orange/results_v5_analysis
```

结果目录包含逐 episode、逐步、聚合指标和逐地图配对差值。
