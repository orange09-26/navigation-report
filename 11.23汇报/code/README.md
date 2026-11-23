# v10 复现入口

环境：Ubuntu 22.04，Python 3.10，NumPy。

最终方法：

```bash
chmod +x run_v10_final.sh
MAPS=/home/orange/benchmark_maps_v10.json \
STEPS=100 \
TAG=v10_final \
./run_v10_final.sh
```

快速检查：

```bash
MAPS=/home/orange/benchmark_maps_v10.json \
STEPS=100 \
TAG=v10_final_smoke \
./run_v10_final.sh
```

最终 profile 的参数：

- `macro_v10`：21×21 使用宏观/微观统一成本，大图只使用宏观分支；
- `geometry`：窄走廊 7×7，开放区/岔路口 9×9，小图 5×5；
- `pred-weight=-2`：小图采用乐观距离，大图采用预测软距离；
- `route-replan-steps=-1`：小图执行到航点再规划，大图每成功移动一格重新规划；
- 默认使用 `map_predictor_v8_multires.json`；
- 不调用 LLM。

所有参数在决策时只使用任务指令、当前观测、历史轨迹和当前预测地图；真实最短路径只用于评测 SPL。
