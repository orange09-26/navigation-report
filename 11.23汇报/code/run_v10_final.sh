#!/usr/bin/env bash
set -euo pipefail

ROOT="${MAPGPT_ROOT:-/home/orange}"
MAPS="${MAPS:-$ROOT/benchmark_maps_v10.json}"
STEPS="${STEPS:-100}"
TAG="${TAG:-v10_final}"

cd "$ROOT"
MAPGPT_ROOT="$ROOT" \
MAPGPT_SLEEP_OK=0 \
/usr/bin/python3 mapgpt_v7.py \
  --condition macro_v10 \
  --maps-file "$MAPS" \
  --predictor-model "$ROOT/map_predictor_v8_multires.json" \
  --num-ep 8 \
  --max-steps "$STEPS" \
  --obs-size 7 \
  --obs-policy geometry \
  --pred-weight -2 \
  --route-replan-steps -1 \
  --waypoint-mode entry \
  --tag "$TAG"
