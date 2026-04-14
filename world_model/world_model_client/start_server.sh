#!/bin/bash
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

module load cuda/12.8
source activate uni-plan

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-/data/home/scwb314/run/models/BAGEL-7B-MoT}"
MODEL_PATH="${MODEL_PATH:-/data/home/scwb314/run/bagel-result/libero/libero/bagel/libero-run2/ckpt/0005000}"
ACTION_NORM_PATH="${ACTION_NORM_PATH:-/data/home/scwb314/run/data/bagel_data/dynamics/libero_spatial_with_wrist/action_normalizer.json}"
PORT="${PORT:-8000}"
GPU_IDS="0 1 2 3"         # GPU ids, one worker per GPU.
NUM_WORKERS=4       # Worker count, usually equal to the GPU count.
MAX_MEM_PER_GPU=80GiB  # Per-GPU memory cap.

echo "========================================="
echo "Starting World Model Inference Server (Multi-Worker)"
echo "========================================="
echo "Model Path: $MODEL_PATH"
echo "Action Norm Path: $ACTION_NORM_PATH"
echo "Port: $PORT"
echo "GPU IDs: $GPU_IDS"
echo "Num Workers: $NUM_WORKERS"
echo "========================================="

python "$SCRIPT_DIR/websocket_world_model_server.py" \
    --model-config-path $CONFIG_PATH \
    --model-weights-path $MODEL_PATH \
    --action-norm-path "$ACTION_NORM_PATH" \
    --max-mem-per-gpu "$MAX_MEM_PER_GPU" \
    --num-workers "$NUM_WORKERS" \
    --gpu-ids $GPU_IDS \
    --start-method spawn \
    --host 0.0.0.0 \
    --port "$PORT" \
    --edit-cfg-text-scale 4.0 \
    --edit-cfg-img-scale 2.0 \
    --edit-timestep-shift 3.0 \
    --edit-num-timesteps 50 \
    --edit-cfg-renorm-type text_channel \
    --understand-max-tokens 1000 \
    --understand-temperature 0.3
