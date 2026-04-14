#!/bin/bash
#SBATCH --gpus=4
#SBATCH -p gpu_h200
cd ~/run/VLA_MBRL/openpi/world_model
if ! module load miniforge3/25.11.0-1>/dev/null; then
    # Fall back to conda directly if module loading fails.
    source ~/.bashrc 2>/dev/null || true
fi

# Activate the conda environment.
source $(conda info --base)/etc/profile.d/conda.sh
module load cuda/12.8
source activate uni-plan

CONFIG_PATH=/data/home/scwb314/run/models/BAGEL-7B-MoT
MODEL_PATH=/data/home/scwb314/run/bagel-result/libero/libero/bagel/libero_goal-run0/ckpt/0005000
ACTION_NORM_PATH=/data/home/scwb314/run/data/bagel_data/dynamics/libero_goal_with_wrist/action_normalizer.json
WM_PORT=8112
ENV_PORT=8113
GPU_IDS="0 1 2 3"         # GPU ids, one worker per GPU.
NUM_WORKERS=4       # Worker count, usually equal to the GPU count.
MAX_MEM_PER_GPU=80GiB  # Per-GPU memory cap.

python world_model_client/websocket_world_model_server.py \
    --model-config-path $CONFIG_PATH \
    --model-weights-path $MODEL_PATH \
    --action-norm-path "$ACTION_NORM_PATH" \
    --max-mem-per-gpu "$MAX_MEM_PER_GPU" \
    --num-workers "$NUM_WORKERS" \
    --gpu-ids $GPU_IDS \
    --start-method spawn \
    --host 0.0.0.0 \
    --port "$WM_PORT" \
    --edit-cfg-text-scale 6.0 \
    --edit-cfg-img-scale 2.0 \
    --edit-timestep-shift 4.0 \
    --edit-num-timesteps 50 \
    --edit-cfg-renorm-type text_channel \
    --understand-max-tokens 1000 \
    --understand-temperature 0.3 &

sleep 30
deactivate

# LIBERO Remote Inference Script
cd /data/home/scwb314/run/VLA_MBRL/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python libero_client/libero_websocket_server.py --task-suite-name libero_goal --num-envs 32 --port $ENV_PORT --use-rel-reward &

deactivate

cd /data/home/scwb314/run/VLA_MBRL/openpi
source .venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=3
export TF_TRT_DISABLE=1

cfg=pi05_libero_goal

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_API_KEY="${WANDB_API_KEY:-}" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
python scripts/train_libero_rl.py ${cfg} \
    --algorithm ppo \
    --exp_name libero_ppo_goal \
    --pretrained_path /data/home/scwb314/run/models/openpi/pi05_libero/libero_one_shot/5000/params \
    --port $ENV_PORT \
    --world_model_port $WM_PORT \
    --overwrite 
