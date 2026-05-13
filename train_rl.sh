#!/usr/bin/env bash
# Example end-to-end launcher for VLA-MBPO on LIBERO.
#
# Set the required artifact paths below through environment variables before
# running this script. Keep cluster-specific module and Slurm settings in your
# own wrapper if they differ from this template.
#
# Required:
#   MODEL_CONFIG_PATH=/path/to/BAGEL-7B-MoT
#   WORLD_MODEL_CKPT=/path/to/world_model_checkpoint
#   ACTION_NORM_PATH=/path/to/action_normalizer.json
#   PRETRAINED_POLICY_PATH=/path/to/pretrained_policy/params
#
# Example:
#   MODEL_CONFIG_PATH=/path/to/BAGEL-7B-MoT \
#   WORLD_MODEL_CKPT=/path/to/world_model_checkpoint \
#   ACTION_NORM_PATH=/path/to/action_normalizer.json \
#   PRETRAINED_POLICY_PATH=/path/to/pretrained_policy/params \
#   CONFIG_NAME=pi05_libero_goal \
#   EXP_NAME=libero_goal_vla_mbpo \
#   bash train_rl.sh

#SBATCH --gpus=4
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

MODEL_CONFIG_PATH=${MODEL_CONFIG_PATH:-/data/home/scwb314/run/models/BAGEL-7B-MoT}
WORLD_MODEL_CKPT=${WORLD_MODEL_CKPT:-/data/home/scwb314/run/bagel-result/libero/libero_data/bagel/libero_goal-run0/ckpt/0002500}
ACTION_NORM_PATH=${ACTION_NORM_PATH:-/data/home/scwb314/run/data/bagel_data/dynamics/libero_goal_with_wrist_v2/action_normalizer.json}
PRETRAINED_POLICY_PATH=${PRETRAINED_POLICY_PATH:-/data/home/scwb314/run/models/openpi/pi05_libero/libero_one_shot/5000/params}

WORLD_MODEL_ENV="${WORLD_MODEL_ENV:-uni-plan}"
RL_VENV="${RL_VENV:-$ROOT_DIR/.venv}"
LIBERO_VENV="${LIBERO_VENV:-$ROOT_DIR/examples/libero/.venv}"

WORLD_MODEL_PORT="${WORLD_MODEL_PORT:-8112}"
LIBERO_PORT="${LIBERO_PORT:-8113}"
WORLD_MODEL_HOST="${WORLD_MODEL_HOST:-0.0.0.0}"
LIBERO_HOST="${LIBERO_HOST:-0.0.0.0}"

GPU_IDS="${GPU_IDS:-0 1 2 3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_ENVS="${NUM_ENVS:-32}"
MAX_MEM_PER_GPU="${MAX_MEM_PER_GPU:-80GiB}"
SERVER_STARTUP_SECONDS="${SERVER_STARTUP_SECONDS:-30}"

CONFIG_NAME="${CONFIG_NAME:-pi05_libero_goal}"
EXP_NAME="${EXP_NAME:-vla_mbpo_libero_goal}"
LIBERO_TASK_SUITE="${LIBERO_TASK_SUITE:-libero_goal}"
ALGORITHM="${ALGORITHM:-ppo}"
OVERWRITE="${OVERWRITE:-1}"
RESUME="${RESUME:-0}"
USE_REL_REWARD="${USE_REL_REWARD:-1}"

PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap cleanup EXIT

if [[ "$OVERWRITE" == "1" && "$RESUME" == "1" ]]; then
    echo "OVERWRITE and RESUME cannot both be enabled." >&2
    exit 1
fi

if [[ -n "${CUDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
    module load "$CUDA_MODULE"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate the world-model environment." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORLD_MODEL_ENV"

cd "$ROOT_DIR/world_model"
python world_model_client/websocket_world_model_server.py \
    --model-config-path "$MODEL_CONFIG_PATH" \
    --model-weights-path "$WORLD_MODEL_CKPT" \
    --action-norm-path "$ACTION_NORM_PATH" \
    --max-mem-per-gpu "$MAX_MEM_PER_GPU" \
    --num-workers "$NUM_WORKERS" \
    --gpu-ids $GPU_IDS \
    --start-method spawn \
    --host "$WORLD_MODEL_HOST" \
    --port "$WORLD_MODEL_PORT" \
    --edit-cfg-text-scale "${EDIT_CFG_TEXT_SCALE:-6.0}" \
    --edit-cfg-img-scale "${EDIT_CFG_IMG_SCALE:-2.0}" \
    --edit-timestep-shift "${EDIT_TIMESTEP_SHIFT:-4.0}" \
    --edit-num-timesteps "${EDIT_NUM_TIMESTEPS:-50}" \
    --edit-cfg-renorm-type "${EDIT_CFG_RENORM_TYPE:-text_channel}" \
    --understand-max-tokens "${UNDERSTAND_MAX_TOKENS:-1000}" \
    --understand-temperature "${UNDERSTAND_TEMPERATURE:-0.3}" &
PIDS+=("$!")

sleep "$SERVER_STARTUP_SECONDS"
conda deactivate

cd "$ROOT_DIR"
if [[ ! -f "$LIBERO_VENV/bin/activate" ]]; then
    echo "LIBERO virtual environment not found: $LIBERO_VENV" >&2
    exit 1
fi

source "$LIBERO_VENV/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/third_party/libero"

REL_REWARD_FLAG=()
if [[ "$USE_REL_REWARD" == "1" ]]; then
    REL_REWARD_FLAG+=(--use-rel-reward)
fi

python libero_client/libero_websocket_server.py \
    --task-suite-name "$LIBERO_TASK_SUITE" \
    --num-envs "$NUM_ENVS" \
    --host "$LIBERO_HOST" \
    --port "$LIBERO_PORT" \
    "${REL_REWARD_FLAG[@]}" &
PIDS+=("$!")

deactivate

if [[ ! -f "$RL_VENV/bin/activate" ]]; then
    echo "Root training virtual environment not found: $RL_VENV" >&2
    exit 1
fi

source "$RL_VENV/bin/activate"

export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TF_TRT_DISABLE="${TF_TRT_DISABLE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}"

TRAIN_FLAGS=(
    --algorithm "$ALGORITHM"
    --exp_name "$EXP_NAME"
    --pretrained_path "$PRETRAINED_POLICY_PATH"
    --port "$LIBERO_PORT"
    --world_model_port "$WORLD_MODEL_PORT"
)

if [[ "$OVERWRITE" == "1" ]]; then
    TRAIN_FLAGS+=(--overwrite)
fi
if [[ "$RESUME" == "1" ]]; then
    TRAIN_FLAGS+=(--resume)
fi

python scripts/train_libero_rl.py "$CONFIG_NAME" "${TRAIN_FLAGS[@]}" "$@"
