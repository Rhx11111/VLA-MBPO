#!/bin/bash
#SBATCH --gpus=4
#SBATCH -p gpu_h100


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
# Initialize conda. Fall back to the local shell setup if module loading fails.
if ! module load miniforge3/24.11 2>/dev/null; then
    # Fall back to conda directly if module loading fails.
    source ~/.bashrc 2>/dev/null || true
fi

# Activate the conda environment.
source $(conda info --base)/etc/profile.d/conda.sh
source activate bagel

export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_PROCID=${SLURM_PROCID:-0}
export MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
export MASTER_PORT=${MASTER_PORT:-15423}

MODEL_PATH="${MODEL_PATH:-/data/home/scwb314/run/models/BAGEL-7B-MoT}"
RESULTS_DIR="${RESULTS_DIR:-/data/home/scwb314/run/bagel-result/aloha/fold_the_towel_new}"
CKPT_DIR="${CKPT_DIR:-$RESULTS_DIR/checkpoints}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY="${WANDB_API_KEY:-}"

torchrun \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_PROCID \
  --nproc_per_node=4 \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/aloha/fold_the_towel_new.yaml \
  --wandb_name "aloha_fold_the_towel_new" \
  --wandb_runid "3" \
  --wandb_offline True \
  --model_path $MODEL_PATH \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $MODEL_PATH \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 10 \
  --ce_weight 0.001 \
  --lr 2e-5 \
  --num_shard 4 \
  --warmup_steps 1000 \
  --total_steps 10000 \
  --save_every 1000 \
  --expected_num_tokens 32768 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 32768 \
  --sharding_strategy "FULL_SHARD" \
  --freeze_vit False \
  --freeze_vae True \
  --visual_und True \
  --visual_gen True \
