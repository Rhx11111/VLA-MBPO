#!/bin/bash
##SBATCH --gpus=4
##SBATCH -p gpu_h100

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"
# Initialize conda. Fall back to the local shell setup if module loading fails.
if ! module load miniforge3/25.11.0-1>/dev/null; then
    # Fall back to conda directly if module loading fails.
    source ~/.bashrc 2>/dev/null || true
fi

# Activate the conda environment.
source $(conda info --base)/etc/profile.d/conda.sh
module load cuda/13.0 nccl/2.28_cuda13.0
cp /data/apps/nccl/nccl-2.28.9-1/build/lib/libnccl.so.2.28.9 \
   ~/.conda/envs/uni-plan/lib/python3.10/site-packages/nvidia/nccl/lib/
ln -sf libnccl.so.2.28.9 \
   ~/.conda/envs/uni-plan/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2
source activate uni-plan

unset CUDA_VISIBLE_DEVICES
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_PROCID=${SLURM_PROCID:-0}
export MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
export MASTER_PORT=${MASTER_PORT:-29502}

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export OMP_NUM_THREADS=1

MODEL_PATH="${MODEL_PATH:-/data/home/scwb314/run/models/BAGEL-7B-MoT}"
RESULTS_DIR="${RESULTS_DIR:-/data/home/scwb314/run/bagel-result/libero/libero_data}"
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
  --dataset_config_file ./data/configs/libero/libero_data.yaml \
  --wandb_name "libero_data" \
  --wandb_runid "0" \
  --wandb_offline True \
  --model_path $MODEL_PATH \
  --results_dir $RESULTS_DIR \
  --checkpoint_dir $CKPT_DIR \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --finetune_from_hf True \
  --auto_resume False \
  --resume-model-only True \
  --resume-from $MODEL_PATH \
  --finetune-from-ema True \
  --log_every 10 \
  --ce_weight 0.01 \
  --lr 2e-5 \
  --num_shard 4 \
  --warmup_steps 100 \
  --total_steps 20000 \
  --save_every 5000 \
  --expected_num_tokens 32768 \
  --max_num_tokens 32768 \
  --max_num_tokens_per_sample 32768 \
  --sharding_strategy "FULL_SHARD" \
  --freeze_vit True \
  --freeze_vae True \
  --visual_und True \
  --visual_gen True \
