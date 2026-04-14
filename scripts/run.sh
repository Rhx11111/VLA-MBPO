source .venv/bin/activate # replace 'uv run' with 'python' commmand

cfg=pi05_robotwin

module load cuda/13.0 nccl/2.28_cuda13.0
NCCL_LIB_DIR="$(python -c 'import site; print(site.getsitepackages()[0] + "/nvidia/nccl/lib")')"
mkdir -p "${NCCL_LIB_DIR}"
cp /data/apps/nccl/nccl-2.28.9-1/build/lib/libnccl.so.2.28.9 \
  "${NCCL_LIB_DIR}/"
ln -sf libnccl.so.2.28.9 "${NCCL_LIB_DIR}/libnccl.so.2"

# python examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir ~/run/data/robotwin_data/ --repo_id robotwin_expert_data_with_randomized

python scripts/compute_norm_stats.py --config-name pi05_robotwin

# /home/yejunyin/Packages/clash/clash & # connect to web

# wandb login 48747959a93fef99824392f1516ed9e6db1312d1

WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_API_KEY="${WANDB_API_KEY:-}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 python scripts/train.py ${cfg} \
    --exp_name pi05_robotwin_with_randomized \
    --resume \

