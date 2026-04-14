# VLA-MBPO

<p align="center">
  <strong>Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2603.20607">Paper</a> |
  <a href="https://rhx11111.github.io/VLA-MBPO/">Project Page</a>
</p>

<p align="center">
  VLA-MBPO is a practical world model-based RL framework for finetuning VLA policies with
  unified multimodal world models, interleaved multi-view decoding, and chunk-level branched rollout.
</p>

## Environment Setup

This project uses two environments:

- the root `openpi` environment for RL training
- the `world_model` environment for world-model training and serving

### 1. Root environment for RL

The root project uses `uv` and Python 3.11.

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
git submodule update --init --recursive
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

### 2. World-model environment

```bash
cd world_model
conda create -n uni-plan python=3.10 -y
conda activate uni-plan
pip install -r requirements.txt
```

If you use the websocket world-model server, set path-dependent variables through the environment:

```bash
export MODEL_PATH=/path/to/model
export CONFIG_PATH=/path/to/config
export ACTION_NORM_PATH=/path/to/action_normalizer.json
export BAGEL_DATA_ROOT=/path/to/bagel_data
export WANDB_API_KEY=...
```

## Training

### RL

The main RL entrypoint is:

```bash
python scripts/train_libero_rl.py <config_name> \
  --exp_name <run_name> \
  --port <libero_server_port> \
  --world_model_port <world_model_port> \
  --overwrite
```

The integrated launcher is:

```bash
bash train_rl.sh
```

This workflow starts the world-model server, starts the LIBERO rollout server, and then launches RL training.

### World Model

World-model training scripts are under `world_model/scripts/`.

Typical entrypoints:

```bash
bash world_model/scripts/train_libero.sh
bash world_model/scripts/train_robotwin.sh
bash world_model/scripts/train_aloha.sh
```
