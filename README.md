# VLA-MBPO

<p align="center">
  <strong>Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2603.20607">Paper</a> |
  <a href="https://rhx11111.github.io/VLA-MBPO/">Project Page</a>
</p>

VLA-MBPO is a practical world model-based reinforcement learning framework for
fine-tuning vision-language-action (VLA) policies. It trains and serves a unified
multimodal world model for pixel-level dynamics and rewards, uses interleaved
multi-view decoding for LIBERO-style observations, and updates VLA policies with
chunk-level branched rollouts in the learned world model.

This repository contains the official implementation. It is built on OpenPI for
VLA policy training and includes the world-model training and serving code used
by VLA-MBPO.

## Highlights

- Unified multimodal world model for joint visual dynamics and reward prediction.
- Interleaved head-view and wrist-view decoding for multi-view consistency.
- Chunk-level branched rollout for stable world model-based policy optimization.
- Websocket-based world-model and LIBERO environment servers for distributed RL.
- Training configs for LIBERO Spatial, Object, Goal, and Long-Horizon suites.

## Repository Layout

```text
.
|-- src/openpi/                 # VLA policy models, data pipelines, and RL configs
|-- scripts/train_libero_rl.py  # Main VLA-MBPO policy optimization entrypoint
|-- train_rl.sh                 # End-to-end launcher template
|-- libero_client/              # Websocket LIBERO environment server/client
|-- world_model/                # Unified multimodal world-model training code
|-- world_model/world_model_client/
|   `-- websocket_world_model_server.py
|-- examples/                   # OpenPI examples and LIBERO evaluation helpers
`-- packages/openpi-client/     # Policy client package inherited from OpenPI
```

## Setup

VLA-MBPO currently uses two Python environments:

- root environment: OpenPI policy training and VLA-MBPO RL
- `world_model` environment: unified multimodal world-model training and serving

### Root Environment

The root project uses Python 3.11 and `uv`.

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
export PYTHONPATH="${PYTHONPATH:-}:$PWD/third_party/libero"
```

Install the OpenPI client package when running remote clients or examples:

```bash
uv pip install -e packages/openpi-client
```

### LIBERO Environment

The LIBERO server depends on the in-repository LIBERO checkout.

```bash
uv pip install -e third_party/libero
```

Some users prefer a separate LIBERO virtual environment because MuJoCo,
robosuite, and rendering dependencies can be sensitive to CUDA/EGL setup. The
launcher template supports this with `LIBERO_VENV=/path/to/venv`.

### World-Model Environment

The world-model code uses Python 3.10.

```bash
cd world_model
conda create -n uni-plan python=3.10 -y
conda activate uni-plan
pip install -r requirements.txt
```

If `flash_attn` cannot be built from source, see `world_model/README.md` for
wheel-based installation guidance.

## Required Artifacts

The public model and dataset artifacts are not bundled in this repository yet.
Before running the full training workflow, prepare the following local paths:

| Variable | Description | Status |
| --- | --- | --- |
| `MODEL_CONFIG_PATH` | Base UMM/Bagel config directory | Coming soon |
| `WORLD_MODEL_CKPT` | Fine-tuned world-model checkpoint | Coming soon |
| `ACTION_NORM_PATH` | Action normalizer JSON for the selected task suite | Coming soon |
| `PRETRAINED_POLICY_PATH` | Initial VLA policy checkpoint for RL fine-tuning | Coming soon |
| LIBERO LeRobot datasets | Converted offline data used by `pi05_libero_*` configs | Coming soon |

The active LIBERO RL configs are defined in `src/openpi/training/config.py`:

- `pi05_libero_spatial`
- `pi05_libero_object`
- `pi05_libero_goal`
- `pi05_libero_long`

## Training Workflow

VLA-MBPO policy optimization runs three services in sequence:

1. Start the world-model websocket server.
2. Start the LIBERO websocket rollout server.
3. Run `scripts/train_libero_rl.py`.

The integrated launcher is:

```bash
MODEL_CONFIG_PATH=/path/to/BAGEL-7B-MoT \
WORLD_MODEL_CKPT=/path/to/world_model_checkpoint \
ACTION_NORM_PATH=/path/to/action_normalizer.json \
PRETRAINED_POLICY_PATH=/path/to/pretrained_policy/params \
CONFIG_NAME=pi05_libero_goal \
EXP_NAME=libero_goal_vla_mbpo \
bash train_rl.sh
```

Useful launcher variables:

| Variable | Default | Description |
| --- | --- | --- |
| `CONFIG_NAME` | `pi05_libero_goal` | RL config name |
| `EXP_NAME` | `vla_mbpo_libero` | Checkpoint and logging run name |
| `LIBERO_TASK_SUITE` | `libero_goal` | LIBERO suite served by `libero_client` |
| `LIBERO_PORT` | `8113` | LIBERO websocket server port |
| `WORLD_MODEL_PORT` | `8112` | World-model websocket server port |
| `GPU_IDS` | `0 1 2 3` | World-model worker GPU ids |
| `NUM_WORKERS` | `4` | Number of world-model workers |
| `NUM_ENVS` | `32` | Parallel LIBERO environments |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3` | Devices used by policy training |
| `WANDB_MODE` | `offline` | Weights and Biases mode |

You can also run the RL entrypoint directly after starting both servers:

```bash
python scripts/train_libero_rl.py pi05_libero_goal \
  --algorithm ppo \
  --exp_name libero_goal_vla_mbpo \
  --pretrained_path /path/to/pretrained_policy/params \
  --port 8113 \
  --world_model_port 8112 \
  --overwrite
```

## World-Model Training

World-model training scripts are under `world_model/scripts/`:

```bash
cd world_model
bash scripts/train_libero.sh
bash scripts/train_robotwin.sh
bash scripts/train_aloha.sh
```

Set dataset, checkpoint, output, and logging paths through environment
variables before launching the scripts. See `world_model/README.md` for the
underlying UniPlan/Bagel setup and training details.

## LIBERO Evaluation

For standalone LIBERO policy evaluation, see `examples/libero/README.md`.
For VLA-MBPO training, the rollout server is:

```bash
python libero_client/libero_websocket_server.py \
  --task-suite-name libero_goal \
  --num-envs 32 \
  --port 8113 \
  --use-rel-reward
```

The server exposes batched `reset`, `step`, and `chunk_step` websocket methods
used by `scripts/train_libero_rl.py`.

## Citation

If you find this code useful, please cite:

```bibtex
@article{zhang2026vlambpo,
  title={Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models},
  author={Zhang, Zhilong and Ren, Haoxiang and Sun, Yihao and Sheng, Yifei and Wang, Haonan and Lin, Haoxin and Wu, Zhichao and Bacon, Pierre-Luc and Yu, Yang},
  journal={arXiv preprint arXiv:2603.20607},
  year={2026}
}
```

## Acknowledgements

This codebase builds on OpenPI, LIBERO, LeRobot, and the Bagel/UniPlan world
model stack. We thank the authors and maintainers of these projects.

## License

This repository is released under the Apache-2.0 license. See `LICENSE` for
details.
