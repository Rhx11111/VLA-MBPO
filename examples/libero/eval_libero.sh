#!/bin/bash
source examples/libero/.venv/bin/activate --active
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

port=8202
# config=pi05_libero_long
config=pi05_libero_long
policy=libero_ppo_long_iter_1
# policy=libero_ppo_long
step=300
# step=50
device=1
task_id=-1
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 uv run \
    scripts/serve_policy.py --port $port policy:checkpoint \
    --policy.config=$config \
    --policy.dir=/data/home/scwb314/run/models/openpi/$config/$policy/$step &

sleep 10

CUDA_VISIBLE_DEVICES=1 python examples/libero/main.py \
    --args.port $port \
    --args.num_trials_per_task 50 \
    --args.video_out_path /data/home/scwb314/run/data/rollout_data/long_iter_1/videos \
    --args.eval_dir /data/home/scwb314/run/data/rollout_data/long_iter_1 \
    --args.replan_steps 10 \
    --args.task_id $task_id \
    # --args.save_data
