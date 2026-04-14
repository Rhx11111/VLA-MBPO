#!/bin/bash
# LIBERO Remote Inference Script
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# CUDA_VISIBLE_DEVICES=0,1,2,3 EGL_VISIBLE_DEVICES=0,1,2,3 
python libero_client/libero_websocket_server.py --task-suite-name libero_goal --num-envs 128 --port 7999 --use-rel-reward
