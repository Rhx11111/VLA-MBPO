# World Model Client

This package provides a WebSocket client and server for remote Bagel-based world-model inference. It supports two runtime modes:

1. **Editing / world-model inference**: current head-camera image, wrist-camera image, and action sequence -> predicted next head and wrist images.
2. **Understanding / reward inference**: image and task prompt -> boolean task-completion reward.

## Files

```text
world_model_client/
|-- __init__.py
|-- base_world_model.py
|-- msgpack_numpy.py
|-- start_server.sh
|-- websocket_world_model_client.py
|-- websocket_world_model_server.py
|-- world_model_inference_server.py
`-- README.md
```

## Installation

Install the runtime dependencies used by the client and server:

```bash
pip install websockets msgpack numpy torch Pillow accelerate
```

The inference server also requires the Bagel model code and checkpoints to be available in the runtime environment.

## Start The Server

The recommended entrypoint is the launcher script:

```bash
cd /data/home/scwb314/run/VLA_MBRL/openpi

MODEL_PATH=/path/to/model/config \
ACTION_NORM_PATH=/path/to/action_norm.json \
PORT=8000 \
bash world_model_client/start_server.sh
```

The launcher starts `websocket_world_model_server.py` with multi-worker inference. Useful environment variables:

- `MODEL_PATH`: Bagel model config/checkpoint path expected by the server launcher.
- `ACTION_NORM_PATH`: JSON file containing action normalization statistics.
- `PORT`: WebSocket server port.

Common server options:

- `--model-path`: model path passed to the inference server.
- `--action-norm-path`: action normalizer JSON path.
- `--max-mem-per-gpu`: per-GPU memory budget, for example `40GiB`.
- `--num-workers`: number of worker processes.
- `--gpu-ids`: GPU IDs assigned to workers.
- `--host`: server bind host.
- `--port`: server bind port.
- `--edit-cfg-text-scale`: text CFG scale for editing.
- `--edit-cfg-img-scale`: image CFG scale for editing.
- `--edit-timestep-shift`: timestep shift for editing.
- `--edit-num-timesteps`: denoising steps for editing.
- `--edit-cfg-renorm-type`: CFG renormalization mode.
- `--understand-max-tokens`: maximum text tokens for understanding.
- `--understand-temperature`: sampling temperature for understanding.

## Client Usage

```python
import numpy as np

from world_model_client import WebsocketWorldModelClient


client = WebsocketWorldModelClient(host="127.0.0.1", port=8000, timeout=1800)

batch_size = 2
head_image = np.random.randint(0, 255, (batch_size, 256, 256, 3), dtype=np.uint8)
wrist_image = np.random.randint(0, 255, (batch_size, 256, 256, 3), dtype=np.uint8)
action = np.random.randn(batch_size, 7).astype(np.float32)

result = client.edit(
    head_image=head_image,
    wrist_image=wrist_image,
    action=action,
    num_timesteps=30,
)

next_head = result["next_head"]
next_wrist = result["next_wrist"]
```

Reward-model usage:

```python
image = np.random.randint(0, 255, (batch_size, 256, 256, 3), dtype=np.uint8)
tasks = [
    "pick up the mug and place it on the plate",
    "open the drawer",
]

result = client.understand(image=image, task=tasks)
rewards = result["reward"]
```

## API

### `WebsocketWorldModelClient`

#### `edit(head_image, wrist_image, action, **kwargs)`

Generates the next head-camera and wrist-camera observations.

Inputs:

- `head_image`: array with shape `(N, H, W, 3)`, dtype `uint8`, RGB.
- `wrist_image`: array with shape `(N, H, W, 3)`, dtype `uint8`, RGB.
- `action`: action array with shape `(N, action_dim)` or `(N, chunk_size, action_dim)`.
- `**kwargs`: optional inference overrides such as `cfg_text_scale`, `cfg_img_scale`, `num_timesteps`, `save_images`, `save_dir`, and `save_prefix`.

Returns:

- `{"next_head": np.ndarray, "next_wrist": np.ndarray}`

#### `understand(image, task, **kwargs)`

Evaluates whether each task is complete in the corresponding image.

Inputs:

- `image`: array with shape `(N, H, W, 3)`, dtype `uint8`, RGB.
- `task`: list of `N` task descriptions.
- `**kwargs`: optional inference overrides such as `max_length`, `temperature`, and `do_sample`.

Returns:

- `{"reward": list[bool]}`

#### `reset()`

Resets server-side worker state.

#### `get_server_metadata()`

Returns metadata advertised by the WebSocket server.

## Notes

- Images must be RGB `uint8` arrays.
- Editing is slower than reward inference because it runs image generation.
- Use a long client timeout for large images or slow generation.
- For best throughput, set `num_workers` to match the number of available GPUs.
- Keep `ACTION_NORM_PATH` consistent with the policy/action space used to produce actions.

## Troubleshooting

- **Connection refused**: verify that the WebSocket server is running and that host/port match the client configuration.
- **CUDA out of memory**: reduce `--max-mem-per-gpu`, reduce worker count, or use smaller inputs.
- **Invalid image format**: ensure all images are RGB arrays with dtype `uint8`.
- **Slow editing**: reduce `num_timesteps` or lower image resolution.
- **Unexpected rewards**: verify task wording and confirm that the reward prompt matches the environment semantics.

## License

Apache-2.0
