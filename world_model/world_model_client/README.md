# Bagel Client

这是一个用于 Bagel 模型的远程推理客户端，支持通过 WebSocket 协议进行远程调用。Bagel 模型提供两种核心能力：

1. **Editing（编辑模式/World Model）**：输入图像 + 文本指令 → 输出下一帧图像
2. **Understanding（理解模式/Reward Model）**：输入图像 + 文本问题 → 输出文本回答

## 文件结构

```
bagel_client/
├── __init__.py                      # 包初始化文件
├── base_bagel_model.py              # Bagel 模型基类接口
├── msgpack_numpy.py                 # NumPy 数组序列化支持
├── bagel_inference_server.py        # Bagel 推理服务器实现
├── websocket_bagel_server.py        # WebSocket 服务器
├── websocket_bagel_client.py        # WebSocket 客户端
├── example_usage.py                 # 使用示例
└── README.md                        # 本文件
```

## 安装依赖

```bash
pip install websockets msgpack numpy torch Pillow accelerate
```

## 使用方法

### 1. 启动服务器

首先启动 Bagel 推理服务器：

```bash
cd /home/work3/VLA_MBRL/Bagel/bagel_client

python websocket_bagel_server.py \
    --model-path /path/to/BAGEL-7B-MoT/weights \
    --max-mem-per-gpu "40GiB" \
    --host 0.0.0.0 \
    --port 8002 \
    --device cuda
```

#### 服务器参数说明：

**模型配置：**
- `--model-path`: Bagel 模型权重目录路径（必需）
- `--max-mem-per-gpu`: 每个 GPU 的最大内存（默认 "40GiB"）
- `--device`: 推理设备（默认 "cuda"）

**服务器配置：**
- `--host`: 服务器绑定的主机地址（默认 "0.0.0.0"）
- `--port`: 服务器绑定的端口（默认 8002）

**Editing（World Model）超参数：**
- `--edit-cfg-text-scale`: 文本 CFG 引导强度（默认 4.0）
- `--edit-cfg-img-scale`: 图像 CFG 引导强度（默认 2.0）
- `--edit-timestep-shift`: 时间步偏移（默认 3.0）
- `--edit-num-timesteps`: 去噪步数（默认 50）
- `--edit-cfg-renorm-type`: CFG 重归一化类型，可选 "global", "channel", "text_channel"（默认 "text_channel"）

**Understanding（Reward Model）超参数：**
- `--understand-max-tokens`: 最大生成 token 数（默认 1000）
- `--understand-temperature`: 采样温度（默认 0.3）

### 2. 使用客户端

#### 基础使用

```python
from bagel_client import WebsocketBagelClient
import numpy as np

# 创建客户端
bagel_client = WebsocketBagelClient(host="0.0.0.0", port=8002)

# 模式 1: Editing（World Model）
# 输入：图像 + 文本指令 → 输出：下一帧图像
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
prompt = "Add a sunset in the background"

result = bagel_client.edit(image=image, prompt=prompt)
next_image = result['next_image']  # shape: (512, 512, 3)

# 模式 2: Understanding（Reward Model）
# 输入：图像 + 文本问题 → 输出：文本回答
question = "Describe what you see in this image"

result = bagel_client.understand(image=image, prompt=question)
text_answer = result['text']
```

#### 覆盖默认超参数

```python
# Editing 时覆盖超参数
result = bagel_client.edit(
    image=image,
    prompt="Make it more colorful",
    cfg_text_scale=5.0,      # 覆盖文本 CFG
    cfg_img_scale=1.5,       # 覆盖图像 CFG
    num_timesteps=30,        # 使用更少的步数加速
)

# Understanding 时覆盖超参数
result = bagel_client.understand(
    image=image,
    prompt="What is happening in this scene?",
    max_think_token_n=500,   # 限制输出长度
    temperature=0.5,         # 调整采样温度
)
```

#### World Model 序列推理

```python
# 模拟 world model 的连续预测
current_image = initial_image.copy()
actions = [
    "Zoom in slightly",
    "Increase brightness",
    "Add motion blur"
]

predicted_frames = [current_image]
for action in actions:
    result = bagel_client.edit(image=current_image, prompt=action)
    current_image = result['next_image']
    predicted_frames.append(current_image)
```

### 3. 运行示例

```bash
cd bagel_client
python example_usage.py
```

## API 接口

### WebsocketBagelClient

#### 方法

**`edit(image, prompt, **kwargs)`** - 图像编辑（World Model 模式）
- **输入：**
  - `image`: 输入图像，形状 (H, W, 3)，dtype uint8，RGB 格式
  - `prompt`: 编辑指令文本
  - `**kwargs`: 可选的超参数覆盖
- **返回：** `{'next_image': np.ndarray}`
  - `next_image`: 预测的下一帧图像，形状 (H, W, 3)，dtype uint8

**`understand(image, prompt, **kwargs)`** - 图像理解（Reward Model 模式）
- **输入：**
  - `image`: 输入图像，形状 (H, W, 3)，dtype uint8，RGB 格式
  - `prompt`: 问题或指令文本
  - `**kwargs`: 可选的超参数覆盖
- **返回：** `{'text': str}`
  - `text`: 理解/回答的文本输出

**`reset()`** - 重置模型状态

**`get_server_metadata()`** - 获取服务器元数据

#### 可覆盖的超参数

**Editing 模式：**
- `cfg_text_scale`: 文本 CFG 引导强度（默认 4.0）
- `cfg_img_scale`: 图像 CFG 引导强度（默认 2.0）
- `cfg_interval`: CFG 应用区间（默认 [0.0, 1.0]）
- `timestep_shift`: 时间步偏移（默认 3.0）
- `num_timesteps`: 去噪步数（默认 50）
- `cfg_renorm_min`: CFG 重归一化最小值（默认 0.0）
- `cfg_renorm_type`: CFG 重归一化类型（默认 "text_channel"）

**Understanding 模式：**
- `max_think_token_n`: 最大 token 数（默认 1000）
- `do_sample`: 是否采样（默认 False）
- `text_temperature`: 采样温度（默认 0.3）

## 特性

- **双模式支持**：同时支持图像编辑（World Model）和图像理解（Reward Model）
- **远程推理**：通过 WebSocket 进行高效的远程推理
- **高效序列化**：使用 msgpack 进行 NumPy 数组的快速序列化
- **灵活配置**：支持运行时覆盖默认超参数
- **错误处理**：完善的错误处理和重连机制
- **易于集成**：与 OpenPI 的 world_model_client 设计理念一致

## 使用场景

### 1. 作为 World Model
```python
# 用于基于模型的强化学习（MBRL）
# 预测给定动作后的下一帧状态
next_state = bagel_client.edit(
    image=current_state,
    prompt=f"Apply action: {action_description}"
)['next_image']
```

### 2. 作为 Reward Model
```python
# 评估当前状态的质量或给出反馈
reward_text = bagel_client.understand(
    image=current_state,
    prompt="Evaluate the quality of this state. Is the task completed?"
)['text']

# 解析 reward_text 以获取数值奖励
```

## 注意事项

1. **图像格式**：所有图像必须是 RGB 格式，dtype 为 uint8，形状为 (H, W, 3)
2. **内存需求**：服务器需要足够的 GPU 内存加载 Bagel 模型（建议至少 40GB）
3. **推理速度**：
   - Editing 模式较慢（~5-10秒/帧），因为需要多步去噪
   - Understanding 模式较快（~1-2秒），只需要文本生成
4. **网络延迟**：建议在本地网络或同一台机器上使用，以减少传输延迟
5. **图像尺寸**：模型会自动调整图像大小，但推荐使用 512x512 或 1024x1024

## 故障排除

### 常见问题

1. **ModuleNotFoundError**：确保 Bagel 根目录在 Python 路径中
2. **连接失败**：检查服务器是否正在运行，端口是否正确
3. **CUDA 内存不足**：减少 `max_mem_per_gpu` 或使用多 GPU 分布式加载
4. **图像格式错误**：确保图像为 uint8，RGB 格式（不是 BGR）
5. **推理过慢**：尝试减少 `num_timesteps` 或使用 TaylorSeer 加速

### 调试

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 与 OpenPI World Model Client 的对比

| 特性 | OpenPI World Model | Bagel Client |
|------|-------------------|--------------|
| 输入 | image + action (numeric) | image + prompt (text) |
| World Model 输出 | next_image + next_qpos | next_image |
| Reward Model | 需要单独实现 | 内置 understanding 模式 |
| 模型类型 | 动力学模型 | 视觉-语言生成模型 |
| 推理速度 | 快 (~0.1s) | 中等 (~5s editing, ~1s understanding) |

## 许可证

Copyright 2025 Bytedance Ltd. and/or its affiliates.
SPDX-License-Identifier: Apache-2.0

