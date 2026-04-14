# LIBERO Client

LIBERO Client是一个基于WebSocket的客户端/服务器实现，用于远程执行LIBERO环境任务。它允许您通过网络连接远程访问LIBERO环境，实现分布式机器人学习和评估。

## 功能特性

- **WebSocket通信**: 基于WebSocket协议的高效双向通信
- **完整的LIBERO环境接口**: 支持所有LIBERO环境的核心功能
- **多任务套件支持**: 支持LIBERO的多个任务套件(libero_spatial, libero_object, libero_goal, libero_10, libero_90)
- **numpy数组序列化**: 高效的numpy数组网络传输
- **错误处理**: 完善的错误处理和重连机制
- **多环境支持**: 可同时连接多个LIBERO环境实例
- **自包含设计**: 无需外部依赖（如rlinf），所有工具函数已内置实现

## 安装要求

```bash
# 核心依赖
pip install websockets msgpack numpy torch

# LIBERO环境依赖（服务器端需要）
pip install libero

# 图像处理（必需）
pip install opencv-python

# Gym环境支持
pip install gym

# 配置管理
pip install omegaconf
```

## 模块说明

### 核心模块

- `libero_websocket_client.py`: WebSocket客户端实现
- `libero_websocket_server.py`: WebSocket服务器实现
- `libero_client_env.py`: Gym风格的客户端环境包装器
- `msgpack_numpy.py`: NumPy数组的MessagePack序列化支持
- `utils.py`: 工具函数集合（图像处理、状态转换等）

所有依赖都已自包含实现，无需外部框架（如rlinf）支持。

## 快速开始

### 1. 启动LIBERO服务器

```bash
# 启动单个LIBERO服务器
python libero_websocket_server.py --task-suite-name libero_spatial --port 8003

# 或者使用示例脚本
python example_usage.py server --task-suite-name libero_spatial --port 8003
```

### 2. 使用LIBERO客户端

```python
from libero_client import LiberoWebsocketClient
import numpy as np

# 连接到LIBERO服务器
with LiberoWebsocketClient(host="localhost", port=8003) as client:
    # 重置环境
    obs = client.reset()
    
    # 执行动作
    action = np.random.randn(7)  # 7自由度动作
    obs, reward, done, info = client.step(action)
    
    # 渲染当前状态
    img = client.render()
```

## API文档

### LiberoWebsocketClient

主要的客户端类，提供与LIBERO环境交互的接口。

#### 方法

- `reset(task_id=None, init_state_id=None)`: 重置环境
- `step(action)`: 执行动作
- `get_obs()`: 获取当前观察
- `get_info()`: 获取当前环境信息
- `render(mode="rgb_array", width=256, height=256)`: 渲染环境
- `check_success()`: 检查任务是否成功
- `set_init_state(init_state)`: 设置初始状态
- `get_task_info()`: 获取任务信息
- `seed(seed)`: 设置随机种子
- `close()`: 关闭连接

#### 观察空间

LIBERO环境返回的观察包含：

```python
{
    'agentview_image': np.ndarray,        # 主相机图像 (H, W, 3)
    'robot0_eye_in_hand_image': np.ndarray,  # 夹爪相机图像 (H, W, 3)
    'robot0_eef_pos': np.ndarray,        # 末端执行器位置 (3,)
    'robot0_eef_quat': np.ndarray,       # 末端执行器四元数 (4,)
    'robot0_gripper_qpos': np.ndarray,   # 夹爪位置 (1,)
}
```

#### 动作空间

LIBERO环境接受7自由度的连续动作：

```python
action = np.array([
    delta_x,      # X轴位移
    delta_y,      # Y轴位移  
    delta_z,      # Z轴位移
    delta_roll,   # 翻滚角
    delta_pitch,  # 俯仰角
    delta_yaw,    # 偏航角
    gripper       # 夹爪开合 (-1关闭, 1打开)
])
```

### LiberoWebsocketServer

服务器端实现，包装LIBERO环境并提供WebSocket接口。

#### 初始化参数

- `task_suite_name`: 任务套件名称，可选值: libero_spatial, libero_object, libero_goal, libero_10, libero_90
- `host`: 绑定的主机地址
- `port`: 绑定的端口号
- `resolution`: 图像分辨率，默认256

## 使用示例

### 基础使用

```python
from libero_client import LiberoWebsocketClient
import numpy as np

# 连接到服务器
client = LiberoWebsocketClient("localhost", 8003)

# 重置到特定任务
obs = client.reset(task_id=0)

# 执行一系列动作
for i in range(100):
    action = np.random.uniform(-0.1, 0.1, 7)
    obs, reward, done, info = client.step(action)
    
    if done:
        obs = client.reset()
        break

client.close()
```

### 多环境并行使用

```python
from libero_client import make_libero_envs

# 创建多个环境客户端
envs = make_libero_envs(host="localhost", port=8003, num_envs=3)

# 并行重置所有环境
observations = [env.reset() for env in envs]

# 并行执行动作
for step in range(50):
    actions = [np.random.randn(7) for _ in envs]
    results = [env.step(action) for env, action in zip(envs, actions)]
    
    for i, (obs, reward, done, info) in enumerate(results):
        if done:
            envs[i].reset()

# 关闭所有环境
for env in envs:
    env.close()
```

### 任务评估

```python
from libero_client import LiberoWebsocketClient

def evaluate_policy(policy, num_episodes=10):
    """评估策略在LIBERO环境中的性能"""
    client = LiberoWebsocketClient("localhost", 8003)
    
    total_reward = 0
    success_count = 0
    
    for episode in range(num_episodes):
        obs = client.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = policy.predict(obs)  # 使用您的策略
            obs, reward, done, info = client.step(action)
            episode_reward += reward
            
        total_reward += episode_reward
        if client.check_success():  # 检查任务成功
            success_count += 1
            
    client.close()
    
    avg_reward = total_reward / num_episodes
    success_rate = success_count / num_episodes
    
    return avg_reward, success_rate
```

## 服务器部署

### 单服务器部署

```bash
# 启动LIBERO服务器
python -m libero_client.libero_websocket_server \
    --task-suite-name libero_spatial \
    --host 0.0.0.0 \
    --port 8003
```

### 多服务器部署

```bash
# 启动多个服务器实例（不同端口）
for port in 8003 8004 8005; do
    python -m libero_client.libero_websocket_server \
        --task-suite-name libero_spatial \
        --port $port &
done
```

## 网络配置

### 防火墙设置

确保服务器端口已开放：

```bash
# Ubuntu/Debian
sudo ufw allow 8003

# CentOS/RHEL
sudo firewall-cmd --add-port=8003/tcp --permanent
sudo firewall-cmd --reload
```

### Docker部署

```dockerfile
FROM python:3.8

# 安装依赖
RUN pip install websockets msgpack numpy opencv-python imageio libero

# 复制代码
COPY libero_client/ /app/libero_client/
WORKDIR /app

# 暴露端口
EXPOSE 8003

# 启动服务器
CMD ["python", "-m", "libero_client.libero_websocket_server", \
     "--task-suite-name", "libero_spatial", "--host", "0.0.0.0", "--port", "8003"]
```

## 故障排除

### 常见问题

1. **连接被拒绝**
   - 确保服务器正在运行
   - 检查端口是否正确
   - 验证防火墙设置

2. **数据传输错误**
   - 检查网络连接稳定性
   - 验证numpy数组格式
   - 确保msgpack版本兼容

3. **LIBERO环境加载失败**
   - 验证LIBERO环境是否正确安装
   - 检查任务套件名称是否正确
   - 确保有足够的系统资源

### 调试模式

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 创建客户端时设置更长的超时
client = LiberoWebsocketClient("localhost", 8003, timeout=60)
```

## 性能优化

### 网络优化

- 使用局域网连接减少延迟
- 调整WebSocket缓冲区大小
- 启用数据压缩（如需要）

### 内存优化

- 限制图像分辨率
- 使用适当的numpy数据类型
- 及时释放大型数组

## 开发指南

### 扩展功能

您可以通过继承基类来扩展功能：

```python
from libero_client import LiberoWebsocketClient

class CustomLiberoClient(LiberoWebsocketClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def custom_method(self):
        # 添加自定义功能
        pass
```

### 测试

```bash
# 运行单元测试
python -m pytest libero_client/tests/

# 运行示例
python libero_client/example_usage.py client --host localhost --port 8003
```

## 许可证

本项目基于MIT许可证发布。详见LICENSE文件。
