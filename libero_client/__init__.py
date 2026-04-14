"""
LIBERO Client Package

This package provides websocket-based client and server implementations for LIBERO environments,
allowing remote execution of LIBERO tasks over a network connection.

Components:
- libero_websocket_client: Client for connecting to LIBERO websocket servers
- libero_websocket_server: Server that wraps LIBERO environments with websocket interface
- msgpack_numpy: MessagePack serialization with NumPy array support

Example usage:

Client side:
```python
from libero_client import LiberoWebsocketClient

# Connect to LIBERO server
with LiberoWebsocketClient(host="localhost", port=8003) as client:
    obs = client.reset()
    action = np.random.randn(7)
    obs, reward, done, info = client.step(action)
```

Server side:
```python
from libero_client import LiberoServerEnv, LiberoWebsocketServer

# Create and run LIBERO server
server_env = LiberoServerEnv(task_suite_name="libero_spatial", num_envs=1)
websocket_server = LiberoWebsocketServer(server_env, host="0.0.0.0", port=8003)
websocket_server.run()
```
"""

from .libero_websocket_client import LiberoWebsocketClient, make_libero_envs
from .libero_websocket_server import LiberoServerEnv, LiberoWebsocketServer
from . import msgpack_numpy

__version__ = "0.1.0"
__all__ = [
    "LiberoWebsocketClient",
    "LiberoServerEnv",
    "LiberoWebsocketServer",
    "make_libero_envs",
    "msgpack_numpy"
]
