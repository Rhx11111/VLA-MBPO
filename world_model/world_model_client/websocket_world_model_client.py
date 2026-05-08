# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Dict, Tuple, Optional

import websockets.sync.client
from typing_extensions import override
import numpy as np

try:
    from . import base_world_model as _base_world_model
    from . import msgpack_numpy
except ImportError:
    # Fallback for when running as script
    import base_world_model as _base_world_model
    import msgpack_numpy


class WebsocketWorldModelClient(_base_world_model.BaseWorldModel):
    """
    World Model Client that communicates with server over websocket.
    
    Supports two modes:
    1. Editing (world model): image + prompt -> next_image
    2. Understanding (reward model): image + prompt -> text
    
    See WebsocketWorldModelServer for the corresponding server implementation.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002, timeout: int = 60) -> None:
        """
        Initialize WorldModel client.
        
        Args:
            host: Server host address
            port: Server port
            timeout: Connection timeout in seconds
        """
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._timeout = timeout
        self._ws, self._server_metadata = self._wait_for_server()
    
    def get_server_metadata(self) -> Dict:
        """Get server metadata including capabilities."""
        return self._server_metadata
    
    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """Wait for server to be available and connect."""
        logging.info(f"Waiting for WorldModel server at {self._uri}...")
        while True:
            try:
                # Configure client with appropriate timeouts for large data transfers
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=self._timeout,
                    close_timeout=self._timeout,
                    ping_timeout=None,  
                    ping_interval=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logging.info(f"Connected to WorldModel server: {metadata}")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for WorldModel server...")
                time.sleep(5)
    
    def _send_request(self, request: Dict) -> Dict:
        """Send request to server and get response."""
        data = self._packer.pack(request)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in WorldModel inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
    
    @override
    def edit(self, head_image: np.ndarray, wrist_image: np.ndarray, action: np.ndarray, **kwargs) -> Dict:
        """
        Edit/generate next frame images based on current multi-view images and action (world model mode).
        
        Uses a two-stage generation process:
        1. Generate next head camera view: [current_head, current_wrist] + action -> next_head
        2. Generate next wrist camera view: [next_head, current_wrist] -> next_wrist
        
        Args:
            head_image: Current head camera image, shape (H, W, 3), dtype uint8, RGB format
            wrist_image: Current wrist camera image, shape (H, W, 3), dtype uint8, RGB format
            action: Action array, continuous values that will be normalized
            **kwargs: Additional inference parameters to override defaults
                     (e.g., cfg_text_scale, cfg_img_scale, num_timesteps, etc.)
        
        Returns:
            Dict containing:
                - 'next_head': Next predicted head camera image, shape (H, W, 3), dtype uint8
                - 'next_wrist': Next predicted wrist camera image, shape (H, W, 3), dtype uint8
        """
        request = {
            "method": "edit",
            "head_image": head_image,
            "wrist_image": wrist_image,
            "action": action,
            "kwargs": kwargs,
        }
        return self._send_request(request)
    
    @override
    def understand(self, image: np.ndarray, task: list, **kwargs) -> Dict:
        """
        Understand image and answer question (reward model mode).
        
        Args:
            image: Input image, shape (H, W, 3), dtype uint8, RGB format
            task: Task prompt
            **kwargs: Additional inference parameters to override defaults
                     (e.g., max_think_token_n, temperature, etc.)
        
        Returns:
            Dict containing:
                - 'text': Understanding/reward text output
        """
        request = {
            "method": "understand",
            "image": image,
            "task": task,
            "kwargs": kwargs,
        }
        return self._send_request(request)
    
    @override
    def reset(self) -> None:
        """Reset the WorldModel model to its initial state."""
        request = {"method": "reset"}
        self._send_request(request)

