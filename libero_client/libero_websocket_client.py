"""
Simplified LIBERO WebSocket client.

Supports fully parallel inference over batch_size * num_trajectory environments.
"""

import logging
import time
import numpy as np
from typing import Dict, Tuple
import websockets.sync.client
import websockets.exceptions

try:
    from . import msgpack_numpy
except ImportError:
    import msgpack_numpy


class LiberoWebsocketClient:
    """WebSocket client for LIBERO environments."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8003, timeout: int = 150):
        """Initialize the WebSocket client.
        
        Args:
            host: Server host.
            port: Server port.
            timeout: Connection timeout in seconds.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self._packer = msgpack_numpy.Packer()
        
        # Connect to the server.
        self._ws, self._server_metadata = self._wait_for_server()
        
    def _wait_for_server(self):
        """Connect to the WebSocket server."""
        uri = f"ws://{self.host}:{self.port}"
        
        logging.info(f"Waiting for LIBERO server at {uri}...")
        while True:
            try:
                # Configure client with appropriate timeouts
                conn = websockets.sync.client.connect(
                    uri, 
                    compression=None, 
                    max_size=None,
                    open_timeout=self.timeout,
                    close_timeout=self.timeout,
                    ping_timeout=None,  # Disable ping timeout to avoid keepalive failures
                    ping_interval=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logging.info("Successfully connected to LIBERO server")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for LIBERO server...")
                time.sleep(5)
            
    def _reconnect(self):
        """Reconnect to the server."""
        logging.warning("Connection lost. Attempting to reconnect...")
        try:
            if self._ws is not None:
                try:
                    self._ws.close()
                except:
                    pass
        except:
            pass
        
        self._ws, self._server_metadata = self._wait_for_server()
        logging.info("Successfully reconnected to LIBERO server")
    
    def _send_recv(self, data: Dict, max_retries: int = 3) -> Dict:
        """Send a request and wait for the response with auto-retry.
        
        Args:
            data: Request payload.
            max_retries: Maximum retry count.
            
        Returns:
            Response dictionary.
        """
        for attempt in range(max_retries):
            try:
                packed_data = self._packer.pack(data)
                self._ws.send(packed_data)
                response_msg = self._ws.recv()
                if isinstance(response_msg, str):
                    raise RuntimeError(f"Error in LIBERO server:\n{response_msg}")
                
                response = msgpack_numpy.unpackb(response_msg)
                return response
            except (websockets.exceptions.ConnectionClosedError, 
                    websockets.exceptions.ConnectionClosed,
                    OSError,
                    BrokenPipeError) as e:
                logging.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    self._reconnect()
                    continue
                else:
                    raise RuntimeError(f"Failed to send/receive after {max_retries} attempts: {e}")
            except Exception as e:
                logging.error(f"Unexpected error in _send_recv: {e}")
                raise
            
    def reset(self, sim_state: np.ndarray, sim_state_len: np.ndarray, task_id: np.ndarray) -> Dict:
        """Reset all environments to the same scene.
        
        Args:
            sim_state: Simulator state array.
            sim_state_len: Simulator state length.
            task_id: Task id.
        Returns:
            Initial observation dictionary.
        """
        data = {
            "method": "reset",
            "sim_state": sim_state,
            "sim_state_len": sim_state_len,
            "task_id": task_id,
        }
        
        response = self._send_recv(data)
        return response["observation"]
        
    def step(self, actions: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Run one environment step.
        
        Args:
            actions: Action array with shape [num_envs, action_dim].
            
        Returns:
            Tuple of (observation, rewards, terminations, truncations).
        """
        data = {
            "method": "step",
            "actions": actions,
        }
        
        response = self._send_recv(data)
        
        return (
            response["observation"],
            response["rewards"],
            response["terminations"],
            response["truncations"],
        )
        
    def chunk_step(self, chunk_actions: np.ndarray) -> Dict:
        """Run a chunk of actions.
        
        Args:
            chunk_actions: Action array with shape [num_envs, chunk_size, action_dim].
            
        Returns:
            Dictionary containing observations, rewards, and done flags.
        """
        data = {
            "method": "chunk_step",
            "chunk_actions": chunk_actions,
        }
        
        response = self._send_recv(data)
        
        return {
            "observation": response["observation"],
            "rewards": response["rewards"],
            "terminations": response["terminations"],
            "truncations": response["truncations"],
        }
        
    def get_server_metadata(self) -> Dict:
        """Return server metadata."""
        return self._server_metadata
        
    def close(self):
        """Close the connection."""
        if self._ws is not None:
            try:
                # Send the close command.
                data = {"method": "close"}
                packed_data = self._packer.pack(data)
                self._ws.send(packed_data)
                
                # Wait for the acknowledgement.
                response_msg = self._ws.recv()
                
                # Close the socket.
                self._ws.close()
                self._ws = None
                
                logging.info("Closed connection to LIBERO server")
            except Exception as e:
                logging.warning(f"Error during close: {e}")
                if self._ws is not None:
                    self._ws.close()
                    self._ws = None
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        

if __name__ == "__main__":
    # Example usage.
    logging.basicConfig(level=logging.INFO)
    
    # Create a client.
    client = LiberoWebsocketClient()
    
    # Reset to a specific scene.
    obs = client.reset(sim_state=np.array([0]))
    logging.info(f"Initial observation keys: {obs.keys()}")
    
    # Run a random action test.
    num_envs = client.get_server_metadata()["num_envs"]
    random_actions = np.random.randn(num_envs, 7) * 0.1
    obs, rewards, terminations, truncations = client.step(random_actions)
    logging.info(f"Step completed, rewards: {rewards}")
    
    client.close()
