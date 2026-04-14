"""
LIBERO Environment for RL Training

This module provides a wrapper around libero_client for LIBERO environment interactions.

Supports fully parallel inference over batch_size * num_trajectory environments.
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import jax
import jax.numpy as jnp

from openpi.models.model import Observation, Actions
import openpi.models.model as _model
import openpi.shared.array_typing as at
from openpi.shared import image_tools
import openpi.training.utils as training_utils
import openpi.transforms as _transforms
import openpi.training.sharding as _sharding

# Import libero_client
import sys
from pathlib import Path
libero_client_path = Path(__file__).parent.parent.parent.parent / "libero_client"
if str(libero_client_path) not in sys.path:
    sys.path.insert(0, str(libero_client_path))

from libero_websocket_client import LiberoWebsocketClient

class LiberoRLEnvironment:
    """LIBERO environment wrapper for RL training.
    
    Connects to one LIBERO server that manages batch_size * num_trajectories parallel environments.
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8003,
        chunk_size: int = 10,
        sequence_len: int = 3,
        camera_views: List[str] = None,
        batch_size: int = 4,
        num_trajectories: int = 8,
        norm_stats: Optional[Dict] = None,
        use_quantile_norm: bool = False,
        timeout: int = 150,
        input_transforms: Optional[Any] = None,
        world_model_client: Optional[Any] = None,
    ):
        """Initialize LIBERO RL environment.
        
        Args:
            host: LIBERO server host
            port: LIBERO server port
            chunk_size: Number of actions per chunk
            sequence_len: Sequence length for trajectories
            camera_views: List of camera view names
            batch_size: Batch size for training
            num_trajectories: Number of trajectories per batch
            norm_stats: Action normalization statistics
            use_quantile_norm: Whether to use quantile normalization
            timeout: Connection timeout
            input_transforms: Transform pipeline for processing observations
            world_model_client: Optional world model client for simulation
        """
        self.host = host
        self.port = port
        self.chunk_size = chunk_size
        self.sequence_len = sequence_len
        self.batch_size = batch_size
        self.num_trajectories = num_trajectories
        self.total_envs = batch_size * num_trajectories 
        self.norm_stats = norm_stats
        self.world_model_client = world_model_client
        self.unnormalize_transform = None

        if norm_stats is not None:
            self.unnormalize_transform = _transforms.Unnormalize(
                norm_stats=norm_stats,
                use_quantiles=use_quantile_norm
            )
        self.camera_views = camera_views
        
        # Connect to a single LIBERO server.
        try:
            self.client = LiberoWebsocketClient(host=host, port=port, timeout=timeout)
            logging.info(f"Connected to LIBERO server at {host}:{port}")
        except Exception as e:
            logging.error(f"Failed to connect to LIBERO server at {host}:{port}: {e}")
            raise
        
        # Read server metadata.
        self.metadata = self.client.get_server_metadata()
        
        # Verify that the server has enough environments.
        server_num_envs = self.metadata.get('num_envs', 1)
        if server_num_envs < self.total_envs:
            raise ValueError(
                f"Server only supports {server_num_envs} environments, "
                f"but {self.total_envs} are required (batch_size={batch_size} * "
                f"num_trajectories={num_trajectories}). Please start server with "
                f"--num-envs {self.total_envs}" 
            )
        
        # Transform pipeline for observations.
        self.input_transforms = input_transforms
        
        self.current_observations = None
        self.initialized = False
        
        # Timing stats.
        self.timing_info = {
            'action_inference_time': 0.0,
            'env_step_time': 0.0,
            'obs_processing_time': 0.0,
            'world_model_time': 0.0,
        }
        
        logging.info(
            f"Initialized LIBERO environment with {self.total_envs} parallel environments "
            f"(batch_size={batch_size}, num_trajectories={num_trajectories})"
        )
        if self.input_transforms is not None:
            logging.info(f"Using input transforms for observation processing")
    
    def reset(self, sim_state: np.ndarray, sim_state_len: np.ndarray, task_id: np.ndarray) -> Dict:
        """Reset all environments to the same scene."""
        observations = self.client.reset(sim_state=sim_state,sim_state_len=sim_state_len,task_id=task_id)
        self.initialized = True
        return observations
    
    def chunk_step(self, chunk_actions: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Execute chunk actions in all environments.
        
        Args:
            chunk_actions: Chunk actions array [total_envs, chunk_size, action_dim]
            
        Returns:
            Tuple of (observations, rewards, terminations, truncations)
        """
        step_start = time.time()
        
        response = self.client.chunk_step(chunk_actions)
        observations = response["observation"]
        rewards = response["rewards"]
        terminations = response["terminations"]
        truncations = response["truncations"]
        
        self.timing_info['env_step_time'] += time.time() - step_start
        
        return observations, rewards, terminations, truncations
    
    def step_with_world_model(
        self, 
        action: np.ndarray, 
        current_obs: _model.Observation
    ) -> _model.Observation:
        """
        Step environment with action using model observation format.
        This method handles the world-model conversion internally.
        
        Args:
            action: Action array of shape (batch_size, chunk_size, action_dim)
            current_obs: Current model observation
            
        Returns:
            Next model observation.
        """
        if self.world_model_client is None:
            raise ValueError("World model client not initialized")
        
        # Update current observations.
        self.current_observations = current_obs
        
        # Use the stored current observations.
        if self.current_observations is None:
            raise RuntimeError("No current observations available. Initialize current_observations first.")
        
        next_observations = {}
        # Predict the next frame for each camera view.
        current_head_image = np.array(jax.device_get(self.current_observations.images['base_0_rgb']))  # (batch_size, H, W, 3)
        current_wrist_image = np.array(jax.device_get(self.current_observations.images['left_wrist_0_rgb']))  # (batch_size, H, W, 3)

        current_head_image = ((current_head_image + 1.0) * 127.5).astype(np.uint8)
        current_wrist_image = ((current_wrist_image + 1.0) * 127.5).astype(np.uint8)
        # Time world-model inference.
        wm_start = time.time()
        result = self.world_model_client.edit(
            head_image=current_head_image,     # (batch_size, 1, H, W, 3)
            wrist_image=current_wrist_image,     # (batch_size, 1, H, W, 3)
            action=action,    # (batch_size, 1, chunk_size, action_dim)
        )
        wm_time = time.time() - wm_start
        
        # Store timing stats.
        self.timing_info['world_model_time'] = self.timing_info.get('world_model_time', 0.0) + wm_time
        
        # Extract results.
        predicted_image = image_tools.resize_with_pad(result['next_head'], 224, 224)  # (batch_size, H, W, 3)
        predicted_wrist_image = image_tools.resize_with_pad(result['next_wrist'], 224, 224)  # (batch_size, H, W, 3)
        predicted_image = predicted_image / 127.5 - 1.0
        predicted_wrist_image = predicted_wrist_image / 127.5 - 1.0
        next_observations['base_0_rgb'] = predicted_image
        next_observations['left_wrist_0_rgb'] = predicted_wrist_image
        
        # Convert back to model observation format.
        next_images = {}
        for model_key in current_obs.images.keys():
            if model_key in next_observations:
                next_images[model_key] = jnp.array(next_observations[model_key])
            else:
                # Keep the current image if it was not predicted.
                next_images[model_key] = current_obs.images[model_key]
        
        # Build the next observation.
        next_obs = _model.Observation(
            images=next_images,
            image_masks=current_obs.image_masks,
            state=current_obs.state,  # Keep the current state until a learned state model is added.
            tokenized_prompt=current_obs.tokenized_prompt,
            tokenized_prompt_mask=current_obs.tokenized_prompt_mask,
        )
        
        return next_obs

    def compute_reward(
        self, 
        observation: _model.Observation,
        task: List[str]
    ) -> np.ndarray:
        """
        Compute current observation's reward.
        
        Args:
            observation: Current model observation
            task: List of task prompts
            
        Returns:
            Reward array.
        """
        if self.world_model_client is None:
            raise ValueError("World model client not initialized")
        
        # Predict the reward for the current observation.
        current_image = np.array(jax.device_get(observation.images['base_0_rgb']))  # (batch_size, H, W, 3)

        current_image = ((current_image + 1.0) * 127.5).astype(np.uint8)
        
        # Time world-model inference.
        wm_start = time.time()
        result = self.world_model_client.understand(
            image=current_image,     # (batch_size, 1, H, W, 3)
            task=task,
        )
        wm_time = time.time() - wm_start
        
        # Store timing stats.
        self.timing_info['reward_model_time'] = self.timing_info.get('reward_model_time', 0.0) + wm_time
        
        reward = np.array(result['reward'])
        
        return reward
    
    def unnormalize(self, actions: np.ndarray, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unnormalize actions using the stored normalization statistics.
        
        Args:
            actions: Normalized actions array
            state: State array
            
        Returns:
            Tuple of (unnormalized_actions, unnormalized_state)
        """
        if self.unnormalize_transform is None:
            logging.warning("No normalization statistics provided, actions will not be unnormalized")
            return actions, state
            
        # Create a data dict that matches the expected structure for transforms
        data_dict = {"actions": actions, "state": state}
        
        # Apply unnormalization transform
        unnormalized_data = self.unnormalize_transform(data_dict)
        
        return unnormalized_data["actions"], unnormalized_data["state"]
    
    def get_server_metadata(self) -> Dict:
        """Get server metadata."""
        return self.metadata
    
    def close(self):
        """Close environment connection."""
        self.client.close()
        logging.info("LIBERO environment connection closed")
