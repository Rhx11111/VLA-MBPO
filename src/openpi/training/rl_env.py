"""
RL Environment Utilities for Training

This module provides RL-related utilities for collecting trajectories,
computing returns, and managing trajectory buffers for RL training.

Supports fully parallel inference over batch_size * num_trajectory environments.
"""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Literal
import numpy as np
import imageio
import jax
import jax.numpy as jnp
from tqdm import tqdm

from openpi.models.model import Observation, Actions
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.utils as training_utils

# Import concat_observations from utils (uses jnp instead of np)
from openpi.training.utils import concat_observations_np


class TrajectoryBuffer:
    """Buffer for storing and sampling trajectories.
    
    Supports multiple collection rounds and random sampling for training.
    """
    
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        num_trajectories: int,
        seq_len: int,
    ):
        """Initialize trajectory buffer.
        
        Args:
            buffer_size: Total number of samples to store (sample_batch_size)
            batch_size: Batch size for training
            num_trajectories: Number of trajectories per batch
            seq_len: Sequence length
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_trajectories = num_trajectories
        self.seq_len = seq_len
        self.num_collections = buffer_size // (batch_size * num_trajectories)
        
        # Storage for collected data
        self.observations_list = []
        self.actions_list = []
        self.returns_list = []
        self.rewards_list = []
        self.info_list = []
        
        self.current_size = 0
        self.is_full = False
        self._total_samples = 0
        self._batch_ptr = 0
        self._shuffled_indices = None
        self._rng = None
        self._succ_valid_indices = None
        self._succ_batch_ptr = 0
        self._succ_shuffled_indices = None
        
        logging.info(
            f"Initialized TrajectoryBuffer: buffer_size={buffer_size}, "
            f"batch_size={batch_size}, num_trajectories={num_trajectories}, "
            f"num_collections={self.num_collections}"
        )
    
    def add(
        self,
        observations: Observation,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        info: Dict,
    ):
        """Add collected trajectory data to buffer.
        
        Args:
            observations: Collected observations
            actions: Collected actions
            rewards: Trajectory rewards
            info: Additional info
        """
        train_step_size = self.seq_len * self.batch_size * self.num_trajectories # Remove the last observation which is for reward computation
        observations = _model.Observation(
            images={k: v[:train_step_size] for k, v in observations.images.items()},
            image_masks={k: v[:train_step_size] for k, v in observations.image_masks.items()},
            state=observations.state[:train_step_size],
            tokenized_prompt=observations.tokenized_prompt[:train_step_size] if observations.tokenized_prompt is not None else None,
            tokenized_prompt_mask=observations.tokenized_prompt_mask[:train_step_size] if observations.tokenized_prompt_mask is not None else None,
        )
        self.observations_list.append(observations)
        self.actions_list.append(actions)
        self.rewards_list.append(rewards)
        self.info_list.append(info)

        self.all_observations = None
        self.all_actions = None
        self.all_rewards = None
        
        self.current_size += 1
        
        if self.current_size >= self.num_collections:
            self.is_full = True
            logging.info(f"TrajectoryBuffer is full: {self.current_size}/{self.num_collections} collections")
    
    def _compute_advantages(
        self, 
        rewards: np.ndarray,
        values: np.ndarray = None, 
        dones: np.ndarray = None, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95,
        loss_mask: np.ndarray = None,
        tokenized_prompts: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Compute advantages.
        
        Args:
            rewards: Step-level rewards from environment, shape (num_collections * batch_size * seq_len * num_trajectories, chunk_size)
                    Used for PPO to compute chunk-level rewards
            values: Value estimates for PPO, shape (num_collections * batch_size * (seq_len+1) * num_trajectories,)
                   - includes one extra value per trajectory for bootstrapping
            dones: Done flags, shape (num_collections * batch_size * seq_len * num_trajectories,)
                  - True/1 if episode ended at that step, False/0 otherwise
            gamma: Discount factor for GAE (PPO only)
            gae_lambda: GAE lambda parameter (PPO only)
            loss_mask: Loss mask, shape (num_collectioxns * batch_size * seq_len * num_trajectories,)
        Returns:
            Tuple of (advantages, returns, per_sample_std, info_dict)
        """
        info = {}
        if values is None:
            raise ValueError("Value estimates are required to compute advantages.")
        if dones is None:
            raise ValueError("Done flags are required to compute advantages.")
        if loss_mask is None:
            raise ValueError("loss_mask is required to compute advantages.")
        
        num_collections = self.num_collections
        batch_size = self.batch_size
        num_trajectories = self.num_trajectories
        seq_len = self.seq_len
        num_trajs = num_collections * batch_size * num_trajectories
        
        rewards = rewards.sum(axis=-1).reshape(seq_len, num_trajs).astype(np.float32)
        
        dones = dones.reshape(seq_len + 1, num_trajs)
        values = values.reshape(seq_len + 1, num_trajs, *values.shape[1:])
        loss_mask = loss_mask.reshape(rewards.shape)

        # Conservative value estimate: use mean of bottom k values (less conservative than min)
        # Options: 'min', 'bottom_k_mean', 'lcb' (Lower Confidence Bound)
        ensemble_strategy = 'min'  # Change this to adjust conservatism
        
        if ensemble_strategy == 'min':
            values_used = np.min(values, axis=-1)
        else:
            values_used = np.mean(values, axis=-1)
        
        # Initialize GAE computation
        T = rewards.shape[0]  # seq_len
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = np.zeros((num_trajs))
        
        # Backward pass through time steps
        for step in reversed(range(T)):
            delta = (
                rewards[step]
                + gamma * values_used[step + 1] * (~dones[step + 1])
                - values_used[step]
            )
            
            gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
            returns[step] = gae + values_used[step]
        
        # Advantages = returns - values (excluding last value which is for bootstrapping)
        advantages = returns - values_used[:-1]

        # Flatten to step level first (to ensure consistent ordering)
        advantages_per_step = advantages.reshape(-1)
        returns_per_step = returns.reshape(-1)
        loss_mask_flat = loss_mask.reshape(-1)
        
        # Reshape advantages to 4D: (seq_len, num_collections, batch_size, num_trajectories)
        advantages_mean = advantages_per_step[loss_mask_flat].mean()
        advantages_std = advantages_per_step[loss_mask_flat].std() + 1e-5
        advantages_per_step = (advantages_per_step - advantages_mean) / advantages_std

        info.update({
            'advantages_mean': advantages_per_step.mean(),
            'advantages_std': advantages_per_step.std() + 1e-5,
            'advantages_min': advantages_per_step.min(),
            'advantages_max': advantages_per_step.max(),
        })
        per_sample_std = None
        
        return advantages_per_step, returns_per_step, per_sample_std, info

    def _ensure_sampling_cache(self, rng: at.KeyArrayLike) -> None:
        """Build merged tensors and initialize sampling cursors lazily."""
        if self.all_observations is None:
            # Each collection has shape: (seq_len * batch_size * num_trajectories, ...)
            # We need to merge num_collections of them
            batch_traj_size = self.batch_size * self.num_trajectories
            
            # Merge images
            merged_images = {
                k: np.concatenate(
                    [
                        obs.images[k].reshape(self.seq_len, batch_traj_size, *obs.images[k].shape[1:])
                        for obs in self.observations_list
                    ],
                    axis=1
                ).reshape(self.seq_len * self.buffer_size, *self.observations_list[0].images[k].shape[1:])
                for k in self.observations_list[0].images
            }

            # Merge image_masks
            merged_image_masks = {
                k: np.concatenate(
                    [
                        obs.image_masks[k].reshape(self.seq_len, batch_traj_size, *obs.image_masks[k].shape[1:])
                        for obs in self.observations_list
                    ],
                    axis=1
                ).reshape(self.seq_len * self.buffer_size, *self.observations_list[0].image_masks[k].shape[1:])
                for k in self.observations_list[0].image_masks
            }

            # Merge state, tokenized_prompt and mask
            merged_state = np.concatenate(
                [
                    obs.state.reshape(self.seq_len, batch_traj_size, *obs.state.shape[1:])
                    for obs in self.observations_list
                ],
                axis=1
            ).reshape(self.seq_len * self.buffer_size, *self.observations_list[0].state.shape[1:])

            merged_prompt = np.concatenate(
                [
                    obs.tokenized_prompt.reshape(self.seq_len, batch_traj_size, *obs.tokenized_prompt.shape[1:])
                    for obs in self.observations_list
                ],
                axis=1
            ).reshape(self.seq_len * self.buffer_size, *self.observations_list[0].tokenized_prompt.shape[1:])
            
            merged_prompt_mask = np.concatenate(
                [
                    obs.tokenized_prompt_mask.reshape(self.seq_len, batch_traj_size, *obs.tokenized_prompt_mask.shape[1:])
                    for obs in self.observations_list
                ],
                axis=1
            ).reshape(self.seq_len * self.buffer_size, *self.observations_list[0].tokenized_prompt_mask.shape[1:])

            # Token masks (just first non-None, no reshape needed)
            token_ar_mask = next(
                (obs.token_ar_mask for obs in self.observations_list if obs.token_ar_mask is not None),
                None
            )
            token_loss_mask = next(
                (obs.token_loss_mask for obs in self.observations_list if obs.token_loss_mask is not None),
                None
            )

            self.all_observations = _model.Observation(
                images=merged_images,
                image_masks=merged_image_masks,
                state=merged_state,
                tokenized_prompt=merged_prompt,
                tokenized_prompt_mask=merged_prompt_mask,
                token_ar_mask=token_ar_mask,
                token_loss_mask=token_loss_mask
            )
            # Actions
            self.all_actions = np.concatenate(
                [a.reshape(self.seq_len, batch_traj_size, *a.shape[1:]) 
                for a in self.actions_list],
                axis=1
            ).reshape(self.seq_len * self.buffer_size, *self.actions_list[0].shape[1:])

            # Rewards
            self.all_rewards = np.concatenate(
                [r.reshape(self.seq_len, batch_traj_size, *r.shape[1:])
                for r in self.rewards_list],
                axis=1
            ).reshape(self.seq_len * self.buffer_size, *self.rewards_list[0].shape[1:])

            # Loss mask
            self.loss_mask = np.concatenate(
                [info['loss_mask'].reshape(self.seq_len, batch_traj_size, *info['loss_mask'].shape[1:])
                for info in self.info_list],
                axis=1
            ).reshape(self.seq_len * self.buffer_size, *self.info_list[0]['loss_mask'].shape[1:])

            # Values (optional, for PPO)
            self.values = None
            if 'values' in self.info_list[0] and self.info_list[0]['values'] is not None:
                self.values = np.concatenate(
                    [info['values'].reshape(self.seq_len + 1, batch_traj_size, *info['values'].shape[1:])
                    for info in self.info_list],
                    axis=1
                ).reshape((self.seq_len + 1) * self.buffer_size, *self.info_list[0]['values'].shape[1:])

            # Dones
            dones = np.concatenate(
                [info['dones'].reshape(self.seq_len + 1, batch_traj_size, -1)
                for info in self.info_list],
                axis=1
            ).reshape((self.seq_len + 1) * self.buffer_size)
            # Compute advantages
            tokenized_prompts = None
            if self.all_observations.tokenized_prompt is not None:
                tokenized_prompts = self.all_observations.tokenized_prompt
            
            self.advantages, self.returns, per_sample_std, adv_info = self._compute_advantages(
                rewards=self.all_rewards, 
                values=self.values, 
                dones=dones, 
                loss_mask=self.loss_mask,
                tokenized_prompts=tokenized_prompts
            )

            # Merge info dicts
            self.merged_info = adv_info.copy()
            for key in ['old_logprobs', 'chains', 'denoise_inds', 'loss_mask']:
                if key in self.info_list[0].keys() and self.info_list[0][key] is not None:
                    v = [info[key] for info in self.info_list]
                    self.merged_info[key] = np.concatenate(
                        [v.reshape(self.seq_len, batch_traj_size, -1) for v in v],
                        axis=1
                    ).reshape(self.seq_len * self.buffer_size, *v[0].shape[1:])

            # Init data loader - now sample from all data, not just valid
            self._total_samples = self.all_actions.shape[0]
            self._batch_ptr = 0
            self._shuffled_indices = np.arange(self._total_samples)
            self._rng = rng

            # Build success-only valid indices mask (mask out failed trajectories completely).
            rewards_per_step = self.all_rewards.sum(axis=-1).reshape(self.seq_len, self.buffer_size)
            success_per_traj = np.any(rewards_per_step > 0, axis=0)
            success_step_mask = np.tile(success_per_traj[None, :], (self.seq_len, 1)).reshape(-1)
            valid_step_mask = self.loss_mask.astype(bool)
            self._succ_valid_indices = np.where(np.logical_and(success_step_mask, valid_step_mask))[0]
            if self._succ_valid_indices.size == 0:
                logging.warning("No successful transitions found in buffer, falling back to valid-step sampling.")
                self._succ_valid_indices = np.where(valid_step_mask)[0]
            if self._succ_valid_indices.size == 0:
                logging.warning("No valid-step mask found, falling back to all-step sampling.")
                self._succ_valid_indices = np.arange(self._total_samples)

            self._succ_batch_ptr = 0
            self._succ_shuffled_indices = self._succ_valid_indices.copy()

    def _collect_batch_indices(
        self, sample_batch_size: int, *, use_success_only: bool = False
    ) -> jnp.ndarray:
        """Collect batch indices with epoch-style shuffling."""
        batch_indices_list = []
        remaining_needed = sample_batch_size

        if use_success_only:
            total_samples = len(self._succ_valid_indices)
            if total_samples == 0:
                raise ValueError("No successful indices available for sample_succ.")
            while remaining_needed > 0:
                if self._succ_batch_ptr == 0:
                    self._rng, subkey = jax.random.split(self._rng)
                    perm = np.array(jax.random.permutation(subkey, jnp.array(total_samples)))
                    self._succ_shuffled_indices = self._succ_valid_indices[perm]

                start = self._succ_batch_ptr
                end = min(start + remaining_needed, total_samples)
                available_indices = self._succ_shuffled_indices[start:end]
                batch_indices_list.append(available_indices)
                remaining_needed -= len(available_indices)

                if end >= total_samples:
                    self._succ_batch_ptr = 0
                else:
                    self._succ_batch_ptr = end
        else:
            while remaining_needed > 0:
                if self._batch_ptr == 0:
                    self._rng, subkey = jax.random.split(self._rng)
                    self._shuffled_indices = np.array(
                        jax.random.permutation(subkey, jnp.array(self._total_samples))
                    )

                start = self._batch_ptr
                end = min(start + remaining_needed, self._total_samples)
                available_indices = self._shuffled_indices[start:end]
                batch_indices_list.append(available_indices)
                remaining_needed -= len(available_indices)

                if end >= self._total_samples:
                    self._batch_ptr = 0
                else:
                    self._batch_ptr = end

        batch_indices = np.concatenate(batch_indices_list, axis=0)
        return jnp.array(batch_indices)

    def _build_sampled_batch(
        self,
        sample_indices: jnp.ndarray,
        data_sharding: jax.sharding.NamedSharding,
        replicated_sharding: jax.sharding.NamedSharding,
    ) -> Tuple[_model.Observation, np.ndarray, np.ndarray, Dict]:
        sampled_observations = jax.device_put(
            _model.Observation(
                images={k: v[sample_indices] for k, v in self.all_observations.images.items()},
                image_masks={k: v[sample_indices] for k, v in self.all_observations.image_masks.items()},
                state=self.all_observations.state[sample_indices],
                tokenized_prompt=self.all_observations.tokenized_prompt[sample_indices]
                if self.all_observations.tokenized_prompt is not None
                else None,
                tokenized_prompt_mask=self.all_observations.tokenized_prompt_mask[sample_indices]
                if self.all_observations.tokenized_prompt_mask is not None
                else None,
            ),
            data_sharding,
        )

        sampled_actions = jax.device_put(self.all_actions[sample_indices], data_sharding)
        sampled_advantages = jax.device_put(self.advantages[sample_indices], replicated_sharding)

        sampled_info = {}
        for key in self.merged_info:
            if key in ["old_logprobs", "chains", "denoise_inds", "loss_mask"]:
                sampled_info[key] = self.merged_info[key][sample_indices]
        sampled_info["returns"] = self.returns[sample_indices]
        sampled_info["values"] = self.values[sample_indices] if self.values is not None else None
        sampled_info["advantages_mean"] = self.merged_info["advantages_mean"]
        sampled_info["advantages_std"] = self.merged_info["advantages_std"]
        for key in self.merged_info.keys():
            if "task_advantages_mean_" in key:
                sampled_info[key] = self.merged_info[key]

        return sampled_observations, sampled_actions, sampled_advantages, sampled_info

    def sample(
        self,
        rng: at.KeyArrayLike,
        sample_batch_size: int,
        data_sharding: jax.sharding.NamedSharding,
        replicated_sharding: jax.sharding.NamedSharding
    ) -> Tuple[_model.Observation, np.ndarray, np.ndarray, Dict]:
        """Sample a batch from the full buffer with DataLoader-style shuffling."""
        if not self.is_full:
            raise ValueError(f"Buffer not full yet: {self.current_size}/{self.num_collections}")
        self._ensure_sampling_cache(rng)
        sample_indices = self._collect_batch_indices(sample_batch_size, use_success_only=False)
        return self._build_sampled_batch(sample_indices, data_sharding, replicated_sharding)

    def sample_succ(
        self,
        rng: at.KeyArrayLike,
        sample_batch_size: int,
        data_sharding: jax.sharding.NamedSharding,
        replicated_sharding: jax.sharding.NamedSharding,
    ) -> Tuple[_model.Observation, np.ndarray, np.ndarray, Dict]:
        """Sample only successful-trajectory transitions from the buffer."""
        if not self.is_full:
            raise ValueError(f"Buffer not full yet: {self.current_size}/{self.num_collections}")
        self._ensure_sampling_cache(rng)
        sample_indices = self._collect_batch_indices(sample_batch_size, use_success_only=True)
        return self._build_sampled_batch(sample_indices, data_sharding, replicated_sharding)

    
    def recompute_values_and_advantages(
        self,
        compute_values_fn: Any,
        state: Any,
        recompute_batch_size: int = 64,
    ) -> Dict:
        """Recompute values and advantages using current value function.
        
        This is useful for PPO multi-epoch training where the value function
        is updated and we want to use fresh value estimates for advantage computation.
        
        Args:
            compute_values_fn: JIT-compiled value computation function
            state: Current training state with updated value function parameters
            recompute_batch_size: Batch size for value recomputation (to manage memory)
            
        Returns:
            Dict with recomputation statistics
        """
        if self.all_observations is None or self.loss_mask is None:
            raise ValueError("Buffer not initialized. Call sample() first to initialize observations.")
        
        logging.info("Recomputing values and advantages with current value function...")
        
        # We need (seq_len + 1) values per trajectory for bootstrapping
        # Total observations in buffer: seq_len * buffer_size
        # But we need additional bootstrap values for the last step of each trajectory
        
        num_samples = self.seq_len * self.buffer_size
        batch_traj_size = self.batch_size * self.num_trajectories
        
        # Recompute values for all observations (seq_len steps)
        new_values_list = []
        
        for start_idx in range(0, num_samples, recompute_batch_size):
            end_idx = min(start_idx + recompute_batch_size, num_samples)
            batch_indices = np.arange(start_idx, end_idx)
            
            batch_obs = _model.Observation(
                images={k: v[batch_indices] for k, v in self.all_observations.images.items()},
                image_masks={k: v[batch_indices] for k, v in self.all_observations.image_masks.items()},
                state=self.all_observations.state[batch_indices],
                tokenized_prompt=self.all_observations.tokenized_prompt[batch_indices]
                    if self.all_observations.tokenized_prompt is not None else None,
                tokenized_prompt_mask=self.all_observations.tokenized_prompt_mask[batch_indices]
                    if self.all_observations.tokenized_prompt_mask is not None else None,
            )
            
            # Compute values using current value function
            batch_values = jax.lax.stop_gradient(compute_values_fn(state, batch_obs))
            new_values_list.append(np.array(batch_values))
        
        # Concatenate all computed values: shape (seq_len * buffer_size, num_ensembles)
        new_step_values = np.concatenate(new_values_list, axis=0)
        
        # Reshape to (seq_len, num_collections * batch_size * num_trajectories, num_ensembles)
        new_step_values = new_step_values.reshape(self.seq_len, self.buffer_size, -1)
        
        # Reshape original values to get bootstrap values
        original_values_reshaped = self.values.reshape(self.seq_len + 1, self.buffer_size, -1)
        bootstrap_values = original_values_reshaped[-1:]  # Shape: (1, buffer_size, num_ensembles)
        
        # Combine new step values with original bootstrap values
        new_values = np.concatenate([new_step_values, bootstrap_values], axis=0)
        new_values = new_values.reshape((self.seq_len + 1) * self.buffer_size, -1)
        
        # Store old values for comparison
        # Note: loss_mask shape is (seq_len * buffer_size,), values shape is ((seq_len + 1) * buffer_size, num_ensembles)
        # We use the step values (excluding bootstrap) for comparison
        step_values = self.values.reshape(self.seq_len + 1, self.buffer_size, -1)[:-1].reshape(-1, self.values.shape[-1])
        old_values_mean = float(step_values[self.loss_mask].mean()) if hasattr(self, 'loss_mask') else float(step_values.mean())
        
        # Update values
        self.values = new_values
        
        # Recompute dones from info_list
        dones = np.concatenate(
            [info['dones'].reshape(self.seq_len + 1, batch_traj_size, -1)
            for info in self.info_list],
            axis=1
        ).reshape((self.seq_len + 1) * self.buffer_size)
        
        # Recompute advantages with new values
        tokenized_prompts = None
        if self.all_observations.tokenized_prompt is not None:
            tokenized_prompts = self.all_observations.tokenized_prompt
        
        old_advantages_mean = float(self.advantages.mean())
        old_advantages_std = float(self.advantages.std())
        
        self.advantages, self.returns, per_sample_std, adv_info = self._compute_advantages(
            rewards=self.all_rewards, 
            values=self.values, 
            dones=dones, 
            loss_mask=self.loss_mask,
            tokenized_prompts=tokenized_prompts
        )
        
        # Update merged_info with new advantage statistics
        self.merged_info.update(adv_info)
        
        # Compute new values mean using step values (excluding bootstrap)
        new_step_values_flat = self.values.reshape(self.seq_len + 1, self.buffer_size, -1)[:-1].reshape(-1, self.values.shape[-1])
        new_values_mean = float(new_step_values_flat[self.loss_mask].mean()) if hasattr(self, 'loss_mask') else float(new_step_values_flat.mean())
        
        recompute_info = {
            'old_values_mean': old_values_mean,
            'new_values_mean': new_values_mean,
            'values_change': new_values_mean - old_values_mean,
            'old_advantages_mean': old_advantages_mean,
            'old_advantages_std': old_advantages_std,
            'new_advantages_mean': float(self.advantages.mean()),
            'new_advantages_std': float(self.advantages.std()),
        }
        recompute_info.update(adv_info)
        
        logging.info(
            f"Recomputed values and advantages: "
            f"values {old_values_mean:.4f} -> {new_values_mean:.4f}, "
            f"advantages mean {old_advantages_mean:.4f} -> {recompute_info['new_advantages_mean']:.4f}"
        )
        
        return recompute_info
    
    def clear(self):
        """Clear the buffer."""
        self.observations_list = []
        self.actions_list = []
        self.returns_list = []
        self.rewards_list = []
        self.info_list = []
        self.current_size = 0
        self.is_full = False
        self.all_observations = None
        self.all_actions = None
        self.all_rewards = None
        self.advantages = None
        self.merged_info = None
        self.values = None
        self.loss_mask = None
        self._total_samples = 0
        self._batch_ptr = 0
        self._shuffled_indices = None
        self._rng = None
        self._succ_valid_indices = None
        self._succ_batch_ptr = 0
        self._succ_shuffled_indices = None
    
    def offload(self, save_path: str, collection_info: dict = None) -> None:
        """Save buffer data to disk.
        
        Args:
            save_path: Output path, preferably with a .npz suffix.
            collection_info: Optional collection statistics.
        """
        import pickle
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the payload to save.
        save_data = {
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'num_trajectories': self.num_trajectories,
            'seq_len': self.seq_len,
            'current_size': self.current_size,
            'is_full': self.is_full,
            'collection_info': collection_info,
        }
        
        # Save observations_list.
        observations_data = []
        for obs in self.observations_list:
            obs_dict = {
                'images': {k: np.array(v) for k, v in obs.images.items()},
                'image_masks': {k: np.array(v) for k, v in obs.image_masks.items()},
                'state': np.array(obs.state),
                'tokenized_prompt': np.array(obs.tokenized_prompt) if obs.tokenized_prompt is not None else None,
                'tokenized_prompt_mask': np.array(obs.tokenized_prompt_mask) if obs.tokenized_prompt_mask is not None else None,
                'token_ar_mask': np.array(obs.token_ar_mask) if obs.token_ar_mask is not None else None,
                'token_loss_mask': np.array(obs.token_loss_mask) if obs.token_loss_mask is not None else None,
            }
            observations_data.append(obs_dict)
        save_data['observations_list'] = observations_data
        
        # Save actions and rewards.
        save_data['actions_list'] = [np.array(a) for a in self.actions_list]
        save_data['rewards_list'] = [np.array(r) for r in self.rewards_list]
        
        # Save info_list, which may contain mixed value types.
        info_data = []
        for info in self.info_list:
            info_dict = {}
            for k, v in info.items():
                if v is None:
                    info_dict[k] = None
                elif isinstance(v, (np.ndarray, jnp.ndarray)):
                    info_dict[k] = np.array(v)
                elif isinstance(v, list):
                    info_dict[k] = v
                else:
                    info_dict[k] = v
            info_data.append(info_dict)
        save_data['info_list'] = info_data
        
        # Persist everything with pickle.
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logging.info(f"Buffer offloaded to {save_path}")
        logging.info(f"Saved {self.current_size} collections, is_full={self.is_full}")
    
    def load(self, load_path: str) -> dict:
        """Load buffer data from disk.
        
        Args:
            load_path: Input path.
            
        Returns:
            collection_info: Collection statistics saved with the buffer.
        """
        import pickle
        from pathlib import Path
        
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Buffer file not found: {load_path}")
        
        # Load the serialized payload.
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Validate the loaded buffer configuration.
        if (save_data['buffer_size'] != self.buffer_size or
            save_data['batch_size'] != self.batch_size or
            save_data['num_trajectories'] != self.num_trajectories or
            save_data['seq_len'] != self.seq_len):
            logging.warning(
                f"Loaded buffer configuration mismatch! "
                f"Current: buffer_size={self.buffer_size}, batch_size={self.batch_size}, "
                f"num_trajectories={self.num_trajectories}, seq_len={self.seq_len}. "
                f"Loaded: buffer_size={save_data['buffer_size']}, batch_size={save_data['batch_size']}, "
                f"num_trajectories={save_data['num_trajectories']}, seq_len={save_data['seq_len']}."
            )
        
        # Restore buffer state.
        self.current_size = save_data['current_size']
        self.is_full = save_data['is_full']
        
        # Restore observations_list.
        self.observations_list = []
        for obs_dict in save_data['observations_list']:
            obs = _model.Observation(
                images=obs_dict['images'],
                image_masks=obs_dict['image_masks'],
                state=obs_dict['state'],
                tokenized_prompt=obs_dict['tokenized_prompt'],
                tokenized_prompt_mask=obs_dict['tokenized_prompt_mask'],
                token_ar_mask=obs_dict.get('token_ar_mask'),
                token_loss_mask=obs_dict.get('token_loss_mask'),
            )
            self.observations_list.append(obs)
        
        # Restore actions, rewards, and info.
        self.actions_list = save_data['actions_list']
        self.rewards_list = save_data['rewards_list']
        self.info_list = save_data['info_list']
        
        # Reset merged data and rebuild it on the next sample() call.
        self.all_observations = None
        self.all_actions = None
        self.all_rewards = None
        self.advantages = None
        self.merged_info = None
        self.values = None
        self.loss_mask = None
        self._total_samples = 0
        self._batch_ptr = 0
        self._shuffled_indices = None
        self._rng = None
        self._succ_valid_indices = None
        self._succ_batch_ptr = 0
        self._succ_shuffled_indices = None
        
        # Read collection_info.
        collection_info = save_data.get('collection_info', {
            'total_collection_time': 0.0,
            'avg_collection_time': 0.0,
            'num_collections': self.num_collections,
        })
        
        logging.info(f"Buffer loaded from {load_path}")
        logging.info(f"Loaded {self.current_size} collections, is_full={self.is_full}")
        
        return collection_info

    def visualize(
        self,
        save_root: Any,
        *,
        seq_len: int,
        batch_size: int,
        num_trajectories: int,
        step: int | None = None,
        cameras: List[str] | None = None,
        max_trajs_per_collection: int | None = 4,
        fps: int = 10,
    ) -> None:
        """Save per-trajectory videos from concatenated observations.

        Args:
            save_root: Directory to save under. If step is provided, a step subdir is created.
            seq_len: Sequence length used during collection.
            batch_size: Training batch size.
            num_trajectories: Number of trajectories per batch.
            step: Optional global step to include in directory structure.
            cameras: Optional subset of camera keys to export.
            max_trajs_per_collection: Limit number of trajs saved per collection.
            fps: Output video frames per second.
        """
        try:
            from pathlib import Path
            save_root = Path(save_root)
            if step is not None:
                save_root = save_root / f"step_{step:06d}"
            save_root.mkdir(parents=True, exist_ok=True)

            # Ensure concatenated observations exist
            if getattr(self, "all_observations", None) is None or self.all_observations is None:
                if not self.observations_list:
                    logging.warning("TrajectoryBuffer.visualize: no observations to save.")
                    return
                self.all_observations = concat_observations_np(self.observations_list)

            cam_keys = list(self.all_observations.images.keys()) if cameras is None else [k for k in cameras if k in self.all_observations.images]
            num_collections = self.num_collections

            for cam in cam_keys:
                imgs = np.array(self.all_observations.images[cam])
                if imgs.ndim != 4:
                    logging.warning(f"TrajectoryBuffer.visualize: unexpected ndim for camera {cam}: {imgs.ndim}")
                    continue

                # Convert [-1, 1] -> [0, 255]
                imgs_u8 = np.clip(((imgs + 1.0) * 0.5) * 255.0, 0, 255).astype(np.uint8)

                # Reshape to [num_collections, seq_len, batch_size, num_traj, H, W, C]
                try:
                    imgs_u8 = imgs_u8.reshape(num_collections, seq_len, batch_size, num_trajectories, *imgs_u8.shape[-3:])
                except Exception as e:
                    logging.warning(f"TrajectoryBuffer.visualize: reshape failed for camera {cam}: {e}")
                    continue

                cam_dir = save_root / cam
                cam_dir.mkdir(parents=True, exist_ok=True)

                for coll_idx in range(num_collections):
                    coll_dir = cam_dir / f"collection_{coll_idx:03d}"
                    coll_dir.mkdir(parents=True, exist_ok=True)

                    limit_trajs = num_trajectories if max_trajs_per_collection is None else min(num_trajectories, max_trajs_per_collection)

                    for b in range(batch_size):
                        for t in range(limit_trajs):
                            frames = imgs_u8[coll_idx, :, b, t]
                            out_path = coll_dir / f"b{b:02d}_traj{t:02d}.mp4"
                            try:
                                with imageio.get_writer(str(out_path), format="mp4", mode="I", fps=fps) as writer:
                                    for f in range(frames.shape[0]):
                                        writer.append_data(frames[f])
                            except Exception as ve:
                                logging.warning(f"TrajectoryBuffer.visualize: failed writing {out_path}: {ve}")
        except Exception as e:
            logging.warning(f"TrajectoryBuffer.visualize: failed with error: {e}")


def sample_actions_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation: Observation,
    num_steps: int = 10,
    mode: Literal["train", "eval"] = "eval",
    use_old: bool = True,
    use_forward: bool = False,
    return_logprobs: bool = False,
    return_values: bool = False,
) -> Tuple[Actions, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None, jnp.ndarray | None]:
    """Sample actions from the policy using Flow SDE.
    
    Args:
        rng: Random key
        state: Training state
        observation: Current observation
        num_steps: Number of action steps to generate
        mode: "train" to enable SDE, "eval" for deterministic denoising
        use_old: Whether to use old params
        use_forward: Whether to use forward (current) params
        return_logprobs: Whether to return log probabilities
        return_values: Whether to return value estimates (for PPO)
        
    Returns:
        Always returns tuple of (actions, logprobs, chains, values, denoise_inds) where:
        - actions: sampled actions
        - logprobs: log probabilities (None if return_logprobs=False)
        - chains: denoising chains (None if return_logprobs=False)
        - values: value estimates (None if return_values=False)
        - denoise_inds: denoising indices (None if return_logprobs=False)
    """
    import flax.nnx as nnx
    
    # Use Old params for inference if requested and available; otherwise use current params
    if use_old and state.old_params is not None:
        params_for_infer = state.old_params
    elif use_forward:
        params_for_infer = state.params
    elif state.ema_params is not None:
        params_for_infer = state.ema_params 
    else:
        params_for_infer = state.params

    model = nnx.merge(state.model_def, params_for_infer)
    model.eval()
    
    result = model.sample_actions(
        rng,
        observation,
        num_steps=num_steps,
        mode=mode,
        return_logprobs=return_logprobs,
        return_values=return_values,
    )
    # New return style: either actions only, or (actions, info_dict)
    if isinstance(result, tuple) and isinstance(result[1], dict):
        actions, info = result
        prev_logprobs = info.get("prev_logprobs", None)
        chains = info.get("chains", None)
        prev_values = info.get("prev_values", None)
        denoise_inds = info.get("denoise_inds", None)
        return actions, prev_logprobs, chains, prev_values, denoise_inds
    else:
        # Actions only
        return result, None, None, None, None


def collect_trajectory(
    config: Any,
    sample_actions_fn: Any,
    compute_values_fn: Any,
    env: Any,  # LiberoRLEnvironment or any compatible environment
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: Tuple,
    tasks: list,
    seq_len: int = 3,
    max_time_stamp: int = 220,
    use_old: bool = True,
    use_forward: bool = False,
    use_world_model: bool = False,
    num_steps: int = 10,
    mode: Literal["train", "eval"] = "eval",
    return_logprobs: bool = False,
    return_values: bool = False,
) -> Tuple[Observation, jnp.ndarray, jnp.ndarray, Observation, jnp.ndarray, Dict]:
    """Collect trajectories from environment with full parallelization.
    
    Args:
        config: Training configuration
        sample_actions_fn: JIT-compiled action sampling function
        compute_values_fn: JIT-compiled value computation function
        env: Environment (e.g., LiberoRLEnvironment)
        rng: Random key
        state: Training state
        batch: Batch of (observations, actions) for initialization
        tasks: Prompt of tasks
        seq_len: Sequence length
        mode: "train" to enable SDE, "eval" for deterministic denoising
        use_old: Whether to use old params
        use_forward: Whether to use forward params
        use_world_model: Whether to use world model for simulation
        num_steps: Number of action steps to generate
        return_logprobs: Whether to compute and return log probabilities
        return_values: Whether to compute and return value estimates
        
    Returns:
        Tuple of (observations, actions, rewards, info)
    """
    batch_obs, batch_actions, batch_sim_state, batch_sim_state_len, task_id, batch_time_stamp = batch
    
    num_trajectories = env.num_trajectories
    total_envs = env.total_envs
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_dones = [np.zeros((total_envs,), dtype=bool)]
    all_logprobs = [] if return_logprobs else None
    all_chains = [] if return_logprobs else None
    all_values = [] if return_values else None
    all_denoise_inds = [] if return_logprobs else None

    batch_sim_state = np.array(np.repeat(batch_sim_state, num_trajectories, axis=0))
    batch_sim_state_len = np.array(np.repeat(batch_sim_state_len, num_trajectories, axis=0))
    task_id = np.array(np.repeat(task_id, num_trajectories, axis=0))
    tasks = [task for task in tasks for _ in range(num_trajectories)]
    if use_world_model:
        # Repeat batch_obs for num_trajectories times
        model_obs = _model.Observation(
            images={k: v.repeat(num_trajectories, axis=0) for k, v in batch_obs.images.items()},
            image_masks={k: v.repeat(num_trajectories, axis=0) for k, v in batch_obs.image_masks.items()},
            state=batch_obs.state.repeat(num_trajectories, axis=0),
            tokenized_prompt=batch_obs.tokenized_prompt.repeat(num_trajectories, axis=0) if batch_obs.tokenized_prompt is not None else None,
            tokenized_prompt_mask=batch_obs.tokenized_prompt_mask.repeat(num_trajectories, axis=0) if batch_obs.tokenized_prompt_mask is not None else None,
        )
        env.current_observations = model_obs
    else:
        raw_obs = env.reset(batch_sim_state, batch_sim_state_len, task_id)
        
        # Convert initial raw_obs to model_obs
        model_obs = training_utils.libero_obs_to_model_obs(
            raw_obs,
            input_transforms=env.input_transforms,
        )

    prev_rewards = np.zeros((total_envs,), dtype=np.float32)
    
    for step_idx in range(seq_len + 1):
        all_observations.append(model_obs)
        
        # Sample actions (except for last observation)
        if step_idx < seq_len:
            action_rng = jax.random.fold_in(rng, step_idx)

            # Time statistics for sample_action
            result = sample_actions_fn(
                action_rng,
                state,
                model_obs,
                num_steps,
                mode,
                use_old,
                use_forward,
                return_logprobs,
                return_values,
            )
            
            # Unpack result based on return_logprobs and return_values
            if return_logprobs and return_values:
                actions, logprobs, chains, step_values, denoise_inds = result
                all_logprobs.append(logprobs)
                all_chains.append(chains)
                all_values.append(step_values)
                all_denoise_inds.append(denoise_inds)
            elif return_logprobs:
                actions, logprobs, chains, _, denoise_inds = result
                all_logprobs.append(logprobs)
                all_chains.append(chains)
                all_denoise_inds.append(denoise_inds)
            elif return_values:
                actions, _, _, step_values, _ = result
                all_values.append(step_values)
            else:
                actions, _, _, _, _ = result
            
            actions = jnp.pad(actions[:, :, :7], ((0, 0), (0, 0), (0, actions.shape[2] - 7)), mode='constant', constant_values=0)
            all_actions.append(actions)
            
            # Execute actions in environment - record time
            actions_np = np.array(jax.device_get(actions))
            # actions_np shape: [total_envs, chunk_size, action_dim]
            unnormalized_actions, _ = env.unnormalize(actions_np, model_obs.state)
            
            # Choose between real environment and world model
            if use_world_model:
                model_obs = env.step_with_world_model(unnormalized_actions[:, :, :7], model_obs)
                truncations = np.zeros((total_envs, env.chunk_size), dtype=bool)
                # Compute reward using reward model
                rewards = env.compute_reward(model_obs, tasks)
                # Reshape to (total_envs, chunk_size) by broadcasting
                prev_rewards[rewards > 0] = rewards[rewards > 0]
                rewards = np.broadcast_to(rewards[:, None], (total_envs, 1))
                terminations = np.broadcast_to(prev_rewards[:, None], (total_envs, env.chunk_size))
            else:
                # Real environment: step and convert to model_obs
                raw_obs, rewards, terminations, truncations = env.chunk_step(unnormalized_actions[:, :, :7])
                
                # Convert raw_obs to model_obs
                model_obs = training_utils.libero_obs_to_model_obs(
                    raw_obs,
                    input_transforms=env.input_transforms,
                )
            
            # Make a writable copy of rewards to avoid read-only array error
            rewards = np.array(rewards, copy=True)        
            dones = np.any(np.logical_or(terminations, truncations), axis=-1)
            
            all_rewards.append(rewards.astype(np.float32))
            all_dones.append(dones)
    
    # Concatenate all observations and actions
    # all_observations: list of (seq_len+1) observations, each with shape [total_envs, ...]
    # We need to reshape to [batch_size, num_trajectories * (seq_len+1), ...]
    combined_obs = concat_observations_np(all_observations)
    combined_actions = np.concatenate([a for a in all_actions], axis=0)
    combined_dones = np.concatenate([d for d in all_dones], axis=0)
    combined_rewards = np.concatenate([r for r in all_rewards], axis=0)
    
    # Concatenate logprobs if collected
    combined_logprobs = np.concatenate([lp for lp in all_logprobs], axis=0) if return_logprobs and all_logprobs else None
    # Concatenate denoise_inds if collected
    combined_denoise_inds = np.concatenate([di for di in all_denoise_inds], axis=0) if return_logprobs and all_denoise_inds else None
    # Concatenate chains if collected
    combined_chains = np.concatenate([c for c in all_chains], axis=0) if return_logprobs and all_chains else None
    # Concatenate values if collected
    # For PPO, we need (seq_len+1) values: one for each observation including the last bootstrap value
    if return_values:
        last_obs = all_observations[-1]  
        last_value = jax.lax.stop_gradient(compute_values_fn(state, last_obs))  # Shape: (total_envs,)
        all_values.append(last_value)
        
        combined_values = np.concatenate([v for v in all_values], axis=0)
    
    # Compute loss mask (mask out steps where episode ended)
    loss_mask = (combined_dones.reshape(seq_len + 1, total_envs).cumsum(axis=0) == 0)[:-1].flatten()
    
    return_info = {
        "loss_mask": loss_mask,
        "dones": combined_dones, 
        "rewards": combined_rewards,
    }

    success_rate = jnp.mean(np.any(combined_rewards.sum(axis=-1).reshape(seq_len, total_envs), axis=0))
    return_info['success_rate'] = success_rate
    if not use_world_model:

       # Compute value bias
        if mode == "eval":
            for i in np.unique(task_id.flatten()):
                task_mask = task_id.flatten() == i
                success_rate = jnp.mean(np.any(combined_rewards.sum(axis=-1).reshape(seq_len, total_envs), axis=0)[task_mask])
                return_info[f'task_success_rate_{i}'] = success_rate

            gamma = 0.99
            step_rewards = combined_rewards.sum(axis=-1).reshape(seq_len, total_envs)
            dones = combined_dones.reshape(seq_len + 1, total_envs)
            gt_returns = np.zeros_like(step_rewards, dtype=np.float32)
            running_return = np.zeros((total_envs,), dtype=np.float32)
            for t in reversed(range(seq_len)):
                running_return = step_rewards[t] + gamma * running_return * (~dones[t + 1])
                gt_returns[t] = running_return

            pred_values = np.zeros_like(gt_returns)
            for t in range(seq_len):
                ensemble_values = jax.lax.stop_gradient(compute_values_fn(state, all_observations[t]))
                # Use the same ensemble strategy as in advantage computation
                pred_values[t] = np.min(ensemble_values, axis=-1)  # Keep min for bias monitoring
            
            valid_trans_mask = (combined_dones.reshape(seq_len + 1, total_envs).cumsum(axis=0) == 0)[:-1]
            flat_mask = valid_trans_mask.flatten()
            gt_returns_flat = gt_returns.reshape(-1)[flat_mask]
            pred_values_flat = pred_values.reshape(-1)[flat_mask]
            value_bias = (pred_values_flat - gt_returns_flat) / (np.abs(gt_returns_flat).mean() + 1e-2)
            return_info['value_bias_mean'] = value_bias.mean()
            return_info['value_bias_std'] = value_bias.std()

            verbose_info = []
            for traj_idx in range(total_envs):
                info_dict = {'timesteps': [], 'gt_returns': [], 'pred_values': [], 'value_bias': [], 'task_id': []}
                for t in range(seq_len):
                    if not valid_trans_mask[t, traj_idx]:
                        break
                    info_dict['timesteps'].append(t)
                    info_dict['gt_returns'].append(gt_returns[t, traj_idx])
                    info_dict['pred_values'].append(pred_values[t, traj_idx])
                    info_dict['task_id'].append(task_id.flatten()[traj_idx])
                verbose_info.append(info_dict)
            return_info['verbose_info'] = verbose_info
    
    # Add logprobs to return info if collected
    return_info['old_logprobs'] = combined_logprobs if return_logprobs and all_logprobs else None
    return_info['denoise_inds'] = combined_denoise_inds if return_logprobs and all_denoise_inds else None
    return_info['chains'] = combined_chains if return_logprobs and all_chains else None
    if all_values:
        return_info['values'] = combined_values

    return combined_obs, combined_actions, combined_rewards, return_info


def collect_trajectories_to_buffer(
    config: Any,
    sample_actions_fn: Any,
    compute_values_fn: Any,
    env: Any,  # LiberoRLEnvironment or any compatible environment
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch_iterator: Any,
    tasks: Any,
    buffer: TrajectoryBuffer,
    max_time_stamp: int = 220,
    use_world_model: bool = False,
    num_steps: int = 10,
    mode: Literal["train", "eval"] = "eval",
    return_logprobs: bool = False,
    return_values: bool = False,
) -> Dict:
    """Collect multiple trajectory batches and fill the buffer.
    
    Args:
        config: Training configuration
        sample_actions_fn: JIT-compiled action sampling function
        compute_values_fn: JIT-compiled value computation function
        env: Environment (e.g., LiberoRLEnvironment)
        rng: Random key
        state: Training state
        batch_iterator: Iterator for getting initialization batches
        buffer: TrajectoryBuffer to fill
        max_time_stamp: Maximum time stamp for truncation
        use_world_model: Whether to use world model for simulation
        num_steps: Number of action steps to generate
        mode: "train" to enable SDE, "eval" for deterministic denoising
        return_logprobs: Whether to collect log probabilities
        return_values: Whether to compute and return value estimates
        
    Returns:
        Dict with collection statistics
    """
    buffer.clear()
    
    total_collection_time = 0.0
    collection_infos = []
    use_old = False
    use_forward = True
    
    for collection_idx in tqdm(range(buffer.num_collections)):
        collection_start = time.time()
        
        # Get a new batch for initialization
        batch = next(batch_iterator)
        
        # Collect trajectories
        collection_rng = jax.random.fold_in(rng, collection_idx)
        task_id_batch = batch[-2]  
        prompt = [tasks.get(int(t)) for t in task_id_batch]
        observations, actions, rewards, info = collect_trajectory(
            config=config,
            sample_actions_fn=sample_actions_fn,
            compute_values_fn=compute_values_fn,
            env=env,
            rng=collection_rng,
            state=state,
            batch=batch,
            tasks=prompt,
            seq_len=buffer.seq_len,
            max_time_stamp=max_time_stamp,
            use_old=use_old,
            use_forward=use_forward,
            use_world_model=use_world_model,
            num_steps=num_steps,
            mode=mode,
            return_logprobs=return_logprobs,
            return_values=return_values,
        )
        
        # Add to buffer
        buffer.add(observations, actions, rewards, info)
        
        collection_time = time.time() - collection_start
        total_collection_time += collection_time
        collection_infos.append(info)
    
    # Aggregate collection statistics
    aggregated_info = {
        'total_collection_time': total_collection_time,
        'avg_collection_time': total_collection_time / buffer.num_collections,
        'num_collections': buffer.num_collections,
    }
    
    # Average numerical metrics across collections
    for key in collection_infos[0].keys():
        if isinstance(collection_infos[0][key], (int, float, jnp.ndarray, np.ndarray)):
            values = [info[key] for info in collection_infos]
            if isinstance(values[0], jnp.ndarray) and values[0].ndim == 0:
                aggregated_info[f'avg_{key}'] = float(jnp.mean(jnp.stack(values)))
            elif isinstance(values[0], (int, float)):
                aggregated_info[f'avg_{key}'] = np.mean(values)
    
    logging.info(
        f"Buffer filled with {buffer.num_collections} collections in {total_collection_time:.2f}s, "
    )
    
    return aggregated_info


def compute_value_step(
    state: training_utils.TrainState,
    observations: Observation,
) -> jnp.ndarray:
    """Compute value estimates for observations (for PPO).
    
    Args:
        state: Training state
        observations: Observations to compute values for
        
    Returns:
        Value estimates, shape (batch_size,)
    """
    import flax.nnx as nnx
    
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    
    # Call model's compute_values method
    values = model.compute_values(None, observations, True)
    
    return values
