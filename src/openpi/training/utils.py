from collections.abc import Callable
from typing import Any, List, Dict, Optional

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None
    old_params: nnx.State | None = None
    ref_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")

def concat_observations(observations: List[_model.Observation]) -> _model.Observation:
    """Concatenate a list of n observations.
    
    Args:
        observations: List of Observation objects to concatenate
    
    Returns:
        Single concatenated Observation
    """
    # Merge images
    merged_images = {
        k: jnp.concatenate([obs.images[k] for obs in observations], axis=0)
        for k in observations[0].images
    }

    # Merge image_masks
    merged_image_masks = {
        k: jnp.concatenate([obs.image_masks[k] for obs in observations], axis=0)
        for k in observations[0].image_masks
    }

    # Merge state, tokenized_prompt and mask
    merged_state = jnp.concatenate([obs.state for obs in observations], axis=0)
    merged_prompt = jnp.concatenate([obs.tokenized_prompt for obs in observations], axis=0)
    merged_prompt_mask = jnp.concatenate([obs.tokenized_prompt_mask for obs in observations], axis=0)

    # Get token masks (use first non-None value)
    token_ar_mask = next((obs.token_ar_mask for obs in observations if obs.token_ar_mask is not None), None)
    token_loss_mask = next((obs.token_loss_mask for obs in observations if obs.token_loss_mask is not None), None)

    return _model.Observation(
        images=merged_images,
        image_masks=merged_image_masks,
        state=merged_state,
        tokenized_prompt=merged_prompt,
        tokenized_prompt_mask=merged_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask
    )

def concat_observations_np(observations: List[_model.Observation]) -> _model.Observation:
    """Concatenate a list of n observations.
    
    Args:
        observations: List of Observation objects to concatenate
    
    Returns:
        Single concatenated Observation
    """
    # Merge images
    merged_images = {
        k: np.concatenate([obs.images[k] for obs in observations], axis=0)
        for k in observations[0].images
    }

    # Merge image_masks
    merged_image_masks = {
        k: np.concatenate([obs.image_masks[k] for obs in observations], axis=0)
        for k in observations[0].image_masks
    }

    # Merge state, tokenized_prompt and mask
    merged_state = np.concatenate([obs.state for obs in observations], axis=0)
    merged_prompt = np.concatenate([obs.tokenized_prompt for obs in observations], axis=0)
    merged_prompt_mask = np.concatenate([obs.tokenized_prompt_mask for obs in observations], axis=0)

    # Get token masks (use first non-None value)
    token_ar_mask = next((obs.token_ar_mask for obs in observations if obs.token_ar_mask is not None), None)
    token_loss_mask = next((obs.token_loss_mask for obs in observations if obs.token_loss_mask is not None), None)

    return _model.Observation(
        images=merged_images,
        image_masks=merged_image_masks,
        state=merged_state,
        tokenized_prompt=merged_prompt,
        tokenized_prompt_mask=merged_prompt_mask,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask
    )

def libero_obs_to_model_obs(
    raw_obs: Dict,
    input_transforms: Optional[Any] = None,
) -> Dict:
    """Convert LIBERO observation to model observation format.
    
    This function converts raw observations from the LIBERO server into a format compatible with the model
    
    Args:
        raw_obs: Raw observation from LIBERO server (batch format)
                Expected keys: 'images_and_states' (dict with full_image, wrist_image, state),
                              'task_descriptions' (list)
        input_transforms: Transform pipeline to apply (handles prompt tokenization, etc.)
        
    Returns:
        Model observation dict
    """
    # Extract data from raw observation
    images_and_states = raw_obs['images_and_states']
    task_descriptions = raw_obs['task_descriptions']
    
    batch_size = len(task_descriptions)
    
    observations = []
    for i in range(batch_size):
        obs_data = {
            'image': images_and_states['full_image'][i],
            'wrist_image': images_and_states['wrist_image'][i],
            'state': images_and_states['state'][i],
            'prompt': task_descriptions[i],
        }
        obs = input_transforms(obs_data) if input_transforms is not None else obs_data
        observations.append(obs)

    observations_dict = {}
    for k in observations[0].keys():
        if isinstance(observations[0][k], dict):
            observations_dict[k] = {k2: np.stack([obs[k][k2] for obs in observations], axis=0) for k2 in observations[0][k].keys()}
        else:
            observations_dict[k] = np.stack([obs[k] for obs in observations], axis=0)

    return _model.Observation.from_dict(observations_dict)
