"""
PPO Training Script for LIBERO Environment with Buffer

This script implements PPO training with buffered trajectory collection (RLinf-style).
Key features:
- Collects multiple batches into a buffer with logprobs
- Computes group-relative advantages: (return - group_mean) / group_std
- Uses PPO-style importance sampling with clipping
- No critic needed (simpler than standard PPO)
"""

import dataclasses
import functools
import logging
import platform
from typing import Any
import time

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
from jax import lax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import openpi.models.model as _model
from openpi.models.model import Observation
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.training.weight_loaders as weight_loaders
from openpi.training.utils import concat_observations
from openpi.training.libero_env import LiberoRLEnvironment
from openpi.training.rl_env import (
    TrajectoryBuffer,
    sample_actions_step, 
    collect_trajectory,
    collect_trajectories_to_buffer,
    compute_value_step,
)

def plot_values_and_returns(verbose_info):
    if not verbose_info:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    cols = 4
    n_plots = len(verbose_info)
    n_rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(n_rows, cols, figsize=(4 * cols, 3 * n_rows), squeeze=False)

    for idx, info in enumerate(verbose_info):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        timesteps = np.asarray(info.get("timesteps", []))
        gt_returns = np.asarray(info.get("gt_returns", []))
        pred_values = np.asarray(info.get("pred_values", []))
        task = np.asarray(info.get("task_id", []))[0]

        ax.plot(timesteps, gt_returns, label="gt_return")
        ax.plot(timesteps, pred_values, label="pred_value")
        ax.set_xlabel("timestep")
        ax.set_ylabel("value / return")
        ax.set_title(f"Trajectory {idx} Task {task}")   
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots if any
    for idx in range(n_plots, n_rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    image = buf[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    return image

def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.RLTrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            entity="openpi_dis",
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights.
    
    This function allows partial loading - if some parameters are missing from the loaded weights,
    they will be kept from params_shape (and will be randomly initialized).
    """
    loaded_params = loader.load(params_shape)
    
    # Flatten both trees to compare keys
    flat_expected = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    
    # Validate only the keys that exist in both trees
    result = {}
    for k, expected_v in flat_expected.items():
        if k in flat_loaded:
            loaded_v = flat_loaded[k]
            result[k] = loaded_v
            logging.info(f"Loading key {k} from loaded params")
        else:
            # Keep the expected value for missing keys (will be randomly initialized)
            logging.warning(f"Expected key {k} not found in the loaded params, will be randomly initialized")
            result[k] = expected_v
    
    # Remove jax.ShapeDtypeStruct from the result. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in result.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.RLTrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    
    # Determine if we need dual-lr optimizer
    use_dual_lr = True
    
    if use_dual_lr:
        # For dual-lr case, we need to create the optimizer after getting param structure
        # First create a temporary single-lr optimizer for shape inference
        temp_tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
        
        def temp_init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
            rng, model_rng = jax.random.split(rng)
            model = config.model.create(model_rng)

            if partial_params is not None:
                graphdef, state = nnx.split(model)
                state.replace_by_pure_dict(partial_params)
                model = nnx.merge(graphdef, state)

            params = nnx.state(model)
            params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

            return training_utils.TrainState(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                tx=temp_tx,
                opt_state=temp_tx.init(params.filter(config.trainable_filter)),
                ema_decay=config.ema_decay,
                ema_params=params if config.ema_decay else None,
                old_params=None,  
                ref_params=params,  
            )
        
        # Get shape with temporary optimizer
        temp_train_state_shape = jax.eval_shape(temp_init, init_rng)
        
        if resume:
            return temp_train_state_shape, sharding.fsdp_sharding(temp_train_state_shape, mesh, log=True)
        
        # Load partial params
        if config.pretrained_path is not None:
            weight_loader = weight_loaders.CheckpointWeightLoader(config.pretrained_path)
        else:
            weight_loader = config.weight_loader
        partial_params = _load_weights_and_validate(weight_loader, temp_train_state_shape.params.to_pure_dict())
        
        # Create parameter labels for dual-lr optimizer
        trainable_params_shape = temp_train_state_shape.params.filter(config.trainable_filter)
        
        def label_params(path_and_leaf):
            path, _ = path_and_leaf
            keystr = jax.tree_util.keystr(path)
            return 'critic' if 'value_head' in keystr else 'actor'
        
        paths_and_leaves = list(jax.tree_util.tree_flatten_with_path(trainable_params_shape)[0])
        labels = [label_params(pl) for pl in paths_and_leaves]
        
        leaves_and_treedef = jax.tree_util.tree_flatten(trainable_params_shape)
        param_labels = jax.tree_util.tree_unflatten(leaves_and_treedef[1], labels)
        
        # Create the actual dual-lr optimizer
        dual_tx = _optimizer.create_dual_lr_optimizer(
            config.optimizer,
            actor_lr_schedule=config.lr_schedule,
            critic_lr_schedule=config.critic_lr_schedule,
            param_labels=param_labels,
        )
        
        # Define the final init function with dual-lr optimizer
        def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
            rng, model_rng = jax.random.split(rng)
            model = config.model.create(model_rng)

            if partial_params is not None:
                graphdef, state = nnx.split(model)
                state.replace_by_pure_dict(partial_params)
                model = nnx.merge(graphdef, state)

            params = nnx.state(model)
            params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

            return training_utils.TrainState(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                tx=dual_tx,
                opt_state=dual_tx.init(params.filter(config.trainable_filter)),
                ema_decay=config.ema_decay,
                ema_params=params if config.ema_decay else None,
                old_params=None,  
                ref_params=params,  
            )
        
        # Get the correct shape with dual-lr optimizer
        train_state_shape = jax.eval_shape(init, init_rng)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
        
    else:
        # Single-lr case - simpler path
        tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

        def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
            rng, model_rng = jax.random.split(rng)
            model = config.model.create(model_rng)

            if partial_params is not None:
                graphdef, state = nnx.split(model)
                state.replace_by_pure_dict(partial_params)
                model = nnx.merge(graphdef, state)

            params = nnx.state(model)
            params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

            return training_utils.TrainState(
                step=0,
                params=params,
                model_def=nnx.graphdef(model),
                tx=tx,
                opt_state=tx.init(params.filter(config.trainable_filter)),
                ema_decay=config.ema_decay,
                ema_params=params if config.ema_decay else None,
                old_params=None,  
                ref_params=params,  
            )

        train_state_shape = jax.eval_shape(init, init_rng)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

        if resume:
            return train_state_shape, state_sharding

        if config.pretrained_path is not None:
            weight_loader = weight_loaders.CheckpointWeightLoader(config.pretrained_path)
        else:
            weight_loader = config.weight_loader
        partial_params = _load_weights_and_validate(weight_loader, train_state_shape.params.to_pure_dict())

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def ppo_train_step(
    config: _config.RLTrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    trajectories: tuple[_model.Observation, jnp.ndarray, jnp.ndarray, dict],
    clip_ratio: float = 0.2,
    clip_ratio_negative: float | None = None,
    value_clip: float = 0.2,
    huber_delta: float = 10.0,
    critic_warmup: at.Array = False,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """PPO training step with actor-critic loss.
    
    Args:
        config: Training configuration
        rng: Random key
        state: Training state
        trajectories: Tuple of (observations, actions, advantages, info)
                     where info contains 'old_logprobs', 'chains', 'values', 'returns' from buffer
        clip_ratio: PPO clipping ratio for positive advantages (default 0.2)
        clip_ratio_negative: PPO clipping ratio for negative advantages. If None, uses clip_ratio (default None)
        value_clip: Value clipping threshold (default 0.2)
        huber_delta: Huber loss delta parameter (default 10.0)
        critic_warmup: If True, only train critic without actor updates (default False)
        
    Returns:
        Updated training state and info dict
    """
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    
    # Unpack trajectory data from buffer
    observations, actions, advantages, traj_info = trajectories
    
    # Get chains, denoise_inds, old_logprobs, prev_values, and returns from trajectory info
    chains = traj_info['chains']
    denoise_inds = traj_info['denoise_inds'] 
    old_logprobs = traj_info['old_logprobs'] 
    prev_values = traj_info['values']  # Previous value estimates from buffer
    returns = traj_info['returns']  # GAE returns from buffer
    loss_mask = traj_info['loss_mask']  # Loss mask from buffer
    
    def loss_fn(
        model: _model.BaseModel,
        train_rng: at.KeyArrayLike,
        observation: _model.Observation,
        advantages: jnp.ndarray,
        prev_values: jnp.ndarray,
        returns: jnp.ndarray,
        loss_mask: jnp.ndarray,
        chains: jnp.ndarray,
        denoise_inds: jnp.ndarray,
    ):
        """PPO loss with actor-critic."""
        # Compute actor loss using model method
        actor_loss, actor_info = model.compute_ppo_actor_loss(
            train_rng,
            observation,
            chains,
            denoise_inds,
            old_logprobs,
            advantages,
            clip_ratio=clip_ratio,
            clip_ratio_negative=clip_ratio_negative,
            valid_action_dim=7,
            loss_mask=loss_mask,
        )
        
        # Compute critic loss using model method
        critic_loss, critic_info = model.compute_value_loss(
            train_rng, 
            observation, 
            prev_values, 
            returns, 
            value_clip=value_clip, 
            huber_delta=huber_delta,
            loss_mask=loss_mask,
        )

        # Use jnp.where instead of Python if (critic_warmup is a JAX array)
        actor_loss = jnp.where(critic_warmup, 0.0, actor_loss)
        # bc_loss = jnp.where(critic_warmup, 0.0, bc_loss)

        # Total Loss
        total_loss = actor_loss + critic_loss  # + bc_loss * config.kl_weight
        
        # Combine info dicts
        adv_mean = traj_info['advantages_mean']
        adv_std = traj_info['advantages_std']
        info = {**actor_info, **critic_info, 
        "advantages_mean": adv_mean, "advantages_std": adv_std}  # "bc_loss": bc_loss,
        info.update(traj_info)
        return total_loss, info
    
    # Compute gradients
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model,
        train_rng,
        observations,
        advantages,
        prev_values,
        returns,
        loss_mask,
        chains,
        denoise_inds,
    )
    
    # Update model parameters
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Compute gradient norms and update norms separately for actor and critic
    # Get parameter labels to split gradients (same logic as in init_train_state)
    trainable_params_shape = state.params.filter(config.trainable_filter)
    
    def label_params(path_and_leaf):
        path, _ = path_and_leaf
        keystr = jax.tree_util.keystr(path)
        return 'critic' if 'value_head' in keystr else 'actor'
    
    # Create param_labels with same structure as grads/updates
    paths_and_leaves = list(jax.tree_util.tree_flatten_with_path(trainable_params_shape)[0])
    labels = [label_params(pl) for pl in paths_and_leaves]
    leaves_and_treedef = jax.tree_util.tree_flatten(trainable_params_shape)
    param_labels = jax.tree_util.tree_unflatten(leaves_and_treedef[1], labels)
    
    # Split gradients and updates by label
    # Use tree_map to create masked versions
    actor_grads = jax.tree_util.tree_map(
        lambda g, label: g if label == 'actor' else jnp.zeros_like(g),
        grads, param_labels
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g, label: g if label == 'critic' else jnp.zeros_like(g),
        grads, param_labels
    )
    actor_updates = jax.tree_util.tree_map(
        lambda u, label: u if label == 'actor' else jnp.zeros_like(u),
        updates, param_labels
    )
    critic_updates = jax.tree_util.tree_map(
        lambda u, label: u if label == 'critic' else jnp.zeros_like(u),
        updates, param_labels
    )
    
    # Compute norms
    grad_norm_total = optax.global_norm(grads)
    actor_grad_norm = optax.global_norm(actor_grads)
    critic_grad_norm = optax.global_norm(critic_grads)
    
    all_updates = jax.tree_util.tree_leaves(updates)
    update_norm_total = optax.global_norm(all_updates)
    actor_update_norm = optax.global_norm(jax.tree_util.tree_leaves(actor_updates))
    critic_update_norm = optax.global_norm(jax.tree_util.tree_leaves(critic_updates))

    info.update({
        "total_loss": loss,
        "grad_norm": grad_norm_total,
        "actor_grad_norm": actor_grad_norm,
        "critic_grad_norm": critic_grad_norm,
        "update_norm": update_norm_total,  # This will decrease when lr decreases
        "actor_update_norm": actor_update_norm,  # This reflects actual parameter change for actor
        "critic_update_norm": critic_update_norm,  # This reflects actual parameter change for critic
    })
    
    return new_state, info


def main(config: _config.RLTrainConfig):
    init_logging()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax_new").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng, env_rng = jax.random.split(rng, 3)

    mesh = sharding.make_mesh(config.fsdp_devices)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    clip_ratio = config.clip_ratio
    clip_ratio_negative = config.clip_ratio_negative if hasattr(config, 'clip_ratio_negative') else None
    value_clip = config.value_clip if hasattr(config, 'value_clip') else 0.5
    huber_delta = config.huber_delta if hasattr(config, 'huber_delta') else 10.0
    host = config.host
    port = config.port
    num_steps = config.num_steps
    
    logging.info(f"Hyperparameters: clip_ratio={clip_ratio}, clip_ratio_negative={clip_ratio_negative}, value_clip={value_clip}, huber_delta={huber_delta}")

    # Buffer configuration
    sample_batch_size = config.sample_batch_size
    mini_batch_size = config.mini_batch_size
    env_interval = config.env_interval
    eval_interval = config.eval_interval  
    recompute_interval = config.recompute_interval
    warmup_step = config.warmup_step
    use_world_model = config.use_world_model
    
    logging.info(f"Buffer configuration: sample_batch_size={sample_batch_size}, mini_batch_size={mini_batch_size}, env_interval={env_interval}, warmup_step={warmup_step}")
    logging.info(f"use_world_model={use_world_model}")
    
    # Initialize world model client if enabled
    world_model_client = None
    if use_world_model:
        try:
            import sys
            from pathlib import Path
            parent_path = Path(__file__).parent.parent
            if str(parent_path) not in sys.path:
                sys.path.insert(0, str(parent_path))
            from world_model_client import WebsocketWorldModelClient
            
            world_model_host = host
            world_model_port = config.world_model_port
            
            world_model_client = WebsocketWorldModelClient(
                host=world_model_host,
                port=world_model_port,
                timeout=1800,  # 30 minutes timeout to prevent keepalive ping timeout
            )
            logging.info(f"Successfully connected to world model server at {world_model_host}:{world_model_port}")
            logging.info(f"World model metadata: {world_model_client.get_server_metadata()}")
        except Exception as e:
            logging.error(f"Failed to initialize world model client: {e}")
            logging.error(f"Please ensure the world model server is running")
            raise
    
    # Initialize environment
    chunk_size = 10
    sequence_len = config.sequence_len
    eval_sequence_len = config.eval_sequence_len
    num_trajectories = config.num_trajectories
    
    # LIBERO-specific camera views
    camera_views = ['base_0_rgb', 'left_wrist_0_rgb']
    
    # Initialize data loaders first to get data_sharding
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    # Dataset weights path
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    data_iter = iter(data_loader)
    data_batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(data_batch)}")

    eval_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        eval=True
    )
    eval_data_iter = iter(eval_data_loader)
    eval_data = []
    for i in range(config.eval_batch_size // config.batch_size):
        eval_batch = next(eval_data_iter)
        eval_data.append(eval_batch)
    logging.info(f"Initialized eval data loader:\n{training_utils.array_tree_to_info(eval_batch)}")
    
    # Initialize train state
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        
    # Get norm_stats and transforms for environment
    norm_stats = None
    use_quantile_norm = False
    
    data_config = config.data.create(config.assets_dirs, config.model)
    tasks = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=f"/data/home/scwb314/run/data/openpi_dataset/{data_config.repo_id}").tasks

    norm_stats = data_config.norm_stats
    use_quantile_norm = data_config.use_quantile_norm
            
    if norm_stats is not None:
        logging.info("Successfully loaded norm_stats, will enable action unnormalization")
    else:
        logging.warning("Norm stats not found, actions will not be unnormalized")
    
    # Build input transforms pipeline for environment observations
    import openpi.transforms as _transforms
    input_transforms = _transforms.compose([
        _transforms.RepackTransform(
            {
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "prompt": "prompt",
            }
        ),
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])
    logging.info(f"Built input transforms pipeline with {len(input_transforms.transforms)} transforms")
        
    try:
        env = LiberoRLEnvironment(
            host=host, 
            port=port,  
            chunk_size=chunk_size, 
            sequence_len=sequence_len, 
            camera_views=camera_views,
            batch_size=config.batch_size,
            num_trajectories=num_trajectories,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
            timeout=150,
            input_transforms=input_transforms,
            world_model_client=world_model_client,
        )
        logging.info(f"Successfully connected to LIBERO server")
        logging.info(f"Initialized environment with chunk_size={chunk_size}, sequence_len={sequence_len}, num_trajectories={num_trajectories}, batch_size={config.batch_size}")
        logging.info(f"Norm stats enabled: {norm_stats is not None}, Quantile norm: {use_quantile_norm}")
        logging.info(f"Server metadata: {env.get_server_metadata()}")
    except Exception as e:
        logging.error(f"Failed to connect to LIBERO server: {e}")
        logging.error(f"Please ensure the LIBERO server is running on {host}:{port}")
        raise
    
    trajectory_buffer = TrajectoryBuffer(
        buffer_size=sample_batch_size,
        batch_size=config.batch_size,
        num_trajectories=num_trajectories,
        seq_len=sequence_len,
    )
    
    ptrain_step = jax.jit(
        functools.partial(ppo_train_step, config),
        in_shardings=(
            replicated_sharding,      # rng (0)
            train_state_sharding,     # state (1)
            # trajectories (2): (observations, actions, advantages, info)
            (data_sharding, data_sharding, replicated_sharding, replicated_sharding),
            replicated_sharding,      # critic_warmup (7) - after static args removed
        ),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
        static_argnums=(3, 4, 5, 6),  # clip_ratio, clip_ratio_negative, value_clip, huber_delta are static
    )
    logging.info("JIT-compiled PPO training step")
    
    # JIT-compile the action sampling function for efficient trajectory collection
    psample_actions = jax.jit(
        sample_actions_step,
        in_shardings=(
            replicated_sharding,     # rng
            train_state_sharding,    # state
            data_sharding,           # observation
        ),
        out_shardings=replicated_sharding,  # Actions should be replicated
        static_argnums=(3, 4, 5, 6, 7, 8),  # num_steps, mode, use_old, use_forward, return_logprobs, return_values
    )
    
    # JIT-compile the value computation function (for PPO)
    pcompute_values = jax.jit(
        compute_value_step,
        in_shardings=(
            train_state_sharding,    # state
            data_sharding,           # observations
        ),
        out_shardings=replicated_sharding,  # values
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    
    for step in pbar:
        with sharding.set_mesh(mesh):
            # Eval model in real simulator
            if step != 0 and ((step - 1) % env_interval > warmup_step and step % eval_interval == 0):
                buffer_start = time.time()
                buffer_rng = jax.random.fold_in(env_rng, step)

                eval_success_rate = []
                for eval_batch in tqdm.tqdm(eval_data, desc="Evaluating"):
                    eval_task_id_batch = eval_batch[-2] 
                    eval_tasks = [tasks.get(int(t)) for t in eval_task_id_batch]
                    eval_observations, _, eval_rewards, eval_info = collect_trajectory(
                        config=config,
                        sample_actions_fn=psample_actions,
                        compute_values_fn=pcompute_values,
                        env=env,
                        rng=buffer_rng,
                        state=train_state,
                        batch=eval_batch,
                        tasks=eval_tasks,
                        seq_len=config.eval_sequence_len,
                        max_time_stamp=config.max_time_stamp,
                        use_old=False,
                        use_forward=True,
                        use_world_model=False,
                        num_steps=num_steps,
                        mode="eval",
                        return_logprobs=False, 
                        return_values=False,
                    )
                    eval_success_rate.append(eval_info['success_rate'])
                    logging.info(f"eval_success_rate={eval_info['success_rate']:.5f}")
                    
                eval_success_rate = np.array(eval_success_rate).mean()
                logging.info(f"overall_eval_success_rate={eval_success_rate:.5f}")
                if 'value_bias_mean' in eval_info:
                    wandb.log({
                        "eval/value_bias_mean": eval_info.get('value_bias_mean', 0.0),
                        "eval/value_bias_std": eval_info.get('value_bias_std', 0.0),
                    }, step=step)
                    for i in range(len(eval_info.get("verbose_info", [])) // 16):
                        wandb.log({f"eval/return_value_figures_{i}": wandb.Image(plot_values_and_returns(eval_info.get("verbose_info", [])[i * 16:(i + 1) * 16]))}, step=step)
                for k, v in eval_info.items():
                    if "task_success_rate_" in k:
                        wandb.log({
                            f"eval/{k}": v,
                        }, step=step)
                wandb.log({
                    "eval/success_rate": eval_success_rate,
                }, step=step)

                train_observations = trajectory_buffer.observations_list[0]
                train_rewards = trajectory_buffer.rewards_list[0]
                
                # Get advantages from buffer
                train_advantages = trajectory_buffer.advantages[:sequence_len * config.batch_size * num_trajectories]
                
                # Choose camera key
                camera_keys = list(train_observations.images.keys())
                camera_key = camera_keys[0]
                
                # train_rewards shape: (seq_len * batch_size * num_trajectories, chunk_size)
                train_rewards_np = np.array(train_rewards)
                # Sum over chunk_size to get step-level rewards
                train_rewards_step = train_rewards_np.sum(axis=-1)  # (seq_len * batch_size * num_trajectories,)
                
                train_traj_images = []
                
                # Log multiple batches to increase image count
                # Record up to 8 batches or all available batches, whichever is smaller
                num_batches_to_log = min(8, config.batch_size)
                
                for batch_idx in range(num_batches_to_log):
                    for traj_idx in range(num_trajectories):
                        for s in range(sequence_len):
                            # Index calculation: observations are stored as (seq_len * batch_size * num_trajectories,)
                            flat_idx = s * config.batch_size * num_trajectories + batch_idx * num_trajectories + traj_idx
                            
                            frame = np.array(train_observations.images[camera_key][flat_idx])
                            frame = np.clip(((frame + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)
                            
                            step_reward = float(train_rewards_step[flat_idx])
                            step_advantage = float(train_advantages[flat_idx])

                            caption = f"train_b{batch_idx}_t{traj_idx}_s{s}_r={step_reward:.3f}_adv={step_advantage:.3f}"
                            
                            train_traj_images.append(wandb.Image(frame, caption=caption))
                
                log_dict = {
                    "trajectories/all_train_images": train_traj_images,
                }

                logging.info(
                    f"policy_loss={info.get('policy_loss', 0.0):.5f}"
                )
                logging.info(
                    f"value_loss={info.get('value_loss', 0.0):.5f}"
                )
                
                wandb.log(log_dict, step=step)

                try:
                    # Choose camera key
                    camera_keys = list(eval_observations.images.keys())
                    camera_key = camera_keys[0]
                    
                    eval_rewards_np = jax.device_get(eval_rewards)
                    total_eval_envs = config.batch_size * num_trajectories
                    eval_rewards_step = eval_rewards_np.sum(axis=-1)
                    eval_rewards_reshaped = eval_rewards_step.reshape(eval_sequence_len, total_eval_envs)
                    trajectory_returns_np = eval_rewards_reshaped[:eval_sequence_len].sum(axis=0)
                    trajectory_returns_np = trajectory_returns_np.reshape(config.batch_size, num_trajectories)

                    # Log all trajectories for the first batch sample
                    traj_images = []
                    traj_return_values = []

                    for traj_idx in range(num_trajectories):
                        traj_return_value = float(trajectory_returns_np[0, traj_idx])
                        traj_return_values.append(traj_return_value)
                        
                        for s in range(eval_sequence_len + 1):
                            flat_idx = s * num_trajectories * config.batch_size + traj_idx
                            
                            frame = jax.device_get(eval_observations.images[camera_key][flat_idx])
                            frame = np.clip(((np.array(frame) + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)
                            
                            traj_images.append(wandb.Image(frame, caption=f"traj_{traj_idx}_step_{s}_return={traj_return_value:.4f}"))

                    wandb.log({
                        "trajectories/all_eval_images": traj_images,
                    }, step=step)
                    
                except Exception as e:
                    logging.warning(f"Failed to log trajectory images: {e}")

            # Refill buffer periodically
            if step % env_interval == 0 and (step == 0 or step > warmup_step) and step != config.num_train_steps - 1:
                buffer_start = time.time()
                buffer_rng = jax.random.fold_in(env_rng, step)

                # Collect trajectories to buffer with logprobs
                collection_info = collect_trajectories_to_buffer(
                    config=config,
                    sample_actions_fn=psample_actions,
                    compute_values_fn=pcompute_values,
                    env=env,
                    rng=buffer_rng,
                    state=train_state,
                    batch_iterator=data_iter,
                    tasks=tasks,
                    buffer=trajectory_buffer,
                    use_world_model=use_world_model,  # Use world model for training if enabled
                    num_steps=num_steps,
                    mode="train",
                    return_logprobs=True, 
                    return_values=True,
                    max_time_stamp=config.max_time_stamp,
                )
                # trajectory_buffer.offload(f"offload/{config.exp_name}_{step // config.env_interval}.npz", collection_info)
                # collection_info = trajectory_buffer.load(f"offload/{config.exp_name}_{step // config.env_interval}.npz")
                
                buffer_time = time.time() - buffer_start
                logging.info(
                    f"Buffer loaded at step {step} in {buffer_time:.2f}s, "
                )
                if 'avg_success_rate' in collection_info:
                    logging.info(
                        f"avg_success_rate={collection_info.get('avg_success_rate', 0.0):.3f}"
                    )
                # Log buffer statistics
                wandb.log({
                    "buffer/refill_time": buffer_time,
                    **{f"buffer/{k}": v for k, v in collection_info.items() if k.startswith('avg_') and not k.startswith('avg_avg_') and k not in ['avg_sample_action_time', 'avg_env_step_time', 'avg_obs_processing_time', 'avg_total_sample_action_time', 'avg_total_env_step_time', 'avg_total_obs_processing_time']},
                }, step=step)
                # Recompute values and advantages using current value function
                
            # Sample from buffer for training
            sample_rng = jax.random.fold_in(train_rng, step)
            trajectories = trajectory_buffer.sample(sample_rng, mini_batch_size, data_sharding, replicated_sharding)

            if step > 0 and recompute_interval > 0 and step % recompute_interval == 0:
                recompute_info = trajectory_buffer.recompute_values_and_advantages(
                    compute_values_fn=pcompute_values,
                    state=train_state,
                    recompute_batch_size=mini_batch_size,
                )
                logging.info(
                    f"Recomputed advantages at step {step}: "
                    f"values {recompute_info['old_values_mean']:.4f} -> {recompute_info['new_values_mean']:.4f}, "
                    f"advantages {recompute_info['old_advantages_mean']:.4f} -> {recompute_info['new_advantages_mean']:.4f}"
                )
                wandb.log({
                    "buffer/recompute_values_change": recompute_info['values_change'],
                    "buffer/recompute_old_values_mean": recompute_info['old_values_mean'],
                    "buffer/recompute_new_values_mean": recompute_info['new_values_mean'],
                    "buffer/recompute_old_advantages_mean": recompute_info['old_advantages_mean'],
                    "buffer/recompute_new_advantages_mean": recompute_info['new_advantages_mean'],
                }, step=step)

            # Training step 
            critic_warmup = jnp.array(step % config.env_interval < warmup_step)
            train_state, info = ptrain_step(
                train_rng,
                train_state,
                trajectories,
                clip_ratio,
                clip_ratio_negative,
                value_clip,
                huber_delta,
                critic_warmup,
            )

        if step % config.log_interval == 0 and step != 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        infos.append(info)

        data_batch = next(data_iter)

        if step % config.save_interval == 0 and step != 0 or step == config.num_train_steps - 1:
            jax.block_until_ready(train_state)
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
            logging.info(f"Saved checkpoint at step {step}")

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    
    # Clean up
    env.close()


if __name__ == "__main__":
    config = _config.cli()
    main(config)
