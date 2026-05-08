import dataclasses
import functools
import logging
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.libero_env import LiberoRLEnvironment
import openpi.training.libero_rl_utils as _libero_rl_utils
import openpi.training.optimizer as _optimizer
from openpi.training.rl_env import TrajectoryBuffer
from openpi.training.rl_env import compute_value_step
from openpi.training.rl_env import sample_actions_step
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def _select_weight_loader(config: _config.RLTrainConfig) -> _weight_loaders.WeightLoader:
    if config.pretrained_path is not None:
        return _weight_loaders.CheckpointWeightLoader(config.pretrained_path)
    return config.weight_loader


def _actor_critic_param_labels(trainable_params_shape: Any) -> Any:
    """Returns a pytree matching trainable params with ``actor`` or ``critic`` labels."""

    def label_params(path_and_leaf: tuple[Any, Any]) -> str:
        path, _ = path_and_leaf
        keystr = jax.tree_util.keystr(path)
        return "critic" if "value_head" in keystr else "actor"

    paths_and_leaves = list(jax.tree_util.tree_flatten_with_path(trainable_params_shape)[0])
    labels = [label_params(path_and_leaf) for path_and_leaf in paths_and_leaves]
    _, treedef = jax.tree_util.tree_flatten(trainable_params_shape)
    return jax.tree_util.tree_unflatten(treedef, labels)


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
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
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
    return traverse_util.unflatten_dict({k: v for k, v in result.items() if not isinstance(v, jax.ShapeDtypeStruct)})


def _create_train_state(
    config: _config.RLTrainConfig,
    tx: Any,
    rng: at.KeyArrayLike,
    partial_params: at.Params | None = None,
) -> training_utils.TrainState:
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
        temp_init = functools.partial(_create_train_state, config, temp_tx)

        # Get shape with temporary optimizer
        temp_train_state_shape = jax.eval_shape(temp_init, init_rng)

        if resume:
            return temp_train_state_shape, sharding.fsdp_sharding(temp_train_state_shape, mesh, log=True)

        # Load partial params
        weight_loader = _select_weight_loader(config)
        partial_params = _load_weights_and_validate(weight_loader, temp_train_state_shape.params.to_pure_dict())

        # Create parameter labels for dual-lr optimizer
        trainable_params_shape = temp_train_state_shape.params.filter(config.trainable_filter)
        param_labels = _actor_critic_param_labels(trainable_params_shape)

        # Create the actual dual-lr optimizer
        dual_tx = _optimizer.create_dual_lr_optimizer(
            config.optimizer,
            actor_lr_schedule=config.lr_schedule,
            critic_lr_schedule=config.critic_lr_schedule,
            param_labels=param_labels,
        )

        init = functools.partial(_create_train_state, config, dual_tx)

        # Get the correct shape with dual-lr optimizer
        train_state_shape = jax.eval_shape(init, init_rng)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    else:
        # Single-lr case - simpler path
        tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
        init = functools.partial(_create_train_state, config, tx)

        train_state_shape = jax.eval_shape(init, init_rng)
        state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

        if resume:
            return train_state_shape, state_sharding

        weight_loader = _select_weight_loader(config)
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
    critic_warmup: at.Array | None = None,
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
    critic_warmup = jnp.zeros((), dtype=jnp.bool_) if critic_warmup is None else critic_warmup

    observations, _, advantages, traj_info = trajectories

    chains = traj_info["chains"]
    denoise_inds = traj_info["denoise_inds"]
    old_logprobs = traj_info["old_logprobs"]
    prev_values = traj_info["values"]
    returns = traj_info["returns"]
    loss_mask = traj_info["loss_mask"]

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
        actor_loss, actor_info = model.compute_ppo_actor_loss(
            train_rng,
            observation,
            chains,
            denoise_inds,
            old_logprobs,
            advantages,
            clip_ratio=clip_ratio,
            clip_ratio_negative=clip_ratio_negative,
            valid_action_dim=_libero_rl_utils.LIBERO_VALID_ACTION_DIM,
            loss_mask=loss_mask,
        )

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
        total_loss = actor_loss + critic_loss

        info = {
            **actor_info,
            **critic_info,
            "advantages_mean": traj_info["advantages_mean"],
            "advantages_std": traj_info["advantages_std"],
        }
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

    # Compute gradient norms separately for actor and critic.
    trainable_params_shape = state.params.filter(config.trainable_filter)
    param_labels = _actor_critic_param_labels(trainable_params_shape)

    # Split gradients by label
    actor_grads = jax.tree_util.tree_map(
        lambda g, label: g if label == "actor" else jnp.zeros_like(g), grads, param_labels
    )
    critic_grads = jax.tree_util.tree_map(
        lambda g, label: g if label == "critic" else jnp.zeros_like(g), grads, param_labels
    )

    # Compute norms
    grad_norm_total = optax.global_norm(grads)
    actor_grad_norm = optax.global_norm(actor_grads)
    critic_grad_norm = optax.global_norm(critic_grads)

    info.update(
        {
            "total_loss": loss,
            "grad_norm": grad_norm_total,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
        }
    )

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
    clip_ratio_negative = getattr(config, "clip_ratio_negative", None)
    value_clip = getattr(config, "value_clip", 0.5)
    huber_delta = getattr(config, "huber_delta", 10.0)
    host = config.host
    port = config.port
    num_steps = config.num_steps

    logging.info(
        "Hyperparameters: clip_ratio=%s, clip_ratio_negative=%s, value_clip=%s, huber_delta=%s",
        clip_ratio,
        clip_ratio_negative,
        value_clip,
        huber_delta,
    )

    # Buffer configuration
    sample_batch_size = config.sample_batch_size
    mini_batch_size = config.mini_batch_size
    env_interval = config.env_interval
    eval_interval = config.eval_interval
    recompute_interval = config.recompute_interval
    warmup_step = config.warmup_step
    use_world_model = config.use_world_model

    logging.info(
        "Buffer configuration: sample_batch_size=%s, mini_batch_size=%s, env_interval=%s, warmup_step=%s",
        sample_batch_size,
        mini_batch_size,
        env_interval,
        warmup_step,
    )
    logging.info("use_world_model=%s", use_world_model)

    world_model_client = _libero_rl_utils.init_world_model_client(config, host)

    # Initialize environment
    chunk_size = _libero_rl_utils.LIBERO_CHUNK_SIZE
    sequence_len = config.sequence_len
    eval_sequence_len = config.eval_sequence_len
    num_trajectories = config.num_trajectories

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

    eval_data = _libero_rl_utils.load_eval_batches(config, data_sharding)
    if eval_data:
        logging.info(f"Initialized eval data loader:\n{training_utils.array_tree_to_info(eval_data[-1])}")

    # Initialize train state
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)

    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    data_config = config.data.create(config.assets_dirs, config.model)
    tasks = _libero_rl_utils.load_libero_tasks(data_config)
    norm_stats = data_config.norm_stats
    use_quantile_norm = data_config.use_quantile_norm

    if norm_stats is not None:
        logging.info("Successfully loaded norm_stats, will enable action unnormalization")
    else:
        logging.warning("Norm stats not found, actions will not be unnormalized")

    input_transforms = _libero_rl_utils.build_input_transforms(data_config)
    logging.info(f"Built input transforms pipeline with {len(input_transforms.transforms)} transforms")

    try:
        env = LiberoRLEnvironment(
            host=host,
            port=port,
            chunk_size=chunk_size,
            sequence_len=sequence_len,
            camera_views=_libero_rl_utils.LIBERO_CAMERA_VIEWS,
            batch_size=config.batch_size,
            num_trajectories=num_trajectories,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
            timeout=_libero_rl_utils.LIBERO_TIMEOUT_SECONDS,
            input_transforms=input_transforms,
            world_model_client=world_model_client,
        )
        logging.info("Successfully connected to LIBERO server")
        logging.info(
            "Initialized environment with chunk_size=%s, sequence_len=%s, num_trajectories=%s, batch_size=%s",
            chunk_size,
            sequence_len,
            num_trajectories,
            config.batch_size,
        )
        logging.info("Norm stats enabled: %s, Quantile norm: %s", norm_stats is not None, use_quantile_norm)
        logging.info("Server metadata: %s", env.get_server_metadata())
    except Exception:
        logging.exception("Failed to connect to LIBERO server")
        logging.error("Please ensure the LIBERO server is running on %s:%s", host, port)
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
            replicated_sharding,  # rng (0)
            train_state_sharding,  # state (1)
            # trajectories (2): (observations, actions, advantages, info)
            (data_sharding, data_sharding, replicated_sharding, replicated_sharding),
            replicated_sharding,  # critic_warmup (7) - after static args removed
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
            replicated_sharding,  # rng
            train_state_sharding,  # state
            data_sharding,  # observation
        ),
        out_shardings=replicated_sharding,  # Actions should be replicated
        static_argnums=(3, 4, 5, 6, 7, 8),  # num_steps, mode, use_old, use_forward, return_logprobs, return_values
    )

    # JIT-compile the value computation function (for PPO)
    pcompute_values = jax.jit(
        compute_value_step,
        in_shardings=(
            train_state_sharding,  # state
            data_sharding,  # observations
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
    latest_train_info = None

    for step in pbar:
        with sharding.set_mesh(mesh):
            if _libero_rl_utils.should_run_eval(step, env_interval, warmup_step, eval_interval):
                buffer_rng = jax.random.fold_in(env_rng, step)
                eval_success_rate, eval_observations, eval_rewards, eval_info = _libero_rl_utils.run_evaluation(
                    config=config,
                    sample_actions_fn=psample_actions,
                    compute_values_fn=pcompute_values,
                    env=env,
                    rng=buffer_rng,
                    state=train_state,
                    eval_data=eval_data,
                    tasks=tasks,
                    num_steps=num_steps,
                )
                _libero_rl_utils.log_eval_metrics(eval_info, eval_success_rate, step)
                _libero_rl_utils.log_train_trajectory_images(
                    config=config,
                    trajectory_buffer=trajectory_buffer,
                    sequence_len=sequence_len,
                    num_trajectories=num_trajectories,
                    latest_train_info=latest_train_info,
                    step=step,
                )

                try:
                    _libero_rl_utils.log_eval_trajectory_images(
                        config=config,
                        eval_observations=eval_observations,
                        eval_rewards=eval_rewards,
                        eval_sequence_len=eval_sequence_len,
                        num_trajectories=num_trajectories,
                        step=step,
                    )
                except Exception:
                    logging.warning("Failed to log trajectory images", exc_info=True)

            if _libero_rl_utils.should_refill_buffer(step, env_interval, warmup_step, config.num_train_steps):
                _libero_rl_utils.refill_trajectory_buffer(
                    config=config,
                    sample_actions_fn=psample_actions,
                    compute_values_fn=pcompute_values,
                    env=env,
                    rng=jax.random.fold_in(env_rng, step),
                    state=train_state,
                    batch_iterator=data_iter,
                    tasks=tasks,
                    trajectory_buffer=trajectory_buffer,
                    use_world_model=use_world_model,
                    num_steps=num_steps,
                    step=step,
                )

            sample_rng = jax.random.fold_in(train_rng, step)
            trajectories = trajectory_buffer.sample(sample_rng, mini_batch_size, data_sharding, replicated_sharding)

            if step > 0 and recompute_interval > 0 and step % recompute_interval == 0:
                _libero_rl_utils.recompute_buffer_values(
                    trajectory_buffer=trajectory_buffer,
                    compute_values_fn=pcompute_values,
                    state=train_state,
                    mini_batch_size=mini_batch_size,
                    step=step,
                )

            critic_warmup = jnp.array(step % env_interval < warmup_step)
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
            latest_train_info = info

        if step % config.log_interval == 0 and step != 0:
            infos = _libero_rl_utils.log_training_interval(infos, step, pbar)
        infos.append(info)

        _ = next(data_iter)

        if _libero_rl_utils.should_save_checkpoint(step, config.save_interval, config.num_train_steps):
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
