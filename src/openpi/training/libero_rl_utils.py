import logging
from pathlib import Path
import sys
import time
from typing import Any

from flax.training import common_utils
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.libero_env import LiberoRLEnvironment
from openpi.training.rl_env import TrajectoryBuffer
from openpi.training.rl_env import collect_trajectories_to_buffer
from openpi.training.rl_env import collect_trajectory
import openpi.training.utils as training_utils
import openpi.transforms as _transforms

LIBERO_CAMERA_VIEWS = ("base_0_rgb", "left_wrist_0_rgb")
LIBERO_VALID_ACTION_DIM = 7
LIBERO_CHUNK_SIZE = 10
MAX_TRAIN_IMAGE_BATCHES = 8
WORLD_MODEL_TIMEOUT_SECONDS = 30 * 60
LIBERO_TIMEOUT_SECONDS = 150
DATASET_ROOT_TEMPLATE = "/data/home/scwb314/run/data/openpi_dataset/{repo_id}"
BUFFER_AVG_METRIC_EXCLUDE = frozenset(
    {
        "avg_sample_action_time",
        "avg_env_step_time",
        "avg_obs_processing_time",
        "avg_total_sample_action_time",
        "avg_total_env_step_time",
        "avg_total_obs_processing_time",
    }
)


def plot_values_and_returns(verbose_info: list[dict[str, Any]]) -> np.ndarray:
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
        task_ids = np.asarray(info.get("task_id", []))
        task = task_ids[0] if task_ids.size else "unknown"

        ax.plot(timesteps, gt_returns, label="gt_return")
        ax.plot(timesteps, pred_values, label="pred_value")
        ax.set_xlabel("timestep")
        ax.set_ylabel("value / return")
        ax.set_title(f"Trajectory {idx} Task {task}")
        ax.legend()
        ax.grid(visible=True, alpha=0.3)

    for idx in range(n_plots, n_rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.tight_layout()

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    image = buf[:, :, :3].copy()
    plt.close(fig)
    return image


def log_eval_metrics(eval_info: dict[str, Any], eval_success_rate: float, step: int) -> None:
    if "value_bias_mean" in eval_info:
        wandb.log(
            {
                "eval/value_bias_mean": eval_info.get("value_bias_mean", 0.0),
                "eval/value_bias_std": eval_info.get("value_bias_std", 0.0),
            },
            step=step,
        )

        verbose_info = eval_info.get("verbose_info", [])
        for figure_idx in range(len(verbose_info) // 16):
            start = figure_idx * 16
            figure = plot_values_and_returns(verbose_info[start : start + 16])
            wandb.log({f"eval/return_value_figures_{figure_idx}": wandb.Image(figure)}, step=step)

    task_success_rates = {f"eval/{key}": value for key, value in eval_info.items() if "task_success_rate_" in key}
    if task_success_rates:
        wandb.log(task_success_rates, step=step)

    wandb.log({"eval/success_rate": eval_success_rate}, step=step)


def log_train_trajectory_images(
    *,
    config: _config.RLTrainConfig,
    trajectory_buffer: TrajectoryBuffer,
    sequence_len: int,
    num_trajectories: int,
    latest_train_info: dict[str, Any] | None,
    step: int,
) -> None:
    train_observations = trajectory_buffer.observations_list[0]
    train_rewards = trajectory_buffer.rewards_list[0]
    train_advantages = trajectory_buffer.advantages[: sequence_len * config.batch_size * num_trajectories]

    camera_key = next(iter(train_observations.images))
    train_rewards_step = np.asarray(jax.device_get(train_rewards)).sum(axis=-1)
    train_advantages = np.asarray(jax.device_get(train_advantages))

    train_traj_images = []
    num_batches_to_log = min(MAX_TRAIN_IMAGE_BATCHES, config.batch_size)
    for batch_idx in range(num_batches_to_log):
        for traj_idx in range(num_trajectories):
            for seq_idx in range(sequence_len):
                flat_idx = seq_idx * config.batch_size * num_trajectories + batch_idx * num_trajectories + traj_idx
                frame = _to_uint8_frame(train_observations.images[camera_key][flat_idx])
                step_reward = float(train_rewards_step[flat_idx])
                step_advantage = float(train_advantages[flat_idx])
                caption = f"train_b{batch_idx}_t{traj_idx}_s{seq_idx}_r={step_reward:.3f}_adv={step_advantage:.3f}"
                train_traj_images.append(wandb.Image(frame, caption=caption))

    latest_train_info = latest_train_info or {}
    policy_loss = float(jax.device_get(latest_train_info.get("policy_loss", 0.0)))
    value_loss = float(jax.device_get(latest_train_info.get("value_loss", 0.0)))
    logging.info("policy_loss=%.5f", policy_loss)
    logging.info("value_loss=%.5f", value_loss)

    wandb.log({"trajectories/all_train_images": train_traj_images}, step=step)


def log_eval_trajectory_images(
    *,
    config: _config.RLTrainConfig,
    eval_observations: _model.Observation,
    eval_rewards: jnp.ndarray,
    eval_sequence_len: int,
    num_trajectories: int,
    step: int,
) -> None:
    camera_key = next(iter(eval_observations.images))

    eval_rewards_step = np.asarray(jax.device_get(eval_rewards)).sum(axis=-1)
    total_eval_envs = config.batch_size * num_trajectories
    eval_rewards_by_step = eval_rewards_step.reshape(eval_sequence_len, total_eval_envs)
    trajectory_returns = eval_rewards_by_step[:eval_sequence_len].sum(axis=0)
    trajectory_returns = trajectory_returns.reshape(config.batch_size, num_trajectories)

    traj_images = []
    for traj_idx in range(num_trajectories):
        traj_return_value = float(trajectory_returns[0, traj_idx])
        for seq_idx in range(eval_sequence_len + 1):
            flat_idx = seq_idx * num_trajectories * config.batch_size + traj_idx
            frame = _to_uint8_frame(eval_observations.images[camera_key][flat_idx])
            caption = f"traj_{traj_idx}_step_{seq_idx}_return={traj_return_value:.4f}"
            traj_images.append(wandb.Image(frame, caption=caption))

    wandb.log({"trajectories/all_eval_images": traj_images}, step=step)


def log_training_interval(infos: list[dict[str, Any]], step: int, pbar: Any) -> list[dict[str, Any]]:
    if not infos:
        return infos

    stacked_infos = common_utils.stack_forest(infos)
    reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
    info_str = ", ".join(f"{key}={value:.4f}" for key, value in reduced_info.items())
    pbar.write(f"Step {step}: {info_str}")
    wandb.log(reduced_info, step=step)
    return []


def should_run_eval(step: int, env_interval: int, warmup_step: int, eval_interval: int) -> bool:
    return step != 0 and (step - 1) % env_interval > warmup_step and step % eval_interval == 0


def should_refill_buffer(step: int, env_interval: int, warmup_step: int, num_train_steps: int) -> bool:
    return step % env_interval == 0 and (step == 0 or step > warmup_step) and step != num_train_steps - 1


def should_save_checkpoint(step: int, save_interval: int, num_train_steps: int) -> bool:
    return (step % save_interval == 0 and step != 0) or step == num_train_steps - 1


def init_world_model_client(config: _config.RLTrainConfig, host: str) -> Any | None:
    if not config.use_world_model:
        return None

    try:
        try:
            from world_model_client import WebsocketWorldModelClient
        except ModuleNotFoundError:
            repo_root = Path(__file__).resolve().parents[3]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from world_model_client import WebsocketWorldModelClient

        world_model_client = WebsocketWorldModelClient(
            host=host,
            port=config.world_model_port,
            timeout=WORLD_MODEL_TIMEOUT_SECONDS,
        )
        logging.info("Successfully connected to world model server at %s:%s", host, config.world_model_port)
        logging.info("World model metadata: %s", world_model_client.get_server_metadata())
        return world_model_client
    except Exception:
        logging.exception("Failed to initialize world model client")
        logging.error("Please ensure the world model server is running")
        raise


def build_input_transforms(data_config: Any) -> Any:
    return _transforms.compose(
        [
            _transforms.RepackTransform(
                {
                    "observation/image": "image",
                    "observation/wrist_image": "wrist_image",
                    "observation/state": "state",
                    "prompt": "prompt",
                }
            ),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def load_libero_tasks(data_config: Any) -> Any:
    dataset_root = DATASET_ROOT_TEMPLATE.format(repo_id=data_config.repo_id)
    return lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=dataset_root).tasks


def load_eval_batches(config: _config.RLTrainConfig, data_sharding: jax.sharding.NamedSharding) -> list[Any]:
    eval_data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        eval=True,
    )
    eval_data_iter = iter(eval_data_loader)
    num_eval_batches = config.eval_batch_size // config.batch_size
    return [next(eval_data_iter) for _ in range(num_eval_batches)]


def run_evaluation(
    *,
    config: _config.RLTrainConfig,
    sample_actions_fn: Any,
    compute_values_fn: Any,
    env: LiberoRLEnvironment,
    rng: jax.Array,
    state: training_utils.TrainState,
    eval_data: list[Any],
    tasks: Any,
    num_steps: int,
) -> tuple[float, _model.Observation, jnp.ndarray, dict[str, Any]]:
    eval_success_rates = []
    latest_eval_observations = None
    latest_eval_rewards = None
    latest_eval_info = None

    for eval_batch in tqdm.tqdm(eval_data, desc="Evaluating"):
        eval_task_id_batch = eval_batch[-2]
        eval_tasks = [tasks.get(int(task_id)) for task_id in eval_task_id_batch]
        eval_observations, _, eval_rewards, eval_info = collect_trajectory(
            config=config,
            sample_actions_fn=sample_actions_fn,
            compute_values_fn=compute_values_fn,
            env=env,
            rng=rng,
            state=state,
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
        eval_success_rates.append(eval_info["success_rate"])
        logging.info("eval_success_rate=%.5f", eval_info["success_rate"])
        latest_eval_observations = eval_observations
        latest_eval_rewards = eval_rewards
        latest_eval_info = eval_info

    if latest_eval_observations is None or latest_eval_rewards is None or latest_eval_info is None:
        raise ValueError("Evaluation requires at least one eval batch.")

    eval_success_rate = float(np.asarray(eval_success_rates).mean())
    logging.info("overall_eval_success_rate=%.5f", eval_success_rate)
    return eval_success_rate, latest_eval_observations, latest_eval_rewards, latest_eval_info


def refill_trajectory_buffer(
    *,
    config: _config.RLTrainConfig,
    sample_actions_fn: Any,
    compute_values_fn: Any,
    env: LiberoRLEnvironment,
    rng: jax.Array,
    state: training_utils.TrainState,
    batch_iterator: Any,
    tasks: Any,
    trajectory_buffer: TrajectoryBuffer,
    use_world_model: bool,
    num_steps: int,
    step: int,
) -> None:
    buffer_start = time.time()
    collection_info = collect_trajectories_to_buffer(
        config=config,
        sample_actions_fn=sample_actions_fn,
        compute_values_fn=compute_values_fn,
        env=env,
        rng=rng,
        state=state,
        batch_iterator=batch_iterator,
        tasks=tasks,
        buffer=trajectory_buffer,
        use_world_model=use_world_model,
        num_steps=num_steps,
        mode="train",
        return_logprobs=True,
        return_values=True,
        max_time_stamp=config.max_time_stamp,
    )

    buffer_time = time.time() - buffer_start
    logging.info("Buffer loaded at step %s in %.2fs", step, buffer_time)
    if "avg_success_rate" in collection_info:
        logging.info("avg_success_rate=%.3f", collection_info.get("avg_success_rate", 0.0))
    _log_buffer_metrics(collection_info, buffer_time, step)


def recompute_buffer_values(
    *,
    trajectory_buffer: TrajectoryBuffer,
    compute_values_fn: Any,
    state: training_utils.TrainState,
    mini_batch_size: int,
    step: int,
) -> None:
    recompute_info = trajectory_buffer.recompute_values_and_advantages(
        compute_values_fn=compute_values_fn,
        state=state,
        recompute_batch_size=mini_batch_size,
    )
    logging.info(
        "Recomputed advantages at step %s: values %.4f -> %.4f, advantages %.4f -> %.4f",
        step,
        recompute_info["old_values_mean"],
        recompute_info["new_values_mean"],
        recompute_info["old_advantages_mean"],
        recompute_info["new_advantages_mean"],
    )
    _log_recompute_metrics(recompute_info, step)


def _to_uint8_frame(image: Any) -> np.ndarray:
    frame = np.asarray(jax.device_get(image))
    return np.clip(((frame + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)


def _is_buffer_avg_metric(key: str) -> bool:
    return key.startswith("avg_") and not key.startswith("avg_avg_") and key not in BUFFER_AVG_METRIC_EXCLUDE


def _log_buffer_metrics(collection_info: dict[str, Any], buffer_time: float, step: int) -> None:
    payload = {
        "buffer/refill_time": buffer_time,
        **{f"buffer/{key}": value for key, value in collection_info.items() if _is_buffer_avg_metric(key)},
    }
    wandb.log(payload, step=step)


def _log_recompute_metrics(recompute_info: dict[str, Any], step: int) -> None:
    wandb.log(
        {
            "buffer/recompute_values_change": recompute_info["values_change"],
            "buffer/recompute_old_values_mean": recompute_info["old_values_mean"],
            "buffer/recompute_new_values_mean": recompute_info["new_values_mean"],
            "buffer/recompute_old_advantages_mean": recompute_info["old_advantages_mean"],
            "buffer/recompute_new_advantages_mean": recompute_info["new_advantages_mean"],
        },
        step=step,
    )
