#!/usr/bin/env python3
"""Convert LIBERO rollout trajectories to Bagel world-model data format."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import Counter
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_INPUT_DIR = pathlib.Path(
    "/data/home/scwb314/run/data/rollout_data/goal_with_wrist_v2/data"
)
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "/data/home/scwb314/run/data/bagel_data/dynamics/libero_goal_with_wrist_v2"
)

ROLLOUT_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<task_id>\d+)_(?P<traj_id>\d+)_(?P<status>success|failure)\.npy$"
)


def parse_rollout_name(path: pathlib.Path) -> tuple[str, int, int, str]:
    match = ROLLOUT_RE.match(path.name)
    if match is None:
        return (path.stem, -1, -1, "")
    return (
        match.group("prefix"),
        int(match.group("task_id")),
        int(match.group("traj_id")),
        match.group("status"),
    )


def is_success_file(path: pathlib.Path) -> bool:
    match = ROLLOUT_RE.match(path.name)
    if match is not None:
        return match.group("status") == "success"
    return "success" in path.stem and "failure" not in path.stem


def coerce_step(raw_step: Any, path: pathlib.Path, frame_idx: int) -> dict[str, Any]:
    if isinstance(raw_step, np.ndarray) and raw_step.shape == ():
        raw_step = raw_step.item()
    if not isinstance(raw_step, dict):
        raise TypeError(f"{path}: frame {frame_idx} is {type(raw_step)}, expected dict")

    missing = {"image", "wrist_image", "action", "prompt"} - set(raw_step)
    if missing:
        raise KeyError(f"{path}: frame {frame_idx} missing keys: {sorted(missing)}")
    return raw_step


def normalize_image_array(image: Any, path: pathlib.Path, key: str, frame_idx: int) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"{path}: frame {frame_idx} {key} has shape {arr.shape}, expected HWC image")
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.size > 0 and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_actions(path: pathlib.Path) -> tuple[np.ndarray, str]:
    arr = np.load(path, allow_pickle=True)
    actions: list[np.ndarray] = []
    task_prompt = ""

    for frame_idx, raw_step in enumerate(arr):
        step = coerce_step(raw_step, path, frame_idx)
        action = np.asarray(step["action"], dtype=np.float64).reshape(-1)
        actions.append(action)
        if not task_prompt:
            task_prompt = str(step.get("prompt", ""))

    if not actions:
        raise ValueError(f"{path}: empty trajectory")
    action_dim = actions[0].shape[0]
    for frame_idx, action in enumerate(actions):
        if action.shape[0] != action_dim:
            raise ValueError(
                f"{path}: frame {frame_idx} action dim {action.shape[0]} != {action_dim}"
            )
    return np.stack(actions, axis=0), task_prompt


def load_episode(path: pathlib.Path) -> list[dict[str, Any]]:
    arr = np.load(path, allow_pickle=True)
    episode: list[dict[str, Any]] = []

    for frame_idx, raw_step in enumerate(arr):
        step = coerce_step(raw_step, path, frame_idx)
        episode.append(
            {
                "image": normalize_image_array(step["image"], path, "image", frame_idx),
                "wrist_image": normalize_image_array(
                    step["wrist_image"], path, "wrist_image", frame_idx
                ),
                "action": np.asarray(step["action"], dtype=np.float64).reshape(-1),
                "prompt": str(step["prompt"]),
            }
        )

    if not episode:
        raise ValueError(f"{path}: empty trajectory")
    return episode


def compute_action_normalizer(
    action_chunks: list[np.ndarray], percentile_clip: float
) -> dict[str, np.ndarray]:
    all_actions = np.concatenate(action_chunks, axis=0)
    action_dim = all_actions.shape[1]

    lower_percentile = 100.0 - percentile_clip
    clip_min = np.zeros(action_dim, dtype=np.float64)
    clip_max = np.zeros(action_dim, dtype=np.float64)

    print(f"Computing action normalizer with percentile_clip={percentile_clip}")
    for dim in range(action_dim):
        clip_min[dim] = np.percentile(all_actions[:, dim], lower_percentile)
        clip_max[dim] = np.percentile(all_actions[:, dim], percentile_clip)
        clipped = np.sum((all_actions[:, dim] < clip_min[dim]) | (all_actions[:, dim] > clip_max[dim]))
        print(
            f"  dim {dim}: [{clip_min[dim]:.6f}, {clip_max[dim]:.6f}], "
            f"clipped {clipped}/{len(all_actions)}"
        )

    clipped_actions = np.clip(all_actions, clip_min, clip_max)
    return {
        "min": clip_min.copy(),
        "max": clip_max.copy(),
        "clip_min": clip_min,
        "clip_max": clip_max,
        "raw_actions": all_actions,
        "clipped_actions": clipped_actions,
    }


def format_action(action: np.ndarray, normalizer: dict[str, np.ndarray]) -> str:
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action.reshape(1, -1)

    min_vals = normalizer["min"]
    max_vals = normalizer["max"]
    clip_min = normalizer["clip_min"]
    clip_max = normalizer["clip_max"]

    timestep_strs: list[str] = []
    for step_idx, step_action in enumerate(action):
        step_action = np.clip(step_action, clip_min, clip_max)
        normalized: list[int] = []
        for dim, value in enumerate(step_action):
            range_val = max_vals[dim] - min_vals[dim]
            if range_val == 0:
                normalized.append(128)
            else:
                scaled = int((value - min_vals[dim]) / range_val * 256)
                normalized.append(int(np.clip(scaled, 0, 256)))
        action_str = ", ".join(str(x) for x in normalized)
        timestep_strs.append(f"Step {step_idx}: [{action_str}]")
    return "; ".join(timestep_strs)


def save_image(image: np.ndarray, image_dir: pathlib.Path, filename: str) -> str:
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(image_dir / filename)
    return filename


def convert_to_dynamics_format(
    episode: list[dict[str, Any]],
    normalizer: dict[str, np.ndarray],
    chunk_size: int,
    episode_id: int,
    image_dir: pathlib.Path,
    task_name: str,
    task_prompt: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    trajectory_len = len(episode)

    for start_frame in range(trajectory_len - chunk_size):
        target_frame = start_frame + chunk_size
        action_chunk = np.stack(
            [episode[start_frame + offset]["action"] for offset in range(chunk_size)],
            axis=0,
        )
        formatted_action = format_action(action_chunk, normalizer)

        current_head_filename = f"episode_{episode_id:06d}_frame_{start_frame:06d}_head.jpg"
        current_wrist_filename = f"episode_{episode_id:06d}_frame_{start_frame:06d}_wrist.jpg"
        next_head_filename = f"episode_{episode_id:06d}_frame_{target_frame:06d}_head.jpg"
        next_wrist_filename = f"episode_{episode_id:06d}_frame_{target_frame:06d}_wrist.jpg"

        save_image(episode[start_frame]["image"], image_dir, current_head_filename)
        save_image(episode[start_frame]["wrist_image"], image_dir, current_wrist_filename)
        save_image(episode[target_frame]["image"], image_dir, next_head_filename)
        save_image(episode[target_frame]["wrist_image"], image_dir, next_wrist_filename)

        sample_id = len(entries) // 2
        entries.append(
            {
                "id": episode_id * 25600 + sample_id * 2,
                "episode_id": episode_id,
                "task_name": task_name,
                "task_prompt": task_prompt,
                "images": [
                    [current_head_filename, current_wrist_filename],
                    [next_head_filename],
                ],
                "action_sequence": [
                    formatted_action
                    + ". Predict next head camera view according to the current observation and action."
                ],
                "start_frame": start_frame,
                "end_frame": target_frame,
                "action_chunk_size": chunk_size,
                "prediction_type": "head_camera",
            }
        )
        entries.append(
            {
                "id": episode_id * 25600 + sample_id * 2 + 1,
                "episode_id": episode_id,
                "task_name": task_name,
                "task_prompt": task_prompt,
                "images": [
                    [next_head_filename, current_wrist_filename],
                    [next_wrist_filename],
                ],
                "action_sequence": [
                    "Predict current wrist camera view according to history wrist camera view and current head camera view."
                ],
                "start_frame": start_frame,
                "end_frame": target_frame,
                "action_chunk_size": chunk_size,
                "prediction_type": "wrist_camera",
            }
        )

    return entries


def convert_to_vlm_reward_format(
    episode: list[dict[str, Any]],
    task_name: str,
    task_prompt: str,
    is_success: bool,
    episode_id: int,
    image_dir: pathlib.Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    trajectory_len = len(episode)

    for frame_idx, item in enumerate(episode):
        img_filename = f"episode_{episode_id:06d}_frame_{frame_idx:06d}.jpg"
        save_image(item["image"], image_dir, img_filename)

        answer = "Yes." if is_success and frame_idx == trajectory_len - 1 else "No."
        question_text = (
            f"Determine whether the task: {task_prompt} is successfully completed, "
            "answer with Yes or No"
        )
        entries.append(
            {
                "id": episode_id * 2560 + frame_idx,
                "episode_id": episode_id,
                "frame": frame_idx,
                "task_name": task_name,
                "task_prompt": task_prompt,
                "image": [img_filename],
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n<prompt>\n{question_text}",
                    },
                    {"from": "gpt", "value": answer},
                ],
            }
        )

    return entries


def balance_vlm_labels(entries: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    yes_samples = []
    no_samples = []
    for entry in tqdm(entries, desc="Splitting VLM labels", unit="sample"):
        if entry["conversations"][1]["value"] == "Yes.":
            yes_samples.append(entry)
        else:
            no_samples.append(entry)
    print(f"VLM labels before balancing: {len(yes_samples)} Yes, {len(no_samples)} No")

    if not yes_samples or not no_samples or len(yes_samples) == len(no_samples):
        balanced = list(entries)
    else:
        rng = np.random.default_rng(seed)
        if len(no_samples) > len(yes_samples):
            sampled = rng.choice(yes_samples, size=len(no_samples), replace=True).tolist()
            balanced = sampled + no_samples
        else:
            sampled = rng.choice(no_samples, size=len(yes_samples), replace=True).tolist()
            balanced = yes_samples + sampled

    print("Sorting balanced VLM labels...")
    balanced.sort(key=lambda item: (item["episode_id"], item["frame"]))
    counts = Counter(entry["conversations"][1]["value"] for entry in balanced)
    print(f"VLM labels after balancing: {counts.get('Yes.', 0)} Yes, {counts.get('No.', 0)} No")
    return balanced


def visualize_action_distribution(
    normalizer: dict[str, np.ndarray], output_dir: pathlib.Path, action_dim: int
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skip action distribution plots")
        return

    raw_actions = normalizer["raw_actions"]
    clipped_actions = normalizer["clipped_actions"]
    clip_min = normalizer["clip_min"]
    clip_max = normalizer["clip_max"]
    min_vals = normalizer["min"]
    max_vals = normalizer["max"]

    normalized_actions = np.zeros_like(clipped_actions, dtype=np.float64)
    for dim in range(action_dim):
        range_val = max_vals[dim] - min_vals[dim]
        if range_val == 0:
            normalized_actions[:, dim] = 128
        else:
            normalized_actions[:, dim] = (clipped_actions[:, dim] - min_vals[dim]) / range_val * 256
            normalized_actions[:, dim] = np.clip(normalized_actions[:, dim], 0, 256)

    n_cols = min(7, action_dim)
    n_rows = (action_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    fig.suptitle("Action Distribution (Raw with Clip Bounds)", fontsize=16, fontweight="bold")
    for dim in range(action_dim):
        ax = axes_array[dim]
        ax.hist(raw_actions[:, dim], bins=50, alpha=0.7, color="skyblue", edgecolor="black", linewidth=0.5)
        ax.axvline(clip_min[dim], color="red", linestyle="--", linewidth=2, label="clip bound")
        ax.axvline(clip_max[dim], color="red", linestyle="--", linewidth=2)
        ax.set_title(f"Dim {dim}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Original Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.legend(fontsize=7, loc="upper right")
    for idx in range(action_dim, len(axes_array)):
        axes_array[idx].axis("off")
    plt.tight_layout()
    output_path = output_dir / "action_distribution_raw_with_clip_bounds.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes_array = np.atleast_1d(axes).reshape(-1)
    fig.suptitle("Normalized Action Distribution [0-256]", fontsize=16, fontweight="bold")
    for dim in range(action_dim):
        ax = axes_array[dim]
        ax.hist(normalized_actions[:, dim], bins=50, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xlim([0, 256])
        ax.set_title(f"Dim {dim}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Normalized Value", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    for idx in range(action_dim, len(axes_array)):
        axes_array[idx].axis("off")
    plt.tight_layout()
    output_path = output_dir / "action_distribution_normalized.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_dynamics_prompt_file(output_dir: pathlib.Path) -> pathlib.Path:
    prompt_file = output_dir / "dynamics_prompt.txt"
    prompt_content = """You are now acting as a **world model** that simulates robot manipulation task execution.
Your task is to predict the **next frame of visual observation**, given the following inputs:
- **Multiple current observation images** from the robot's cameras (head camera and wrist camera)
- An **action sequence** describing the manipulation to execute
- Optionally, the **next frame from the head camera** (for predicting wrist camera views)

You will receive images from different camera viewpoints and need to predict the next frame according to the provided action sequence and instruction."""
    prompt_file.write_text(prompt_content, encoding="utf-8")
    return prompt_file


def create_vlm_reward_prompt_file(output_dir: pathlib.Path) -> pathlib.Path:
    prompt_file = output_dir / "vlm_reward_prompt.txt"
    prompt_content = """You are a vision-language model with advanced reasoning abilities.
Your task is to carefully observe the image and determine whether the task is successfully completed.

### Environment description:
- You are observing a robot workspace with manipulation capabilities
- The environment is from the LIBERO dataset, containing simulated manipulation tasks
- The robot can manipulate objects in the scene
- Common tasks include: picking, placing, arranging objects, etc.

### Task:
Given an image and a task description, determine whether the task has been successfully completed.

### Response format:
- Answer with "Yes." if the task is successfully completed
- Answer with "No." if the task is not yet completed or failed

### Guidelines:
- Carefully examine the state of objects in the scene
- Check if the goal state matches the task description
- Consider the spatial arrangement and object states
- Be precise in your judgment

**Your response must be either "Yes." or "No." without additional explanation.**"""
    prompt_file.write_text(prompt_content, encoding="utf-8")
    return prompt_file


def write_action_normalizer(normalizer: dict[str, np.ndarray], output_dir: pathlib.Path) -> pathlib.Path:
    normalizer_path = output_dir / "action_normalizer.json"
    serializable = {
        "min": normalizer["min"].tolist(),
        "max": normalizer["max"].tolist(),
        "clip_min": normalizer["clip_min"].tolist(),
        "clip_max": normalizer["clip_max"].tolist(),
    }
    normalizer_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return normalizer_path


def discover_files(input_dir: pathlib.Path, max_files: int | None) -> list[pathlib.Path]:
    files = sorted(input_dir.glob("*.npy"), key=parse_rollout_name)
    if max_files is not None:
        files = files[:max_files]
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LIBERO rollout .npy files into Bagel dynamics and VLM reward data."
    )
    parser.add_argument("--input-dir", "--input_dir", type=pathlib.Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", "--output_dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task-name", "--task_name", default="libero")
    parser.add_argument("--action-chunk-size", "--action_chunk_size", type=int, default=10)
    parser.add_argument("--percentile-clip", "--percentile_clip", type=float, default=100)
    parser.add_argument("--max-files", "--max_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--success-only", "--success_only", action="store_true")
    parser.set_defaults(balance_labels=True)
    parser.add_argument("--balance-labels", "--balance_labels", dest="balance_labels", action="store_true")
    parser.add_argument("--no-balance-labels", "--no_balance_labels", dest="balance_labels", action="store_false")
    args = parser.parse_args()

    if args.action_chunk_size <= 0:
        raise ValueError("--action-chunk-size must be positive")
    if not (50.0 < args.percentile_clip <= 100.0):
        raise ValueError("--percentile-clip must be in (50, 100]")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("--max-files must be positive")
    return args


def main() -> None:
    args = parse_args()
    input_dir: pathlib.Path = args.input_dir
    output_dir: pathlib.Path = args.output_dir
    dynamics_images_dir = output_dir / "dynamics_images"
    vlm_images_dir = output_dir / "vlm_images"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = discover_files(input_dir, args.max_files)
    if args.success_only:
        files = [path for path in files if is_success_file(path)]
    if not files:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dynamics_images_dir.mkdir(parents=True, exist_ok=True)
    vlm_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} trajectories")
    action_chunks: list[np.ndarray] = []
    skipped = 0
    action_dim: int | None = None
    for path in tqdm(files, desc="Loading actions", unit="traj"):
        try:
            actions, _ = load_actions(path)
        except Exception as exc:
            skipped += 1
            tqdm.write(f"Skip {path}: {exc}")
            continue
        if action_dim is None:
            action_dim = actions.shape[1]
        elif actions.shape[1] != action_dim:
            raise ValueError(f"{path}: action dim {actions.shape[1]} != {action_dim}")
        action_chunks.append(actions)

    if not action_chunks or action_dim is None:
        raise RuntimeError("No valid trajectories after action loading")
    if skipped:
        print(f"Skipped {skipped} trajectories during action loading")

    normalizer = compute_action_normalizer(action_chunks, args.percentile_clip)
    normalizer_path = write_action_normalizer(normalizer, output_dir)
    if not args.no_plots:
        visualize_action_distribution(normalizer, output_dir, action_dim)

    dynamics_jsonl_path = output_dir / "libero_dynamics.jsonl"
    vlm_jsonl_path = output_dir / "libero_vlm_reward.jsonl"
    vlm_entries_all: list[dict[str, Any]] = []
    dynamics_count = 0
    head_count = 0
    wrist_count = 0
    converted_episodes = 0

    with dynamics_jsonl_path.open("w", encoding="utf-8") as dynamics_file:
        for path in tqdm(files, desc="Converting trajectories", unit="traj"):
            try:
                episode = load_episode(path)
            except Exception as exc:
                tqdm.write(f"Skip {path}: {exc}")
                continue
            if len(episode) <= args.action_chunk_size:
                tqdm.write(
                    f"Skip {path}: trajectory length {len(episode)} <= chunk size {args.action_chunk_size}"
                )
                continue

            task_prompt = episode[0]["prompt"]
            is_success = is_success_file(path)
            episode_id = converted_episodes

            dynamics_entries = convert_to_dynamics_format(
                episode=episode,
                normalizer=normalizer,
                chunk_size=args.action_chunk_size,
                episode_id=episode_id,
                image_dir=dynamics_images_dir,
                task_name=args.task_name,
                task_prompt=task_prompt,
            )
            for entry in dynamics_entries:
                dynamics_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            dynamics_count += len(dynamics_entries)
            head_count += sum(1 for entry in dynamics_entries if entry["prediction_type"] == "head_camera")
            wrist_count += sum(1 for entry in dynamics_entries if entry["prediction_type"] == "wrist_camera")

            vlm_entries_all.extend(
                convert_to_vlm_reward_format(
                    episode=episode,
                    task_name=args.task_name,
                    task_prompt=task_prompt,
                    is_success=is_success,
                    episode_id=episode_id,
                    image_dir=vlm_images_dir,
                )
            )

            converted_episodes += 1

    if args.balance_labels:
        vlm_entries_all = balance_vlm_labels(vlm_entries_all, args.seed)

    with vlm_jsonl_path.open("w", encoding="utf-8") as vlm_file:
        for entry in tqdm(vlm_entries_all, desc="Writing VLM JSONL", unit="sample"):
            vlm_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    dynamics_prompt_path = create_dynamics_prompt_file(output_dir)
    vlm_prompt_path = create_vlm_reward_prompt_file(output_dir)
    vlm_counts = Counter(entry["conversations"][1]["value"] for entry in vlm_entries_all)

    print("\nConversion complete")
    print(f"  episodes: {converted_episodes}")
    print(f"  dynamics: {dynamics_jsonl_path} ({dynamics_count} records)")
    print(f"    head_camera: {head_count}")
    print(f"    wrist_camera: {wrist_count}")
    print(f"  vlm reward: {vlm_jsonl_path} ({len(vlm_entries_all)} records)")
    print(f"    Yes.: {vlm_counts.get('Yes.', 0)}")
    print(f"    No.: {vlm_counts.get('No.', 0)}")
    print(f"  action normalizer: {normalizer_path}")
    print(f"  dynamics prompt: {dynamics_prompt_path}")
    print(f"  vlm reward prompt: {vlm_prompt_path}")


if __name__ == "__main__":
    main()
