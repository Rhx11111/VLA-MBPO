#!/usr/bin/env python3
"""
Data conversion script: Convert LIBERO evaluation data into bagel_example format.

Original data: data/libero/eval_result/spatial_1000/data/*.npy
Target format 1: editing format (parquet)
Target format 2: vlm format (jsonl)
"""

import argparse
import json
import pathlib
import io
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import tqdm


def format_action(action: np.ndarray, normalizer: Dict[str, np.ndarray] = None) -> str:
    """
    Format an action array into a string.
    
    Args:
        action: Action array (can be a chunk of actions, flattened or not)
        normalizer: Dictionary with 'min' and 'max' arrays for normalization
                    Shape: (action_dim,) - min/max for each action dimension
    
    Returns:
        Formatted action string with normalized values (0-1000 integers)
    """
    action = np.array(action)
    
    # Normalize to 0-1000 if normalizer is provided
    if normalizer is not None:
        min_vals = normalizer['min']  # Shape: (action_dim,)
        max_vals = normalizer['max']  # Shape: (action_dim,)
        
        normalized = ((action - min_vals) / (max_vals - min_vals) * 1000).astype(int)
        # Clip to [0, 1000]
        normalized = np.clip(normalized, 0, 1000)
        
        # Flatten back for string formatting
        normalized = normalized.flatten()
        action_str = ", ".join([str(x) for x in normalized])
    else:
        action_str = ", ".join([f"{x:.4f}" for x in action])
    return f"[{action_str}]"


def convert_to_editing_format(
    data: List[Dict[str, Any]], 
    normalizer: Dict[str, np.ndarray] = None,
    chunk_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Convert to editing format using action chunks.
    
    Args:
        data: A list of raw data entries, each containing 'image', 'action' (single action per step),
              'prompt', etc.
        normalizer: Dictionary with 'min' and 'max' arrays for action normalization
        chunk_size: Number of actions to predict ahead (default: 10)
    
    Returns:
        A list of dictionaries, each containing 'image_list' and 'instruction_list'.
        Format: img[t], action[t:t+chunk_size] -> img[t+chunk_size]
    """
    editing_entries = []
    trajectory_len = len(data)
    
    if trajectory_len == 0:
        return editing_entries

    # Create chunks: img[t] + actions[t:t+chunk_size] -> img[t+chunk_size]
    for i in range(trajectory_len - chunk_size):
        image_list = []
        instruction_list = []
        
        # Collect action chunk from current timestep to t+chunk_size
        action_chunk = []
        for j in range(chunk_size):
            if i + j < trajectory_len:
                action_chunk.append(np.array(data[i + j]['action']))
        
        # Stack actions into chunk
        actions = np.array(action_chunk)  # Shape: (chunk_size, action_dim)
        
        # Add current image (convert to JPEG bytes)
        img = data[i]['image']
        img_array = np.array(img).astype(np.uint8)
        img_pil = Image.fromarray(img_array)
        
        # Encode as JPEG bytes
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        image_list.append(img_bytes)
        
        # Format action chunk with normalization

        a = format_action(actions, normalizer)

        instruction = f"""You are now acting as a **world model** that simulates robot manipulation task execution. Current action is {a}, Your task is to **predict the next image** that results from executing the given action from the current observation."""
        
        # Each element in instruction_list should be a list containing a single string
        instruction_list.append([instruction])
        
        # Add the target frame image (t+chunk_size timestep)
        target_idx = i + chunk_size
        img = data[target_idx]['image']
        img_array = np.array(img).astype(np.uint8)
        img_pil = Image.fromarray(img_array)
        
        # Encode as JPEG bytes
        img_buffer = io.BytesIO()
        img_pil.save(img_buffer, format='JPEG')
        img_bytes = img_buffer.getvalue()
        image_list.append(img_bytes)
        
        editing_entries.append({
            'image_list': np.array(image_list, dtype=object),
            'instruction_list': instruction_list
        })
    
    return editing_entries


def convert_to_vlm_format(
    data: List[Dict[str, Any]], 
    task_prompt: str, 
    is_success: bool,
    image_id: int,
    image_dir: pathlib.Path
) -> Dict[str, Any]:
    """
    Convert to VLM format.
    
    Args:
        data: Raw data list.
        task_prompt: Task description.
        is_success: Whether this is a successful trajectory.
        image_id: ID used to name images.
        image_dir: Directory where images are stored.
    
    Returns:
        A list of VLM-formatted dictionary entries (one per image).
    """
    vlm_entries = []
    
    for idx, item in enumerate(data):
        # Save image
        img = item['image']
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        else:
            img_pil = Image.fromarray(np.array(img))
        
        # Generate image filename
        img_filename = f"libero_spatial_{image_id}_{idx}.jpg"
        img_path = image_dir / img_filename
        img_pil.save(img_path)
        
        # Determine if this is the last frame
        is_last_frame = (idx >= len(data) - 5)
        is_not_last_frame = (idx < len(data) - 10)
        
        # Build human instruction
        human_instruction = f"<image>\nDetermine whether the task: {task_prompt} is successfully completed, answer with Yes or No"
        
        # Build model response
        if is_last_frame and is_success:
            gpt_response = "Yes."
        elif is_not_last_frame:
            gpt_response = "No."
        else:
            continue
        
        # Final VLM entry
        vlm_entry = {
            "id": image_id * 1000 + idx,  # ensure unique ID
            "image": img_filename,
            "conversations": [
                {"from": "human", "value": human_instruction},
                {"from": "gpt", "value": gpt_response}
            ]
        }
        
        vlm_entries.append(vlm_entry)
    
    return vlm_entries


def compute_action_normalizer(data_list: List[List[Dict[str, Any]]]) -> Dict[str, np.ndarray]:
    """
    Compute action normalizer (min and max) across all trajectories.
    
    Args:
        data_list: List of trajectories, each is a list of data entries
    
    Returns:
        Dictionary with 'min' and 'max' arrays for normalization
    """
    all_actions = []
    
    for data in data_list:
        for item in data:
            action = item['action']
            all_actions.append(np.array(action))
    
    # Stack all actions
    all_actions = np.array(all_actions)
    
    # Compute min and max for each dimension
    min_vals = np.min(all_actions, axis=0)
    max_vals = np.max(all_actions, axis=0)

    print(min_vals, max_vals)

    return {
        'min': min_vals,
        'max': max_vals
    }


def main():
    parser = argparse.ArgumentParser(description="Convert LIBERO data into bagel_example format.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/home/scwb314/run/data/libero_data",
        help="Input data directory"
    )
    parser.add_argument(
        "--output_editing_dir",
        type=str,
        default="/data/home/scwb314/run/data/bagel_data/editing/libero_spatial_slide",
        help="Output directory for editing format"
    )
    parser.add_argument(
        "--output_vlm_dir",
        type=str,
        default="/data/home/scwb314/run/data/bagel_data/vlm/libero_spatial_slide",
        help="Output directory for VLM format"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Parquet chunk size"
    )
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=10,
        help="Number of actions in each prediction chunk"
    )
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        help="Balance success/failure samples by undersampling the majority class"
    )
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_editing_dir = pathlib.Path(args.output_editing_dir)
    editing_dataset_name = args.output_editing_dir.split("/")[-1]
    output_vlm_dir = pathlib.Path(args.output_vlm_dir)
    vlm_dataset_name = args.output_vlm_dir.split("/")[-1]
    output_vlm_images_dir = output_vlm_dir / "images"
    
    # Create output directories
    output_editing_dir.mkdir(parents=True, exist_ok=True)
    output_vlm_dir.mkdir(parents=True, exist_ok=True)
    output_vlm_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Find .npy files
    npy_files = sorted(input_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} data files.")
    
    # First pass: Load all data to compute action normalizer
    print("Computing action normalizer...")
    all_data = []
    for npy_file in tqdm.tqdm(npy_files, desc="Loading data for normalization"):
        data = np.load(npy_file, allow_pickle=True)
        all_data.append(data)
    
    # Compute normalizer
    normalizer = compute_action_normalizer(all_data)
    print(f"Action normalizer computed: min shape={normalizer['min'].shape}, max shape={normalizer['max'].shape}")
    
    # Save normalizer
    normalizer_path = output_editing_dir / "action_normalizer.json"
    normalizer_dict = {
        'min': normalizer['min'].tolist(),
        'max': normalizer['max'].tolist()
    }
    with open(normalizer_path, 'w') as f:
        json.dump(normalizer_dict, f, indent=2)
    print(f"Saved action normalizer: {normalizer_path}")
    
    # Prepare editing data
    editing_data_list = []
    
    # Prepare VLM data
    vlm_entries_all = []
    global_image_id = 0
    
    # Process each file
    for npy_file, data in zip(tqdm.tqdm(npy_files, desc="Processing data files"), all_data):
        # Determine success
        is_success = "success" in npy_file.stem
        
        # Extract task prompt
        task_prompt = data[0]['prompt']
        
        # Convert to editing format (returns list of entries)
        editing_entries = convert_to_editing_format(data, normalizer, args.action_chunk_size)
        editing_data_list.extend(editing_entries)
        
        # Convert to VLM format
        vlm_entries = convert_to_vlm_format(
            data, 
            task_prompt, 
            is_success,
            global_image_id,
            output_vlm_images_dir
        )
        vlm_entries_all.extend(vlm_entries)
        
        global_image_id += 1
    
    # Save editing format as parquet
    print(f"\nSaving editing format data...")
    editing_df_list = []
    for entry in editing_data_list:
        editing_df_list.append({
            'image_list': entry['image_list'],
            'instruction_list': entry['instruction_list']
        })
    
    editing_df = pd.DataFrame(editing_df_list)
    
    # Save in chunks
    num_chunks = (len(editing_df) + args.chunk_size - 1) // args.chunk_size
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * args.chunk_size
        end_idx = min((chunk_idx + 1) * args.chunk_size, len(editing_df))
        chunk_df = editing_df.iloc[start_idx:end_idx]
        
        chunk_path = output_editing_dir / f"chunk_{chunk_idx}.parquet"
        chunk_df.to_parquet(chunk_path, index=False)
        print(f"Saved editing chunk {chunk_idx}: {chunk_path} ({len(chunk_df)} records)")
    
    # Balance VLM labels if requested (sample-level balancing)
    print(f"\nProcessing VLM format data...")
    if args.balance_labels:
        # Separate Yes and No samples
        yes_samples = [entry for entry in vlm_entries_all 
                      if entry['conversations'][1]['value'] == "Yes."]
        no_samples = [entry for entry in vlm_entries_all 
                     if entry['conversations'][1]['value'] == "No."]
        
        print(f"Original VLM samples: {len(yes_samples)} Yes, {len(no_samples)} No")
        
        # Balance by oversampling Yes samples
        if len(no_samples) > len(yes_samples):
            np.random.seed(42)
            # Keep all No samples and replicate Yes samples to match
            num_replications = len(no_samples) // len(yes_samples) // 5
            remainder = len(no_samples) % len(yes_samples)
            
            # Replicate Yes samples
            yes_samples_balanced = yes_samples * num_replications
            if remainder > 0:
                # Add random samples to match exactly
                additional_samples = list(np.random.choice(yes_samples, remainder, replace=True))
                yes_samples_balanced.extend(additional_samples)
            
            vlm_entries_all = yes_samples_balanced + no_samples
            print(f"After balancing: {len(yes_samples_balanced)} Yes, {len(no_samples)} No")
        else:
            vlm_entries_all = yes_samples + no_samples
            print(f"No balancing needed (Yes >= No)")
        
        vlm_entries_all = list(np.array(vlm_entries_all)[np.random.randint(0, len(vlm_entries_all), len(vlm_entries_all))]) 
    
    # Save VLM format as jsonl
    print(f"Saving VLM format data...")
    vlm_jsonl_path = output_vlm_dir / f"{vlm_dataset_name}.jsonl"
    with open(vlm_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in vlm_entries_all:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved VLM data: {vlm_jsonl_path} ({len(vlm_entries_all)} records)")
    
    # Create parquet_info.json
    # Use pyarrow metadata API to avoid reading nested data
    parquet_info = {}
    for chunk_idx in range(num_chunks):
        chunk_path = output_editing_dir / f"chunk_{chunk_idx}.parquet"
        parquet_file = pq.ParquetFile(chunk_path)
        metadata = parquet_file.metadata
        parquet_info[str(chunk_path)] = {
            "num_row_groups": metadata.num_row_groups,
            "num_rows": metadata.num_rows
        }
    
    parquet_info_path = output_editing_dir.parent / "parquet_info" / f"{editing_dataset_name}.json"
    parquet_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parquet_info_path, 'w') as f:
        json.dump(parquet_info, f, indent=2)
    
    # Count final VLM label distribution
    final_yes = sum(1 for entry in vlm_entries_all if entry['conversations'][1]['value'] == "Yes.")
    final_no = sum(1 for entry in vlm_entries_all if entry['conversations'][1]['value'] == "No.")
    
    print(f"\nConversion completed!")
    print(f"  - Editing format: {output_editing_dir} ({len(editing_df)} records)")
    print(f"  - VLM format: {vlm_jsonl_path} ({len(vlm_entries_all)} records)")
    print(f"  - Action normalizer: {normalizer_path}")
    print(f"\nSummary:")
    print(f"  - Action chunk size: {args.action_chunk_size}")
    print(f"  - Label balancing (VLM): {args.balance_labels}")
    print(f"  - Total trajectories: {len(npy_files)}")
    print(f"  - Total editing samples: {len(editing_df)}")
    print(f"  - Total VLM samples: {len(vlm_entries_all)} ({final_yes} Yes, {final_no} No)")
    print(f"  - Editing num_files: {num_chunks}")


if __name__ == "__main__":
    main()
