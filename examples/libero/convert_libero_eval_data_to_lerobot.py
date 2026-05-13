import shutil
import glob
import os
from collections import defaultdict
import numpy as np
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import json


HF_LEROBOT_HOME = "/data/home/scwb314/run/data/openpi_dataset"
REPO_NAME = "libero_spatial_chunk_with_wrist" 


def parse_filename(filename):
    basename = os.path.basename(filename)
    name_without_ext = basename.replace('.npy', '')
    
    if name_without_ext.endswith('_success'):
        suffix = 'success'
        name_without_suffix = name_without_ext[:-8]
    elif name_without_ext.endswith('_failure'):
        suffix = 'failure'
        name_without_suffix = name_without_ext[:-8]
    else:
        raise ValueError(f"Unknown filename: {basename}")
    
    parts = name_without_suffix.split('_')
    if len(parts) < 3:
        return None
    
    episode_id = int(parts[-1])
    task_id = int(parts[-2])
    dataset = '_'.join(parts[:-2])
    
    return {
        'dataset': dataset,
        'task_id': task_id,
        'episode_id': episode_id,
        'suffix': suffix 
    }


def create_dataset(repo_id, max_sim_state_dim=150):
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "sim_state": {
                "dtype": "float32",
                "shape": (max_sim_state_dim,),
                "names": ["sim_state"],
            },
            "sim_state_len": {
                "dtype": "int32",
                "shape": (1,), 
                "names": ["sim_state_len"],
            },
            "task_id": {
                "dtype": "int32",
                "shape": (1,),
                "names": ["task_id"],
            },
            "time_stamp": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["time_stamp"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    

def process_episode_to_dataset(episode_data, file_path, dataset, task_id, max_sim_state_dim=150):
    if len(episode_data) == 0:
        return
    
    task_description = episode_data[0]['prompt']
    
    for (idx, frame_data) in enumerate(episode_data):
        action = np.array(frame_data['action'], dtype=np.float32)
        # Convert first image for evaluation
        # if idx != 0: 
        #     continue
        if idx % 10 != 0:
            continue
        
        sim_state = frame_data['sim_state'].astype(np.float32)
        sim_state_len = np.array(sim_state.shape[0]).reshape(1,).astype(np.int32)
        if sim_state.shape[0] < max_sim_state_dim:
            padded_sim_state = np.zeros(max_sim_state_dim, dtype=np.float32)
            padded_sim_state[:sim_state.shape[0]] = sim_state
            sim_state = padded_sim_state
        
        dataset.add_frame({
            "image": frame_data['image'],
            "wrist_image": frame_data['wrist_image'],
            "state": np.asarray(frame_data['state']).reshape(8,).astype(np.float32),
            "actions": action.astype(np.float32),
            "sim_state": sim_state,
            "sim_state_len": sim_state_len,
            "time_stamp": np.array(idx).reshape(1,).astype(np.float32),
            "task": task_description,
            "task_id": np.array(task_id).reshape(1,).astype(np.int32),
        })
    dataset.save_episode()


def main(
    data_dir: str = "path/to/data",
    task_id: int = -1,
    *,
    push_to_hub: bool = False,
    max_episodes_per_task: int = 1000,
    output_repo_name: str = REPO_NAME,
):
    
    data_files = glob.glob(os.path.join(data_dir, "*.npy"))

    episodes = {}
    task_count = {}
    for file_path in data_files:
        info = parse_filename(file_path)
        if info:
            if info['task_id'] not in task_count:
                task_count[info['task_id']] = 0
            if info['episode_id'] >= max_episodes_per_task:
                continue
            task_count[info['task_id']] += 1
            episode_key = (info['dataset'], info['task_id'], info['episode_id'])
            episodes[episode_key] = {
                'file_path': file_path,
                'suffix': info['suffix'],
                'dataset': info['dataset'],
                'task_id': info['task_id'],
                'episode_id': info['episode_id']
            }
    
    output_path = Path(f"/data/home/scwb314/run/data/openpi_dataset/{output_repo_name}")
    if output_path.exists():
        shutil.rmtree(output_path)
    dataset = create_dataset(output_repo_name, max_sim_state_dim=150)
    
    success_episodes = []
    failure_episodes = []
    for episode_key in sorted(episodes.keys()):
        ep = episodes[episode_key]
        if ep['task_id'] != task_id and task_id > 0:
            continue
        if ep['suffix'] == 'success':
            success_episodes.append(ep)
        else:
            failure_episodes.append(ep)
    
    episodes_to_process = success_episodes + failure_episodes
    
    for idx, ep in enumerate(episodes_to_process):
        episode_data = np.load(ep['file_path'], allow_pickle=True)
        if isinstance(episode_data, np.ndarray):
            episode_data = episode_data.tolist()
        if not isinstance(episode_data, list):
            continue
        
        process_episode_to_dataset(episode_data, ep['file_path'], dataset, ep['task_id'], 150)
        
    
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "eval", "sim_state"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
