# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

# from .interleave_datasets import UnifiedEditIterableDataset
# from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .dynamics_dataset import DynamicsJSONLIterableDataset


_LEGACY_BAGEL_DATA_ROOT = Path("/data/home/scwb314/run/data/bagel_data")
_BAGEL_DATA_ROOT = Path(os.environ.get("BAGEL_DATA_ROOT", _LEGACY_BAGEL_DATA_ROOT))


def _bagel_data_path(*parts: str) -> str:
    return str(_BAGEL_DATA_ROOT.joinpath(*parts))


DATASET_REGISTRY = {
    'vlm_sft': SftJSONLIterableDataset,
    'dynamics_sft': DynamicsJSONLIterableDataset
}


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
        },
    },
    'vlm_sft': {
        'libero_spatial_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_spatial_with_wrist', 'vlm_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_spatial_with_wrist', 'libero_vlm_reward.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_spatial_with_wrist', 'vlm_reward_prompt.txt'),
			'num_total_samples': 253692
        },
        'libero_object_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_object_with_wrist', 'vlm_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_object_with_wrist', 'libero_vlm_reward.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_object_with_wrist', 'vlm_reward_prompt.txt'),
			'num_total_samples': 293664
        },
        'libero_goal_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_goal_with_wrist', 'vlm_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_goal_with_wrist', 'libero_vlm_reward.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_goal_with_wrist', 'vlm_reward_prompt.txt'),
			'num_total_samples': 260780
        },
        'libero_long_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_long_with_wrist', 'vlm_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_long_with_wrist', 'libero_vlm_reward.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_long_with_wrist', 'vlm_reward_prompt.txt'),
			'num_total_samples': 750210
        },
    },
    'dynamics_sft': {
        'libero_spatial_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_spatial_with_wrist', 'dynamics_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_spatial_with_wrist', 'libero_dynamics.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_spatial_with_wrist', 'dynamics_prompt.txt'),
			'num_total_samples': 235348
        },
        'libero_object_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_object_with_wrist', 'dynamics_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_object_with_wrist', 'libero_dynamics.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_object_with_wrist', 'dynamics_prompt.txt'),
			'num_total_samples': 275544
        },
        'libero_goal_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_goal_with_wrist', 'dynamics_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_goal_with_wrist', 'libero_dynamics.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_goal_with_wrist', 'dynamics_prompt.txt'),
			'num_total_samples': 242530
        },
        'libero_long_with_wrist': {
            'data_dir': _bagel_data_path('dynamics', 'libero_long_with_wrist', 'dynamics_images'),
			'jsonl_path': _bagel_data_path('dynamics', 'libero_long_with_wrist', 'libero_dynamics.jsonl'),
            'prompt_path': _bagel_data_path('dynamics', 'libero_long_with_wrist', 'dynamics_prompt.txt'),
			'num_total_samples': 731382
        },
    },
}
