# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Dict, Optional
import numpy as np
from PIL import Image


class BaseWorldModel(abc.ABC):
    """
    Base interface for WorldModel model inference.
    Supports two modes:
    1. Editing (world model): multi-view images + action -> next frame images
    2. Understanding (reward model): image + prompt -> text reward
    
    The editing mode uses a two-stage generation process:
    - Stage 1: Generate next head camera view from current head + wrist + action
    - Stage 2: Generate next wrist camera view from generated head + current wrist
    """
    
    @abc.abstractmethod
    def edit(self, head_image: np.ndarray, wrist_image: np.ndarray, action: np.ndarray, **kwargs) -> Dict:
        """
        Edit/generate next frame images based on current multi-view images and action (world model).
        
        Uses a two-stage generation process:
        1. Generate next head camera view: [current_head, current_wrist] + action -> next_head
        2. Generate next wrist camera view: [next_head, current_wrist] -> next_wrist
        
        Args:
            head_image: Current head camera image, shape (H, W, 3), dtype uint8, RGB format
            wrist_image: Current wrist camera image, shape (H, W, 3), dtype uint8, RGB format
            action: Action array, continuous values that will be normalized
            **kwargs: Additional inference parameters
            
        Returns:
            Dict containing:
                - 'next_head': Next predicted head camera image, shape (H, W, 3), dtype uint8
                - 'next_wrist': Next predicted wrist camera image, shape (H, W, 3), dtype uint8
        """
        pass
    
    @abc.abstractmethod
    def understand(self, image: np.ndarray, prompt: str, **kwargs) -> Dict:
        """
        Understand image and answer question (reward model).
        
        Args:
            image: Input image, shape (H, W, 3), dtype uint8, RGB format
            prompt: Question/instruction text
            **kwargs: Additional inference parameters
            
        Returns:
            Dict containing:
                - 'text': Understanding/reward text output
        """
        pass
    
    def reset(self) -> None:
        """Reset the model to its initial state."""
        pass

