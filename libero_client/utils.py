"""
LIBERO Utility Functions

This module provides utility functions for LIBERO environments.
"""

import os
from typing import Dict, List, Any
import numpy as np
import torch
import cv2


def get_libero_image(obs: Dict) -> np.ndarray:
    """Extract agentview image from LIBERO observation.
    
    Args:
        obs: Observation dictionary from LIBERO environment
        
    Returns:
        RGB image as numpy array
    """
    if "agentview_image" in obs:
        return obs["agentview_image"]
    elif "image" in obs:
        return obs["image"]
    else:
        raise KeyError("No image found in observation")


def get_libero_wrist_image(obs: Dict) -> np.ndarray:
    """Extract wrist camera image from LIBERO observation.
    
    Args:
        obs: Observation dictionary from LIBERO environment
        
    Returns:
        RGB image as numpy array
    """
    if "robot0_eye_in_hand_image" in obs:
        return obs["robot0_eye_in_hand_image"]
    elif "wrist_image" in obs:
        return obs["wrist_image"]
    else:
        raise KeyError("No wrist image found in observation")


def list_of_dict_to_dict_of_list(list_of_dict: List[Dict]) -> Dict:
    """Convert list of dictionaries to dictionary of lists.
    
    Args:
        list_of_dict: List of dictionaries with same keys
        
    Returns:
        Dictionary where each key maps to a list of values
    """
    if not list_of_dict:
        return {}
    
    keys = list_of_dict[0].keys()
    dict_of_list = {key: [] for key in keys}
    
    for d in list_of_dict:
        for key in keys:
            dict_of_list[key].append(d[key])
    
    # Convert lists to numpy arrays for numeric data
    for key in dict_of_list:
        values = dict_of_list[key]
        if values and isinstance(values[0], (int, float, np.ndarray)):
            dict_of_list[key] = np.array(values)
    
    return dict_of_list


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation.
    
    Args:
        quat: Quaternion as [w, x, y, z] or [x, y, z, w]
        
    Returns:
        Axis-angle representation as 3D vector
    """
    # Assume quaternion format is [w, x, y, z]
    if len(quat) != 4:
        raise ValueError(f"Quaternion must have 4 elements, got {len(quat)}")
    
    w, x, y, z = quat
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-8:
        return np.zeros(3)
    
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to axis-angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    
    if angle < 1e-8:
        return np.zeros(3)
    
    sin_half_angle = np.sin(angle / 2)
    if abs(sin_half_angle) < 1e-8:
        return np.zeros(3)
    
    axis = np.array([x, y, z]) / sin_half_angle
    axis_angle = axis * angle
    
    return axis_angle


def to_tensor(data: Any) -> Any:
    """Convert numpy arrays to PyTorch tensors recursively.
    
    Args:
        data: Data to convert (can be dict, list, numpy array, or primitive)
        
    Returns:
        Data with numpy arrays converted to tensors
    """
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, dict):
        return {key: to_tensor(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_tensor(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_tensor(item) for item in data)
    else:
        return data


def put_info_on_image(image: np.ndarray, info: Dict) -> np.ndarray:
    """Put text information on image.
    
    Args:
        image: RGB image as numpy array
        info: Dictionary of information to display
        
    Returns:
        Image with text overlay
    """
    img = image.copy()
    
    # Convert to uint8 if needed
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Add text
    y_offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 255, 255)
    thickness = 1
    
    for key, value in info.items():
        if isinstance(value, (int, float, bool)):
            text = f"{key}: {value}"
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                text = f"{key}: {value.item()}"
            else:
                continue
        elif isinstance(value, str):
            text = f"{key}: {value}"
        else:
            text = f"{key}: {str(value)}"
        
        cv2.putText(img, text, (5, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += 15
    
    return img


def tile_images(images: List[np.ndarray], nrows: int = None, ncols: int = None) -> np.ndarray:
    """Tile a list of images into a grid.
    
    Args:
        images: List of images (all same size)
        nrows: Number of rows (if None, computed from ncols or sqrt)
        ncols: Number of columns (if None, computed from nrows or sqrt)
        
    Returns:
        Tiled image grid
    """
    if not images:
        raise ValueError("Image list is empty")
    
    n_images = len(images)
    
    # Determine grid size
    if nrows is None and ncols is None:
        nrows = int(np.ceil(np.sqrt(n_images)))
        ncols = int(np.ceil(n_images / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n_images / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n_images / nrows))
    
    # Get image dimensions
    img_h, img_w = images[0].shape[:2]
    n_channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    # Create canvas
    if n_channels == 1:
        canvas = np.zeros((nrows * img_h, ncols * img_w), dtype=images[0].dtype)
    else:
        canvas = np.zeros((nrows * img_h, ncols * img_w, n_channels), dtype=images[0].dtype)
    
    # Place images
    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        
        y_start = row * img_h
        y_end = y_start + img_h
        x_start = col * img_w
        x_end = x_start + img_w
        
        canvas[y_start:y_end, x_start:x_end] = img
    
    return canvas


def save_rollout_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str = "rollout",
    fps: int = 10,
) -> None:
    """Save a list of images as a video file.
    
    Args:
        images: List of RGB images
        output_dir: Output directory path
        video_name: Video file name (without extension)
        fps: Frames per second
    """
    if not images:
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare video writer
    output_path = os.path.join(output_dir, f"{video_name}.mp4")
    height, width = images[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for img in images:
        # Convert RGB to BGR for OpenCV
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img
        
        video_writer.write(img_bgr)
    
    video_writer.release()
    print(f"Video saved to {output_path}")

