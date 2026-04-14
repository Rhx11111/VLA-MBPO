#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of the World Model Client.

This script demonstrates how to use the WebsocketWorldModelClient to perform
remote inference with a World Model server for both:
1. Editing (world model): multi-view images + action -> next frame images
2. Understanding (reward model): image + prompt -> text
"""

import numpy as np
import logging
from PIL import Image
from websocket_world_model_client import WebsocketWorldModelClient


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create World Model client
    # Make sure the server is running before creating the client
    world_model_client = WebsocketWorldModelClient(
        host="0.0.0.0",  # Server host
        port=8002        # Server port (default for World Model)
    )
    
    print("Connected to World Model server!")
    print("Server metadata:", world_model_client.get_server_metadata())
    
    # Example 1: Multi-view Image Editing (World Model Mode)
    print("\n=== Example 1: Multi-view Image Editing (World Model) ===")
    
    # Create or load multi-view images (head camera + wrist camera)
    dummy_head_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_wrist_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create a dummy action (e.g., 7-dimensional action for LIBERO)
    dummy_action = np.random.randn(7).astype(np.float32)
    
    print(f"Input head image shape: {dummy_head_image.shape}")
    print(f"Input wrist image shape: {dummy_wrist_image.shape}")
    print(f"Action shape: {dummy_action.shape}")
    
    # Perform editing (two-stage generation)
    # Stage 1: Generate next head from [head, wrist] + action
    # Stage 2: Generate next wrist from [next_head, wrist]
    edit_result = world_model_client.edit(
        head_image=dummy_head_image,
        wrist_image=dummy_wrist_image,
        action=dummy_action,
        # Optional: override default hyperparameters
        # cfg_text_scale=5.0,
        # cfg_img_scale=2.0,
        # num_timesteps=50,
    )
    
    next_head = edit_result['next_head']
    next_wrist = edit_result['next_wrist']
    print(f"Generated next head image shape: {next_head.shape}")
    print(f"Generated next wrist image shape: {next_wrist.shape}")
    
    # Example 2: Image Understanding (Reward Model Mode)
    print("\n=== Example 2: Image Understanding (Reward Model) ===")
    
    # Use head camera image for understanding
    task_prompt = "pick up the red block and place it on the blue block"
    
    print(f"Task prompt: {task_prompt}")
    
    # Perform understanding
    understand_result = world_model_client.understand(
        image=dummy_head_image,
        task=[task_prompt],
        # Optional: override default hyperparameters
        # max_think_token_n=500,
        # temperature=0.3,
    )
    
    reward = understand_result['reward']
    print(f"Task success (reward): {reward}")
    
    # Example 3: Sequential multi-view editing (simulating world model rollout)
    print("\n=== Example 3: Sequential Multi-view Editing (World Model Rollout) ===")
    
    current_head = dummy_head_image.copy()
    current_wrist = dummy_wrist_image.copy()
    
    # Simulate a sequence of actions
    num_steps = 5
    predicted_heads = [current_head]
    predicted_wrists = [current_wrist]
    
    for i in range(num_steps):
        # Generate random action
        action = np.random.randn(7).astype(np.float32)
        
        print(f"\nStep {i+1}: Executing action...")
        result = world_model_client.edit(
            head_image=current_head,
            wrist_image=current_wrist,
            action=action
        )
        
        current_head = result['next_head']
        current_wrist = result['next_wrist']
        predicted_heads.append(current_head)
        predicted_wrists.append(current_wrist)
        
        print(f"  Generated head frame shape: {current_head.shape}")
        print(f"  Generated wrist frame shape: {current_wrist.shape}")
    
    print(f"\nTotal predicted frames: {len(predicted_heads)} (head), {len(predicted_wrists)} (wrist)")
    
    # Example 4: Batch processing
    print("\n=== Example 4: Batch Processing ===")
    
    batch_size = 4
    batch_head_images = np.random.randint(0, 255, (batch_size, 480, 640, 3), dtype=np.uint8)
    batch_wrist_images = np.random.randint(0, 255, (batch_size, 480, 640, 3), dtype=np.uint8)
    batch_actions = np.random.randn(batch_size, 7).astype(np.float32)
    
    print(f"Batch head images shape: {batch_head_images.shape}")
    print(f"Batch wrist images shape: {batch_wrist_images.shape}")
    print(f"Batch actions shape: {batch_actions.shape}")
    
    # Process batch (internally distributed to multiple workers)
    batch_result = world_model_client.edit(
        head_image=batch_head_images,
        wrist_image=batch_wrist_images,
        action=batch_actions
    )
    
    batch_next_heads = batch_result['next_head']
    batch_next_wrists = batch_result['next_wrist']
    print(f"Batch next heads shape: {batch_next_heads.shape}")
    print(f"Batch next wrists shape: {batch_next_wrists.shape}")
    
    # Example 5: Reset the model
    print("\n=== Example 5: Reset Model ===")
    world_model_client.reset()
    print("Model reset successfully!")
    
    print("\nWorld Model Client example completed successfully!")


if __name__ == "__main__":
    main()

