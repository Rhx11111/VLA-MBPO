# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os
from pickle import NEXT_BUFFER
from re import A
import sys
import json
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Dict, Optional, List
from datetime import datetime
import copy
import numpy as np
from PIL import Image
import torch
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
import time

# Add Bagel root to path
bagel_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bagel_root)

from data.transforms import ImageTransform
from data.data_utils import pil_img2rgb, add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.autoencoder import load_ae


def get_model_device(model):
    """获取模型的实际设备 - 对于使用 accelerate 加载的模型更可靠"""
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'embed_tokens'):
        device = model.language_model.model.embed_tokens.weight.device
        return device
    else:
        device = next(model.parameters()).device
        return device


def move_to_device(generation_input, device):
    """将 generation_input 字典中的所有张量移动到模型设备"""
    for k, v in generation_input.items():
        if isinstance(v, torch.Tensor):
            generation_input[k] = v.to(device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            generation_input[k] = [v_i.to(device) for v_i in v]
    return generation_input


def _world_model_worker(
    remote: Connection,
    parent_remote: Connection,
    model_config_path: str,
    model_weights_path: str,
    action_norm_path: str,
    max_mem_per_gpu: str,
    worker_id: int,
    gpu_id: int,
) -> None:
    """
    Worker 进程函数，运行单个 Bagel 模型实例。
    
    Args:
        remote: 子进程的连接对象
        parent_remote: 父进程的连接对象
        model_config_path: 模型配置文件路径（包含 llm_config.json, vit_config.json, ae.safetensors, tokenizer）
        model_weights_path: 模型权重文件路径（model.safetensors）
        action_norm_path: 动作归一化配置路径
        max_mem_per_gpu: 每个 GPU 的最大内存
        worker_id: Worker ID
        gpu_id: GPU ID
    """
    parent_remote.close()
    
    try:
        torch.cuda.set_device(gpu_id)
        
        logging.info(f"[Worker {worker_id}] Initializing on GPU {gpu_id}")
        
        # 加载 action normalizer
        with open(action_norm_path, 'r') as f:
            action_normalizer = json.load(f)
            action_normalizer['min'] = np.array(action_normalizer['min'])
            action_normalizer['max'] = np.array(action_normalizer['max'])
            # 加载 clip_min 和 clip_max（如果存在）
            if 'clip_min' in action_normalizer:
                action_normalizer['clip_min'] = np.array(action_normalizer['clip_min'])
            else:
                # 如果没有 clip_min/clip_max，使用 min/max 作为默认值
                action_normalizer['clip_min'] = action_normalizer['min'].copy()
            if 'clip_max' in action_normalizer:
                action_normalizer['clip_max'] = np.array(action_normalizer['clip_max'])
            else:
                action_normalizer['clip_max'] = action_normalizer['max'].copy()
        
        # 加载模型
        model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids = \
            _load_model_for_worker(model_config_path, model_weights_path, max_mem_per_gpu, gpu_id)
        
        logging.info(f"[Worker {worker_id}] Model loaded successfully on GPU {gpu_id}")
        remote.send(("ready", None))
        
        # 处理请求
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == "edit":
                    # 解包参数：现在接收 head_images, wrist_images, actions
                    head_images, wrist_images, actions, inference_hyper = data
                    
                    # 准备 action 字符串（使用与 convert_libero_data.py 中 format_action 相同的逻辑）
                    # 确保 actions 是 2D (chunk_size, action_dim)
                    actions_array = np.array(actions)
                    if actions_array.ndim == 1:
                        actions_array = actions_array.reshape(1, -1)
                    
                    min_vals = action_normalizer['min']
                    max_vals = action_normalizer['max']
                    clip_min = action_normalizer['clip_min']
                    clip_max = action_normalizer['clip_max']
                    
                    # 格式化每个时间步
                    action_str = []
                    for batch_idx in range(actions_array.shape[0]):
                        timestep_strs = []
                        # 假设每个 batch 的 action 是一个 chunk，需要按时间步格式化
                        # 如果 actions_array 是 (batch_size, action_dim)，则每个 batch 只有一个时间步
                        # 如果 actions_array 是 (batch_size, chunk_size, action_dim)，则需要遍历 chunk_size
                        if actions_array.ndim == 2:
                            # (batch_size, action_dim) - 每个样本只有一个时间步
                            step_action = actions_array[batch_idx].copy()
                            # 裁剪
                            step_action = np.clip(step_action, clip_min, clip_max)
                            # 归一化到 [0, 256]
                            action_dim = len(step_action)
                            normalized = np.zeros(action_dim, dtype=int)
                            for dim in range(action_dim):
                                range_val = max_vals[dim] - min_vals[dim]
                                if range_val == 0:
                                    normalized[dim] = 128
                                else:
                                    normalized[dim] = int((step_action[dim] - min_vals[dim]) / range_val * 256)
                                    normalized[dim] = np.clip(normalized[dim], 0, 256)
                            # 格式化为字符串
                            action_str_val = ", ".join([str(x) for x in normalized])
                            timestep_strs.append(f"Step 0: [{action_str_val}]")
                        else:
                            # (batch_size, chunk_size, action_dim) - 每个样本有多个时间步
                            chunk_size = actions_array.shape[1]
                            for step_idx in range(chunk_size):
                                step_action = actions_array[batch_idx, step_idx].copy()
                                # 裁剪
                                step_action = np.clip(step_action, clip_min, clip_max)
                                # 归一化到 [0, 256]
                                action_dim = len(step_action)
                                normalized = np.zeros(action_dim, dtype=int)
                                for dim in range(action_dim):
                                    range_val = max_vals[dim] - min_vals[dim]
                                    if range_val == 0:
                                        normalized[dim] = 128
                                    else:
                                        normalized[dim] = int((step_action[dim] - min_vals[dim]) / range_val * 256)
                                        normalized[dim] = np.clip(normalized[dim], 0, 256)
                                # 格式化为字符串
                                action_str_val = ", ".join([str(x) for x in normalized])
                                timestep_strs.append(f"Step {step_idx}: [{action_str_val}]")
                        action_str.append("; ".join(timestep_strs))
                    
                    # 转换为 PIL images
                    pil_head_images = [Image.fromarray(img) for img in head_images]
                    pil_wrist_images = [Image.fromarray(img) for img in wrist_images]
                    
                    # 提取保存图像的参数
                    save_images = inference_hyper.get('save_images', True)
                    save_dir = inference_hyper.get('save_dir', None)
                    base_prefix = inference_hyper.get('save_prefix', 'generated')
                    save_prefix_head = f'{base_prefix}_head_worker{worker_id}'
                    save_prefix_wrist = f'{base_prefix}_wrist_worker{worker_id}'
                    
                    # ========== Stage 1: 生成 next_head ==========
                    # 输入: [current_head, current_wrist] + action -> next_head
                    stage1_prompt = """You are now acting as a **world model** that simulates robot manipulation task execution.
Your task is to predict the **next frame of visual observation**, given the following inputs:
- **Multiple current observation images** from the robot's cameras (head camera and wrist camera)
- An **action sequence** describing the manipulation to execute
- Optionally, the **next frame from the head camera** (for predicting wrist camera views)

You will receive images from different camera viewpoints and need to predict the next frame according to the provided action sequence and instruction."""
                    
                    stage1_actions = [
                        f"{a}. Predict next head camera view according to the current observation and action."
                        for a in action_str
                    ]
                    # 准备输入：每个样本是 [head, wrist] 的列表
                    stage1_input_images = [[head, wrist] for head, wrist in zip(pil_head_images, pil_wrist_images)]
                    
                    next_head_images = batch_pred_next_imgs_cfg_multi_input(
                        model=model,
                        vae_model=vae_model,
                        tokenizer=tokenizer,
                        new_token_ids=new_token_ids,
                        vae_transform=vae_transform,
                        vit_transform=vit_transform,
                        prompt=stage1_prompt,
                        images=stage1_input_images,
                        actions=stage1_actions,
                        num_timesteps=inference_hyper['num_timesteps'],
                        cfg_text_scale=inference_hyper['cfg_text_scale'],
                        cfg_img_scale=inference_hyper['cfg_img_scale'],
                        cfg_interval=inference_hyper['cfg_interval'],
                        cfg_renorm_min=inference_hyper['cfg_renorm_min'],
                        cfg_renorm_type=inference_hyper['cfg_renorm_type'],
                        timestep_shift=inference_hyper['timestep_shift'],
                        enable_taylorseer=True,
                        save_images=save_images,
                        save_dir=save_dir,
                        save_prefix=save_prefix_head,
                    )
                    # ========== Stage 2: 生成 next_wrist ==========
                    # 输入: [next_head, current_wrist] -> next_wrist
                    stage2_prompt = """You are now acting as a **world model** that simulates robot manipulation task execution.
Your task is to predict the **next frame of visual observation**, given the following inputs:
- **Multiple current observation images** from the robot's cameras (head camera and wrist camera)
- An **action sequence** describing the manipulation to execute
- Optionally, the **next frame from the head camera** (for predicting wrist camera views)

You will receive images from different camera viewpoints and need to predict the next frame according to the provided action sequence and instruction."""
                    
                    stage2_actions = [
                        "Predict current wrist camera view according to history wrist camera view and current head camera view."
                        for _ in range(len(pil_wrist_images))
                    ]
                    
                    # 转换生成的 next_head 为 PIL
                    pil_next_head_images = [Image.fromarray(img) for img in next_head_images]
                    
                    # 准备输入：每个样本是 [next_head, current_wrist] 的列表
                    stage2_input_images = [[next_head, wrist] for next_head, wrist in zip(pil_next_head_images, pil_wrist_images)]
                    
                    next_wrist_images = batch_pred_next_imgs_cfg_multi_input(
                        model=model,
                        vae_model=vae_model,
                        tokenizer=tokenizer,
                        new_token_ids=new_token_ids,
                        vae_transform=vae_transform,
                        vit_transform=vit_transform,
                        prompt=stage2_prompt,
                        images=stage2_input_images,
                        actions=stage2_actions,
                        num_timesteps=inference_hyper['num_timesteps'],
                        cfg_text_scale=inference_hyper['cfg_text_scale'],
                        cfg_img_scale=inference_hyper['cfg_img_scale'],
                        cfg_interval=inference_hyper['cfg_interval'],
                        cfg_renorm_min=inference_hyper['cfg_renorm_min'],
                        cfg_renorm_type=inference_hyper['cfg_renorm_type'],
                        timestep_shift=inference_hyper['timestep_shift'],
                        enable_taylorseer=True,
                        save_images=save_images,
                        save_dir=save_dir,
                        save_prefix=save_prefix_wrist,
                    )
                    # 返回两个结果
                    remote.send(("success", (np.array(next_head_images), np.array(next_wrist_images))))
                
                elif cmd == "understand":
                    # 解包参数
                    images, tasks, inference_hyper = data
                    
                    # 转换为 PIL images
                    pil_images = [Image.fromarray(img) for img in images]

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
                    
                    prompts = [f"\n{prompt_content}\nDetermine whether the task: {t} is successfully completed, answer with Yes or No" for t in tasks]
                    
                    # 获取模型设备
                    device = get_model_device(model)
                    
                    rewards = []
                    for i in range(len(pil_images)):
                        outputs = vlm_pred(
                            model=model,
                            tokenizer=tokenizer,
                            new_token_ids=new_token_ids,
                            vit_transform=vit_transform,
                            prompt=prompts[i],
                            images=[pil_images[i]],
                            original_image_size=pil_images[i].size,
                            num_samples=1,
                            do_sample=False,
                            temperature=0.3,
                            max_length=512,
                            device=device
                        )
                        # 检查输出是否包含 'Yes'
                        # 取第一个输出（如果 num_samples > 1，可以取多数投票）
                        output_text = outputs[0] if outputs else ""
                        reward = 'Yes' in output_text or 'yes' in output_text.lower()
                        rewards.append(reward)
                    remote.send(("success", rewards))
                
                elif cmd == "reset":
                    remote.send(("success", None))
                
                elif cmd == "close":
                    remote.close()
                    break
                
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
            
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                remote.send(("error", str(e)))
    
    except Exception as e:
        logging.error(f"[Worker {worker_id}] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        remote.send(("error", str(e)))
    finally:
        remote.close()


def _load_model_for_worker(model_config_path: str, model_weights_path: str, max_mem_per_gpu: str, gpu_id: int):
    """Load model for worker
    
    Args:
        model_config_path: 模型配置文件路径（包含 llm_config.json, vit_config.json, ae.safetensors, tokenizer）
        model_weights_path: 模型权重文件路径（model.safetensors）
        max_mem_per_gpu: 每个 GPU 的最大内存
        gpu_id: 要使用的 GPU ID
    """
    # LLM config
    llm_config = Qwen2Config.from_json_file(os.path.join(model_config_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    
    # ViT config
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_config_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1
    
    # VAE loading
    vae_model, vae_config = load_ae(local_path=os.path.join(model_config_path, "ae.safetensors"))
    vae_model = vae_model.to(dtype=torch.bfloat16, device=f"cuda:{gpu_id}").eval()
    
    # Bagel config
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    
    # Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_config_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    
    # Image transforms
    vae_transform = ImageTransform(1024, 256, 16)
    vit_transform = ImageTransform(518, 224, 14)
    
    device_map = infer_auto_device_map(
        model,
        max_memory={gpu_id: max_mem_per_gpu},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    
    same_device_modules = [
        'language_model.model.embed_tokens',
        'time_embedder',
        'latent_pos_embed',
        'vae2llm',
        'llm2vae',
        'connector',
        'vit_pos_embed'
    ]
    
    first_device = gpu_id
    for k in same_device_modules:
        device_map[k] = first_device  

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=model_weights_path,
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        offload_folder=None,
        force_hooks=False,
    )

    model = model.eval()
    
    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


@torch.no_grad()
def batch_pred_next_imgs_cfg_multi_input(
    model, vae_model, tokenizer, new_token_ids, vae_transform, vit_transform,
    prompt, images, actions,
    num_timesteps=50,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_type="parallel",
    cfg_interval=[0.4, 1.0],
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
    timestep_shift=1.0,
    enable_taylorseer=False,
    # Image saving params
    save_images=True,
    save_dir="./generated_images",
    save_prefix="generated",
):
    """
    Generate next images with multi-image input support (e.g., [head, wrist]).
    
    Args:
        model: The Bagel model
        vae_model: The VAE model
        tokenizer: Tokenizer for text processing
        new_token_ids: Special token IDs
        vae_transform: Transform for VAE input
        vit_transform: Transform for VIT input
        prompt: Text prompt (single string, applied to all samples)
        images: List of image lists. Each element is a list of PIL images for one sample.
                Example: [[head1, wrist1], [head2, wrist2], ...]
        actions: List of action strings, one per sample
        num_timesteps: Number of diffusion steps
        cfg_text_scale: Text CFG scale
        cfg_img_scale: Image CFG scale
        cfg_interval: CFG interval
        cfg_renorm_min: CFG renorm min
        cfg_renorm_type: CFG renorm type
        timestep_shift: Timestep shift for diffusion
        enable_taylorseer: Enable TaylorSeer optimization
        save_images: Whether to save generated images
        save_dir: Directory to save images
        save_prefix: Prefix for saved image filenames
        
    Returns:
        List of generated images (numpy arrays)
    """
    assert len(images) == len(actions)
    batch_size = len(images)
    
    # Get image size from first image
    first_img = images[0][0]
    w, h = first_img.size
    vae_resize = vae_transform.resize_transform
    
    def _make_divisible(value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))
    
    scale = min(vae_resize.max_size / max(w, h), 1.0)
    scale = max(scale, vae_resize.min_size / min(w, h))
    w = _make_divisible(round(w * scale), vae_resize.stride)
    h = _make_divisible(round(h * scale), vae_resize.stride)
    
    if max(w, h) > vae_resize.max_size:
        scale = vae_resize.max_size / max(w, h)
        w = _make_divisible(round(w * scale), vae_resize.stride)
        h = _make_divisible(round(h * scale), vae_resize.stride)
    
    # 获取模型设备
    device = get_model_device(model)
    
    # ========== Main branch: prompt + images + actions ==========
    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0] * batch_size
    new_rope = [0] * batch_size
    
    # 1. Forward prompt
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)
    
    # 2. Forward each condition image (e.g., head, wrist)
    num_cond = len(images[0])
    for cond_idx in range(num_cond):
        cond_images = [sample_imgs[cond_idx] for sample_imgs in images]
        
        # VAE images
        generation_input, newlens, new_rope = model.prepare_vae_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=cond_images,
            transforms=vae_transform,
            new_token_ids=new_token_ids,
            timestep=0.0,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_vae(vae_model, past_key_values, **generation_input)
        
        # VIT images
        generation_input, newlens, new_rope = model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            images=cond_images,
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)
    
    # 3. Forward action instructions
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        prompts=actions,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)
    
    # 4. Prepare VAE latent for main branch
    generation_input = model.prepare_vae_latent(
        curr_kvlens=newlens,
        curr_rope=new_rope,
        image_sizes=[(h, w)] * batch_size,
        new_token_ids=new_token_ids,
    )
    
    # ========== Text CFG branch: images only (no prompt, no actions) ==========
    cfg_text_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_text_newlens = [0] * batch_size
    cfg_text_new_rope = [0] * batch_size

    generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope, 
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
        cfg_text_past_key_values = model.forward_cache_update_text(cfg_text_past_key_values, **generation_input_cfg_text)
    
    for cond_idx in range(num_cond):
        cond_images = [sample_imgs[cond_idx] for sample_imgs in images]
        
        # VAE images
        generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vae_images(
            curr_kvlens=cfg_text_newlens,
            curr_rope=cfg_text_new_rope,
            images=cond_images,
            transforms=vae_transform,
            new_token_ids=new_token_ids,
            timestep=0.0,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
            cfg_text_past_key_values = model.forward_cache_update_vae(
                vae_model, cfg_text_past_key_values, **generation_input_cfg_text
            )
        
        # VIT images
        generation_input_cfg_text, cfg_text_newlens, cfg_text_new_rope = model.prepare_vit_images(
            curr_kvlens=cfg_text_newlens,
            curr_rope=cfg_text_new_rope,
            images=cond_images,
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input_cfg_text = move_to_device(generation_input_cfg_text, device)
            cfg_text_past_key_values = model.forward_cache_update_vit(cfg_text_past_key_values, **generation_input_cfg_text)
    
    generation_input_cfg_text = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_newlens,
        curr_rope=cfg_text_new_rope,
        image_sizes=[(h, w)] * batch_size,
    )
    
    # ========== Image CFG branch: prompt + actions only (no images) ==========
    cfg_img_past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    cfg_img_newlens = [0] * batch_size
    cfg_img_new_rope = [0] * batch_size
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        prompts=[prompt] * batch_size,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)
        cfg_img_past_key_values = model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    # Forward actions
    generation_input_cfg_img, cfg_img_newlens, cfg_img_new_rope = model.prepare_prompts(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        prompts=actions,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input_cfg_img = move_to_device(generation_input_cfg_img, device)
        cfg_img_past_key_values = model.forward_cache_update_text(cfg_img_past_key_values, **generation_input_cfg_img)
    
    generation_input_cfg_img = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_newlens,
        curr_rope=cfg_img_new_rope,
        image_sizes=[(h, w)] * batch_size,
    )
    
    # Extract packed positions and indexes for CFGs
    cfg_text_args = {
        'cfg_text_packed_position_ids': generation_input_cfg_text['cfg_packed_position_ids'],
        'cfg_text_packed_query_indexes': generation_input_cfg_text['cfg_packed_query_indexes'],
        'cfg_text_key_values_lens': generation_input_cfg_text['cfg_key_values_lens'],
        'cfg_text_packed_key_value_indexes': generation_input_cfg_text['cfg_packed_key_value_indexes'],
    }
    
    cfg_img_args = {
        'cfg_img_packed_position_ids': generation_input_cfg_img['cfg_packed_position_ids'],
        'cfg_img_packed_query_indexes': generation_input_cfg_img['cfg_packed_query_indexes'],
        'cfg_img_key_values_lens': generation_input_cfg_img['cfg_key_values_lens'],
        'cfg_img_packed_key_value_indexes': generation_input_cfg_img['cfg_packed_key_value_indexes'],
    }
    
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_to_device(generation_input, device)
        cfg_text_args = move_to_device(cfg_text_args, device)
        cfg_img_args = move_to_device(cfg_img_args, device)
        unpacked_latent = model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_type=cfg_type,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            **cfg_text_args,
            **cfg_img_args,
            enable_taylorseer=enable_taylorseer,
        )
    
    # Process and decode the latent representation
    image_list = []
    for latent in unpacked_latent:
        latent = latent.reshape(1, h // 16, w // 16, 2, 2, 16).to(torch.bfloat16)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, 16, h // 8, w // 8)
        image = vae_model.decode(latent)
        
        # Convert to image
        image = ((image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image_list.append(image)
    
    # Save images if requested
    if save_images and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, img_array in enumerate(image_list):
            img_pil = Image.fromarray(img_array)
            filename = f"{save_prefix}_{timestamp}_batch{idx}.png"
            save_path = os.path.join(save_dir, filename)
            img_pil.save(save_path)
            logging.info(f"Saved generated image to: {save_path}")
    
    return image_list

@torch.no_grad()
def vlm_pred(
    model, tokenizer, new_token_ids, vit_transform,
    prompt, images,
    original_image_size=(640, 360),
    num_samples=16,
    do_sample=True,
    temperature=0.3,
    max_length=512,
    device='cuda:0'
):
    """
    Visual Language Model prediction function.
    
    Args:
        model: The Bagel model
        tokenizer: Tokenizer for text processing
        new_token_ids: Special token IDs
        vit_transform: Transform for VIT input
        prompt: Text prompt (single string)
        images: List of PIL images
        original_image_size: Original image size for resizing
        num_samples: Number of samples to generate
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        max_length: Maximum generation length
        device: Device to run on
        
    Returns:
        List of generated text outputs
    """
    # Initialize cache and setup
    past_key_values = NaiveCache(model.config.llm_config.num_hidden_layers)
    newlens = [0]
    new_rope = [0]
        
    # prepare & forward VIT images
    for image in images:
        generation_input, newlens, new_rope = model.prepare_vit_images(
            curr_kvlens=newlens,
            curr_rope=new_rope, 
            images=[image.resize(original_image_size)],
            transforms=vit_transform, 
            new_token_ids=new_token_ids,
        )
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generation_input = move_to_device(generation_input, device)
            past_key_values = model.forward_cache_update_vit(past_key_values, **generation_input)

    # Prepare & forward prompt for main branch
    generation_input, newlens, new_rope = model.prepare_prompts(
        curr_kvlens=newlens,
        curr_rope=new_rope, 
        prompts=[prompt],
        tokenizer=tokenizer, 
        new_token_ids=new_token_ids,
    )
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = move_to_device(generation_input, device)
        past_key_values = model.forward_cache_update_text(past_key_values, **generation_input)

    outputs = []

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        generation_input = model.prepare_start_tokens(newlens, new_rope, new_token_ids)
        
        generation_input = move_to_device(generation_input, device)

        for i in range(num_samples):
            unpacked_latent = model.generate_text(
                past_key_values=copy.deepcopy(past_key_values),
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
            output = tokenizer.decode(unpacked_latent[:,0])
            think_output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
            outputs.append(think_output)
        
    return outputs


class WorldModelInferenceServer:
    """
    Bagel Inference Server that provides:
    1. Editing capability as world model (image + prompt -> next_image)
    2. Understanding capability as reward model (image + prompt -> text)
    """
    
    def __init__(
        self, 
        model_config_path: str,
        model_weights_path: str,
        action_norm_path: str,
        num_workers: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        max_mem_per_gpu: str = "40GiB",
        start_method: Optional[str] = None,
        # Editing (world model) default hyperparameters
        edit_cfg_text_scale: float = 4.0,
        edit_cfg_img_scale: float = 2.0,
        edit_cfg_interval: tuple = (0.4, 1.0),
        edit_timestep_shift: float = 4.0,
        edit_num_timesteps: int = 50,
        edit_cfg_renorm_min: float = 0.0,
        edit_cfg_renorm_type: str = "text_channel",
        # Image saving default parameters
        save_images: bool = False,
        save_dir: str = "./generated_images",
        save_prefix: str = "generated",
        # Understanding (reward model) default hyperparameters
        understand_max_tokens: int = 1000,
        understand_do_sample: bool = False,
        understand_temperature: float = 0.3,
    ):
        """
        Initialize World Model Inference Server with multi-worker architecture.
        
        Args:
            model_config_path: Path to World Model config directory (contains llm_config.json, vit_config.json, ae.safetensors, tokenizer)
            model_weights_path: Path to model weights file (model.safetensors)
            action_norm_path: Path to action normalizer JSON file
            num_workers: Number of worker processes (default: number of GPUs)
            gpu_ids: List of GPU IDs to use (default: all available GPUs)
            max_mem_per_gpu: Maximum memory per GPU (e.g., "40GiB")
            start_method: Multiprocessing start method ('spawn' or 'forkserver')
            edit_*: Default hyperparameters for editing (world model)
            save_images: 是否默认保存生成的图像 (default: True)
            save_dir: 默认保存图像的目录 (default: "./generated_images")
            save_prefix: 默认保存图像的文件名前缀 (default: "generated")
            understand_*: Default hyperparameters for understanding (reward model)
        """
        self.model_config_path = model_config_path
        self.model_weights_path = model_weights_path
        self.action_norm_path = action_norm_path
        self.max_mem_per_gpu = max_mem_per_gpu
        self.closed = False
        
        # 确定使用的 GPU
        if gpu_ids is None:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs available")
            gpu_ids = list(range(num_gpus))
        self.gpu_ids = gpu_ids
        
        # 确定 worker 数量
        if num_workers is None:
            num_workers = len(gpu_ids)
        self.num_workers = num_workers
        
        logging.info(f"Initializing Bagel Inference Server with {num_workers} workers on GPUs {gpu_ids}")
        
        # Store default hyperparameters
        self.edit_hyper = {
            'cfg_text_scale': edit_cfg_text_scale,
            'cfg_img_scale': edit_cfg_img_scale,
            'cfg_interval': list(edit_cfg_interval),
            'timestep_shift': edit_timestep_shift,
            'num_timesteps': edit_num_timesteps,
            'cfg_renorm_min': edit_cfg_renorm_min,
            'cfg_renorm_type': edit_cfg_renorm_type,
            'save_images': save_images,
            'save_dir': save_dir,
            'save_prefix': save_prefix,
        }
        
        self.understand_hyper = {
            'num_samples': 1,
            'do_sample': understand_do_sample,
            'temperature': understand_temperature,
            'max_length': understand_max_tokens,
        }
        
        # 设置多进程启动方法
        if start_method is None:
            start_method = "spawn"
        
        logging.info(f"Using multiprocessing start method: {start_method}")
        ctx = mp.get_context(start_method)
        
        # 创建 worker 进程
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_workers)])
        self.processes = []
        
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            args = (
                work_remote, 
                remote, 
                model_config_path,
                model_weights_path,
                action_norm_path,
                max_mem_per_gpu,
                i,  # worker_id
                gpu_id,  # gpu_id
            )
            process = ctx.Process(target=_world_model_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
            logging.info(f"Started worker process {i} on GPU {gpu_id}")
        
        # 等待所有 worker 准备就绪
        logging.info("Waiting for all workers to load models...")
        self._wait_for_workers_ready()
        
        logging.info("Bagel Inference Server initialized successfully!")
    
    def _wait_for_workers_ready(self, timeout: float = 300.0):
        """等待所有 worker 进程准备就绪。"""
        import time
        start_time = time.time()
        
        for i in range(self.num_workers):
            while time.time() - start_time < timeout:
                if self.remotes[i].poll(timeout=1.0):
                    status, _ = self.remotes[i].recv()
                    if status == "ready":
                        logging.info(f"Worker {i} is ready")
                        break
                    elif status == "error":
                        raise RuntimeError(f"Worker {i} failed to initialize")
            else:
                raise RuntimeError(f"Worker {i} failed to start within {timeout} seconds")
        
        logging.info("All worker processes are ready")
    
    @torch.no_grad()
    def edit(self, head_image: np.ndarray, wrist_image: np.ndarray, action: np.ndarray, **kwargs) -> Dict:
        """
        Edit/generate next frame images based on current multi-view images and action (world model mode).
        数据会自动分发到不同的 worker 进行并行处理。
        
        Uses a two-stage generation process:
        1. Generate next head camera view: [current_head, current_wrist] + action -> next_head
        2. Generate next wrist camera view: [next_head, current_wrist] -> next_wrist
        
        Args:
            head_image: Input head camera images, shape (N, H, W, 3), dtype uint8, RGB
            wrist_image: Input wrist camera images, shape (N, H, W, 3), dtype uint8, RGB
            action: Actions, shape (N, action_dim)
            **kwargs: Override default editing hyperparameters
                - save_images (bool): 是否保存生成的图像 (default: True)
                - save_dir (str): 保存图像的目录路径 (default: "./generated_images")
                - save_prefix (str): 保存图像的文件名前缀 (default: "generated")
            
        Returns:
            Dict with:
                - 'next_head': np.ndarray (N, H, W, 3), dtype uint8
                - 'next_wrist': np.ndarray (N, H, W, 3), dtype uint8
        """
        # Merge kwargs with default hyperparameters
        inference_hyper = {**self.edit_hyper, **kwargs}
        
        batch_size = len(head_image)
        
        # 将数据分配到不同的 worker
        # 计算每个 worker 应该处理的样本数量
        samples_per_worker = [batch_size // self.num_workers] * self.num_workers
        for i in range(batch_size % self.num_workers):
            samples_per_worker[i] += 1
        
        # 分发数据到 worker
        start_idx = 0
        for worker_id in range(self.num_workers):
            end_idx = start_idx + samples_per_worker[worker_id]
            if samples_per_worker[worker_id] > 0:
                worker_head_images = head_image[start_idx:end_idx]
                worker_wrist_images = wrist_image[start_idx:end_idx]
                worker_actions = action[start_idx:end_idx]
                self.remotes[worker_id].send(("edit", (worker_head_images, worker_wrist_images, worker_actions, inference_hyper)))
            start_idx = end_idx
        
        # 收集结果
        next_head_images = []
        next_wrist_images = []
        start_idx = 0
        for worker_id in range(self.num_workers):
            if samples_per_worker[worker_id] > 0:
                status, result = self.remotes[worker_id].recv()
                if status == "error":
                    raise RuntimeError(f"Worker {worker_id} error: {result}")
                # result 现在是 (next_head, next_wrist) 的元组
                next_head, next_wrist = result
                next_head_images.append(next_head)
                next_wrist_images.append(next_wrist)
        
        # 拼接结果
        next_head_images = np.concatenate(next_head_images, axis=0)
        next_wrist_images = np.concatenate(next_wrist_images, axis=0)
        
        return {'next_head': next_head_images, 'next_wrist': next_wrist_images}
    
    @torch.no_grad()
    def understand(self, image: np.ndarray, task: list, **kwargs) -> Dict:
        """
        Understand image and answer question (reward model mode).
        数据会自动分发到不同的 worker 进行并行处理。
        
        Args:
            image: Input images, shape (N, H, W, 3), dtype uint8, RGB
            task: Task prompts, list of N strings
            **kwargs: Override default understanding hyperparameters
            
        Returns:
            Dict with 'reward': list of N booleans
        """
        # Merge kwargs with default hyperparameters
        inference_hyper = {**self.understand_hyper, **kwargs}
        
        batch_size = len(image)
        
        # 将数据分配到不同的 worker
        samples_per_worker = [batch_size // self.num_workers] * self.num_workers
        for i in range(batch_size % self.num_workers):
            samples_per_worker[i] += 1
        
        # 分发数据到 worker
        start_idx = 0
        for worker_id in range(self.num_workers):
            end_idx = start_idx + samples_per_worker[worker_id]
            if samples_per_worker[worker_id] > 0:
                worker_images = image[start_idx:end_idx]
                worker_tasks = task[start_idx:end_idx]
                self.remotes[worker_id].send(("understand", (worker_images, worker_tasks, inference_hyper)))
            start_idx = end_idx
        
        # 收集结果
        rewards = []
        start_idx = 0
        for worker_id in range(self.num_workers):
            if samples_per_worker[worker_id] > 0:
                status, result = self.remotes[worker_id].recv()
                if status == "error":
                    raise RuntimeError(f"Worker {worker_id} error: {result}")
                rewards.extend(result)
        
        return {'reward': rewards}
    
    def handle_request(self, request: Dict) -> Dict:
        """Handle incoming requests."""
        method = request.get('method')
        
        if method == 'edit':
            head_image = request['head_image']
            wrist_image = request['wrist_image']
            action = request['action']
            kwargs = request.get('kwargs', {})
            return self.edit(head_image, wrist_image, action, **kwargs)
        
        elif method == 'understand':
            image = request['image']
            task = request['task']
            kwargs = request.get('kwargs', {})
            return self.understand(image, task, **kwargs)
        
        elif method == 'reset':
            # 发送 reset 命令到所有 worker
            for worker_id in range(self.num_workers):
                self.remotes[worker_id].send(("reset", None))
            
            # 等待所有 worker 完成 reset
            for worker_id in range(self.num_workers):
                status, _ = self.remotes[worker_id].recv()
                if status == "error":
                    raise RuntimeError(f"Worker {worker_id} reset failed")
            
            return {'status': 'reset_complete'}
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def close(self) -> Dict:
        """关闭所有 worker 进程。"""
        if self.closed:
            return {"status": "already_closed"}
        
        # 发送关闭命令到所有 worker
        for i, remote in enumerate(self.remotes):
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, Exception) as e:
                logging.warning(f"Error sending close command to worker {i}: {e}")
        
        # 等待所有进程结束
        for i, process in enumerate(self.processes):
            try:
                process.join(timeout=5)
                if process.is_alive():
                    logging.warning(f"Worker {i} did not terminate, forcing...")
                    process.terminate()
                    process.join(timeout=2)
            except Exception as e:
                logging.warning(f"Error closing worker {i}: {e}")
        
        # 关闭所有 remote 连接
        for i, remote in enumerate(self.remotes):
            try:
                remote.close()
            except Exception as e:
                logging.warning(f"Error closing remote {i}: {e}")
        
        self.closed = True
        logging.info("All worker processes closed")
        return {"status": "closed"}
    
    def __del__(self):
        """析构函数。"""
        if not self.closed:
            self.close()
