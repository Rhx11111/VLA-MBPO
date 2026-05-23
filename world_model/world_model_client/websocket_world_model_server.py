# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import argparse
import os

import websockets

try:
    from . import msgpack_numpy
    from .world_model_inference_server import WorldModelInferenceServer
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import msgpack_numpy
    from world_model_inference_server import WorldModelInferenceServer


class WebsocketWorldModelServer:
    """
    WebSocket server for WorldModel model inference.
    Supports two modes:
    1. Editing (world model): image + prompt -> next_image
    2. Understanding (reward model): image + prompt -> text
    """
    
    def __init__(
        self, 
        inference_server: WorldModelInferenceServer, 
        host: str = "0.0.0.0", 
        port: int = 8002
    ):
        self.inference_server = inference_server
        self.host = host
        self.port = port
        self.metadata = {
            'server_type': 'world_model',
            'model_config_path': inference_server.model_config_path,
            'model_weights_path': inference_server.model_weights_path,
            'num_workers': inference_server.num_workers,
            'gpu_ids': inference_server.gpu_ids,
            'capabilities': ['edit', 'understand'],
        }
    
    async def handle_client(self, websocket):
        """Handle a client connection."""
        logging.info(f"Client connected from {websocket.remote_address}")
        
        try:
            # Send metadata to client
            await websocket.send(msgpack_numpy.packb(self.metadata))
            
            async for message in websocket:
                try:
                    # Decode request
                    request = msgpack_numpy.unpackb(message)
                    
                    # Process request
                    response = self.inference_server.handle_request(request)
                    
                    # Send response
                    await websocket.send(msgpack_numpy.packb(response))
                    
                except Exception as e:
                    error_msg = f"Error processing request: {str(e)}"
                    logging.error(error_msg)
                    import traceback
                    traceback.print_exc()
                    await websocket.send(error_msg)
        
        except Exception as e:
            logging.error(f"Error handling client: {str(e)}")
        finally:
            logging.info(f"Client {websocket.remote_address} disconnected")
    
    async def start_server(self):
        """Start the websocket server."""
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(f"Starting WorldModel server on {local_ip}:{self.port}")
        
        # Configure server with appropriate timeouts for large data transfers
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            close_timeout=30
        ):
            logging.info("WorldModel server started successfully!")
            logging.info("Capabilities: edit (world model), understand (reward model)")
            await asyncio.Future()  # Run forever
    
    def run(self):
        """Run the server."""
        asyncio.run(self.start_server())


def main():
    parser = argparse.ArgumentParser(description="WorldModel Model WebSocket Server with Multi-Worker Architecture")
    
    # Model configuration
    parser.add_argument("--model-config-path", type=str, required=True,
                        help="Path to BAGEL base model config directory containing llm_config.json, vit_config.json, ae.safetensors, and tokenizer files")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to finetuned WorldModel checkpoint directory containing model.safetensors")
    parser.add_argument("--action-norm-path", type=str, 
                        default=os.environ.get("WORLD_MODEL_ACTION_NORM_PATH"),
                        help="Action normalizer path")
    parser.add_argument("--max-mem-per-gpu", type=str, default="40GiB",
                        help="Maximum memory per GPU (e.g., '40GiB')")
    
    # Worker configuration
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of worker processes (default: number of GPUs)")
    parser.add_argument("--gpu-ids", type=int, nargs='+', default=None,
                        help="List of GPU IDs to use (default: all available GPUs)")
    parser.add_argument("--start-method", type=str, default=None, 
                        choices=['spawn', 'forkserver'],
                        help="Multiprocessing start method")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8002,
                        help="Port to bind server to")
    
    # Editing (world model) hyperparameters
    parser.add_argument("--edit-cfg-text-scale", type=float, default=4.0,
                        help="Text CFG scale for editing")
    parser.add_argument("--edit-cfg-img-scale", type=float, default=2.0,
                        help="Image CFG scale for editing")
    parser.add_argument("--edit-timestep-shift", type=float, default=3.0,
                        help="Timestep shift for editing")
    parser.add_argument("--edit-num-timesteps", type=int, default=50,
                        help="Number of timesteps for editing")
    parser.add_argument("--edit-cfg-renorm-type", type=str, default="text_channel",
                        choices=["global", "channel", "text_channel"],
                        help="CFG renorm type for editing")
    
    # Understanding (reward model) hyperparameters
    parser.add_argument("--understand-max-tokens", type=int, default=1000,
                        help="Max tokens for understanding")
    parser.add_argument("--understand-temperature", type=float, default=0.3,
                        help="Temperature for understanding")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create inference server with multi-worker architecture
    inference_server = WorldModelInferenceServer(
        model_config_path=args.model_config_path,
        model_weights_path=args.model_path,
        action_norm_path=args.action_norm_path,
        num_workers=args.num_workers,
        gpu_ids=args.gpu_ids,
        max_mem_per_gpu=args.max_mem_per_gpu,
        start_method=args.start_method,
        edit_cfg_text_scale=args.edit_cfg_text_scale,
        edit_cfg_img_scale=args.edit_cfg_img_scale,
        edit_cfg_interval=(0.4, 1.0),
        edit_timestep_shift=args.edit_timestep_shift,
        edit_num_timesteps=args.edit_num_timesteps,
        edit_cfg_renorm_min=0.0,
        edit_cfg_renorm_type=args.edit_cfg_renorm_type,
        understand_max_tokens=args.understand_max_tokens,
        understand_do_sample=False,
        understand_temperature=args.understand_temperature,
    )
    
    # Create and run websocket server
    websocket_server = WebsocketWorldModelServer(
        inference_server=inference_server,
        host=args.host,
        port=args.port
    )
    
    try:
        websocket_server.run()
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
        inference_server.close()


if __name__ == "__main__":
    main()
