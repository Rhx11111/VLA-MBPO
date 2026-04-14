#!/usr/bin/env python3
"""
LIBERO Client Example Usage

This script demonstrates how to use the LIBERO client and server.
"""

import argparse
import logging
import numpy as np
import time
from libero_client import LiberoWebsocketClient, LiberoServerEnv, LiberoWebsocketServer


def run_client_example(host="localhost", port=8003):
    """Run client example."""
    logging.info(f"Connecting to LIBERO server at {host}:{port}")
    
    with LiberoWebsocketClient(host=host, port=port) as client:
        # Get server metadata
        metadata = client.get_server_metadata()
        logging.info(f"Server metadata: {metadata}")
        
        # Reset environment
        obs = client.reset()
        logging.info(f"Reset observation keys: {list(obs.keys())}")
        
        # Get task info
        task_info = client.get_task_info()
        logging.info(f"Task info: {task_info}")
        
        # Take some random actions
        for step in range(10):
            action = np.random.uniform(-0.1, 0.1, 7)  # Random 7-DOF actions
            obs, reward, done, info = client.step(action)
            logging.info(f"Step {step}: reward={reward:.3f}, done={done}")
            
            if done:
                logging.info("Task completed!")
                break
                
        # Check success
        success = client.check_success()
        logging.info(f"Task success: {success}")
                
        # Render current state
        img = client.render()
        logging.info(f"Rendered image shape: {img.shape}")


def run_server_example(task_suite_name="libero_spatial", host="0.0.0.0", port=8003):
    """Run server example."""
    logging.info(f"Starting LIBERO server with task suite: {task_suite_name}")
    
    try:
        # Create server environment
        server_env = LiberoServerEnv(
            task_suite_name=task_suite_name,
            num_envs=1,
            resolution=256,
            seed=7
        )
        
        # Create and run websocket server
        websocket_server = LiberoWebsocketServer(
            env=server_env,
            host=host,
            port=port
        )
        
        logging.info(f"LIBERO server started on {host}:{port}")
        websocket_server.run()
        
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
        server_env.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LIBERO Client Example")
    parser.add_argument("mode", choices=["client", "server"], 
                       help="Mode to run: client or server")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Host address")
    parser.add_argument("--port", type=int, default=8003,
                       help="Port number")
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial",
                       help="LIBERO task suite name (for server mode)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.mode == "client":
        run_client_example(args.host, args.port)
    elif args.mode == "server":
        run_server_example(args.task_suite_name, args.host, args.port)


if __name__ == "__main__":
    main()
