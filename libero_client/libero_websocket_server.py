import asyncio
import logging
import warnings
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import websockets

from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import Benchmark
from libero.libero.envs import OffScreenRenderEnv

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from . import msgpack_numpy
    from .utils import (
        get_libero_image,
        get_libero_wrist_image,
        list_of_dict_to_dict_of_list,
        quat2axisangle,
        to_tensor,
    )
except ImportError:
    import msgpack_numpy
    from utils import (
        get_libero_image,
        get_libero_wrist_image,
        list_of_dict_to_dict_of_list,
        quat2axisangle,
        to_tensor,
    )


def _libero_worker(
    remote: Connection,
    parent_remote: Connection,
    task_suite_name: str,
    resolution: int,
    seed: int,
    worker_id: int
) -> None:
    """
    Worker process function that runs one LIBERO environment instance.
    
    Args:
        remote: Child-process connection.
        parent_remote: Parent-process connection.
        task_suite_name: LIBERO task suite name.
        resolution: Camera resolution.
        seed: Random seed.
        worker_id: Worker id.
    """
    parent_remote.close()
    current_env = None
    
    try:
        # Initialize the task suite in the child process.
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite: Benchmark = benchmark_dict[task_suite_name]()
        available_gpus = [0, 1, 2, 3]
        device_id = available_gpus[worker_id % len(available_gpus)]
        import os 
        os.environ["EGL_VISIBLE_DEVICES"] = str(device_id) 
        print(f"[Worker {worker_id}] Using device: {device_id}")
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == "reset":
                    sim_state, sim_state_len, task_id = data
                    if current_env:
                        current_env.close()
                        current_env = None
                        
                    # Create a new environment.
                    task = task_suite.get_task(task_id)
                    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                    env_args = {
                        "bddl_file_name": str(task_bddl_file),
                        "camera_heights": resolution,
                        "camera_widths": resolution,
                        "render_gpu_device_id": device_id,
                    }
                    current_env = OffScreenRenderEnv(**env_args)
                    current_env.seed(seed + worker_id)
                    current_env.reset()
                    current_task_id = task_id

                    if sim_state_len > 0:
                        obs = current_env.set_init_state(sim_state[:sim_state_len])
                    else:
                        obs = current_env.reset()

                    for _ in range(10):
                        zero_actions = np.zeros(7)
                        zero_actions[-1] = -1
                        obs, _reward, terminations, truncations = current_env.step(zero_actions)

                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    obs["agentview_image"] = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    obs["robot0_eye_in_hand_image"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    obs['task_description'] = task.language
                    
                    remote.send(("success", obs.copy()))
                    
                elif cmd == "step":
                    # data = action
                    action = data
                    obs, reward, done, info = current_env.step(action.tolist())

                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    obs["agentview_image"] = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    obs["robot0_eye_in_hand_image"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    obs['task_description'] = task.language
                    truncation = np.zeros(1)
                    remote.send(("success", (obs.copy(), reward, done, truncation)))
                    
                elif cmd == "close":
                    if current_env is not None:
                        current_env.close()
                        current_env = None
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
                break
    
    finally:
        # Ensure final cleanup.
        if current_env is not None:
            current_env.close()
        
        remote.close()


class LiberoInferenceServer:
    """
    Multiprocess LIBERO inference server.

    This simplified version focuses on parallel execution and omits metrics.
    """
    
    def __init__(
        self,
        task_suite_name: str = "libero_spatial",
        num_envs: int = 1,
        resolution: int = 256,
        seed: int = 7,
        start_method: Optional[str] = None,
        use_rel_reward: bool = False,
    ):
        """Initialize the LIBERO inference server.
        
        Args:
            task_suite_name: LIBERO task suite name.
            num_envs: Number of parallel environments.
            resolution: Camera resolution.
            seed: Random seed.
            start_method: Multiprocessing start method ('spawn' or 'forkserver').
            use_rel_reward: Whether to use relative rewards.
        """
        self.task_suite_name = task_suite_name
        self.num_envs = num_envs
        self.resolution = resolution
        self.seed = seed
        self.closed = False
        self.use_rel_reward = use_rel_reward
        self.prev_step_reward = np.zeros(self.num_envs)
        
        # Set the multiprocessing start method to avoid GPU context conflicts.
        if start_method is None:
            start_method = "spawn"  # Force spawn to avoid OpenGL context conflicts.
        
        logging.info(f"Using multiprocessing start method: {start_method}")
        ctx = mp.get_context(start_method)
        
        # Initialize the task suite in the main process.
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite: Benchmark = benchmark_dict[task_suite_name]()
        self.num_tasks = self.task_suite.n_tasks
        self.task_descriptions: List[str] = [""] * num_envs
        
        # Create one worker process per environment.
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            args = (work_remote, remote, task_suite_name, resolution, self.seed, i)
            process = ctx.Process(target=_libero_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
            logging.info(f"Started worker process {i} for LIBERO environment")
        
        # Wait for all worker processes to be fully ready.
        logging.info("Waiting for all worker processes to be ready...")
        self._wait_for_workers_ready()
        
        logging.info(f"LiberoInferenceServer initialized with {num_envs} parallel workers")
        logging.info(f"Task suite: {task_suite_name}, Num tasks: {self.num_tasks}")
    
    def _wait_for_workers_ready(self, timeout: float = 60.0):
        """Wait until all worker processes are ready."""
        import time
        start_time = time.time()
        
        for i, process in enumerate(self.processes):
            while time.time() - start_time < timeout:
                if process.is_alive():
                    # The process is alive; give it a moment to finish initialization.
                    time.sleep(0.1)
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Worker {i} failed to start within {timeout} seconds")
        
        # Wait a bit longer to ensure full initialization.
        time.sleep(2.0)
        logging.info("All worker processes are ready")

    def _extract_image_and_state(self, obs: Dict) -> Dict:
        """Extract images and state from an observation."""
        return {
            "full_image": get_libero_image(obs),
            'wrist_image': get_libero_wrist_image(obs),
            "state": np.concatenate([
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]),
        }

    def _wrap_obs(self, obs_list: List[Dict]) -> Dict:
        """Pack a list of observations into the server format."""
        images_and_states_list = []
        task_descriptions = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)
            task_descriptions.append(obs['task_description'])
        
        obs = {
            "images_and_states": to_tensor(
                list_of_dict_to_dict_of_list(images_and_states_list)
            ),
            "task_descriptions": task_descriptions,
        }
        return obs

    def reset(
        self,
        sim_state: np.ndarray,
        sim_state_len: np.ndarray,
        task_id: np.ndarray,
    ) -> Dict:
        """Reset all environments in parallel.
        
        Args:
            sim_state: Simulator state array.
            sim_state_len: Simulator state length.
            task_id: Task ids.
        Returns:
            observations
        """
        # Reset relative reward state.
        self.prev_step_reward = np.zeros(self.num_envs)
        
        # Dispatch reset commands to all workers.
        for env_id in range(self.num_envs):
            self.remotes[env_id].send(("reset", (sim_state[env_id], sim_state_len[env_id], task_id[env_id])))
        
        # Wait for all workers to finish reset.
        obs_list = []
        for env_id in range(self.num_envs):
            status, obs = self.remotes[env_id].recv()
            if status == "error":
                raise RuntimeError(f"Worker error during reset: {obs}")
            obs_list.append(obs)
        
        # Pack observations.
        obs = self._wrap_obs(obs_list)
        
        return obs

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Run one parallel environment step.
        
        Args:
            actions: Action array with shape [num_envs, action_dim].
            
        Returns:
            (obs, rewards, terminations, truncations)
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        
        # Send the step command to all workers.
        for env_id in range(self.num_envs):
            action = actions[env_id] if actions.ndim > 1 else actions
            self.remotes[env_id].send(("step", action))
        
        # Wait for all workers to finish the step.
        obs_list = []
        rewards = np.zeros(self.num_envs)
        terminations = np.zeros(self.num_envs)
        truncations = np.zeros(self.num_envs)        
        for env_id in range(self.num_envs):
            status, result = self.remotes[env_id].recv()
            if status == "error":
                raise RuntimeError(f"Worker error during step: {result}")
            obs, reward, done, truncation = result
            obs_list.append(obs)
            rewards[env_id] = done
            terminations[env_id] = done
            truncations[env_id] = truncation

        # Make sure there is no -1 reward and terminations after suceed, (avoid libero's bug)
        if self.use_rel_reward:
            reward_diff = terminations - self.prev_step_reward
            self.prev_step_reward[self.prev_step_reward == 0] = terminations[self.prev_step_reward == 0]
            rewards = reward_diff
            rewards[rewards < 0] = 0
            terminations = self.prev_step_reward
        
        # Pack observations.
        obs = self._wrap_obs(obs_list)
        
        return obs, rewards, terminations, truncations

    def chunk_step(
        self,
        chunk_actions: np.ndarray,
    ) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """Run a chunk of actions.
        
        Args:
            chunk_actions: Chunk actions with shape [num_envs, chunk_size, action_dim].
            
        Returns:
            (obs, rewards, terminations, truncations)
        """
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        
        if chunk_actions.ndim == 3:
            chunk_size = chunk_actions.shape[1]
        else:
            raise ValueError(f"Expected shape [num_envs, chunk_size, action_dim], got {chunk_actions.shape}")
        
        all_rewards = []
        all_terminations = []
        all_truncations = []
        
        # Execute each chunk step.
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            obs, rewards, terminations, truncations = self.step(actions)
            all_rewards.append(rewards)
            all_terminations.append(terminations)
            all_truncations.append(truncations)
        
        # Stack results into [num_envs, chunk_size].
        all_rewards = np.stack(all_rewards, axis=1)
        all_terminations = np.stack(all_terminations, axis=1)
        all_truncations = np.stack(all_truncations, axis=1)
        
        return obs, all_rewards, all_terminations, all_truncations

    def close(self) -> Dict:
        """Close all environments."""
        if self.closed:
            return {"status": "already_closed"}
        
        # Send the close command to all workers.
        for i, remote in enumerate(self.remotes):
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, Exception) as e:
                logging.warning(f"Error sending close command to worker {i}: {e}")
        
        # Wait for all processes to exit.
        for i, process in enumerate(self.processes):
            try:
                process.join(timeout=5)
                if process.is_alive():
                    logging.warning(f"Worker {i} did not terminate, forcing...")
                    process.terminate()
                    process.join(timeout=2)
            except Exception as e:
                logging.warning(f"Error closing worker {i}: {e}")
        
        # Close all remote connections.
        for i, remote in enumerate(self.remotes):
            try:
                remote.close()
            except Exception as e:
                logging.warning(f"Error closing remote {i}: {e}")
        
        self.closed = True
        logging.info("All worker processes closed")
        return {"status": "closed"}

    def __del__(self):
        """Destructor."""
        if not self.closed:
            self.close()


class LiberoWebsocketServer:
    """WebSocket server for LIBERO environments."""
    
    def __init__(
        self,
        inference_server: LiberoInferenceServer,
        host: str = "0.0.0.0",
        port: int = 8003,
    ):
        """Initialize the WebSocket server.
        
        Args:
            inference_server: LIBERO inference server instance.
            host: Host address.
            port: Port number.
        """
        self.inference_server = inference_server
        self.host = host
        self.port = port
        self.metadata = {
            "server_type": "libero_env",
            "task_suite_name": inference_server.task_suite_name,
            "num_tasks": inference_server.num_tasks,
            "resolution": inference_server.resolution,
            "num_envs": inference_server.num_envs,
        }

    def _tensor_to_numpy(self, data):
        """Recursively convert tensors to NumPy arrays."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, dict):
            return {k: self._tensor_to_numpy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._tensor_to_numpy(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._tensor_to_numpy(item) for item in data)
        else:
            return data

    async def handle_client(self, websocket, path):
        """Handle a client connection."""
        logging.info(f"LIBERO client connected from {websocket.remote_address}")
        
        try:
            # Send metadata first.
            await websocket.send(msgpack_numpy.packb(self.metadata))
            
            # Process client messages.
            async for message in websocket:
                request = msgpack_numpy.unpackb(message)
                method = request.get("method")
                
                try:
                    if method == "reset":
                        # Read reset arguments.
                        sim_state = request.get("sim_state")
                        sim_state_len = request.get("sim_state_len")
                        task_id = request.get("task_id")
                        
                        # Reset all environments in batch.
                        obs = self.inference_server.reset(
                            sim_state=sim_state,
                            sim_state_len=sim_state_len,
                            task_id=task_id,
                        )
                        response = {
                            "observation": obs,
                        }
                            
                    elif method == "step":
                        # Step all environments in batch.
                        actions = request.get("actions")  # [num_envs, action_dim]
                        obs, rewards, terminations, truncations = self.inference_server.step(actions)
                        
                        # Convert tensors to NumPy.
                        response = {
                            "observation": obs,
                            "rewards": self._tensor_to_numpy(rewards),
                            "terminations": self._tensor_to_numpy(terminations),
                            "truncations": self._tensor_to_numpy(truncations),
                        }
                            
                    elif method == "chunk_step":
                        # Batch chunk step with shape [num_envs, chunk_size, action_dim].
                        chunk_actions = request.get("chunk_actions")
                        obs, rewards, terminations, truncations = self.inference_server.chunk_step(chunk_actions)
                        
                        # Convert tensors to NumPy.
                        response = {
                            "observation": obs,
                            "rewards": self._tensor_to_numpy(rewards),
                            "terminations": self._tensor_to_numpy(terminations),
                            "truncations": self._tensor_to_numpy(truncations),
                        }
                        
                    elif method == "close":
                        self.inference_server.close()
                        response = {"status": "closed"}
                        
                    else:
                        response = {"error": f"Unknown method: {method}"}
                    
                    # Make sure the response is NumPy-only.
                    response = self._tensor_to_numpy(response)
                    await websocket.send(msgpack_numpy.packb(response))
                    
                except Exception as e:
                    logging.error(f"Error handling request: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send(msgpack_numpy.packb({"error": str(e)}))
        
        except Exception as e:
            logging.error(f"Error handling client: {e}")
        finally:
            logging.info(f"LIBERO client {websocket.remote_address} disconnected")

    async def start_server(self):
        """Start the WebSocket server."""
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logging.info(f"Starting LIBERO server on {local_ip}:{self.port}")
        
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            close_timeout=150,
            open_timeout=150,  # Important: increase the handshake timeout.
            ping_interval=60,
            ping_timeout=120,
        ):
            logging.info("LIBERO server started successfully!")
            await asyncio.Future()  # Run forever

    def run(self):
        """Run the server."""
        asyncio.run(self.start_server())


def main():
    """Main entrypoint for the LIBERO WebSocket server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LIBERO WebSocket Server with Multiprocessing")
    parser.add_argument("--task-suite-name", type=str, default="libero_spatial", help="LIBERO task suite name")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--resolution", type=int, default=256, help="Camera resolution")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind to")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--start-method", type=str, default=None, choices=['spawn', 'forkserver'], 
                        help="Multiprocessing start method")
    parser.add_argument("--use-rel-reward", action="store_true", default=False,
                        help="Use relative rewards (difference between current and previous step rewards)")
    
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(level=logging.INFO)
    
    # Create the inference server.
    inference_server = LiberoInferenceServer(
        task_suite_name=args.task_suite_name,
        num_envs=args.num_envs,
        resolution=args.resolution,
        seed=args.seed,
        start_method=args.start_method,
        use_rel_reward=args.use_rel_reward,
    )
    
    # Create and run the WebSocket server.
    server = LiberoWebsocketServer(inference_server, host=args.host, port=args.port)
    
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
        inference_server.close()


if __name__ == "__main__":
    main()
