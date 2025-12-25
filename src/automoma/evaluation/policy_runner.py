"""Policy runner for model evaluation with async LeRobot communication."""

import os
import time
import threading
import queue
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import torch

from automoma.core.config import EvalConfig
from automoma.evaluation.metrics import MetricsCalculator, EvaluationMetrics


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: int
    observation: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: int
    action: np.ndarray
    inference_time: float
    success: bool = True
    error_message: str = ""


class AsyncModelClient:
    """
    Async client for communicating with LeRobot model server.
    
    Uses a request-response pattern with async queues for non-blocking inference.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self._request_queue = queue.Queue()
        self._request_id = 0
        self._running = False
        self._worker_thread = None
        self._lock = threading.Lock()
        
        self._pending_requests: Dict[int, InferenceRequest] = {}
        self._responses: Dict[int, InferenceResponse] = {}
        self._response_events: Dict[int, threading.Event] = {}
    
    def start(self) -> None:
        """Start the async client worker thread."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print(f"AsyncModelClient started on {self.host}:{self.port}")
    
    def stop(self) -> None:
        """Stop the async client."""
        self._running = False
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        print("AsyncModelClient stopped")
    
    def _worker_loop(self) -> None:
        """Worker loop for processing inference requests."""
        while self._running:
            try:
                # Get request from queue with timeout
                try:
                    request = self._request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the request
                response = self._process_request(request)
                
                # Store response and signal event (thread-safe)
                with self._lock:
                    self._responses[response.request_id] = response
                    event = self._response_events.get(response.request_id)
                
                # Signal outside lock to avoid holding lock while event triggers
                if event is not None:
                    event.set()
                
            except Exception as e:
                print(f"Error in worker loop: {e}")
    
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """
        Process an inference request.
        
        This is where the actual model inference happens.
        Override this method for different model backends.
        """
        start_time = time.time()
        
        try:
            # Default implementation - should be overridden
            # For now, return a dummy action
            observation = request.observation
            
            # Get action dimension from observation if available
            if "joint_positions" in observation and observation["joint_positions"] is not None:
                action_dim = len(observation["joint_positions"])
            else:
                action_dim = 10  # Default
            
            # Dummy action (should be replaced with actual model inference)
            action = np.zeros(action_dim)
            
            inference_time = time.time() - start_time
            
            return InferenceResponse(
                request_id=request.request_id,
                action=action,
                inference_time=inference_time,
                success=True,
            )
            
        except Exception as e:
            return InferenceResponse(
                request_id=request.request_id,
                action=np.array([]),
                inference_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )
    
    def submit_request(self, observation: Dict[str, Any]) -> int:
        """
        Submit an async inference request.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Request ID for tracking
        """
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
        
        request = InferenceRequest(
            request_id=request_id,
            observation=observation,
        )
        
        # Create event for this request
        event = threading.Event()
        with self._lock:
            self._pending_requests[request_id] = request
            self._response_events[request_id] = event
        
        self._request_queue.put(request)
        return request_id
    
    def get_response(self, request_id: int, timeout: float = None) -> Optional[InferenceResponse]:
        """
        Get response for a specific request ID using event-driven mechanism.
        
        Args:
            request_id: The request ID to get response for
            timeout: Timeout in seconds
            
        Returns:
            InferenceResponse or None if timeout
        """
        if timeout is None:
            timeout = self.timeout
        
        # Get the event for this request
        with self._lock:
            event = self._response_events.get(request_id)
        
        if event is None:
            return None
        
        # Wait for the response
        if event.wait(timeout=timeout):
            # Response is ready
            with self._lock:
                response = self._responses.pop(request_id, None)
                self._pending_requests.pop(request_id, None)
                self._response_events.pop(request_id, None)
            return response
        
        # Timeout - cleanup
        with self._lock:
            self._pending_requests.pop(request_id, None)
            self._response_events.pop(request_id, None)
        
        return None
    
    def infer_sync(self, observation: Dict[str, Any]) -> InferenceResponse:
        """
        Synchronous inference (blocking).
        
        Args:
            observation: Observation dictionary
            
        Returns:
            InferenceResponse
        """
        request_id = self.submit_request(observation)
        response = self.get_response(request_id)
        
        if response is None:
            return InferenceResponse(
                request_id=request_id,
                action=np.array([]),
                inference_time=self.timeout,
                success=False,
                error_message="Inference timeout",
            )
        
        return response


class LeRobotModelClient(AsyncModelClient):
    """
    Model client specifically for LeRobot policies.
    
    Supports loading LeRobot checkpoints and running inference.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        policy_type: str = "diffusion",
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.policy_type = policy_type
        self.device = device
        self.policy = None
    
    def load_model(self) -> None:
        """Load the LeRobot policy model."""
        if not os.path.exists(self.checkpoint_path):
            print(f"Checkpoint not found: {self.checkpoint_path}")
            return
        
        try:
            # Import LeRobot components
            from lerobot.common.policies.factory import make_policy
            from omegaconf import OmegaConf
            
            # Load checkpoint
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Create policy from config
            if "config" in checkpoint:
                config = OmegaConf.create(checkpoint["config"])
                self.policy = make_policy(config)
                self.policy.load_state_dict(checkpoint["state_dict"])
            else:
                # Try to load as raw state dict
                print("Loading checkpoint as raw state dict")
                self.policy = checkpoint
            
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            if hasattr(self.policy, "to"):
                self.policy.to(self.device)
            
            print(f"Model loaded from {self.checkpoint_path}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.policy = None
    
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request using LeRobot policy."""
        start_time = time.time()
        
        if self.policy is None:
            return InferenceResponse(
                request_id=request.request_id,
                action=np.array([]),
                inference_time=time.time() - start_time,
                success=False,
                error_message="Model not loaded",
            )
        
        try:
            observation = request.observation
            
            # Convert observation to tensor format expected by LeRobot
            obs_tensor = self._prepare_observation(observation)
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.policy, "select_action"):
                    action = self.policy.select_action(obs_tensor)
                elif hasattr(self.policy, "forward"):
                    action = self.policy(obs_tensor)
                else:
                    raise ValueError("Policy has no inference method")
            
            # Convert action to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if action.ndim > 1:
                action = action.squeeze()
            
            inference_time = time.time() - start_time
            
            return InferenceResponse(
                request_id=request.request_id,
                action=action,
                inference_time=inference_time,
                success=True,
            )
            
        except Exception as e:
            return InferenceResponse(
                request_id=request.request_id,
                action=np.array([]),
                inference_time=time.time() - start_time,
                success=False,
                error_message=str(e),
            )
    
    def _prepare_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert observation dict to tensor dict for LeRobot."""
        obs_tensor = {}
        
        for key, value in observation.items():
            if value is None:
                continue
            
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float().to(self.device)
            elif isinstance(value, torch.Tensor):
                tensor = value.float().to(self.device)
            elif isinstance(value, (list, tuple)):
                tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
            elif isinstance(value, dict):
                # Recursively process nested dicts (e.g., images)
                obs_tensor[key] = self._prepare_observation(value)
                continue
            else:
                continue
            
            # Add batch dimension if needed
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            
            obs_tensor[key] = tensor
        
        return obs_tensor


class PolicyRunner:
    """
    Policy runner for evaluating trained models.
    
    Supports both synchronous and asynchronous inference modes.
    """
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.model_client = None
        self.env = None
        self.metrics_calculator = MetricsCalculator(
            success_threshold=cfg.success_threshold
        )
    
    def setup_env(self, env_wrapper=None) -> None:
        """Setup the evaluation environment."""
        self.env = env_wrapper
    
    def get_policy(self) -> LeRobotModelClient:
        """Get or create the policy model client."""
        if self.model_client is None:
            self.model_client = LeRobotModelClient(
                checkpoint_path=self.cfg.checkpoint_path,
                policy_type=self.cfg.policy_type,
                device=self.cfg.device,
                host=self.cfg.inference_host,
                port=self.cfg.inference_port,
                timeout=self.cfg.inference_timeout,
            )
            self.model_client.load_model()
            
            if self.cfg.use_async_inference:
                self.model_client.start()
        
        return self.model_client
    
    def load_dataset(self, dataset_path: str = None) -> None:
        """Load evaluation dataset if needed."""
        pass
    
    def run_infer(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Run single inference step.
        
        Args:
            observation: Current observation
            
        Returns:
            Action array
        """
        policy = self.get_policy()
        response = policy.infer_sync(observation)
        
        if not response.success:
            print(f"Inference failed: {response.error_message}")
            return np.zeros(10)  # Return dummy action
        
        return response.action
    
    def run_eval(
        self,
        num_episodes: int = None,
        max_steps: int = None,
        save_videos: bool = None,
    ) -> EvaluationMetrics:
        """
        Run full evaluation loop.
        
        Args:
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            save_videos: Whether to save evaluation videos
            
        Returns:
            EvaluationMetrics object
        """
        if num_episodes is None:
            num_episodes = self.cfg.num_episodes
        if max_steps is None:
            max_steps = self.cfg.max_steps_per_episode
        if save_videos is None:
            save_videos = self.cfg.save_videos
        
        self.metrics_calculator.reset()
        
        for episode_idx in range(num_episodes):
            print(f"Evaluating episode {episode_idx + 1}/{num_episodes}")
            
            episode_metrics = self._run_episode(
                episode_idx=episode_idx,
                max_steps=max_steps,
                save_video=save_videos,
            )
            
            print(f"  Episode {episode_idx + 1}: Success={episode_metrics.get('success', False)}")
        
        # Compute aggregate metrics
        metrics = self.metrics_calculator.compute_metrics()
        
        # Save results
        self._save_results(metrics)
        
        return metrics
    
    def _run_episode(
        self,
        episode_idx: int,
        max_steps: int,
        save_video: bool = False,
    ) -> Dict[str, float]:
        """Run a single evaluation episode."""
        if self.env is None:
            print("Environment not set up")
            return {"success": False}
        
        # Reset environment
        observation = self._reset_environment()
        
        pred_trajectory = []
        gt_trajectory = []
        pred_positions = []
        gt_positions = []
        pred_orientations = []
        gt_orientations = []
        inference_times = []
        
        goal_position = None  # Would be set from task
        
        for step in range(max_steps):
            # Get action from policy
            start_time = time.time()
            action = self.run_infer(observation)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Store trajectory data
            if "joint_positions" in observation and observation["joint_positions"] is not None:
                pred_trajectory.append(action)
                gt_trajectory.append(observation["joint_positions"])
            
            if "eef_position" in observation and observation["eef_position"] is not None:
                pred_positions.append(observation["eef_position"])
            
            if "eef_orientation" in observation and observation["eef_orientation"] is not None:
                pred_orientations.append(observation["eef_orientation"])
            
            # Execute action in environment
            observation, done = self._step_environment(action)
            
            if done:
                break
        
        # Compute episode metrics
        episode_metrics = {}
        
        if pred_trajectory and gt_trajectory:
            pred_traj_np = np.array(pred_trajectory)
            gt_traj_np = np.array(gt_trajectory)
            
            episode_metrics = self.metrics_calculator.add_episode(
                pred_trajectory=pred_traj_np,
                gt_trajectory=gt_traj_np,
                pred_positions=np.array(pred_positions) if pred_positions else None,
                gt_positions=np.array(gt_positions) if gt_positions else None,
                pred_orientations=np.array(pred_orientations) if pred_orientations else None,
                gt_orientations=np.array(gt_orientations) if gt_orientations else None,
                goal_position=goal_position,
                inference_time=np.mean(inference_times) if inference_times else None,
                completed=True,
            )
        
        return episode_metrics
    
    def _reset_environment(self) -> Dict[str, Any]:
        """Reset the environment and return initial observation."""
        if self.env is not None and hasattr(self.env, "reset"):
            return self.env.reset()
        return {}
    
    def _step_environment(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool]:
        """Step the environment with action."""
        if self.env is not None and hasattr(self.env, "step"):
            obs, reward, done, info = self.env.step(action)
            return obs, done
        return {}, True
    
    def _save_results(self, metrics: EvaluationMetrics) -> None:
        """Save evaluation results."""
        import json
        
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        results_path = os.path.join(self.cfg.output_dir, "eval_results.json")
        with open(results_path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"Results saved to {results_path}")
        print(f"Success rate: {metrics.success_rate:.2%}")
        print(f"Position error: {metrics.position_error_mean:.4f} +/- {metrics.position_error_std:.4f}")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.model_client is not None:
            self.model_client.stop()


def get_model(
    checkpoint_path: str,
    policy_type: str = "diffusion",
    device: str = "cuda",
    async_mode: bool = True,
) -> LeRobotModelClient:
    """
    Factory function to get a LeRobot model client.
    
    Uses async communication mechanism for non-blocking inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        policy_type: Type of policy (diffusion, act, vq_bet)
        device: Device to run on
        async_mode: Whether to use async inference
        
    Returns:
        LeRobotModelClient instance
    """
    client = LeRobotModelClient(
        checkpoint_path=checkpoint_path,
        policy_type=policy_type,
        device=device,
    )
    client.load_model()
    
    if async_mode:
        client.start()
    
    return client
