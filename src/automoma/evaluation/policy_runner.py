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
from automoma.utils.logging import logger


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
        dataset_id: str = None,
        dataset_root: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.policy_type = policy_type
        self.device = device
        self.dataset_id = dataset_id
        self.dataset_root = dataset_root
        self.policy = None
        self.preprocess = None
        self.postprocess = None
    
    def load_model(self) -> None:
        """Load the LeRobot policy model with preprocessing/postprocessing."""
        if not os.path.exists(self.checkpoint_path):
            print(f"Checkpoint not found: {self.checkpoint_path}")
            return
        
        try:
            from automoma.utils.file_utils import get_abs_path
            
            # Check if checkpoint is a directory (pretrained model format)
            if not os.path.isdir(self.checkpoint_path):
                raise ValueError(f"Checkpoint path must be a directory for pretrained models: {self.checkpoint_path}")
            
            print(f"Loading pretrained model from directory: {self.checkpoint_path}")
            
            # Load policy
            if self.policy_type == "act":
                from lerobot.policies.act.modeling_act import ACTPolicy
                self.policy = ACTPolicy.from_pretrained(self.checkpoint_path)
            elif self.policy_type == "diffusion":
                from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
                self.policy = DiffusionPolicy.from_pretrained(self.checkpoint_path)
            elif self.policy_type == "dp3":
                from lerobot.policies.dp3.modeling_dp3 import DP3Policy
                self.policy = DP3Policy.from_pretrained(self.checkpoint_path)
            elif self.policy_type == "pi0":
                from lerobot.policies.pi0.modeling_pi0 import PI0Policy
                self.policy = PI0Policy.from_pretrained(self.checkpoint_path)
            elif self.policy_type == "pi05":
                from lerobot.policies.pi05.modeling_pi05 import PI05Policy
                self.policy = PI05Policy.from_pretrained(self.checkpoint_path)
            else:
                raise ValueError(f"Unsupported policy type for pretrained loading: {self.policy_type}")
        
            if hasattr(self.policy, "eval"):
                self.policy.eval()
            if hasattr(self.policy, "to"):
                self.policy.to(self.device)
            
            print(f"✓ Model loaded from {self.checkpoint_path}")
            
            # Load metadata and create preprocessors if dataset info provided
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.policies.factory import make_pre_post_processors
            
            print(f"Loading dataset metadata for preprocessing from ID: {self.dataset_id}")
            print(f"Using dataset root: {self.dataset_root}")
            
            dataset_id_abs = get_abs_path(os.path.join("data", self.dataset_id))
            dataset_metadata = LeRobotDatasetMetadata(
                repo_id=dataset_id_abs, 
                root=self.dataset_root
            )
            
            self.preprocess, self.postprocess = make_pre_post_processors(
                self.policy.config, 
                self.checkpoint_path,
                dataset_stats=dataset_metadata.stats
            )
            print(f"✓ Loaded preprocessing/postprocessing from dataset: {self.dataset_id}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.policy = None
    
    def _flatten_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten observation dict to match LeRobot expected format."""
        flat_obs = {}
        
        # Map joint_data to observation.state
        if "joint_data" in observation:
            flat_obs["observation.state"] = observation["joint_data"]
            
        # Map eef_pose_data to observation.eef
        if "eef_pose_data" in observation:
            flat_obs["observation.eef"] = observation["eef_pose_data"]
            
        # Map images
        if "obs_data" in observation and "images" in observation["obs_data"]:
            for name, img in observation["obs_data"]["images"].items():
                # Transpose HWC -> CHW
                if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
                    img = img.transpose(2, 0, 1)
                    # Normalize to [0, 1] if in [0, 255]
                    if img.dtype == np.uint8:
                        img = img.astype(np.float32) / 255.0
                    elif img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                flat_obs[f"observation.images.{name}"] = img
                
        # Map depth
        if "obs_data" in observation and "depth" in observation["obs_data"]:
            for name, depth in observation["obs_data"]["depth"].items():
                # Add channel dim if needed
                if isinstance(depth, np.ndarray) and depth.ndim == 2:
                    depth = depth[np.newaxis, ...]
                flat_obs[f"observation.depth.{name}"] = depth
        
        # Map pointcloud
        if "obs_data" in observation and "pointcloud" in observation["obs_data"]:
            for name, pc in observation["obs_data"]["pointcloud"].items():
                # Map to observation.pointcloud (standard for DP3)
                # If multiple cameras, the last one will overwrite unless we have logic to merge
                # But usually there is one main sensor for PC
                flat_obs["observation.pointcloud"] = pc
                # Also keep named version for flexibility
                flat_obs[f"observation.pointcloud.{name}"] = pc
                
        return flat_obs
    
    def reset(self) -> None:
        """Reset any internal state if needed."""
        self.policy.reset()
    def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request using LeRobot policy."""
        start_time = time.time()
        
        observation = request.observation
        
        # Flatten observation if it comes from SimEnvWrapper (has obs_data)
        if "obs_data" in observation:
            observation = self._flatten_observation(observation)
        
        # DEBUG: import matplotlib.pyplot as plt; import numpy as np; img = np.transpose(observation['observation.images.ego_topdown'], (1,2,0)); plt.imshow(img); plt.axis('off'); plt.show()
        # DEBUG: import matplotlib.pyplot as plt; import numpy as np; pc = observation['observation.pointcloud.ego_topdown']; fig = plt.figure(); ax = fig.add_subplot(111, projection='3d'); ax.scatter(pc[:,0], pc[:,1], pc[:,2], c=pc[:,3:6]/np.max(pc[:,3:6]), s=1); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); plt.show()
        
        # Convert observation to tensor format expected by LeRobot
        obs_tensor = self._prepare_observation(observation)
        
        # Run inference with preprocessing/postprocessing
        with torch.no_grad():
            # Apply preprocessing (normalization) if configured
            obs_tensor = self.preprocess(obs_tensor)
            
            # Policy Inference
            action = self.policy.select_action(obs_tensor)
            
            # Apply postprocessing (denormalization) if configured
            action = self.postprocess(action)
        
        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze()
        
        # Log action stats occasionally
        if np.random.random() < 0.05:
            print(f"Action stats: mean={action.mean():.4f}, std={action.std():.4f}, min={action.min():.4f}, max={action.max():.4f}")
        
        inference_time = time.time() - start_time
        
        return InferenceResponse(
            request_id=request.request_id,
            action=action,
            inference_time=inference_time,
            success=True,
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
            
            # Add batch dimension
            # We assume input is a single observation, so we always add batch dim 0
            tensor = tensor.unsqueeze(0)
            
            obs_tensor[key] = tensor
        
        return obs_tensor

def get_model(
    checkpoint_path: str,
    policy_type: str = "diffusion",
    device: str = "cuda",
    async_mode: bool = True,
    dataset_id: str = None,
    dataset_root: str = None,
) -> LeRobotModelClient:
    """
    Factory function to get a LeRobot model client.
    
    Uses async communication mechanism for non-blocking inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        policy_type: Type of policy (diffusion, act, vq_bet)
        device: Device to run on
        async_mode: Whether to use async inference
        dataset_id: Dataset ID for preprocessing/postprocessing
        dataset_root: Dataset root path for preprocessing/postprocessing
        
    Returns:
        LeRobotModelClient instance
    """
    client = LeRobotModelClient(
        checkpoint_path=checkpoint_path,
        policy_type=policy_type,
        device=device,
        dataset_id=dataset_id,
        dataset_root=dataset_root,
    )
    client.load_model()
    
    if async_mode:
        client.start()
    
    return client
