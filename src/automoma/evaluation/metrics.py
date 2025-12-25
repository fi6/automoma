"""Evaluation metrics for model evaluation."""

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    # Success metrics
    success_rate: float = 0.0
    completion_rate: float = 0.0
    
    # Position metrics
    position_error_mean: float = 0.0
    position_error_std: float = 0.0
    position_error_max: float = 0.0
    
    # Orientation metrics
    orientation_error_mean: float = 0.0
    orientation_error_std: float = 0.0
    orientation_error_max: float = 0.0
    
    # Trajectory metrics
    trajectory_length_mean: float = 0.0
    trajectory_length_std: float = 0.0
    smoothness_mean: float = 0.0
    
    # Joint metrics
    joint_error_mean: float = 0.0
    joint_error_std: float = 0.0
    
    # Timing metrics
    inference_time_mean: float = 0.0
    inference_time_std: float = 0.0
    
    # Episode counts
    num_episodes: int = 0
    num_successful: int = 0
    num_failed: int = 0
    
    # Per-episode results
    episode_successes: List[bool] = field(default_factory=list)
    episode_position_errors: List[float] = field(default_factory=list)
    episode_orientation_errors: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "position_error": {
                "mean": self.position_error_mean,
                "std": self.position_error_std,
                "max": self.position_error_max,
            },
            "orientation_error": {
                "mean": self.orientation_error_mean,
                "std": self.orientation_error_std,
                "max": self.orientation_error_max,
            },
            "trajectory": {
                "length_mean": self.trajectory_length_mean,
                "length_std": self.trajectory_length_std,
                "smoothness_mean": self.smoothness_mean,
            },
            "joint_error": {
                "mean": self.joint_error_mean,
                "std": self.joint_error_std,
            },
            "inference_time": {
                "mean": self.inference_time_mean,
                "std": self.inference_time_std,
            },
            "counts": {
                "total": self.num_episodes,
                "successful": self.num_successful,
                "failed": self.num_failed,
            },
        }


def compute_position_error(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute position error between predicted and ground truth positions.
    
    Args:
        pred_positions: Predicted positions (N, 3)
        gt_positions: Ground truth positions (N, 3)
        
    Returns:
        Tuple of (mean, std, max) position errors
    """
    errors = np.linalg.norm(pred_positions - gt_positions, axis=-1)
    return float(np.mean(errors)), float(np.std(errors)), float(np.max(errors))


def compute_orientation_error(
    pred_quats: np.ndarray,
    gt_quats: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute orientation error between predicted and ground truth quaternions.
    Uses geodesic distance on SO(3).
    
    Args:
        pred_quats: Predicted quaternions (N, 4) [qw, qx, qy, qz]
        gt_quats: Ground truth quaternions (N, 4) [qw, qx, qy, qz]
        
    Returns:
        Tuple of (mean, std, max) orientation errors in radians
    """
    # Normalize quaternions
    pred_quats = pred_quats / (np.linalg.norm(pred_quats, axis=-1, keepdims=True) + 1e-8)
    gt_quats = gt_quats / (np.linalg.norm(gt_quats, axis=-1, keepdims=True) + 1e-8)
    
    # Compute inner product
    dot_products = np.abs(np.sum(pred_quats * gt_quats, axis=-1))
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Geodesic distance
    errors = 2.0 * np.arccos(dot_products)
    
    return float(np.mean(errors)), float(np.std(errors)), float(np.max(errors))


def compute_trajectory_smoothness(trajectory: np.ndarray) -> float:
    """
    Compute trajectory smoothness using second derivative.
    
    Args:
        trajectory: Trajectory array (T, D)
        
    Returns:
        Smoothness metric (lower is smoother)
    """
    if len(trajectory) < 3:
        return 0.0
    
    # Compute second derivative (acceleration)
    velocity = np.diff(trajectory, axis=0)
    acceleration = np.diff(velocity, axis=0)
    
    # Sum of squared accelerations
    smoothness = np.mean(np.sum(acceleration ** 2, axis=-1))
    
    return float(smoothness)


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """
    Compute total trajectory length.
    
    Args:
        trajectory: Trajectory array (T, D)
        
    Returns:
        Total path length
    """
    if len(trajectory) < 2:
        return 0.0
    
    diffs = np.diff(trajectory, axis=0)
    lengths = np.linalg.norm(diffs, axis=-1)
    
    return float(np.sum(lengths))


def compute_joint_error(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute joint angle error.
    
    Args:
        pred_joints: Predicted joint angles (N, num_joints)
        gt_joints: Ground truth joint angles (N, num_joints)
        
    Returns:
        Tuple of (mean, std) joint errors
    """
    errors = np.abs(pred_joints - gt_joints)
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    
    return mean_error, std_error


def compute_success(
    final_position: np.ndarray,
    goal_position: np.ndarray,
    success_threshold: float = 0.05,
) -> bool:
    """
    Determine if episode was successful based on final position.
    
    Args:
        final_position: Final end-effector position (3,)
        goal_position: Goal position (3,)
        success_threshold: Distance threshold for success
        
    Returns:
        True if successful
    """
    distance = np.linalg.norm(final_position - goal_position)
    return distance <= success_threshold


class MetricsCalculator:
    """Calculate comprehensive evaluation metrics."""
    
    def __init__(self, success_threshold: float = 0.05):
        self.success_threshold = success_threshold
        self.reset()
    
    def reset(self) -> None:
        """Reset all collected data."""
        self.position_errors = []
        self.orientation_errors = []
        self.joint_errors = []
        self.trajectory_lengths = []
        self.smoothness_values = []
        self.inference_times = []
        self.successes = []
        self.completions = []
    
    def add_episode(
        self,
        pred_trajectory: np.ndarray,
        gt_trajectory: np.ndarray,
        pred_positions: Optional[np.ndarray] = None,
        gt_positions: Optional[np.ndarray] = None,
        pred_orientations: Optional[np.ndarray] = None,
        gt_orientations: Optional[np.ndarray] = None,
        goal_position: Optional[np.ndarray] = None,
        inference_time: Optional[float] = None,
        completed: bool = True,
    ) -> Dict[str, float]:
        """
        Add an episode's data for metrics calculation.
        
        Args:
            pred_trajectory: Predicted joint trajectory (T, num_joints)
            gt_trajectory: Ground truth joint trajectory (T, num_joints)
            pred_positions: Predicted end-effector positions (T, 3)
            gt_positions: Ground truth end-effector positions (T, 3)
            pred_orientations: Predicted orientations (T, 4) [qw, qx, qy, qz]
            gt_orientations: Ground truth orientations (T, 4)
            goal_position: Goal position for success evaluation
            inference_time: Time for inference
            completed: Whether episode completed
            
        Returns:
            Dictionary of per-episode metrics
        """
        episode_metrics = {}
        
        # Joint error
        joint_mean, joint_std = compute_joint_error(pred_trajectory, gt_trajectory)
        self.joint_errors.append(joint_mean)
        episode_metrics["joint_error"] = joint_mean
        
        # Trajectory metrics
        traj_length = compute_trajectory_length(pred_trajectory)
        self.trajectory_lengths.append(traj_length)
        episode_metrics["trajectory_length"] = traj_length
        
        smoothness = compute_trajectory_smoothness(pred_trajectory)
        self.smoothness_values.append(smoothness)
        episode_metrics["smoothness"] = smoothness
        
        # Position error
        if pred_positions is not None and gt_positions is not None:
            pos_mean, pos_std, pos_max = compute_position_error(pred_positions, gt_positions)
            self.position_errors.append(pos_mean)
            episode_metrics["position_error"] = pos_mean
        
        # Orientation error
        if pred_orientations is not None and gt_orientations is not None:
            ori_mean, ori_std, ori_max = compute_orientation_error(pred_orientations, gt_orientations)
            self.orientation_errors.append(ori_mean)
            episode_metrics["orientation_error"] = ori_mean
        
        # Success evaluation
        if goal_position is not None and pred_positions is not None:
            final_pos = pred_positions[-1]
            success = compute_success(final_pos, goal_position, self.success_threshold)
        else:
            success = completed  # Fallback to completion status
        
        self.successes.append(success)
        self.completions.append(completed)
        episode_metrics["success"] = success
        episode_metrics["completed"] = completed
        
        # Inference time
        if inference_time is not None:
            self.inference_times.append(inference_time)
            episode_metrics["inference_time"] = inference_time
        
        return episode_metrics
    
    def compute_metrics(self) -> EvaluationMetrics:
        """
        Compute aggregate metrics from all collected episodes.
        
        Returns:
            EvaluationMetrics object
        """
        metrics = EvaluationMetrics()
        
        num_episodes = len(self.successes)
        if num_episodes == 0:
            return metrics
        
        metrics.num_episodes = num_episodes
        metrics.num_successful = sum(self.successes)
        metrics.num_failed = num_episodes - metrics.num_successful
        
        metrics.success_rate = metrics.num_successful / num_episodes
        metrics.completion_rate = sum(self.completions) / num_episodes
        
        # Position metrics
        if self.position_errors:
            metrics.position_error_mean = float(np.mean(self.position_errors))
            metrics.position_error_std = float(np.std(self.position_errors))
            metrics.position_error_max = float(np.max(self.position_errors))
        
        # Orientation metrics
        if self.orientation_errors:
            metrics.orientation_error_mean = float(np.mean(self.orientation_errors))
            metrics.orientation_error_std = float(np.std(self.orientation_errors))
            metrics.orientation_error_max = float(np.max(self.orientation_errors))
        
        # Trajectory metrics
        if self.trajectory_lengths:
            metrics.trajectory_length_mean = float(np.mean(self.trajectory_lengths))
            metrics.trajectory_length_std = float(np.std(self.trajectory_lengths))
        
        if self.smoothness_values:
            metrics.smoothness_mean = float(np.mean(self.smoothness_values))
        
        # Joint metrics
        if self.joint_errors:
            metrics.joint_error_mean = float(np.mean(self.joint_errors))
            metrics.joint_error_std = float(np.std(self.joint_errors))
        
        # Timing metrics
        if self.inference_times:
            metrics.inference_time_mean = float(np.mean(self.inference_times))
            metrics.inference_time_std = float(np.std(self.inference_times))
        
        # Store per-episode data
        metrics.episode_successes = list(self.successes)
        metrics.episode_position_errors = list(self.position_errors)
        metrics.episode_orientation_errors = list(self.orientation_errors)
        
        return metrics
