"""Evaluation module for AutoMoMa framework."""

from automoma.evaluation.metrics import (
    EvaluationMetrics,
    MetricsCalculator,
    compute_position_error,
    compute_orientation_error,
    compute_trajectory_smoothness,
    compute_trajectory_length,
    compute_joint_error,
    compute_success,
)
from automoma.evaluation.policy_runner import (
    AsyncModelClient,
    LeRobotModelClient,
    InferenceRequest,
    InferenceResponse,
    get_model,
)

__all__ = [
    # Metrics
    "EvaluationMetrics",
    "MetricsCalculator",
    "compute_position_error",
    "compute_orientation_error",
    "compute_trajectory_smoothness",
    "compute_trajectory_length",
    "compute_joint_error",
    "compute_success",
    # Policy Runner
    "AsyncModelClient",
    "LeRobotModelClient",
    "InferenceRequest",
    "InferenceResponse",
    "get_model",
]
