#!/usr/bin/env python3
"""
Script 4: Evaluate trained policy model.

This script:
1. Loads the trained policy checkpoint
2. Sets up the evaluation environment
3. Runs policy inference with async communication
4. Computes evaluation metrics

Usage:
    python 4_evaluate.py --config configs/exps/multi_object_open/eval.yaml
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automoma.core.config import EvalConfig
from automoma.evaluation.policy_runner import PolicyRunner, get_model
from automoma.evaluation.metrics import EvaluationMetrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_eval_config(config: Dict[str, Any]) -> EvalConfig:
    """Create EvalConfig from dictionary."""
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    async_cfg = config.get("async_inference", {})
    output_cfg = config.get("output", {})
    hardware_cfg = config.get("hardware", {})
    
    return EvalConfig(
        checkpoint_path=model_cfg.get("checkpoint_path", ""),
        policy_type=model_cfg.get("policy_type", "diffusion"),
        num_episodes=eval_cfg.get("num_episodes", 50),
        max_steps_per_episode=eval_cfg.get("max_steps_per_episode", 500),
        success_threshold=eval_cfg.get("success_threshold", 0.05),
        use_async_inference=async_cfg.get("enabled", True),
        inference_host=async_cfg.get("host", "localhost"),
        inference_port=async_cfg.get("port", 50051),
        inference_timeout=async_cfg.get("timeout", 30.0),
        output_dir=output_cfg.get("output_dir", "outputs/eval"),
        save_videos=output_cfg.get("save_videos", True),
        save_trajectories=output_cfg.get("save_trajectories", True),
        device=hardware_cfg.get("device", "cuda"),
    )


def run_evaluation(
    eval_config: EvalConfig,
    env_config: Optional[Dict[str, Any]] = None,
) -> EvaluationMetrics:
    """
    Run policy evaluation.
    
    Args:
        eval_config: Evaluation configuration
        env_config: Optional environment configuration
        
    Returns:
        EvaluationMetrics object
    """
    print("="*80)
    print("POLICY EVALUATION")
    print("="*80)
    print(f"Checkpoint: {eval_config.checkpoint_path}")
    print(f"Policy type: {eval_config.policy_type}")
    print(f"Num episodes: {eval_config.num_episodes}")
    print(f"Async inference: {eval_config.use_async_inference}")
    print("="*80)
    
    # Create policy runner
    runner = PolicyRunner(eval_config)
    
    # Setup environment if configuration provided
    if env_config is not None:
        try:
            from automoma.simulation.env_wrapper import SimEnvWrapper
            env_wrapper = SimEnvWrapper(env_config)
            env_wrapper.setup_env()
            runner.setup_env(env_wrapper)
            print("Environment setup complete")
        except ImportError as e:
            print(f"Could not setup environment: {e}")
            print("Running evaluation without environment (metrics only)")
    
    # Run evaluation
    try:
        metrics = runner.run_eval(
            num_episodes=eval_config.num_episodes,
            max_steps=eval_config.max_steps_per_episode,
            save_videos=eval_config.save_videos,
        )
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        print(f"Success Rate: {metrics.success_rate:.2%}")
        print(f"Completion Rate: {metrics.completion_rate:.2%}")
        print(f"Position Error: {metrics.position_error_mean:.4f} ± {metrics.position_error_std:.4f}")
        print(f"Orientation Error: {metrics.orientation_error_mean:.4f} rad ± {metrics.orientation_error_std:.4f}")
        print(f"Joint Error: {metrics.joint_error_mean:.4f} ± {metrics.joint_error_std:.4f}")
        print(f"Inference Time: {metrics.inference_time_mean*1000:.1f} ms ± {metrics.inference_time_std*1000:.1f}")
        print(f"Total Episodes: {metrics.num_episodes}")
        print(f"Successful: {metrics.num_successful}, Failed: {metrics.num_failed}")
        print("="*80)
        
        return metrics
        
    finally:
        runner.cleanup()


def run_offline_evaluation(
    eval_config: EvalConfig,
    traj_dir: str,
) -> EvaluationMetrics:
    """
    Run offline evaluation on recorded trajectories.
    
    Args:
        eval_config: Evaluation configuration
        traj_dir: Directory containing trajectory data
        
    Returns:
        EvaluationMetrics object
    """
    import torch
    from automoma.evaluation.metrics import MetricsCalculator
    
    print("="*80)
    print("OFFLINE EVALUATION")
    print("="*80)
    print(f"Trajectory directory: {traj_dir}")
    print("="*80)
    
    # Load model
    print("Loading model...")
    model = get_model(
        checkpoint_path=eval_config.checkpoint_path,
        policy_type=eval_config.policy_type,
        device=eval_config.device,
        async_mode=eval_config.use_async_inference,
    )
    
    # Find trajectory files
    traj_path = Path(traj_dir)
    traj_files = list(traj_path.glob("**/filtered_traj_data.pt"))
    if not traj_files:
        traj_files = list(traj_path.glob("**/traj_data.pt"))
    
    print(f"Found {len(traj_files)} trajectory files")
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(success_threshold=eval_config.success_threshold)
    
    # Evaluate on trajectories
    from tqdm import tqdm
    
    for traj_file in tqdm(traj_files[:eval_config.num_episodes], desc="Evaluating"):
        try:
            # Note: weights_only=False is required for loading complex tensor dicts
            # Ensure trajectory files come from trusted sources
            traj_data = torch.load(traj_file, weights_only=True)
            trajectories = traj_data["trajectories"]
            success = traj_data["success"]
            
            # Get successful trajectories
            success_mask = success.bool()
            successful_trajs = trajectories[success_mask]
            
            for traj in successful_trajs[:5]:  # Limit per file
                # Create mock observation for model
                observation = {
                    "joint_positions": traj[0].numpy(),
                }
                
                # Get model prediction
                response = model.infer_sync(observation)
                
                if response.success:
                    # Add episode metrics
                    pred_traj = response.action.reshape(1, -1)
                    gt_traj = traj[0].numpy().reshape(1, -1)
                    
                    metrics_calc.add_episode(
                        pred_trajectory=pred_traj,
                        gt_trajectory=gt_traj,
                        inference_time=response.inference_time,
                        completed=True,
                    )
                    
        except Exception as e:
            print(f"Error evaluating {traj_file}: {e}")
    
    # Compute final metrics
    metrics = metrics_calc.compute_metrics()
    
    # Save results
    output_dir = Path(eval_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "offline_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("OFFLINE EVALUATION RESULTS")
    print("="*80)
    print(f"Episodes evaluated: {metrics.num_episodes}")
    print(f"Success Rate: {metrics.success_rate:.2%}")
    print(f"Inference Time: {metrics.inference_time_mean*1000:.1f} ms")
    print("="*80)
    
    # Cleanup
    model.stop()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exps/multi_object_open/eval.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes (overrides config)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run offline evaluation on trajectories",
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="data/run_1223/traj",
        help="Trajectory directory for offline evaluation",
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {
            "model": {"checkpoint_path": "", "policy_type": "diffusion"},
            "evaluation": {"num_episodes": 50, "max_steps_per_episode": 500, "success_threshold": 0.05},
            "async_inference": {"enabled": True, "host": "localhost", "port": 50051, "timeout": 30.0},
            "output": {"output_dir": "outputs/eval", "save_videos": True, "save_trajectories": True},
            "hardware": {"device": "cuda"},
        }
    
    # Override with command line arguments
    if args.checkpoint:
        config.setdefault("model", {})["checkpoint_path"] = args.checkpoint
    if args.output_dir:
        config.setdefault("output", {})["output_dir"] = args.output_dir
    if args.num_episodes:
        config.setdefault("evaluation", {})["num_episodes"] = args.num_episodes
    
    # Create eval config
    eval_config = create_eval_config(config)
    
    # Check checkpoint exists
    if eval_config.checkpoint_path and not os.path.exists(eval_config.checkpoint_path):
        print(f"Warning: Checkpoint not found at {eval_config.checkpoint_path}")
        if not args.offline:
            print("Consider using --offline mode for trajectory evaluation")
    
    # Run evaluation
    if args.offline:
        traj_dir = str(PROJECT_ROOT / args.traj_dir)
        metrics = run_offline_evaluation(eval_config, traj_dir)
    else:
        env_config = config.get("env_cfg")
        metrics = run_evaluation(eval_config, env_config)
    
    print("\nEvaluation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
