#!/usr/bin/env python3
"""
Script 4: Evaluate trained policy.

This script evaluates a trained policy:
1. Load evaluation configuration
2. Initialize SimulationApp (required for Isaac Sim)
3. Load trained model with async LeRobot communication
4. Initialize robot to test states (from trajectory data)
5. Run policy inference and compute metrics

Usage:
    python 4_evaluate.py --exp multi_object_open
    python 4_evaluate.py --exp multi_object_open --headless
"""

import os
import sys
print(f"PYTHON EXECUTABLE: {sys.executable}")
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# --- CRITICAL INITIALIZATION ---
# Isaac Sim's SimulationApp MUST be initialized before any other imports that might touch pxr or omni.
# We do a quick parse of sys.argv to get the headless flag before importing anything else.
import argparse
temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument("--headless", action="store_true")
temp_args, _ = temp_parser.parse_known_args()

from automoma.utils.sim_utils import get_simulation_app
sim_app = get_simulation_app(headless=temp_args.headless)
# -------------------------------

# Now safe to import other modules
from automoma.core.config_loader import load_config, Config
from automoma.tasks.factory import create_task
from automoma.evaluation.policy_runner import get_model
from automoma.evaluation.metrics import MetricsCalculator


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(cfg: Config, headless: bool = False, checkpoint_path: str = None, dry_run: bool = False):
    """
    Run evaluation pipeline.
    
    For task evaluation:
    - Load test data to get initial states
    - Initialize robot to task-specific start state
      (e.g., for open task, robot starts grasping handle)
    - Run policy inference
    - Compute metrics
    
    Args:
        cfg: Configuration object
        headless: Whether to run in headless mode
        checkpoint_path: Optional override for checkpoint path
        dry_run: If True, skip actual evaluation
    """
    # Create task
    task = create_task(cfg)
    
    # Get eval config
    eval_cfg = cfg.eval_cfg if cfg.eval_cfg else cfg.evaluation
    
    # Checkpoint path from policy_cfg or override
    if checkpoint_path is None:
        # Try to get from policy_cfg based on policy type
        policy_type = eval_cfg.policy_type if hasattr(eval_cfg, 'policy_type') else "diffusion"
        if cfg.policy_cfg and hasattr(cfg.policy_cfg, policy_type):
            policy_config = getattr(cfg.policy_cfg, policy_type)
            checkpoint_path = policy_config.checkpoint_path if hasattr(policy_config, 'checkpoint_path') else None
    
    if checkpoint_path is None:
        # Default path
        checkpoint_path = f"outputs/train/{cfg.info_cfg.task}/model_final.pt"
    
    checkpoint_path = str(PROJECT_ROOT / checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        if not dry_run:
            logger.error("Cannot run evaluation without checkpoint")
            return
    
    # Evaluation parameters
    num_episodes = eval_cfg.num_episodes if eval_cfg else 50
    max_steps = eval_cfg.max_steps_per_episode if eval_cfg else 500
    success_threshold = eval_cfg.success_threshold if eval_cfg else 0.05
    use_async = eval_cfg.use_async_inference if eval_cfg else True
    
    # Get policy type from eval_cfg or policy_cfg
    policy_type = "diffusion"  # default
    if hasattr(eval_cfg, 'policy_type'):
        policy_type = eval_cfg.policy_type
    elif cfg.policy_cfg:
        # Infer from available policies in policy_cfg
        if hasattr(cfg.policy_cfg, 'act'):
            policy_type = "act"
        elif hasattr(cfg.policy_cfg, 'diffusion'):
            policy_type = "diffusion"
    
    logger.info(f"Evaluation configuration:")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Max steps: {max_steps}")
    logger.info(f"  Success threshold: {success_threshold}")
    logger.info(f"  Async inference: {use_async}")
    
    # Setup environment
    if not dry_run:
        try:
            # SimulationApp is already initialized at the top of the script
            
            # Now safe to import SimEnvWrapper
            from automoma.simulation.env_wrapper import SimEnvWrapper
            
            env = SimEnvWrapper(cfg)
            env.setup_env()
            task.setup_env(env)
            logger.info("Environment setup complete")
        except Exception as e:
            logger.warning(f"Could not setup environment: {e}")
            dry_run = True
    
    # Get test data directory from eval_cfg.data_cfg
    if eval_cfg and eval_cfg.data_cfg and eval_cfg.data_cfg.test_data_dir:
        test_data_dir = eval_cfg.data_cfg.test_data_dir
    else:
        test_data_dir = "data/output/traj"
    test_data_dir = str(PROJECT_ROOT / test_data_dir)
    
    logger.info(f"Loading test data from: {test_data_dir}")
    
    # Get initial states for evaluation
    # For different tasks, the initial state is different:
    # - open task: robot already grasping handle at start angle
    # - pick task: robot at home position
    initial_states = task.get_test_initial_states(test_data_dir)
    logger.info(f"Found {len(initial_states)} test initial states")
    
    if not initial_states:
        logger.warning("No test initial states found")
        return
    
    if dry_run:
        logger.info("Dry run - skipping actual evaluation")
        return
    
    # Load model with async communication
    logger.info("Loading model...")
    policy_type = eval_cfg.policy_type if eval_cfg else "diffusion"
    
    model = get_model(
        checkpoint_path=checkpoint_path,
        policy_type=policy_type,
        async_mode=use_async,
        device=eval_cfg.device if eval_cfg else "cuda",
    )
    model.start()
    
    # Metrics calculator with config from eval_cfg.metrics_cfg
    metrics_config = {}
    if eval_cfg and eval_cfg.metrics_cfg:
        metrics_config = eval_cfg.metrics_cfg.to_dict() if hasattr(eval_cfg.metrics_cfg, 'to_dict') else eval_cfg.metrics_cfg
    
    metrics_calc = MetricsCalculator(
        success_threshold=success_threshold,
        **metrics_config
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    
    for ep_idx in range(min(num_episodes, len(initial_states))):
        initial_state = initial_states[ep_idx % len(initial_states)]
        
        logger.info(f"\nEpisode {ep_idx + 1}/{num_episodes}")
        
        # Reset environment with initial state
        task.env.reset(initial_state)
        
        episode_traj = []
        episode_success = False
        
        for step in range(max_steps):
            # Get observation
            obs = task.env.get_data()
            
            # Inference
            response = model.infer_sync(obs)
            
            if not response.success:
                logger.warning(f"  Inference failed at step {step}: {response.error_message}")
                break
            
            action = response.action
            
            # Execute action
            task.env.step(action)
            episode_traj.append(action)
            
            # Check task completion
            obs_after = task.env.get_data()
            
            # Debug: print object angle
            env_state = obs_after.get("env_state", 0.0)
            if step % 10 == 0:
                logger.info(f"  Step {step}: Object Angle = {env_state:.4f}")
            
            if task._check_task_complete(obs_after):
                episode_success = True
                logger.info(f"  Task completed at step {step + 1}")
                break
        
        # Record metrics
        metrics_calc.add_episode(
            pred_trajectory=episode_traj,
            gt_trajectory=episode_traj,
            completed=episode_success,
        )
        
        logger.info(f"  Steps: {len(episode_traj)}, Success: {episode_success}")
    
    # Stop model
    model.stop()
    
    # Compute and display metrics
    metrics = metrics_calc.compute_metrics()
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Success rate: {metrics.success_rate*100:.1f}%")
    logger.info(f"Completion rate: {metrics.completion_rate*100:.1f}%")
    logger.info(f"Avg inference time: {metrics.inference_time_mean*1000:.1f}ms")
    
    # Save results
    output_dir = eval_cfg.output_dir if eval_cfg else "outputs/eval"
    output_dir = Path(PROJECT_ROOT / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"{cfg.info_cfg.task}_results.json"
    
    import json
    with open(results_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path override",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without simulation",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default=None,
        help="Policy type override (act or diffusion)",
    )
    parser.add_argument(
        "--initial-state-path",
        type=str,
        default=None,
        help="Path to initial state file (start_iks.pt)",
    )
    args = parser.parse_args()
    
    # Load evaluation configuration
    from automoma.core.config_loader import load_eval_config
    cfg = load_eval_config(args.exp, PROJECT_ROOT)
    
    # Override config with args
    eval_cfg = cfg.eval_cfg if cfg.eval_cfg else cfg.evaluation
    if args.policy_type:
        if hasattr(eval_cfg, 'policy_type'):
            eval_cfg.policy_type = args.policy_type
        else:
            # If using DictConfig or similar that allows adding keys
            try:
                eval_cfg['policy_type'] = args.policy_type
            except:
                setattr(eval_cfg, 'policy_type', args.policy_type)
                
    if args.initial_state_path:
        if hasattr(eval_cfg, 'initial_state_path'):
            eval_cfg.initial_state_path = args.initial_state_path
        else:
            try:
                eval_cfg['initial_state_path'] = args.initial_state_path
            except:
                setattr(eval_cfg, 'initial_state_path', args.initial_state_path)
    
    # Run evaluation
    run_evaluation(cfg, args.headless, args.checkpoint, args.dry_run)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
