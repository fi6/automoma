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
    python 4_evaluate.py --policy-type act \
        --checkpoint-dir outputs/train/act_multi_object_open_7221_scene_0_seed_0 \
        --initial-state-path data/multi_object_open/traj/summit_franka/scene_0_seed_0/7221/grasp_0000/stage_0/start_iks.pt \
        --scene scene_0_seed_0 --object 7221 --headless
"""

import os
import sys
print(f"PYTHON EXECUTABLE: {sys.executable}")
import argparse
import logging
import re
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CRITICAL INITIALIZATION ---
# Isaac Sim's SimulationApp MUST be initialized before any other imports that might touch pxr or omni.
# We do a quick parse of sys.argv to get the headless flag before importing anything else.
# Check for help flag to avoid starting sim unnecessarily
if "--help" in sys.argv or "-h" in sys.argv:
    pass
else:
    import argparse
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--headless", action="store_true")
    temp_args, _ = temp_parser.parse_known_args()

    # Pre-check for ninja for curobo JIT
    import shutil
    if not shutil.which("ninja"):
        logger.warning("Ninja not found. Curobo JIT compilation may fail. Consider installing ninja.")

    from automoma.utils.sim_utils import get_simulation_app
    sim_app = get_simulation_app(headless=temp_args.headless)
# -------------------------------

# Now safe to import other modules
from automoma.core.config_loader import load_config, Config
from automoma.tasks.factory import create_task
from automoma.evaluation.policy_runner import get_model
from automoma.utils.file_utils import load_object_from_metadata, get_latest_checkpoint
def run_evaluation(
    cfg: Config,
    policy_type: str,
    checkpoint_path: str,
    initial_state_path: str,
    headless: bool = False,
    scene_name: str = None,
    object_id: str = None,
    dataset_root: str = None,
):
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
        checkpoint_path: Full path to latest checkpoint
        initial_state_path: Path to a start_iks.pt file or a directory
        scene_name: Scene name for evaluation
        object_id: Object ID for evaluation
        dataset_root: Optional dataset root override
    """
    # Create task
    task = create_task(cfg)
    
    # Get eval config
    eval_cfg = cfg.eval_cfg if cfg.eval_cfg else cfg.evaluation

    # Validate scene is in info_cfg
    info_cfg = cfg.info_cfg
    if scene_name not in info_cfg.scene:
        logger.error(f"Scene '{scene_name}' not found in info_cfg.scene: {info_cfg.scene}")
        return

    # Validate object is in info_cfg
    if object_id not in info_cfg.object:
        logger.error(f"Object '{object_id}' not found in info_cfg.object: {info_cfg.object}")
        return

    # Get scene config
    scene_cfg = cfg.env_cfg.scene_cfg[scene_name] if cfg.env_cfg and cfg.env_cfg.scene_cfg else None
    if scene_cfg is None:
        logger.error(f"Scene config not found: {scene_name}")
        return
    
    metadata_path = scene_cfg.metadata_path if hasattr(scene_cfg, 'metadata_path') else None
    
    # Get object config
    object_cfg = cfg.env_cfg.object_cfg[object_id] if cfg.env_cfg and cfg.env_cfg.object_cfg else None
    if object_cfg is None:
        logger.error(f"Object config not found: {object_id}")
        return
    
    # Load object with metadata if available
    if metadata_path:
        object_cfg = load_object_from_metadata(metadata_path, object_cfg)
    
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = str(PROJECT_ROOT / checkpoint_path)

    if not os.path.isabs(initial_state_path):
        initial_state_path = str(PROJECT_ROOT / initial_state_path)

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return 1
    
    # Evaluation parameters
    num_episodes = eval_cfg.num_episodes if eval_cfg else 50
    use_async = eval_cfg.use_async_inference if eval_cfg else True
    
    # Record policy type in eval config (optional)
    if eval_cfg and hasattr(eval_cfg, 'policy_type'):
        eval_cfg.policy_type = policy_type
    
    logger.info(f"Evaluation configuration:")
    logger.info(f"  Policy type: {policy_type}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Async inference: {use_async}")
    logger.info(f"  Scene: {scene_name}")
    logger.info(f"  Object: {object_id}")
    logger.info(f"  Initial state path: {initial_state_path}")
    if dataset_root:
        logger.info(f"  Dataset root: {dataset_root}")
    
    # Setup environment
    try:
        # SimulationApp is already initialized at the top of the script

        # Now safe to import SimEnvWrapper
        from automoma.simulation.env_wrapper import SimEnvWrapper

        env = SimEnvWrapper(cfg.env_cfg)

        # Convert Config objects to dict for setup_env
        scene_cfg_dict = scene_cfg.to_dict() if hasattr(scene_cfg, 'to_dict') else scene_cfg
        object_cfg_dict = object_cfg.to_dict() if hasattr(object_cfg, 'to_dict') else object_cfg

        env.setup_env(object_cfg=object_cfg_dict, scene_cfg=scene_cfg_dict)
        task.setup_env(env)
        logger.info("Environment setup complete")
    except Exception as e:
        logger.error(f"Could not setup environment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load model with async communication
    logger.info("Loading model...")

    # Convert relative paths to absolute
    if dataset_root and not os.path.isabs(dataset_root):
        dataset_root = str(PROJECT_ROOT / dataset_root)

    dataset_id = os.path.basename(os.path.normpath(dataset_root)) if dataset_root else None
    
    model = get_model(
        checkpoint_path=checkpoint_path,
        policy_type=policy_type,
        async_mode=use_async,
        device=eval_cfg.device if eval_cfg else "cuda",
        dataset_id=dataset_id,
        dataset_root=dataset_root,
    )
    model.start()
    
    # Run evaluation using task pipeline
    logger.info("Starting evaluation task pipeline...")
    
    try:
        metrics = task.run_evaluation_pipeline(
            policy_model=model,
            initial_state_path=initial_state_path,
            num_episodes=num_episodes
        )
    except Exception as e:
        logger.error(f"Error during evaluation loop: {e}")
        import traceback
        traceback.print_exc()
        metrics = None

    # Stop model
    model.stop()
    print("DEBUG: Model stopped.")
    
    if metrics is None:
        return

    logger.info(f"\n{'='*60}")
    print("EVALUATION RESULTS") # Forced print
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    if hasattr(metrics, 'success_rate'):
        logger.info(f"Success rate: {metrics.success_rate*100:.1f}%")
        print(f"Success rate: {metrics.success_rate*100:.1f}%")
        logger.info(f"Completion rate: {metrics.completion_rate*100:.1f}%")
        logger.info(f"Avg inference time: {metrics.inference_time_mean*1000:.1f}ms")
    else:
        logger.info(f"Metrics: {metrics}")
    
    # Save results
    output_dir = eval_cfg.output_dir if eval_cfg else "outputs/eval"
    output_dir = Path(PROJECT_ROOT / output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / f"{cfg.info_cfg.task}_{object_id}_{scene_name}_results.json"
    print(f"DEBUG: Saving results to {results_path}")
    
    import json
    with open(results_path, 'w') as f:
        json.dump(metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy")
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment name (optional; inferred from checkpoint-dir if possible)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to training run directory (used to locate latest checkpoint)",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        required=True,
        help="Policy type (act or diffusion)",
    )
    parser.add_argument(
        "--initial-state-path",
        type=str,
        required=True,
        help="Path to initial state file or directory (start_iks.pt)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene name (must be in info_cfg.scene)",
    )
    parser.add_argument(
        "--object",
        type=str,
        required=True,
        help="Object ID (must be in info_cfg.object)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to the dataset root for preprocessing/postprocessing (defaults to checkpoint-dir)",
    )
    args = parser.parse_args()
    
    # Resolve latest checkpoint from checkpoint-dir
    checkpoint = get_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint:
        parser.error(f"No checkpoint found in {args.checkpoint_dir}")

    # Default dataset_root to checkpoint-dir
    if not args.dataset_root:
        args.dataset_root = args.checkpoint_dir
    
    # Load configuration (merges plan, record, train, eval) to ensure we have full info_cfg
    from automoma.core.config_loader import load_config
    cfg = load_config(args.exp, PROJECT_ROOT)
    
    # Run evaluation
    run_evaluation(
        cfg,
        args.policy_type,
        checkpoint,
        args.initial_state_path,
        args.headless,
        scene_name=args.scene,
        object_id=args.object,
        dataset_root=args.dataset_root
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
