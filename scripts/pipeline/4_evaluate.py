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
import re
import glob
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
from automoma.evaluation.metrics import MetricsCalculator
from automoma.utils.robot_utils import adjust_pose_for_robot
from automoma.utils.file_utils import load_object_from_metadata


def parse_run_dir_info(run_dir_name):
    """
    Parse experiment info from run directory name.
    Expected format: {policy}_{exp}_{object}_{scene}
    Example: act_multi_object_open_7221_scene_0_seed_0
    """
    # Regex for scene at end: (scene_\d+_seed_\d+)$
    scene_match = re.search(r'(scene_\d+_seed_\d+)$', run_dir_name)
    if not scene_match: 
        return None
    scene = scene_match.group(1)
    
    # Remove scene substring
    remain = run_dir_name[:-len(scene)-1] # remove _scene...
    
    # Get object (digits at end of remain)
    obj_match = re.search(r'_(\d+)$', remain)
    if not obj_match: 
        return None
    obj = obj_match.group(1)
    
    remain = remain[:-len(obj)-1]
    
    # Get policy at start
    policy_match = re.match(r'^([a-z0-9]+)_', remain)
    if not policy_match: 
        return None
    policy = policy_match.group(1)
    
    exp = remain[len(policy)+1:]
    
    return {
        "policy": policy,
        "exp": exp,
        "object": obj,
        "scene": scene
    }


def get_latest_checkpoint(run_dir):
    """Find the latest checkpoint in run_dir (searching recursively for checkpoints/)."""
    # Pattern: .../checkpoints/XXXXXX/pretrained_model
    # or .../checkpoints/*.pt
    
    run_dir_path = Path(run_dir)
    
    # Search for any directory named 'checkpoints' recursively
    ckpt_dirs = list(run_dir_path.rglob("checkpoints"))
    if not ckpt_dirs:
        return None
        
    # Pick the one with the most subdirectories if multiple exist (unlikely in standard setup)
    # or just use the first one found that has numeric subdirs
    best_pretrained = None
    best_step = -1
    
    for ckpt_dir in ckpt_dirs:
        # Look for numeric directories (000000, 010000, etc.)
        subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if subdirs:
            # Sort by step number
            subdirs.sort(key=lambda x: int(x.name))
            latest = subdirs[-1]
            step = int(latest.name)
            
            if step > best_step:
                pretrained = latest / "pretrained_model"
                if pretrained.exists():
                    best_pretrained = str(pretrained)
                    best_step = step
            
        # Fallback: look for .pt files in this checkpoints dir
        pt_files = list(ckpt_dir.glob("*.pt"))
        if pt_files and best_pretrained is None:
            # naive sort by mtime if steps are not available
            pt_files.sort(key=lambda x: x.stat().st_mtime)
            best_pretrained = str(pt_files[-1])
            
    return best_pretrained


def run_evaluation(
    cfg: Config, 
    headless: bool = False, 
    checkpoint_path: str = None, 
    dry_run: bool = False,
    scene_name: str = None,
    object_id: str = None,
    dataset_id: str = None,
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
        checkpoint_path: Optional override for checkpoint path
        dry_run: If True, skip actual evaluation
        scene_name: Scene name for evaluation
        object_id: Object ID for evaluation
        dataset_id: Optional dataset ID override
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
    
    # Checkpoint path from policy_cfg or override
    if checkpoint_path is None:
        # Try to get from policy_cfg based on policy type
        policy_type = eval_cfg.policy_type if hasattr(eval_cfg, 'policy_type') else "diffusion"
        if cfg.policy_cfg:
            policy_config = getattr(cfg.policy_cfg, policy_type, None)
            if policy_config:
                # Try model_path first (for pretrained models), then checkpoint_path
                checkpoint_path = getattr(policy_config, 'model_path', None) or getattr(policy_config, 'checkpoint_path', None)
    
    if checkpoint_path is None:
        logger.error("No checkpoint_path or model_path found in config")
        return 1

    # Override dataset_id and dataset_root in policy_cfg if provided
    policy_type = eval_cfg.policy_type if hasattr(eval_cfg, 'policy_type') else "diffusion"
    if cfg.policy_cfg:
        policy_config = getattr(cfg.policy_cfg, policy_type, None)
        if policy_config:
            if dataset_id:
                policy_config.dataset_id = dataset_id
            if dataset_root:
                policy_config.dataset_root = dataset_root
        else:
            logger.warning(f"No policy configuration found for '{policy_type}', skipping dataset ID overrides")
    
    checkpoint_path = str(PROJECT_ROOT / checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        if not dry_run:
            logger.error("Cannot run evaluation without checkpoint")
            return
    
    # Evaluation parameters
    num_episodes = eval_cfg.num_episodes if eval_cfg else 50
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
    logger.info(f"  Async inference: {use_async}")
    logger.info(f"  Scene: {scene_name}")
    logger.info(f"  Object: {object_id}")
    
    # Setup environment
    if not dry_run:
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
            logger.warning(f"Could not setup environment: {e}")
            import traceback
            traceback.print_exc()
            dry_run = True
    
    # Get test data directory
    # Similar logic to 2_render_dataset.py logic for finding traj dir
    if eval_cfg and eval_cfg.data_cfg and eval_cfg.data_cfg.test_data_dir:
        test_data_dir = eval_cfg.data_cfg.test_data_dir
    else:
        # Fallback to a constructed path if not explicitly set in evaluation config
        # Assuming we want to evaluate on trajectories for this specific scene/object
        traj_dir = "data/output/traj" # Default base
        if cfg.record_cfg and cfg.record_cfg.data_cfg and cfg.record_cfg.data_cfg.traj_dir:
            traj_dir = cfg.record_cfg.data_cfg.traj_dir
            
        test_data_dir = str(PROJECT_ROOT / traj_dir / cfg.env_cfg.robot_cfg.robot_type / scene_name / object_id)

    test_data_dir = str(PROJECT_ROOT / test_data_dir)
    
    logger.info(f"Loading test data from: {test_data_dir}")
    
    if dry_run:
        logger.info("Dry run - skipping actual evaluation")
        return
    
    # Load model with async communication
    logger.info("Loading model...")
    policy_type = eval_cfg.policy_type if eval_cfg else "diffusion"
    
    # Use dataset configuration from arguments if provided, otherwise from policy_cfg
    if dataset_id is None or dataset_root is None:
        if cfg.policy_cfg:
            policy_cfg_for_type = getattr(cfg.policy_cfg, policy_type, None)
            if policy_cfg_for_type:
                if dataset_id is None:
                    dataset_id = getattr(policy_cfg_for_type, 'dataset_id', None)
                if dataset_root is None:
                    dataset_root = getattr(policy_cfg_for_type, 'dataset_root', None)
    
    # Convert relative paths to absolute
    if dataset_root and not os.path.isabs(dataset_root):
        dataset_root = str(PROJECT_ROOT / dataset_root)
    
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
            test_data_dir=test_data_dir,
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
        help="Experiment name (required if --run-dir is not set)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to training run directory (implies exp, scene, object, checkpoint)",
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
    parser.add_argument(
        "--scene",
        type=str,
        help="Scene name (must be in info_cfg.scene)",
    )
    parser.add_argument(
        "--object",
        type=str,
        help="Object ID (must be in info_cfg.object)",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Path to the dataset root for preprocessing/postprocessing (id derived from basename)",
    )
    args = parser.parse_args()
    
    # Infer parameters from run-dir if provided
    if args.run_dir:
        run_name = os.path.basename(os.path.normpath(args.run_dir))
        info = parse_run_dir_info(run_name)
        
        if info:
            if not args.exp:
                logger.info(f"Inferring exp='{info['exp']}' from run-dir")
                args.exp = info['exp']
            if not args.scene:
                logger.info(f"Inferring scene='{info['scene']}' from run-dir")
                args.scene = info['scene']
            if not args.object:
                logger.info(f"Inferring object='{info['object']}' from run-dir")
                args.object = info['object']
            
            # Map dp to diffusion; keep dp3 as its own policy type
            if not args.policy_type:
                inferred_policy = info['policy']
                if inferred_policy == 'dp':
                    logger.info(f"Inferring policy_type='diffusion' (mapped from 'dp')")
                    args.policy_type = "diffusion"
                else:
                    logger.info(f"Inferring policy_type='{inferred_policy}'")
                    args.policy_type = inferred_policy
        
        if not args.checkpoint:
            latest = get_latest_checkpoint(args.run_dir)
            if latest:
                logger.info(f"Inferring checkpoint='{latest}' from run-dir")
                args.checkpoint = latest
            else:
                logger.warning(f"No checkpoint found in {args.run_dir}")

        # Infer dataset_id and dataset_root from train_config.json if not provided via CLI
        if not args.dataset_root and args.checkpoint and os.path.isdir(args.checkpoint):
            train_config_path = os.path.join(args.checkpoint, "train_config.json")
            if os.path.exists(train_config_path):
                try:
                    with open(train_config_path, 'r') as f:
                        train_data = json.load(f)
                    
                    dataset_root_inferred = train_data.get("dataset", {}).get("root")
                    if dataset_root_inferred:
                        args.dataset_root = dataset_root_inferred
                        logger.info(f"Inferred dataset_root='{args.dataset_root}' from train_config.json")
                except Exception as e:
                    logger.warning(f"Failed to parse train_config.json: {e}")

    # Derive dataset_id from dataset_root if available
    print(f"DEBUG: args.dataset_root = {args.dataset_root}")
    if args.dataset_root:
        args.inferred_dataset_id = os.path.basename(os.path.normpath(args.dataset_root))
        args.inferred_dataset_root = args.dataset_root
        logger.info(f"Using dataset_id='{args.inferred_dataset_id}' from dataset_root='{args.dataset_root}'")

    # Validation
    if not args.exp:
        parser.error("argument --exp is required (or derive from --run-dir)")
    if not args.scene:
        parser.error("argument --scene is required (or derive from --run-dir)")
    if not args.object:
        parser.error("argument --object is required (or derive from --run-dir)")
    
    # Load configuration (merges plan, record, train, eval) to ensure we have full info_cfg
    from automoma.core.config_loader import load_config
    cfg = load_config(args.exp, PROJECT_ROOT)
    
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
    run_evaluation(
        cfg, 
        args.headless, 
        args.checkpoint, 
        args.dry_run,
        scene_name=args.scene,
        object_id=args.object,
        dataset_id=args.inferred_dataset_id,
        dataset_root=args.dataset_root
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
