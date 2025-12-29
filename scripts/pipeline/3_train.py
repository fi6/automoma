#!/usr/bin/env python3
"""
Script 3: Train policy model.

This script trains a policy using LeRobot:
1. Load training configuration
2. Build lerobot-train command based on policy type
3. Execute training via subprocess

Usage:
    # Train with ACT policy (default from config)
    python 3_train.py --exp single_object_open_test
    
    # Train with specific policy type
    python 3_train.py --exp single_object_open_test --policy act
    python 3_train.py --exp single_object_open_test --policy diffusion
    
    # Override output directory
    python 3_train.py --exp single_object_open_test --policy act --output-dir outputs/train/my_experiment
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from automoma.core.config_loader import load_config, Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_lerobot_command(cfg: Config, policy_type: str = None, output_dir: str = None) -> list:
    """
    Build lerobot-train command from configuration.
    
    Args:
        cfg: Configuration object
        policy_type: Policy type override (act, diffusion, pi0, etc.)
        output_dir: Output directory override
        
    Returns:
        List of command arguments for subprocess
    """
    train_cfg = cfg.train_cfg
    
    # Determine policy type
    if policy_type is None:
        # Try to infer from config
        policy_type = "act"  # default
        if hasattr(train_cfg, 'policy_type'):
            policy_type = train_cfg.policy_type
    
    # Get policy config
    policy_cfg = cfg.policy_cfg
    if not hasattr(policy_cfg, policy_type):
        logger.error(f"Policy type '{policy_type}' not found in policy_cfg")
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    policy_config = getattr(policy_cfg, policy_type)
    
    # Build base command
    cmd = ["lerobot-train"]
    
    # Policy type
    cmd.extend(["--policy.type", policy_config.type])
    
    # Dataset configuration
    dataset_repo_id = train_cfg.dataset_repo_id
    dataset_root = train_cfg.dataset_root
    cmd.extend(["--dataset.repo_id", dataset_repo_id])
    cmd.extend(["--dataset.root", str(PROJECT_ROOT / dataset_root)])
    
    # Training parameters
    cmd.extend(["--batch_size", str(train_cfg.batch_size)])
    cmd.extend(["--steps", str(train_cfg.steps)])
    cmd.extend(["--log_freq", str(train_cfg.log_freq)])
    cmd.extend(["--eval_freq", str(train_cfg.eval_freq)])
    cmd.extend(["--save_freq", str(train_cfg.save_freq)])
    
    # Job naming
    job_name = f"{policy_type}_{cfg.info_cfg.task}"
    if hasattr(train_cfg, 'job_name'):
        job_name = train_cfg.job_name
    cmd.extend(["--job_name", job_name])
    
    # Output directory
    if output_dir is None:
        output_dir = train_cfg.output_dir
    cmd.extend(["--output_dir", str(PROJECT_ROOT / output_dir)])
    
    # Policy-specific parameters
    if policy_type == "act":
        if hasattr(policy_config, 'chunk_size'):
            cmd.extend(["--policy.chunk_size", str(policy_config.chunk_size)])
        if hasattr(policy_config, 'n_action_steps'):
            cmd.extend(["--policy.n_action_steps", str(policy_config.n_action_steps)])
        if hasattr(policy_config, 'optimizer_lr'):
            cmd.extend(["--policy.optimizer_lr", str(policy_config.optimizer_lr)])
        if hasattr(policy_config, 'kl_weight'):
            cmd.extend(["--policy.kl_weight", str(policy_config.kl_weight)])
        if hasattr(policy_config, 'hidden_dim'):
            cmd.extend(["--policy.hidden_dim", str(policy_config.hidden_dim)])
        if hasattr(policy_config, 'dim_feedforward'):
            cmd.extend(["--policy.dim_feedforward", str(policy_config.dim_feedforward)])
            
    elif policy_type == "diffusion":
        if hasattr(policy_config, 'n_obs_steps'):
            cmd.extend(["--policy.n_obs_steps", str(policy_config.n_obs_steps)])
        if hasattr(policy_config, 'n_action_steps'):
            cmd.extend(["--policy.n_action_steps", str(policy_config.n_action_steps)])
        if hasattr(policy_config, 'horizon'):
            cmd.extend(["--policy.horizon", str(policy_config.horizon)])
        if hasattr(policy_config, 'optimizer_lr'):
            cmd.extend(["--policy.optimizer_lr", str(policy_config.optimizer_lr)])
        if hasattr(policy_config, 'num_inference_steps'):
            cmd.extend(["--policy.num_inference_steps", str(policy_config.num_inference_steps)])
    
    # Device
    if hasattr(policy_config, 'device'):
        cmd.extend(["--policy.device", policy_config.device])
    
    # Push to hub
    if hasattr(policy_config, 'push_to_hub'):
        cmd.extend(["--policy.push_to_hub", str(policy_config.push_to_hub).lower()])
    
    # Wandb configuration
    if hasattr(train_cfg, 'wandb_enable'):
        cmd.extend(["--wandb.enable", str(train_cfg.wandb_enable).lower()])
    if hasattr(train_cfg, 'wandb_project'):
        cmd.extend(["--wandb.project", train_cfg.wandb_project])
    if hasattr(train_cfg, 'wandb_entity') and train_cfg.wandb_entity:
        cmd.extend(["--wandb.entity", train_cfg.wandb_entity])
    
    return cmd


def train_policy(cfg: Config, policy_type: str = None, output_dir: str = None):
    """
    Train a policy model using LeRobot.
    
    Args:
        cfg: Configuration object
        policy_type: Policy type override (act, diffusion, etc.)
        output_dir: Output directory override
    """
    # Build command
    cmd = build_lerobot_command(cfg, policy_type, output_dir)
    
    # Log command
    logger.info("="*60)
    logger.info("Training Command:")
    logger.info("="*60)
    logger.info(" ".join(cmd))
    logger.info("="*60)
    
    # Execute training
    try:
        logger.info("Starting training...")
        result = subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        
        if result.returncode == 0:
            logger.info("="*60)
            logger.info("Training completed successfully!")
            logger.info("="*60)
        else:
            logger.error(f"Training failed with return code: {result.returncode}")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        raise
    except FileNotFoundError:
        logger.error("lerobot-train command not found. Make sure LeRobot is installed.")
        logger.info("Install LeRobot: pip install lerobot")
        raise


def main():
    parser = argparse.ArgumentParser(description="Train policy using LeRobot")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name (e.g., single_object_open_test)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        choices=["act", "diffusion", "pi0"],
        help="Policy type to train (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    args = parser.parse_args()
    
    # Load training configuration only
    logger.info(f"Loading training configuration: {args.exp}")
    from automoma.core.config_loader import load_train_config
    cfg = load_train_config(args.exp, PROJECT_ROOT)
    
    # Train policy
    train_policy(cfg, args.policy, args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
