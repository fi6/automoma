#!/usr/bin/env python3
"""
Script 3: Train policy model.

This script trains a policy using LeRobot:
1. Load training configuration
2. Load dataset
3. Train policy
4. Save checkpoints

Usage:
    python 3_train.py --exp multi_object_open
"""

import os
import sys
import argparse
import logging
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


def train_policy(cfg: Config):
    """
    Train a policy model.
    
    Args:
        cfg: Configuration object
    """
    import torch
    
    # Get training config
    train_cfg = cfg.train_cfg if cfg.train_cfg else cfg.training
    if train_cfg is None:
        logger.error("No training configuration found")
        return
    
    # Dataset config
    dataset_repo = train_cfg.dataset_repo_id if train_cfg.dataset_repo_id else f"automoma/{cfg.info_cfg.task}"
    dataset_root = train_cfg.dataset_root if train_cfg.dataset_root else "./datasets"
    
    # Policy config
    policy_type = train_cfg.policy_type if train_cfg.policy_type else "diffusion"
    
    # Training params
    batch_size = train_cfg.batch_size if train_cfg.batch_size else 64
    learning_rate = train_cfg.learning_rate if train_cfg.learning_rate else 1e-4
    num_epochs = train_cfg.num_epochs if train_cfg.num_epochs else 100
    
    # Output
    output_dir = train_cfg.output_dir if train_cfg.output_dir else "outputs/train"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = train_cfg.device if train_cfg.device else "cuda"
    
    logger.info(f"Training configuration:")
    logger.info(f"  Dataset: {dataset_repo}")
    logger.info(f"  Policy: {policy_type}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Device: {device}")
    
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.factory import make_policy
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = LeRobotDataset(repo_id=dataset_repo, root=dataset_root)
        logger.info(f"Dataset loaded: {len(dataset)} frames")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        # Create policy
        logger.info(f"Creating {policy_type} policy...")
        policy_config = {
            "name": policy_type,
            "input_shapes": {},
            "output_shapes": {},
        }
        
        if hasattr(dataset, "features"):
            for key, feature in dataset.features.items():
                if key.startswith("observation"):
                    policy_config["input_shapes"][key] = feature.get("shape", [])
                elif key == "action":
                    policy_config["output_shapes"][key] = feature.get("shape", [])
        
        policy = make_policy(policy_config)
        policy.to(device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info("Starting training...")
        global_step = 0
        save_freq = train_cfg.save_freq if train_cfg.save_freq else 10000
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                optimizer.zero_grad()
                loss_dict = policy.forward(batch)
                loss = loss_dict.get("loss", sum(loss_dict.values()))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
                if global_step % save_freq == 0:
                    ckpt_path = output_dir / f"checkpoint_{global_step:08d}.pt"
                    torch.save({
                        "step": global_step,
                        "state_dict": policy.state_dict(),
                        "config": cfg.to_dict(),
                    }, ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
            
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        # Save final model
        final_path = output_dir / "model_final.pt"
        torch.save({
            "step": global_step,
            "state_dict": policy.state_dict(),
            "config": cfg.to_dict(),
        }, final_path)
        logger.info(f"Training complete! Model saved to {final_path}")
        
    except ImportError as e:
        logger.error(f"LeRobot not available: {e}")
        logger.info("Install LeRobot to train policies")


def main():
    parser = argparse.ArgumentParser(description="Train policy")
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name",
    )
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.exp, PROJECT_ROOT)
    
    # Train
    train_policy(cfg)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
