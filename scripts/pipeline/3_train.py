#!/usr/bin/env python3
"""
Script 3: Train policy model on recorded dataset.

This script:
1. Loads the LeRobot dataset
2. Configures the policy model
3. Trains the model
4. Saves checkpoints

Usage:
    python 3_train.py --config configs/exps/multi_object_open/train.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_wandb(config: Dict[str, Any]) -> None:
    """Setup Weights & Biases logging."""
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("use_wandb", False):
        return
    
    try:
        import wandb
        wandb.init(
            project=wandb_cfg.get("project", "automoma"),
            entity=wandb_cfg.get("entity"),
            config=config,
        )
        print("W&B initialized")
    except ImportError:
        print("wandb not installed, skipping logging")


def train_lerobot_policy(config: Dict[str, Any]) -> None:
    """
    Train a LeRobot policy using the training configuration.
    
    Args:
        config: Training configuration dictionary
    """
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.factory import make_policy
        from lerobot.common.utils.utils import init_hydra_config
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as e:
        print(f"Error importing LeRobot modules: {e}")
        print("Make sure LeRobot is installed in the environment")
        return
    
    # Extract configuration
    dataset_cfg = config.get("dataset", {})
    policy_cfg = config.get("policy", {})
    training_cfg = config.get("training", {})
    checkpoint_cfg = config.get("checkpoint", {})
    hardware_cfg = config.get("hardware", {})
    
    device = hardware_cfg.get("device", "cuda")
    
    # Load dataset
    print("Loading dataset...")
    dataset_repo = dataset_cfg.get("repo_id", "")
    dataset_root = dataset_cfg.get("root", "./datasets")
    
    if not dataset_repo:
        print("No dataset repo_id specified")
        return
    
    try:
        dataset = LeRobotDataset(
            repo_id=dataset_repo,
            root=dataset_root,
        )
        print(f"Dataset loaded: {len(dataset)} frames")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=hardware_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    
    # Create policy
    print("Creating policy...")
    policy_type = policy_cfg.get("type", "diffusion")
    
    # Build policy config based on dataset
    policy_config = {
        "name": policy_type,
        "input_shapes": {},
        "output_shapes": {},
    }
    
    # Configure input/output shapes from dataset features
    if hasattr(dataset, "features"):
        for key, feature in dataset.features.items():
            if key.startswith("observation"):
                policy_config["input_shapes"][key] = feature.get("shape", [])
            elif key == "action":
                policy_config["output_shapes"][key] = feature.get("shape", [])
    
    try:
        policy = make_policy(policy_config)
        policy.to(device)
        print(f"Policy created: {policy_type}")
    except Exception as e:
        print(f"Error creating policy: {e}")
        # Fallback: create a simple policy manually
        print("Creating simple fallback policy...")
        policy = create_simple_policy(policy_config, device)
    
    # Load pretrained weights if available
    pretrained_path = policy_cfg.get("pretrained_path")
    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        policy.load_state_dict(checkpoint.get("state_dict", checkpoint))
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=training_cfg.get("learning_rate", 1e-4),
    )
    
    # Create output directory
    output_dir = Path(checkpoint_cfg.get("output_dir", "outputs/train"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    num_epochs = training_cfg.get("num_epochs", 100)
    save_freq = checkpoint_cfg.get("save_freq", 10000)
    log_freq = checkpoint_cfg.get("log_freq", 100)
    grad_clip = training_cfg.get("grad_clip", 10.0)
    
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                loss_dict = policy.forward(batch)
                loss = loss_dict.get("loss", loss_dict.get("total_loss", sum(loss_dict.values())))
            except Exception as e:
                # Simple MSE loss fallback
                if hasattr(policy, "predict"):
                    pred = policy.predict(batch)
                    loss = torch.nn.functional.mse_loss(pred, batch.get("action", pred))
                else:
                    continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Logging
            if global_step % log_freq == 0:
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                # Log to wandb if available
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"loss": loss.item(), "step": global_step})
                except:
                    pass
            
            # Save checkpoint
            if global_step % save_freq == 0:
                checkpoint_path = output_dir / f"checkpoint_{global_step:08d}.pt"
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "state_dict": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }, checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
    
    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save({
        "step": global_step,
        "epoch": num_epochs,
        "state_dict": policy.state_dict(),
        "config": config,
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")


def create_simple_policy(config: Dict, device: str):
    """Create a simple MLP policy as fallback."""
    import torch
    import torch.nn as nn
    
    # Get input/output dimensions
    input_dim = sum(
        torch.tensor(shape).prod().item() 
        for shape in config.get("input_shapes", {}).values()
    )
    output_dim = sum(
        torch.tensor(shape).prod().item()
        for shape in config.get("output_shapes", {}).values()
    )
    
    if input_dim == 0:
        input_dim = 256  # Default
    if output_dim == 0:
        output_dim = 10  # Default
    
    class SimplePolicy(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim),
            )
        
        def forward(self, batch):
            # Flatten observations
            obs_list = []
            for k, v in batch.items():
                if k.startswith("observation") and isinstance(v, torch.Tensor):
                    obs_list.append(v.view(v.shape[0], -1))
            
            if not obs_list:
                return {"loss": torch.tensor(0.0)}
            
            x = torch.cat(obs_list, dim=-1)
            pred = self.net(x)
            
            target = batch.get("action", pred)
            loss = nn.functional.mse_loss(pred, target)
            
            return {"loss": loss}
        
        def predict(self, batch):
            obs_list = []
            for k, v in batch.items():
                if k.startswith("observation") and isinstance(v, torch.Tensor):
                    obs_list.append(v.view(v.shape[0], -1))
            
            if not obs_list:
                return torch.zeros(1, self.net[-1].out_features)
            
            x = torch.cat(obs_list, dim=-1)
            return self.net(x)
    
    return SimplePolicy(int(input_dim), int(output_dim)).to(device)


def main():
    parser = argparse.ArgumentParser(description="Train policy model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/exps/multi_object_open/train.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset repo_id (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    args = parser.parse_args()
    
    # Load configuration
    config_path = PROJECT_ROOT / args.config
    
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {
            "dataset": {"repo_id": "automoma_dataset", "root": "./datasets"},
            "policy": {"type": "diffusion"},
            "training": {"batch_size": 64, "learning_rate": 1e-4, "num_epochs": 100},
            "checkpoint": {"output_dir": "outputs/train"},
            "hardware": {"device": "cuda", "num_workers": 4},
        }
    
    # Override with command line arguments
    if args.dataset:
        config.setdefault("dataset", {})["repo_id"] = args.dataset
    if args.output_dir:
        config.setdefault("checkpoint", {})["output_dir"] = args.output_dir
    if args.epochs:
        config.setdefault("training", {})["num_epochs"] = args.epochs
    
    # Setup wandb
    setup_wandb(config)
    
    # Train
    train_lerobot_policy(config)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
