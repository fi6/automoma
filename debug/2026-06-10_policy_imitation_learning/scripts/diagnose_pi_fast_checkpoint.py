#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC_ROOT = REPO_ROOT / "third_party" / "lerobot" / "src"
if str(LEROBOT_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC_ROOT))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION_TOKEN_MASK


def parse_indices(raw: str) -> list[int]:
    indices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-force diagnostic for PI-FAST checkpoints on a LeRobot dataset."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--indices", default="0,1,2,10,100,500,1000,2000,3000,4000")
    parser.add_argument("--text-tokenizer-name", default=None)
    parser.add_argument("--action-tokenizer-name", default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    indices = parse_indices(args.indices)

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.dataset.root = dataset_root
    cfg.policy.pretrained_path = checkpoint
    cfg.policy.device = args.device
    if args.text_tokenizer_name:
        cfg.policy.text_tokenizer_name = args.text_tokenizer_name
    if args.action_tokenizer_name:
        cfg.policy.action_tokenizer_name = args.action_tokenizer_name

    print(f"checkpoint={checkpoint}", flush=True)
    print(f"dataset_root={dataset_root}", flush=True)
    print(f"policy_type={cfg.policy.type}", flush=True)
    print(f"use_peft={getattr(cfg.policy, 'use_peft', None)}", flush=True)
    print(f"use_relative_actions={getattr(cfg.policy, 'use_relative_actions', None)}", flush=True)
    print(f"pretrained_path={cfg.policy.pretrained_path}", flush=True)
    print(f"text_tokenizer_name={getattr(cfg.policy, 'text_tokenizer_name', None)}", flush=True)
    print(f"action_tokenizer_name={getattr(cfg.policy, 'action_tokenizer_name', None)}", flush=True)

    dataset = make_dataset(cfg)
    print(f"dataset_num_frames={len(dataset)}", flush=True)
    print(f"dataset_num_episodes={dataset.num_episodes}", flush=True)
    print(f"delta_timestamps={getattr(dataset, 'delta_timestamps', None)}", flush=True)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(cfg.policy.device)}},
    )

    print(f"loaded_policy_class={policy.__class__.__module__}.{policy.__class__.__name__}", flush=True)
    print(f"policy_config_pretrained_path={getattr(policy.config, 'pretrained_path', None)}", flush=True)
    print(f"policy_config_use_peft={getattr(policy.config, 'use_peft', None)}", flush=True)

    losses: list[float] = []
    with torch.inference_mode():
        for idx in indices:
            item = dataset[idx]
            raw_action_shape = tuple(item["action"].shape)
            batch = torch.utils.data.default_collate([item])
            for cam_key in dataset.meta.camera_keys:
                if cam_key in batch and batch[cam_key].dtype == torch.uint8:
                    batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
            batch = preprocessor(batch)
            processed_action_shape = tuple(batch["action"].shape)
            token_mask_sum = int(batch.get(ACTION_TOKEN_MASK, torch.zeros(1)).sum().item())
            loss, loss_dict = policy(batch)
            loss_value = float(loss.item())
            losses.append(loss_value)
            print(
                "sample",
                idx,
                "raw_action_shape",
                raw_action_shape,
                "processed_action_shape",
                processed_action_shape,
                "action_token_mask_sum",
                token_mask_sum,
                "loss",
                loss_value,
                "details",
                loss_dict,
                flush=True,
            )

    if losses:
        print(
            "loss_summary",
            {
                "mean": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
                "count": len(losses),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
