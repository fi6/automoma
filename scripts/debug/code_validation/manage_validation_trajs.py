#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

TRAJ_KEYS = [
    "start_robot",
    "start_obj",
    "goal_robot",
    "goal_obj",
    "traj_robot",
    "traj_obj",
    "traj_success",
]


def load_traj(path: Path) -> dict[str, Any]:
    return torch.load(path, weights_only=False)


def count_stats(data: dict[str, Any]) -> dict[str, int]:
    total = 0
    successful = 0
    if "traj_success" in data:
        success = data["traj_success"]
        total = int(success.shape[0])
        successful = int(success.sum().item())
    elif "traj_robot" in data:
        total = int(data["traj_robot"].shape[0])
        successful = total
    return {"total": total, "successful": successful}


def merge_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No payloads to merge")

    merged: dict[str, Any] = {}
    for key in TRAJ_KEYS:
        values = [payload[key] for payload in payloads if key in payload]
        if not values:
            continue
        merged[key] = torch.cat(values, dim=0)
    return merged


def cmd_count(args: argparse.Namespace) -> int:
    path = Path(args.traj_file)
    data = load_traj(path)
    stats = count_stats(data)
    stats["path"] = str(path)
    print(json.dumps(stats, indent=2))
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    src = Path(args.traj_file)
    if not src.exists():
        raise FileNotFoundError(src)

    backup_dir = src.parent / "code_validation_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"{src.stem}_{ts}{src.suffix}"
    shutil.copy2(src, dst)

    metadata = {
        "source": str(src),
        "backup": str(dst),
        "timestamp": ts,
        "stats": count_stats(load_traj(src)),
    }
    meta_path = dst.with_suffix(dst.suffix + ".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    output = Path(args.output)
    input_paths = [Path(p) for p in args.inputs]
    payloads = [load_traj(path) for path in input_paths]
    merged = merge_payloads(payloads)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)

    report = {
        "output": str(output),
        "inputs": [str(p) for p in input_paths],
        "stats": count_stats(merged),
    }
    print(json.dumps(report, indent=2))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.traj_file)
    data = load_traj(path)
    stats = count_stats(data)
    report = {
        "path": str(path),
        "total": stats["total"],
        "successful": stats["successful"],
        "meets_target": stats["successful"] >= args.target,
        "target": args.target,
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage validation trajectory files")
    subparsers = parser.add_subparsers(dest="command", required=True)

    count = subparsers.add_parser("count")
    count.add_argument("traj_file")
    count.set_defaults(func=cmd_count)

    backup = subparsers.add_parser("backup")
    backup.add_argument("traj_file")
    backup.set_defaults(func=cmd_backup)

    merge = subparsers.add_parser("merge")
    merge.add_argument("output")
    merge.add_argument("inputs", nargs="+")
    merge.set_defaults(func=cmd_merge)

    status = subparsers.add_parser("status")
    status.add_argument("traj_file")
    status.add_argument("--target", type=int, required=True)
    status.set_defaults(func=cmd_status)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))
