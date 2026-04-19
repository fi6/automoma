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

REQUIRED_TRAJ_KEYS = tuple(TRAJ_KEYS)


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


def shape_summary(data: dict[str, Any]) -> dict[str, list[int]]:
    summary: dict[str, list[int]] = {}
    for key in TRAJ_KEYS:
        value = data.get(key)
        if hasattr(value, "shape"):
            summary[key] = list(value.shape)
    return summary


def validate_payload(data: dict[str, Any], *, label: str) -> None:
    missing = [key for key in REQUIRED_TRAJ_KEYS if key not in data]
    if missing:
        raise ValueError(f"{label} is missing required trajectory keys: {missing}")

    for key in TRAJ_KEYS:
        value = data[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{label} key '{key}' must be a torch.Tensor, got {type(value).__name__}")


def validate_merge_compatibility(payloads: list[dict[str, Any]], labels: list[str]) -> None:
    if len(payloads) != len(labels):
        raise ValueError("Payload/label length mismatch")
    if not payloads:
        raise ValueError("No payloads to validate")

    for payload, label in zip(payloads, labels):
        validate_payload(payload, label=label)

    reference = payloads[0]
    reference_label = labels[0]
    for payload, label in zip(payloads[1:], labels[1:]):
        for key in TRAJ_KEYS:
            ref_tensor = reference[key]
            tensor = payload[key]
            if ref_tensor.ndim != tensor.ndim:
                raise ValueError(
                    f"Cannot merge {label} into {reference_label}: key '{key}' rank mismatch "
                    f"{tensor.ndim} != {ref_tensor.ndim}"
                )
            if ref_tensor.shape[1:] != tensor.shape[1:]:
                raise ValueError(
                    f"Cannot merge {label} into {reference_label}: key '{key}' shape mismatch "
                    f"{tuple(tensor.shape)} vs {tuple(ref_tensor.shape)}"
                )
            if ref_tensor.dtype != tensor.dtype:
                raise ValueError(
                    f"Cannot merge {label} into {reference_label}: key '{key}' dtype mismatch "
                    f"{tensor.dtype} vs {ref_tensor.dtype}"
                )


def merge_payloads(payloads: list[dict[str, Any]], labels: list[str] | None = None) -> dict[str, Any]:
    if not payloads:
        raise ValueError("No payloads to merge")

    labels = labels or [f"payload[{idx}]" for idx in range(len(payloads))]
    validate_merge_compatibility(payloads, labels)

    merged: dict[str, Any] = {}
    for key in TRAJ_KEYS:
        merged[key] = torch.cat([payload[key] for payload in payloads], dim=0)
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
    labels = [str(path) for path in input_paths]
    merged = merge_payloads(payloads, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output)

    report = {
        "output": str(output),
        "inputs": [str(p) for p in input_paths],
        "input_stats": {str(path): count_stats(payload) for path, payload in zip(input_paths, payloads)},
        "input_shapes": {str(path): shape_summary(payload) for path, payload in zip(input_paths, payloads)},
        "stats": count_stats(merged),
        "shapes": shape_summary(merged),
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
