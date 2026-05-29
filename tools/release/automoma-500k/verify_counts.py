#!/usr/bin/env python3
"""Verify AutoMoMa-500k trajectory counts by parallel-loading traj_data_train.pt files."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch


def count_file(path: Path) -> tuple[str, str, int, int]:
    """Returns (object, scene, total_traj, success_count)."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        traj_robot = payload.get("traj_robot")
        total = int(traj_robot.shape[0]) if isinstance(traj_robot, torch.Tensor) else 0

        success = payload.get("traj_success")
        if isinstance(success, torch.Tensor):
            success_count = int(success.bool().sum().item())
        elif isinstance(success, torch.Tensor):
            success_count = int(success.sum().item())
        else:
            success_count = total  # fall back to total if no success field

        obj = path.parts[-4]
        scene = path.parts[-3]
        return (obj, scene, total, success_count)
    except Exception as e:
        obj = path.parts[-4]
        scene = path.parts[-3]
        return (obj, scene, -1, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify AutoMoMa-500k trajectory counts")
    parser.add_argument("--root", type=Path, default=Path("data/trajs/summit_franka"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target-per-object", type=int, default=100_000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    files = list(args.root.glob(f"*/*/{args.split}/traj_data_{args.split}.pt"))
    print(f"Found {len(files)} trajectory files under {args.root}")
    print(f"Using {args.workers} parallel workers\n")

    by_obj_scene = defaultdict(lambda: defaultdict(int))
    by_obj = defaultdict(int)
    scene_totals = defaultdict(int)
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as exc:
        futures = {exc.submit(count_file, f): f for f in files}
        for fut in as_completed(futures):
            obj, scene, total, success = fut.result()
            if total == -1:
                errors.append(f"{obj}/{scene}")
            else:
                by_obj_scene[obj][scene] = success
                by_obj[obj] += success
                scene_totals[scene] += success

    print(f"{'Object':<25} {'Scenes':>7} {'Success':>10} {'Target':>10} {'Remaining':>10} {'%':>8}")
    print("-" * 75)

    grand_success = 0
    for obj in sorted(by_obj):
        s = by_obj[obj]
        n_scenes = len(by_obj_scene[obj])
        rem = max(0, args.target_per_object - s)
        pct = (s / args.target_per_object * 100) if args.target_per_object else 0
        print(f"{obj:<25} {n_scenes:>7} {s:>10} {args.target_per_object:>10} {rem:>10} {pct:>7.2f}%")
        grand_success += s

    print("-" * 75)
    print(f"{'TOTAL':<25} {len(scene_totals):>7} {grand_success:>10}")

    if args.verbose:
        print("\n" + "=" * 90)
        print(f"{'Scene':<35} {'Success':>10}")
        print("-" * 90)
        for scene in sorted(scene_totals):
            print(f"{scene:<35} {scene_totals[scene]:>10}")
        print("-" * 90)

        print("\n" + "=" * 90)
        print("Per-object, per-scene breakdown:")
        print("-" * 90)
        for obj in sorted(by_obj_scene):
            print(f"\n  {obj}:")
            for scene in sorted(by_obj_scene[obj]):
                cnt = by_obj_scene[obj][scene]
                print(f"    {scene:<35} {cnt:>6}")
            print(f"    {'TOTAL':<35} {by_obj[obj]:>6}")

    if errors:
        print(f"\nErrors loading {len(errors)} files:")
        for e in errors[:10]:
            print(f"  {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())