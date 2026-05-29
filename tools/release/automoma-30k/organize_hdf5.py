#!/usr/bin/env python3
"""Create a read-only symlink view for the AutoMoMa-30K source HDF5 files."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "automoma_30scenes"
DEFAULT_SOURCE_ROOT = DATA_ROOT / "traj" / "summit_franka"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "automoma-30k-sort"

SCENE_RE = re.compile(r"scene_(\d+)_seed_(\d+)$")
EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


def natural_scene_key(path: Path) -> tuple[int, int, str]:
    match = SCENE_RE.fullmatch(path.name)
    if not match:
        return (10**9, 10**9, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def episode_key(path: Path) -> tuple[int, str]:
    match = EPISODE_RE.fullmatch(path.name)
    if not match:
        return (10**9, path.name)
    return (int(match.group(1)), path.name)


def make_writable(path: Path) -> None:
    if not path.exists():
        return
    for root, dirs, _files in os.walk(path):
        for dirname in dirs:
            os.chmod(Path(root) / dirname, 0o755)
        os.chmod(root, 0o755)


def chmod_read_only(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            if not file_path.is_symlink():
                os.chmod(file_path, 0o444)
        for dirname in dirs:
            os.chmod(Path(root) / dirname, 0o555)
        os.chmod(root, 0o555)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--object-id", default="7221")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "manifest"),
        default="symlink",
        help=(
            "How to build the sorted view. 'manifest' writes only manifest/summary "
            "when the data filesystem does not permit links."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate output-root. Only the managed symlink view is removed.",
    )
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=30,
        help="Expected number of non-empty scenes for summary warnings.",
    )
    parser.add_argument(
        "--expected-episodes-per-scene",
        type=int,
        default=1000,
        help="Expected episode count for a complete scene.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root

    if not source_root.exists():
        raise FileNotFoundError(f"source root does not exist: {source_root}")

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"output root already exists: {output_root}; use --force to recreate")
        make_writable(output_root)
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    scene_summaries: list[dict[str, object]] = []

    scene_dirs = sorted(
        [path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("scene_")],
        key=natural_scene_key,
    )
    for scene_dir in scene_dirs:
        source_episode_dir = scene_dir / args.object_id / "camera_data"
        episodes = sorted(source_episode_dir.glob("*.hdf5"), key=episode_key) if source_episode_dir.exists() else []
        if not episodes:
            scene_summaries.append({
                "scene": scene_dir.name,
                "source_dir": str(source_episode_dir),
                "num_episodes": 0,
                "status": "missing_or_empty",
            })
            continue

        out_scene_dir = output_root / scene_dir.name
        out_scene_dir.mkdir()
        for episode in episodes:
            dst = out_scene_dir / episode.name
            link_status = args.link_mode
            if args.link_mode == "symlink":
                try:
                    os.symlink(str(episode.resolve()), dst)
                except PermissionError as exc:
                    raise PermissionError(
                        f"filesystem refused symlink creation under {output_root}. "
                        "Re-run with --link-mode manifest to write a source manifest only."
                    ) from exc
            elif args.link_mode == "hardlink":
                try:
                    os.link(episode.resolve(), dst)
                except PermissionError as exc:
                    raise PermissionError(
                        f"filesystem refused hardlink creation under {output_root}. "
                        "Re-run with --link-mode manifest to write a source manifest only."
                    ) from exc
            else:
                link_status = "manifest_only"
            manifest_rows.append({
                "scene": scene_dir.name,
                "episode": episode.name,
                "source": str(episode.resolve()),
                "link": str(dst.absolute()),
                "link_status": link_status,
            })

        status = "complete" if len(episodes) == args.expected_episodes_per_scene else "partial"
        scene_summaries.append({
            "scene": scene_dir.name,
            "source_dir": str(source_episode_dir),
            "num_episodes": len(episodes),
            "status": status,
        })

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for row in manifest_rows:
            f.write(json.dumps(row) + "\n")

    non_empty_scenes = sum(1 for row in scene_summaries if row["num_episodes"])
    total_episodes = len(manifest_rows)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root.resolve()),
        "object_id": args.object_id,
        "link_mode": args.link_mode,
        "expected_scenes": args.expected_scenes,
        "expected_episodes_per_scene": args.expected_episodes_per_scene,
        "non_empty_scenes": non_empty_scenes,
        "total_episodes": total_episodes,
        "complete_scenes": sum(1 for row in scene_summaries if row["status"] == "complete"),
        "partial_scenes": [row for row in scene_summaries if row["status"] == "partial"],
        "missing_or_empty_scenes": [row for row in scene_summaries if row["status"] == "missing_or_empty"],
        "scenes": scene_summaries,
    }
    write_json(output_root / "summary.json", summary)
    chmod_read_only(output_root)

    if args.link_mode == "manifest":
        print(f"Created manifest-only index: {output_root}")
    else:
        print(f"Created {args.link_mode} view: {output_root}")
    print(f"Linked episodes: {total_episodes}")
    print(f"Non-empty scenes: {non_empty_scenes}")
    if non_empty_scenes != args.expected_scenes:
        print(f"WARNING: expected {args.expected_scenes} non-empty scenes, found {non_empty_scenes}")
    incomplete = [row for row in scene_summaries if row["num_episodes"] and row["status"] != "complete"]
    if incomplete:
        print("WARNING: incomplete non-empty scenes:")
        for row in incomplete:
            print(f"  {row['scene']}: {row['num_episodes']} episodes")


if __name__ == "__main__":
    main()
