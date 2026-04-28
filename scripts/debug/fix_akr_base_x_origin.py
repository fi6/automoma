#!/usr/bin/env python3
"""Remove the erroneous -1.5 Y offset from AKR summit_franka URDF base_x joints.

The script scans ``assets/object`` for ``summit_franka_*.urdf`` files, finds the
``base_x`` prismatic joint, and changes its origin from:

    xyz="0.0 -1.5 0.0"

to:

    xyz="0.0 0.0 0.0"

It defaults to dry-run mode. Use ``--apply`` to write changes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


JOINT_RE = re.compile(
    r'(<joint\s+name="base_x"\s+type="prismatic"\s*>.*?</joint>)',
    re.DOTALL,
)
ORIGIN_RE = re.compile(r'(<origin\b[^>]*\bxyz=")([^"]+)("[^>]*/?>)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix AKR summit_franka URDF base_x origin Y offset.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("assets/object"),
        help="Root directory to scan. Default: assets/object",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag, only prints what would change.",
    )
    parser.add_argument(
        "--from-y",
        type=float,
        default=-1.5,
        help="Y offset to replace. Default: -1.5",
    )
    parser.add_argument(
        "--to-y",
        default="0.0",
        help='Replacement Y value text. Default: "0.0"',
    )
    return parser.parse_args()


def fix_urdf_text(text: str, from_y: float, to_y: str) -> tuple[str, int]:
    replacements = 0

    def fix_joint(match: re.Match[str]) -> str:
        nonlocal replacements
        joint_block = match.group(1)

        def fix_origin(origin_match: re.Match[str]) -> str:
            nonlocal replacements
            xyz = origin_match.group(2).split()
            if len(xyz) != 3:
                return origin_match.group(0)
            try:
                y_value = float(xyz[1])
            except ValueError:
                return origin_match.group(0)
            if abs(y_value - from_y) > 1e-9:
                return origin_match.group(0)
            xyz[1] = to_y
            replacements += 1
            return f"{origin_match.group(1)}{' '.join(xyz)}{origin_match.group(3)}"

        return ORIGIN_RE.sub(fix_origin, joint_block, count=1)

    fixed = JOINT_RE.sub(fix_joint, text)
    return fixed, replacements


def main() -> int:
    args = parse_args()
    assets_root = args.assets_root
    if not assets_root.exists():
        raise FileNotFoundError(f"Assets root not found: {assets_root}")

    files = sorted(assets_root.rglob("summit_franka_*.urdf"))
    changed_files: list[Path] = []
    total_replacements = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        fixed, replacements = fix_urdf_text(text, args.from_y, args.to_y)
        if replacements == 0:
            continue
        changed_files.append(path)
        total_replacements += replacements
        print(f"{'fix' if args.apply else 'would fix'} {path} ({replacements})")
        if args.apply:
            path.write_text(fixed, encoding="utf-8")

    mode = "updated" if args.apply else "would update"
    print(
        f"{mode} {len(changed_files)} files; "
        f"{total_replacements} base_x origin replacements"
    )
    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
