#!/usr/bin/env python3
"""Convert AutoMoMa HDF5 datasets between merged and split layouts."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path

import h5py


SPLIT_FILE_RE = re.compile(r"^(?:episode|chunk)_(\d{6})(?:_\d{6})?\.hdf5$")


def _demo_sort_key(name: str) -> tuple[int, str]:
    prefix, _, suffix = name.rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return int(suffix), name
    return 1_000_000_000, name


def _episode_file_sort_key(path: Path) -> tuple[int, str]:
    match = SPLIT_FILE_RE.match(path.name)
    if match:
        return int(match.group(1)), path.name
    return 1_000_000_000, path.name


def _copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def _require_data_group(root: h5py.File, path: Path) -> h5py.Group:
    if "data" not in root:
        raise KeyError(f"{path} is missing the HDF5 'data' group")
    return root["data"]


def _demo_keys(data_group: h5py.Group) -> list[str]:
    keys = sorted(data_group.keys(), key=_demo_sort_key)
    if not keys:
        raise ValueError("HDF5 'data' group has no demo_* episodes")
    return keys


def _split_file_path(output_dir: Path, episode_number: int) -> Path:
    return output_dir / f"episode_{episode_number:06d}.hdf5"


def split_merged_hdf5(input_file: Path, output_dir: Path, *, move: bool, overwrite: bool) -> None:
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output exists and is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}")
        if overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_file, "r") as src:
        src_data = _require_data_group(src, input_file)
        keys = _demo_keys(src_data)
        for offset, demo_key in enumerate(keys, start=1):
            target = _split_file_path(output_dir, offset)
            if target.exists() and not overwrite:
                raise FileExistsError(target)
            tmp_target = target.with_suffix(".hdf5.tmp")
            if tmp_target.exists():
                tmp_target.unlink()

            with h5py.File(tmp_target, "w") as dst:
                _copy_attrs(src.attrs, dst.attrs)
                dst_data = dst.create_group("data")
                _copy_attrs(src_data.attrs, dst_data.attrs)
                src.copy(src_data[demo_key], dst_data, name="demo_0")

                demo = dst_data["demo_0"]
                demo.attrs["original_demo_key"] = demo_key
                demo.attrs["episode_index"] = offset - 1
                dst_data.attrs["total"] = int(demo.attrs.get("num_samples", 0))
                dst_data.attrs["split_layout"] = True
                dst_data.attrs["split_source"] = str(input_file)
                dst.flush()

            tmp_target.replace(target)

    if move:
        input_file.unlink()


def _episode_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    files = sorted(
        [path for path in input_dir.glob("*.hdf5") if SPLIT_FILE_RE.match(path.name)],
        key=_episode_file_sort_key,
    )
    if not files:
        raise ValueError(f"No episode_*.hdf5 or chunk_*.hdf5 files found in {input_dir}")
    return files


def merge_split_hdf5(input_dir: Path, output_file: Path, *, move: bool, overwrite: bool) -> None:
    files = _episode_files(input_dir)
    if output_file.exists():
        if not overwrite:
            raise FileExistsError(output_file)
        output_file.unlink()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_file.name}.",
        suffix=".tmp",
        dir=str(output_file.parent),
    )
    os.close(fd)
    Path(tmp_name).unlink()

    total_samples = 0
    try:
        with h5py.File(tmp_name, "w") as dst:
            dst_data = dst.create_group("data")
            next_demo_index = 0
            for source in files:
                with h5py.File(source, "r") as src:
                    src_data = _require_data_group(src, source)
                    if next_demo_index == 0:
                        _copy_attrs(src.attrs, dst.attrs)
                        _copy_attrs(src_data.attrs, dst_data.attrs)
                    for demo_key in _demo_keys(src_data):
                        output_demo_key = f"demo_{next_demo_index}"
                        src.copy(src_data[demo_key], dst_data, name=output_demo_key)
                        demo = dst_data[output_demo_key]
                        demo.attrs["split_source"] = str(source)
                        demo.attrs["split_source_demo_key"] = demo_key
                        total_samples += int(demo.attrs.get("num_samples", 0))
                        next_demo_index += 1
                dst_data.attrs["total"] = total_samples
                dst_data.attrs["split_layout"] = False
                dst.flush()
                if move:
                    source.unlink()

        Path(tmp_name).replace(output_file)
    except Exception:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if move:
        try:
            input_dir.rmdir()
        except OSError:
            pass


def infer_direction(input_path: Path, output_path: Path) -> str:
    if input_path.is_file():
        return "split"
    if input_path.is_dir():
        return "merge"
    if input_path.suffix == ".hdf5":
        return "split"
    if output_path.suffix == ".hdf5":
        return "merge"
    raise ValueError("Could not infer conversion direction; pass --direction split or --direction merge")


def validate_hdf5_file(path: Path) -> tuple[int, int]:
    with h5py.File(path, "r") as root:
        data = _require_data_group(root, path)
        keys = _demo_keys(data)
        total = sum(int(data[key].attrs.get("num_samples", 0)) for key in keys)
    return len(keys), total


def validate_split_dir(path: Path) -> tuple[int, int]:
    count = 0
    total = 0
    files = _episode_files(path)
    for file_path in files:
        demo_count, samples = validate_hdf5_file(file_path)
        count += demo_count
        total += samples
    return count, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AutoMoMa HDF5 datasets between merged and per-episode layouts."
    )
    parser.add_argument("input", type=Path, help="Input .hdf5 file or split episode directory.")
    parser.add_argument("output", type=Path, help="Output split directory or merged .hdf5 file.")
    parser.add_argument(
        "--direction",
        choices=("split", "merge", "auto"),
        default="auto",
        help="'split' converts merged file -> directory; 'merge' converts directory -> merged file.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "move"),
        default="copy",
        help="'copy' keeps the input; 'move' deletes input data after conversion.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output.")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    direction = infer_direction(input_path, output_path) if args.direction == "auto" else args.direction
    move = args.mode == "move"

    if direction == "split" and move:
        print(
            "Warning: merged -> split move deletes the source after all episode files are written; "
            "HDF5 cannot reclaim per-demo space incrementally from a single source file."
        )

    if direction == "split":
        split_merged_hdf5(input_path, output_path, move=move, overwrite=args.overwrite)
        count, samples = validate_split_dir(output_path)
    else:
        merge_split_hdf5(input_path, output_path, move=move, overwrite=args.overwrite)
        count, samples = validate_hdf5_file(output_path)

    print(f"Converted {count} episode(s), {samples} sample(s): {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
