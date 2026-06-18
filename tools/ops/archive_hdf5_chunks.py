#!/usr/bin/env python3
"""Archive completed split/chunk HDF5 files from a local or SSH source."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def inspect_hdf5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as root:
        if "data" not in root:
            raise ValueError("missing /data group")
        data = root["data"]
        keys = sorted(data.keys(), key=_demo_sort_key)
        if not keys:
            raise ValueError("no demo groups under /data")

        samples = 0
        successes = 0
        checked = 0
        traj_indices: list[int] = []
        for key in keys:
            demo = data[key]
            demo_samples = int(demo.attrs.get("num_samples", 0))
            if demo_samples <= 0:
                raise ValueError(f"{key} has non-positive num_samples={demo_samples}")
            samples += demo_samples
            if bool(demo.attrs.get("success", False)):
                successes += 1
            if bool(demo.attrs.get("record_success_checked", False)):
                checked += 1
            if "traj_index" in demo.attrs:
                traj_indices.append(int(demo.attrs["traj_index"]))

    return {
        "demo_count": len(keys),
        "samples": samples,
        "success_count": successes,
        "record_success_checked_count": checked,
        "traj_indices": traj_indices,
        "size_bytes": path.stat().st_size,
    }


def _demo_sort_key(name: str) -> tuple[int, str]:
    prefix, _, suffix = name.rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return int(suffix), name
    return 1_000_000_000, name


REMOTE_INSPECT = r"""
import json
import sys
from pathlib import Path

import h5py

def demo_sort_key(name):
    prefix, _, suffix = name.rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return int(suffix), name
    return 1_000_000_000, name

path = Path(sys.argv[1])
with h5py.File(path, "r") as root:
    if "data" not in root:
        raise ValueError("missing /data group")
    data = root["data"]
    keys = sorted(data.keys(), key=demo_sort_key)
    if not keys:
        raise ValueError("no demo groups under /data")
    samples = 0
    successes = 0
    checked = 0
    traj_indices = []
    for key in keys:
        demo = data[key]
        demo_samples = int(demo.attrs.get("num_samples", 0))
        if demo_samples <= 0:
            raise ValueError(f"{key} has non-positive num_samples={demo_samples}")
        samples += demo_samples
        if bool(demo.attrs.get("success", False)):
            successes += 1
        if bool(demo.attrs.get("record_success_checked", False)):
            checked += 1
        if "traj_index" in demo.attrs:
            traj_indices.append(int(demo.attrs["traj_index"]))
print(json.dumps({
    "demo_count": len(keys),
    "samples": samples,
    "success_count": successes,
    "record_success_checked_count": checked,
    "traj_indices": traj_indices,
    "size_bytes": path.stat().st_size,
}))
"""


REMOTE_LIST = r"""
import json
import sys
import time
from pathlib import Path

source = Path(sys.argv[1])
pattern = sys.argv[2]
min_age = float(sys.argv[3])
recursive = sys.argv[4].lower() == "true"
cutoff = time.time() - min_age
rows = []
if source.is_dir():
    iterator = source.rglob(pattern) if recursive else source.glob(pattern)
    for path in sorted(iterator):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if path.is_file() and stat.st_mtime <= cutoff:
            rows.append({
                "name": path.name,
                "path": str(path),
                "relative": str(path.relative_to(source)),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            })
print(json.dumps(rows))
"""


def remote_python(args: argparse.Namespace, script: str, *script_args: str) -> str:
    quoted_args = " ".join(shlex.quote(value) for value in script_args)
    return f"HDF5_USE_FILE_LOCKING=FALSE {args.remote_python} -c {shlex.quote(script)} {quoted_args}"


def remote_list(args: argparse.Namespace) -> list[dict[str, Any]]:
    completed = run([
        "ssh",
        args.source_host,
        remote_python(
            args,
            REMOTE_LIST,
            args.source_dir,
            args.pattern,
            str(args.min_age_sec),
            "true" if args.recursive else "false",
        ),
    ])
    output = completed.stdout.strip().splitlines()
    return json.loads(output[-1]) if output else []


def remote_inspect(args: argparse.Namespace, remote_path: str) -> dict[str, Any]:
    completed = run(["ssh", args.source_host, remote_python(args, REMOTE_INSPECT, remote_path)])
    return json.loads(completed.stdout.strip().splitlines()[-1])


def local_candidates(source_dir: Path, pattern: str, min_age_sec: float, recursive: bool) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - min_age_sec
    out = []
    iterator = source_dir.rglob(pattern) if recursive else source_dir.glob(pattern)
    for path in sorted(iterator):
        if path.is_file() and path.stat().st_mtime <= cutoff:
            out.append(path)
    return out


def compare_info(source_info: dict[str, Any], dest_info: dict[str, Any], label: str) -> None:
    keys = (
        "size_bytes",
        "demo_count",
        "samples",
        "success_count",
        "record_success_checked_count",
        "traj_indices",
    )
    for key in keys:
        if source_info.get(key) != dest_info.get(key):
            raise RuntimeError(f"{label} {key} mismatch: {dest_info.get(key)} != {source_info.get(key)}")


def tmp_dest_for(dest: Path) -> Path:
    return dest.with_name(f".{dest.name}.tmp-{socket.gethostname()}-{os.getpid()}")


def archive_local(args: argparse.Namespace, source: Path) -> bool:
    source_root = Path(args.source_dir)
    relative = source.relative_to(source_root) if args.recursive else Path(source.name)
    dest = args.dest_dir / relative
    try:
        source_info = inspect_hdf5(source)
        if dest.exists():
            compare_info(source_info, inspect_hdf5(dest), "existing dest")
            if args.delete_source:
                source.unlink()
            append_jsonl(args.ledger, {
                "time": now_iso(),
                "event": "dest_exists_verified",
                "source": str(source),
                "dest": str(dest),
                "info": source_info,
            })
            return True

        tmp = tmp_dest_for(dest)
        if tmp.exists():
            tmp.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, tmp)
        compare_info(source_info, inspect_hdf5(tmp), "local tmp")
        os.replace(tmp, dest)
        compare_info(source_info, inspect_hdf5(dest), "local final")
        if args.delete_source:
            source.unlink()
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "archived_local",
            "source": str(source),
            "dest": str(dest),
            "info": source_info,
        })
        return True
    except Exception as exc:
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "archive_local_failed",
            "source": str(source),
            "dest": str(dest),
            "error": str(exc),
        })
        return False


def archive_remote(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    remote_path = str(row["path"])
    dest = args.dest_dir / str(row.get("relative") or row["name"])
    tmp = tmp_dest_for(dest)
    try:
        source_info = remote_inspect(args, remote_path)
        if dest.exists():
            compare_info(source_info, inspect_hdf5(dest), "existing dest")
            if args.delete_source:
                run(["ssh", args.source_host, f"rm -f {shlex.quote(remote_path)}"])
            append_jsonl(args.ledger, {
                "time": now_iso(),
                "event": "dest_exists_verified_remote_deleted",
                "source": f"{args.source_host}:{remote_path}",
                "dest": str(dest),
                "info": source_info,
            })
            return True

        if tmp.exists():
            tmp.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["rsync", "-a", "--partial", f"{args.source_host}:{remote_path}", str(tmp)])
        compare_info(source_info, inspect_hdf5(tmp), "remote tmp")
        os.replace(tmp, dest)
        compare_info(source_info, inspect_hdf5(dest), "remote final")
        if args.delete_source:
            run(["ssh", args.source_host, f"rm -f {shlex.quote(remote_path)}"])
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "archived_remote",
            "source": f"{args.source_host}:{remote_path}",
            "dest": str(dest),
            "info": source_info,
        })
        return True
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "archive_remote_failed",
            "source": f"{args.source_host}:{remote_path}",
            "dest": str(dest),
            "error": str(exc),
        })
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Local or remote source directory.")
    parser.add_argument("--dest-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--source-host", default="", help="SSH host for remote source. Omit for local source.")
    parser.add_argument("--pattern", default="chunk_*.hdf5")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--min-age-sec", type=float, default=10.0)
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--remote-python", default="python")
    parser.add_argument("--delete-source", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_once(args: argparse.Namespace) -> tuple[int, int]:
    attempted = 0
    archived = 0
    if args.source_host:
        for row in remote_list(args):
            attempted += 1
            archived += int(archive_remote(args, row))
    else:
        for source in local_candidates(Path(args.source_dir), args.pattern, args.min_age_sec, args.recursive):
            attempted += 1
            archived += int(archive_local(args, source))
    return attempted, archived


def main() -> int:
    args = parse_args()
    args.dest_dir = args.dest_dir.expanduser().resolve()
    args.ledger = args.ledger.expanduser().resolve()

    while True:
        attempted, archived = run_once(args)
        if attempted:
            print(f"{now_iso()} attempted={attempted} archived={archived}", flush=True)
        if args.once:
            return 0 if attempted == archived else 1
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    raise SystemExit(main())
