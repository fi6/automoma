#!/usr/bin/env python3
"""Watch split AutoMoMa HDF5 episodes and archive completed files remotely."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def inspect_hdf5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as root:
        if "data" not in root:
            raise ValueError("missing /data group")
        data = root["data"]
        keys = sorted(data.keys())
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
            if "success" in demo.attrs and bool(demo.attrs["success"]):
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


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def remote_inspect(host: str, remote_path: str, python_bin: str) -> dict[str, object]:
    script = r"""
import json
import sys
from pathlib import Path
import h5py

path = Path(sys.argv[1])
with h5py.File(path, "r") as root:
    if "data" not in root:
        raise ValueError("missing /data group")
    data = root["data"]
    keys = sorted(data.keys())
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
        if "success" in demo.attrs and bool(demo.attrs["success"]):
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
    cmd = [
        "ssh",
        host,
        f"HDF5_USE_FILE_LOCKING=FALSE {shlex.quote(python_bin)} -c {shlex.quote(script)} {shlex.quote(remote_path)}",
    ]
    completed = run(cmd)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def compare_hdf5_info(local_info: dict[str, object], remote_info: dict[str, object], label: str) -> None:
    keys = (
        "size_bytes",
        "demo_count",
        "samples",
        "success_count",
        "record_success_checked_count",
        "traj_indices",
    )
    for key in keys:
        if remote_info.get(key) != local_info.get(key):
            raise RuntimeError(
                f"{label} {key} mismatch: {remote_info.get(key)} != {local_info.get(key)}"
            )


def transfer_one(args: argparse.Namespace, source: Path) -> bool:
    try:
        local_info = inspect_hdf5(source)
    except Exception as exc:
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "local_not_ready",
            "file": str(source),
            "error": str(exc),
        })
        return False

    remote_file = f"{args.remote_dir.rstrip('/')}/{source.name}"
    remote_tmp = f"{remote_file}.tmp-{socket.gethostname()}-{os.getpid()}"

    try:
        try:
            remote_info = remote_inspect(args.host, remote_file, args.remote_python)
        except Exception:
            remote_info = None

        if remote_info is None:
            run(["ssh", args.host, f"mkdir -p {shlex.quote(args.remote_dir)}"])
            run(["rsync", "-a", "--partial", str(source), f"{args.host}:{remote_tmp}"])
            tmp_info = remote_inspect(args.host, remote_tmp, args.remote_python)
            compare_hdf5_info(local_info, tmp_info, "remote tmp")
            run(["ssh", args.host, f"mv -f {shlex.quote(remote_tmp)} {shlex.quote(remote_file)}"])
            remote_info = remote_inspect(args.host, remote_file, args.remote_python)

        compare_hdf5_info(local_info, remote_info, "remote final")

        source.unlink()
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "transferred_deleted_local",
            "file": str(source),
            "remote": f"{args.host}:{remote_file}",
            "info": local_info,
        })
        return True
    except Exception as exc:
        run(["ssh", args.host, f"rm -f {shlex.quote(remote_tmp)}"], check=False)
        append_jsonl(args.ledger, {
            "time": now_iso(),
            "event": "transfer_failed",
            "file": str(source),
            "remote": f"{args.host}:{remote_file}",
            "error": str(exc),
        })
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote_dir", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--interval_sec", type=int, default=300)
    parser.add_argument("--min_age_sec", type=int, default=60)
    parser.add_argument("--remote_python", default="/home/xinhai/miniconda3/envs/automoma/bin/python")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    args.ledger = args.ledger.expanduser().resolve()

    while True:
        cutoff = time.time() - args.min_age_sec
        for source in sorted(args.local_dir.glob("episode_*.hdf5")):
            if source.name.endswith(".tmp") or source.stat().st_mtime > cutoff:
                continue
            transfer_one(args, source)
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
