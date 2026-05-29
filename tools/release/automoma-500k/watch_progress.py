#!/usr/bin/env python3
"""Watch AutoMoMa-500k planning progress and append a markdown status log."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]
ERROR_PATTERN = re.compile(
    r"Traceback|ValueError|RuntimeError|CUDA out of memory|No space left|Planning failed|No progress possible",
    re.IGNORECASE,
)
PLAN_PATTERN = re.compile(
    r"^(Round |Planning |Skipping |Object target satisfied|=== Object|===== RESTART|Done\.)"
)
RESTART_PATTERN = re.compile(r"^===== (?:WATCHDOG )?RESTART")


def load_objects(config: Path) -> list[str]:
    cfg = OmegaConf.load(config)
    data = OmegaConf.to_container(cfg, resolve=True)
    objects = []
    for object_id, obj_cfg in data["objects"].items():
        objects.append(f"{obj_cfg['asset_type'].lower()}_{object_id}")
    return objects


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        return None
    if not text.isdigit():
        return None
    return int(text)


def command_output(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout.strip()
    except Exception as exc:
        return f"COMMAND_ERROR {' '.join(cmd)}: {exc}"


def worker_status(log_dir: Path) -> list[dict[str, Any]]:
    workers = []
    for pid_file in sorted(log_dir.glob("gpu*.pid")):
        match = re.fullmatch(r"gpu(\d+)\.pid", pid_file.name)
        if not match:
            continue
        worker = int(match.group(1))
        pid = read_pid(pid_file)
        alive = pid_alive(pid) if pid is not None else False
        ps = ""
        if pid is not None:
            ps = command_output(["ps", "-p", str(pid), "-o", "pid=,ppid=,sid=,stat=,etime=,%cpu=,%mem=,cmd="])
        workers.append(
            {
                "worker": worker,
                "pid": pid,
                "alive": alive,
                "ps": ps,
                "cmd_file": log_dir / f"gpu{worker}.cmd.sh",
                "log_file": log_dir / f"gpu{worker}.log",
            }
        )
    return workers


def gpu_status() -> list[str]:
    out = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def count_progress(output_root: Path, split: str, objects: list[str]) -> dict[str, dict[str, int]]:
    progress = {obj: {"success": 0, "files": 0, "nonzero_scenes": 0} for obj in objects}
    root = output_root / "summit_franka"
    for path in root.glob(f"*/*/{split}/traj_data_{split}.pt"):
        obj = path.parts[-4]
        if obj not in progress:
            progress[obj] = {"success": 0, "files": 0, "nonzero_scenes": 0}
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            success = payload.get("traj_success")
            if isinstance(success, torch.Tensor):
                count = int(success.bool().sum().item())
            else:
                traj_robot = payload.get("traj_robot")
                count = int(traj_robot.shape[0]) if isinstance(traj_robot, torch.Tensor) else 0
        except Exception:
            count = 0
        progress[obj]["success"] += count
        progress[obj]["files"] += 1
        if count > 0:
            progress[obj]["nonzero_scenes"] += 1
    return progress


def matching_tail(path: Path, pattern: re.Pattern[str], limit: int = 12) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return [line for line in lines if pattern.search(line)][-limit:]


def matching_tail_since_restart(path: Path, pattern: re.Pattern[str], limit: int = 12) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    start = 0
    for idx, line in enumerate(lines):
        if RESTART_PATTERN.search(line):
            start = idx + 1
    return [line for line in lines[start:] if pattern.search(line)][-limit:]


def worker_completed_cleanly(log_file: Path) -> bool:
    try:
        lines = log_file.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return False

    start = 0
    for idx, line in enumerate(lines):
        if RESTART_PATTERN.search(line):
            start = idx + 1

    latest_done = None
    latest_error = None
    for idx, line in enumerate(lines[start:], start=start):
        if line.startswith("Done. Manifest:"):
            latest_done = idx
        if ERROR_PATTERN.search(line):
            latest_error = idx

    return latest_done is not None and (latest_error is None or latest_done > latest_error)


def restart_dead_workers(log_dir: Path, workers: list[dict[str, Any]], should_run: bool) -> list[str]:
    actions = []
    if not should_run:
        return actions
    for worker in workers:
        if worker["alive"]:
            continue
        cmd_file = worker["cmd_file"]
        log_file = worker["log_file"]
        if not cmd_file.exists():
            actions.append(f"worker {worker['worker']}: cmd missing, cannot restart")
            continue
        with log_file.open("a", encoding="utf-8") as log:
            log.write(f"\n===== WATCHDOG RESTART gpu{worker['worker']} {datetime.now().astimezone().isoformat()} =====\n")
            log.flush()
            proc = subprocess.Popen(
                ["setsid", "/usr/bin/bash", str(cmd_file)],
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=False,
            )
        (log_dir / f"gpu{worker['worker']}.pid").write_text(f"{proc.pid}\n", encoding="utf-8")
        actions.append(f"worker {worker['worker']}: restarted pid={proc.pid}")
    return actions


def append_report(
    *,
    progress_file: Path,
    log_dir: Path,
    workers: list[dict[str, Any]],
    progress: dict[str, dict[str, int]],
    target_per_object: int,
    target_total: int,
    actions: list[str],
) -> None:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    total_success = sum(item["success"] for item in progress.values())
    total_remaining = max(0, target_total - total_success)
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    if not progress_file.exists():
        progress_file.write_text("# AutoMoMa-500k Progress\n\n", encoding="utf-8")

    lines = [
        f"## {now}",
        "",
        f"- log_dir: `{log_dir}`",
        f"- total: `{total_success}/{target_total}` success, remaining `{total_remaining}`",
        "",
        "| object | success | target | remaining | percent | files | nonzero scenes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for obj in sorted(progress):
        success = progress[obj]["success"]
        remaining = max(0, target_per_object - success)
        pct = (success / target_per_object * 100.0) if target_per_object else 0.0
        lines.append(
            f"| `{obj}` | {success} | {target_per_object} | {remaining} | {pct:.2f}% | "
            f"{progress[obj]['files']} | {progress[obj]['nonzero_scenes']} |"
        )

    lines.extend(
        [
            "",
            "| worker | pid | alive | ps |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for worker in workers:
        ps = worker["ps"].replace("|", "\\|") if worker["ps"] else ""
        lines.append(f"| {worker['worker']} | {worker['pid'] or ''} | {worker['alive']} | `{ps}` |")

    lines.extend(["", "| gpu | used MiB | total MiB | util % |", "| ---: | ---: | ---: | ---: |"])
    for row in gpu_status():
        parts = [part.strip() for part in row.split(",")]
        if len(parts) == 4:
            lines.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} |")
        else:
            lines.append(f"| `{row}` |  |  |  |")

    if actions:
        lines.extend(["", "Actions:"])
        lines.extend(f"- {action}" for action in actions)
    else:
        lines.extend(["", "Actions: none"])

    lines.extend(["", "Recent planning lines:"])
    for worker in workers:
        lines.append(f"- gpu{worker['worker']}:")
        recent = matching_tail(worker["log_file"], PLAN_PATTERN, limit=8)
        if not recent:
            lines.append("  - no recent planning line")
        else:
            lines.extend(f"  - `{line[-240:]}`" for line in recent)

    errors = []
    for worker in workers:
        errors.extend(
            f"gpu{worker['worker']}: {line[-240:]}"
            for line in matching_tail_since_restart(worker["log_file"], ERROR_PATTERN, limit=6)
        )
    if errors:
        lines.extend(["", "Recent errors:"])
        lines.extend(f"- `{line}`" for line in errors[-12:])
    else:
        lines.extend(["", "Recent errors: none"])

    lines.append("")
    progress_file.open("a", encoding="utf-8").write("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append AutoMoMa-500k progress reports and restart dead workers.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, default=REPO_ROOT / "logs" / "automoma-500k-progress.md")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "plan.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "trajs")
    parser.add_argument("--split", default="train")
    parser.add_argument("--target-per-object", type=int, default=100_000)
    parser.add_argument("--target-total", type=int, default=500_000)
    parser.add_argument("--interval-seconds", type=int, default=7200)
    parser.add_argument("--restart-dead", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    objects = load_objects(args.config)

    while True:
        workers = worker_status(args.log_dir)
        progress = count_progress(args.output_root, args.split, objects)
        total_success = sum(item["success"] for item in progress.values())
        all_done = total_success >= args.target_total and all(
            item["success"] >= args.target_per_object for item in progress.values()
        )
        completed_workers = [
            worker
            for worker in workers
            if not worker["alive"] and worker_completed_cleanly(worker["log_file"])
        ]
        dead_workers = [
            worker
            for worker in workers
            if not worker["alive"] and not worker_completed_cleanly(worker["log_file"])
        ]
        actions = []
        if completed_workers and not all_done:
            actions.extend(
                f"worker {worker['worker']}: exited cleanly after shard completion; no restart needed"
                for worker in completed_workers
            )
        if args.restart_dead and not all_done and dead_workers:
            actions.extend(restart_dead_workers(args.log_dir, dead_workers, should_run=True))
        if actions:
            workers = worker_status(args.log_dir)
        append_report(
            progress_file=args.progress_file,
            log_dir=args.log_dir,
            workers=workers,
            progress=progress,
            target_per_object=args.target_per_object,
            target_total=args.target_total,
            actions=actions,
        )
        if args.once or all_done:
            break
        time.sleep(args.interval_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
