#!/usr/bin/env python3
"""Summarize AutoMoMa record ablation debug CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


EXP_RE = re.compile(r"i(?P<interpolated>\d+)_d(?P<decimation>\d+)_init(?P<init_steps>\d+)_len(?P<trajectory_len>\d+)")
JOINT_RE = re.compile(r"(?P<stem>.+)-joint-tracking\.csv$")


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _percentile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _exp_from_path(path: Path) -> dict[str, int]:
    for part in path.parts:
        match = EXP_RE.fullmatch(part)
        if match:
            return {key: int(value) for key, value in match.groupdict().items()}
    match = EXP_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot parse experiment params from {path}")
    return {key: int(value) for key, value in match.groupdict().items()}


def _matching_handle_csv(joint_csv: Path) -> Path:
    match = JOINT_RE.match(joint_csv.name)
    if not match:
        return joint_csv.with_name(joint_csv.name.replace("joint-tracking", "handle-tracking"))
    return joint_csv.with_name(f"{match.group('stem')}-handle-tracking.csv")


def _joint_abs_columns(row: dict[str, str]) -> list[str]:
    return [key for key in row if key.startswith("abs_")]


def _summarize_one(joint_csv: Path) -> dict[str, object]:
    params = _exp_from_path(joint_csv)
    rows = _read_csv(joint_csv)
    if not rows:
        raise ValueError(f"Empty CSV: {joint_csv}")

    all_joint = [_float(row, "all_joint_max_abs") for row in rows]
    mean_abs = [_float(row, "all_joint_mean_abs") for row in rows]
    base = [_float(row, "base_max_abs") for row in rows]
    fk_pos = [_float(row, "fk_eef_pos_distance_m") for row in rows]
    fk_rot = [_float(row, "fk_eef_rot_distance_rad") for row in rows]
    object_open = [_float(row, "object_openness") for row in rows]
    object_target = [_float(row, "object_target_openness") for row in rows]
    object_open_error = [_float(row, "object_open_error") for row in rows]
    abs_cols = _joint_abs_columns(rows[0])

    per_joint_max = {}
    per_joint_mean = {}
    for col in abs_cols:
        values = [_float(row, col) for row in rows]
        per_joint_max[col[4:]] = max(value for value in values if math.isfinite(value))
        finite_values = [value for value in values if math.isfinite(value)]
        per_joint_mean[col[4:]] = mean(finite_values) if finite_values else float("nan")

    worst_row = max(rows, key=lambda row: _float(row, "all_joint_max_abs"))
    worst_joint = max(abs_cols, key=lambda col: _float(worst_row, col)) if abs_cols else ""
    first_episode_rows = [row for row in rows if row.get("episode") == "0"]
    last_episode = rows[-1].get("episode")
    last_episode_rows = [row for row in rows if row.get("episode") == last_episode]

    handle_csv = _matching_handle_csv(joint_csv)
    handle_rows = _read_csv(handle_csv) if handle_csv.exists() else []
    handle_distance = [_float(row, "handle_distance") for row in handle_rows]
    handle_final = _float(handle_rows[-1], "handle_distance") if handle_rows else float("nan")
    handle_min = min((value for value in handle_distance if math.isfinite(value)), default=float("nan"))

    final_rows_by_episode: dict[str, dict[str, str]] = {}
    for row in rows:
        final_rows_by_episode[row["episode"]] = row
    final_open_values = [_float(row, "object_openness") for row in final_rows_by_episode.values()]
    final_open_errors = [_float(row, "object_open_error") for row in final_rows_by_episode.values()]
    final_fk_values = [_float(row, "fk_eef_pos_distance_m") for row in final_rows_by_episode.values()]

    final_handle_by_episode: dict[str, float] = {}
    for row in handle_rows:
        final_handle_by_episode[row["episode"]] = _float(row, "handle_distance")
    episode_success = []
    door_open_success = []
    handle_engaged_success = []
    target_match_success = []
    for episode, final_row in final_rows_by_episode.items():
        door_open = _float(final_row, "object_openness") >= 0.3
        open_match = abs(_float(final_row, "object_open_error")) <= 0.1
        handle_engaged = final_handle_by_episode.get(episode, float("nan")) <= 0.1
        door_open_success.append(door_open)
        target_match_success.append(open_match)
        handle_engaged_success.append(handle_engaged)
        episode_success.append(door_open and open_match and handle_engaged)

    summary = {
        **params,
        "name": next(part for part in joint_csv.parts if EXP_RE.fullmatch(part)),
        "joint_csv": str(joint_csv),
        "handle_csv": str(handle_csv) if handle_csv.exists() else "",
        "samples": len(rows),
        "episodes": len(final_rows_by_episode),
        "all_joint_max": max(all_joint),
        "all_joint_p95": _percentile(all_joint, 0.95),
        "all_joint_mean": mean(value for value in mean_abs if math.isfinite(value)),
        "base_max": max(base),
        "fk_pos_max": max(value for value in fk_pos if math.isfinite(value)),
        "fk_pos_p95": _percentile(fk_pos, 0.95),
        "fk_pos_final_mean": mean(value for value in final_fk_values if math.isfinite(value)),
        "fk_rot_max": max(value for value in fk_rot if math.isfinite(value)),
        "door_open_max": max(value for value in object_open if math.isfinite(value)),
        "door_open_final_mean": mean(value for value in final_open_values if math.isfinite(value)),
        "door_open_error_final_abs_mean": mean(abs(value) for value in final_open_errors if math.isfinite(value)),
        "handle_distance_min": handle_min,
        "handle_distance_final": handle_final,
        "door_open_rate": sum(door_open_success) / len(door_open_success) if door_open_success else 0.0,
        "target_match_rate": sum(target_match_success) / len(target_match_success) if target_match_success else 0.0,
        "handle_engaged_rate": sum(handle_engaged_success) / len(handle_engaged_success) if handle_engaged_success else 0.0,
        "success_rate": sum(episode_success) / len(episode_success) if episode_success else 0.0,
        "success_count": sum(episode_success),
        "worst_sample": int(worst_row["sample"]),
        "worst_episode": int(worst_row["episode"]),
        "worst_step": int(worst_row["step"]),
        "worst_joint": worst_joint[4:] if worst_joint.startswith("abs_") else worst_joint,
        "worst_joint_error": _float(worst_row, worst_joint) if worst_joint else float("nan"),
        "first_episode_max": max((_float(row, "all_joint_max_abs") for row in first_episode_rows), default=float("nan")),
        "last_episode_max": max((_float(row, "all_joint_max_abs") for row in last_episode_rows), default=float("nan")),
        "per_joint_max": per_joint_max,
        "per_joint_mean": per_joint_mean,
    }
    summary["score"] = _score(summary)
    return summary


def _score(row: dict[str, object]) -> float:
    fk = float(row["fk_pos_p95"])
    joint = float(row["all_joint_p95"])
    door_err = float(row["door_open_error_final_abs_mean"])
    handle = float(row["handle_distance_final"])
    door_final = float(row["door_open_final_mean"])
    interpolated = float(row["interpolated"])
    decimation = float(row["decimation"])
    success_rate = float(row.get("success_rate", 0.0))
    handle_rate = float(row.get("handle_engaged_rate", 0.0))
    target_rate = float(row.get("target_match_rate", 0.0))

    penalty = 0.0
    penalty += 20.0 * (1.0 - success_rate)
    penalty += 4.0 * (1.0 - handle_rate)
    penalty += 4.0 * (1.0 - target_rate)
    penalty += 30.0 * min(fk, 1.0)
    penalty += 8.0 * min(joint, 1.0)
    penalty += 10.0 * min(door_err, 1.0)
    if math.isfinite(handle):
        penalty += 5.0 * min(handle, 1.0)
    penalty -= 4.0 * min(max(door_final, 0.0), 1.0)
    penalty += 0.03 * interpolated + 0.02 * decimation
    return penalty


def _write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    scalar_keys = [
        key
        for key in rows[0].keys()
        if key not in {"per_joint_max", "per_joint_mean"}
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in scalar_keys})


def _write_report(rows: list[dict[str, object]], path: Path) -> None:
    best = rows[0]
    lines = [
        "# Record Ablation Summary",
        "",
        f"Experiments analyzed: {len(rows)}",
        "",
        "## Best By Score",
        "",
    ]
    for row in rows[:10]:
        lines.append(
            "- {name}: score={score:.4f}, i={interpolated}, d={decimation}, init={init_steps}, "
            "len={trajectory_len}, fk_p95={fk_pos_p95:.5f}m, joint_p95={all_joint_p95:.5f}, "
            "door_final={door_open_final_mean:.5f}, door_err={door_open_error_final_abs_mean:.5f}, "
            "handle_final={handle_distance_final:.5f}, success={success_count}/{episodes}".format(**row)
        )
    lines.extend(
        [
            "",
            "## Worst Spike In Best Experiment",
            "",
            "- sample={worst_sample}, episode={worst_episode}, step={worst_step}, joint={worst_joint}, abs_error={worst_joint_error:.6f}".format(
                **best
            ),
            "",
            "## Per-Joint Max In Best Experiment",
            "",
        ]
    )
    per_joint = sorted(best["per_joint_max"].items(), key=lambda item: item[1], reverse=True)
    for name, value in per_joint:
        lines.append(f"- {name}: {value:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plot generation unavailable: {exc}")
        return

    x = [float(row["score"]) for row in rows]
    y = [float(row["fk_pos_p95"]) for row in rows]
    colors = [float(row["interpolated"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    scatter = ax.scatter(x, y, c=colors, cmap="viridis", s=36)
    ax.set_xlabel("score (lower is better)")
    ax.set_ylabel("FK EEF p95 error (m)")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="interpolated")
    fig.savefig(output_dir / "score_vs_fk_p95.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.scatter(
        [float(row["interpolated"]) for row in rows],
        [float(row["door_open_final_mean"]) for row in rows],
        c=[float(row["decimation"]) for row in rows],
        cmap="plasma",
        s=36,
    )
    ax.set_xlabel("interpolated")
    ax.set_ylabel("mean final door openness")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "door_final_by_interpolation.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/automoma/ablation_study"))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    joint_csvs = sorted(root.glob("*/debug_curves/*-joint-tracking.csv"))
    if not joint_csvs:
        raise FileNotFoundError(f"No joint tracking CSVs found under {root}")

    rows = [_summarize_one(path) for path in joint_csvs]
    rows.sort(key=lambda row: float(row["score"]))

    _write_summary_csv(rows, output_dir / "summary.csv")
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(rows, output_dir / "report.md")
    _write_plots(rows, output_dir)
    print(f"[done] wrote {output_dir / 'summary.csv'}")
    print(f"[best] {rows[0]['name']} score={rows[0]['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
