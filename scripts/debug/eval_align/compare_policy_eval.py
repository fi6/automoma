#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


JOINT_NAMES = (
    "base_x",
    "base_y",
    "base_z",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)

JOINT_GROUPS = (
    ("base", ("base_x", "base_y", "base_z")),
    ("arm 1-4", ("panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4")),
    ("arm 5-7", ("panda_joint5", "panda_joint6", "panda_joint7")),
    ("gripper", ("panda_finger_joint1", "panda_finger_joint2")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a policy eval alignment report from eval trace artifacts.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run descriptor in LABEL=OUTPUT_DIR or LABEL:OUTPUT_DIR form. Can be passed multiple times.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_run_descriptor(raw: str) -> tuple[str, Path]:
    sep = "=" if "=" in raw else ":"
    if sep not in raw:
        raise ValueError(f"Invalid --run value: {raw}. Use LABEL=OUTPUT_DIR.")
    label, path = raw.split(sep, 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid --run label: {raw}")
    return label, Path(path).expanduser()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fmt(value: float, ndigits: int = 4) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{ndigits}f}"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def rows_by_episode(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["episode_ix"])].append(row)
    return grouped


def trace_arrays(rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    by_joint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_joint[row["joint_name"]].append(row)

    output: dict[str, dict[str, np.ndarray]] = {}
    for joint_name, joint_rows in by_joint.items():
        joint_rows = sorted(joint_rows, key=lambda r: (int(r["sim_trace_step_ix"]), int(r["joint_ix"])))
        raw_steps = np.asarray([int(r["sim_trace_step_ix"]) for r in joint_rows], dtype=np.int32)
        local_steps = raw_steps - int(raw_steps.min()) if raw_steps.size else raw_steps
        output[joint_name] = {
            "step": local_steps,
            "policy": np.asarray([to_float(r.get("policy_action_abs")) for r in joint_rows], dtype=np.float64),
            "target": np.asarray([to_float(r.get("interpolated_action_abs")) for r in joint_rows], dtype=np.float64),
            "state": np.asarray([to_float(r.get("sim_joint_pos_after")) for r in joint_rows], dtype=np.float64),
            "sample": np.asarray([to_bool(r.get("is_policy_action_sample")) for r in joint_rows], dtype=bool),
        }
    return output


def episode_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    arrays = trace_arrays(rows)
    group_errors: dict[str, list[float]] = {"base": [], "arm": [], "gripper": [], "all": []}
    group_max_errors: dict[str, list[float]] = {"base": [], "arm": [], "gripper": [], "all": []}
    policy_jump: list[float] = []

    for joint_name, data in arrays.items():
        err = np.abs(data["state"] - data["target"])
        err = err[np.isfinite(err)]
        if not err.size:
            continue
        if joint_name.startswith("base_"):
            group = "base"
        elif joint_name.startswith("panda_finger"):
            group = "gripper"
        else:
            group = "arm"
        group_errors[group].append(float(np.mean(err)))
        group_max_errors[group].append(float(np.max(err)))
        group_errors["all"].append(float(np.mean(err)))
        group_max_errors["all"].append(float(np.max(err)))

        samples = data["policy"][data["sample"]]
        samples = samples[np.isfinite(samples)]
        if samples.size > 1:
            policy_jump.append(float(np.mean(np.abs(np.diff(samples)))))

    def mean_or_nan(values: list[float]) -> float:
        return float(np.mean(values)) if values else np.nan

    return {
        "mean_abs_track_error": mean_or_nan(group_errors["all"]),
        "max_abs_track_error": mean_or_nan(group_max_errors["all"]),
        "base_mean_abs_track_error": mean_or_nan(group_errors["base"]),
        "base_max_abs_track_error": mean_or_nan(group_max_errors["base"]),
        "arm_mean_abs_track_error": mean_or_nan(group_errors["arm"]),
        "arm_max_abs_track_error": mean_or_nan(group_max_errors["arm"]),
        "gripper_mean_abs_track_error": mean_or_nan(group_errors["gripper"]),
        "gripper_max_abs_track_error": mean_or_nan(group_max_errors["gripper"]),
        "mean_abs_policy_jump": mean_or_nan(policy_jump),
    }


def plot_trace(rows: list[dict[str, str]], title: str, out_path: Path) -> None:
    arrays = trace_arrays(rows)
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (group_name, joints) in zip(axes, JOINT_GROUPS, strict=True):
        plotted = False
        for idx, joint_name in enumerate(joints):
            if joint_name not in arrays:
                continue
            data = arrays[joint_name]
            c = colors[idx % len(colors)]
            ax.plot(data["step"], data["state"], color=c, linewidth=2.0, label=f"{joint_name} state")
            ax.plot(data["step"], data["target"], color=c, linewidth=1.4, linestyle="--", alpha=0.85, label=f"{joint_name} eval target")
            sample_steps = data["step"][data["sample"]]
            sample_policy = data["policy"][data["sample"]]
            ax.scatter(sample_steps, sample_policy, color=c, s=12, marker="x", alpha=0.8, label=f"{joint_name} policy target")
            plotted = True
        ax.set_title(group_name, loc="left", fontsize=11, weight="bold")
        ax.grid(True, alpha=0.22)
        ax.set_ylabel("joint value")
        if plotted:
            ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper right")
    axes[-1].set_xlabel("eval sim step")
    fig.suptitle(title, fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_heatmap(rows: list[dict[str, str]], title: str, out_path: Path) -> None:
    arrays = trace_arrays(rows)
    ordered = [name for name in JOINT_NAMES if name in arrays]
    if not ordered:
        return
    max_len = max(len(arrays[name]["step"]) for name in ordered)
    heat = np.full((len(ordered), max_len), np.nan, dtype=np.float64)
    for row_ix, joint_name in enumerate(ordered):
        data = arrays[joint_name]
        err = np.abs(data["state"] - data["target"])
        heat[row_ix, : len(err)] = err

    fig, ax = plt.subplots(figsize=(15, 5.4))
    finite = heat[np.isfinite(heat)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    vmax = max(vmax, 1e-4)
    im = ax.imshow(heat, aspect="auto", interpolation="nearest", cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_yticks(np.arange(len(ordered)), labels=ordered)
    ax.set_xlabel("eval sim step")
    ax.set_title(title, loc="left", fontsize=13, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("|eval state - eval target|")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def copy_video(src: str, output_dir: Path, label: str, episode_ix: int) -> str:
    if not src:
        return ""
    src_path = Path(src)
    if not src_path.exists():
        return ""
    dst = output_dir / "assets" / "videos" / f"{safe_name(label)}_episode_{episode_ix}{src_path.suffix}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)
    return dst.relative_to(output_dir).as_posix()


def load_run(label: str, output_dir: Path, report_dir: Path) -> dict[str, Any]:
    per_episode_rows = read_csv_rows(output_dir / "per_episode_results.csv")
    trace_rows = read_csv_rows(output_dir / "action_trace_joint_states.csv")
    info = read_json(output_dir / "eval_info.json")
    trace_by_ep = rows_by_episode(trace_rows)
    per_episode: dict[int, dict[str, Any]] = {}
    for row in per_episode_rows:
        episode_ix = int(row["episode_ix"])
        metrics = episode_metrics(trace_by_ep.get(episode_ix, []))
        video_rel = copy_video(row.get("video_path", ""), report_dir, label, episode_ix)
        per_episode[episode_ix] = {
            "row": row,
            "trace_rows": trace_by_ep.get(episode_ix, []),
            "metrics": metrics,
            "video_rel": video_rel,
        }
    return {
        "label": label,
        "output_dir": output_dir,
        "eval_info": info,
        "per_episode": per_episode,
        "trace_rows": trace_rows,
    }


def aggregate_metrics(run: dict[str, Any]) -> dict[str, float]:
    episodes = list(run["per_episode"].values())
    successes = [to_bool(ep["row"].get("success")) for ep in episodes]
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for ep in episodes:
        for key, value in ep["metrics"].items():
            if np.isfinite(value):
                values_by_key[key].append(value)
    final_openness = [to_float(ep["row"].get("final_door_openness")) for ep in episodes]
    final_handle = [to_float(ep["row"].get("final_handle_distance")) for ep in episodes]

    def mean(values: list[float]) -> float:
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else np.nan

    out = {
        "episodes": float(len(episodes)),
        "success_rate": float(np.mean(successes) * 100.0) if successes else np.nan,
        "mean_final_door_openness": mean(final_openness),
        "mean_final_handle_distance": mean(final_handle),
    }
    for key, values in values_by_key.items():
        out[key] = mean(values)
    return out


def max_trace_error(run: dict[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    max_error = -1.0
    for row in run["trace_rows"]:
        target = to_float(row.get("interpolated_action_abs"))
        state = to_float(row.get("sim_joint_pos_after"))
        error = abs(state - target)
        if np.isfinite(error) and error > max_error:
            max_error = error
            best = dict(row)
            best["abs_error"] = error
    return best


def render_auto_analysis(runs: list[dict[str, Any]]) -> str:
    items = []
    for run in runs:
        agg = aggregate_metrics(run)
        episodes = list(run["per_episode"].values())
        door_open = sum(to_bool(ep["row"].get("final_door_open")) for ep in episodes)
        success = sum(to_bool(ep["row"].get("success")) for ep in episodes)
        max_row = max_trace_error(run)
        items.append(
            f"""
            <div class="analysis-item">
              <h3>{html.escape(run["label"])}</h3>
              <p>成功 {success}/{len(episodes)}，final_door_open {door_open}/{len(episodes)}。
              平均 handle distance 是 {fmt(agg.get("mean_final_handle_distance", np.nan), 4)}，
              平均 door openness 是 {fmt(agg.get("mean_final_door_openness", np.nan), 4)}。</p>
              <p>整体 mean tracking error 是 {fmt(agg.get("mean_abs_track_error", np.nan), 5)}，
              base/arm/gripper mean 分别是 {fmt(agg.get("base_mean_abs_track_error", np.nan), 5)} /
              {fmt(agg.get("arm_mean_abs_track_error", np.nan), 5)} /
              {fmt(agg.get("gripper_mean_abs_track_error", np.nan), 5)}。</p>
              <p>最大单点误差出现在 episode {html.escape(str(max_row.get("episode_ix", "n/a")))}
              step {html.escape(str(max_row.get("policy_step_ix", "n/a")))}
              的 {html.escape(str(max_row.get("joint_name", "n/a")))}：
              eval target={html.escape(str(max_row.get("interpolated_action_abs", "n/a")))},
              eval state={html.escape(str(max_row.get("sim_joint_pos_after", "n/a")))},
              |error|={fmt(float(max_row.get("abs_error", np.nan)), 4)}。</p>
            </div>
            """
        )
    return f"""
    <section class="analysis">
      <h2>自动判读摘要</h2>
      <p class="analysis-lead">当前 report 的核心判断逻辑是先看 policy target 和 eval target 是否一致，再看 eval target 和 eval state 是否能跟上。平均 tracking 很小但最大误差很大时，通常说明绝大多数时间 execution 对齐，少量 policy target outlier 或末端跳变拉高了 max。</p>
      <div class="analysis-grid">{''.join(items)}</div>
    </section>
    """


def video_tag(rel_path: str) -> str:
    if not rel_path:
        return '<div class="missing">no video</div>'
    escaped = html.escape(rel_path)
    return f'<video src="{escaped}" controls muted preload="metadata"></video>'


def image_tag(rel_path: str, alt: str) -> str:
    return f'<img src="{html.escape(rel_path)}" alt="{html.escape(alt)}" loading="lazy" />'


def choose_episodes(runs: list[dict[str, Any]], n: int, seed: int) -> list[int]:
    episode_set: set[int] = set()
    for run in runs:
        episode_set.update(run["per_episode"].keys())
    episodes = sorted(episode_set)
    if len(episodes) <= n:
        return episodes
    rng = np.random.default_rng(seed)
    picked = sorted(rng.choice(np.asarray(episodes), size=n, replace=False).tolist())
    return [int(x) for x in picked]


def render_html(runs: list[dict[str, Any]], selected_episodes: list[int], output_dir: Path) -> str:
    cards = []
    for run in runs:
        agg = aggregate_metrics(run)
        align = run["eval_info"].get("alignment", {})
        cards.append(
            f"""
            <section class="card">
              <div class="card-title">{html.escape(run["label"])}</div>
              <div class="metric-grid">
                <div><span>success</span><strong>{fmt(agg.get("success_rate", np.nan), 1)}%</strong></div>
                <div><span>episodes</span><strong>{int(agg.get("episodes", 0))}</strong></div>
                <div><span>door openness</span><strong>{fmt(agg.get("mean_final_door_openness", np.nan), 4)}</strong></div>
                <div><span>handle dist</span><strong>{fmt(agg.get("mean_final_handle_distance", np.nan), 4)}</strong></div>
                <div><span>track mean</span><strong>{fmt(agg.get("mean_abs_track_error", np.nan), 5)}</strong></div>
                <div><span>track max</span><strong>{fmt(agg.get("max_abs_track_error", np.nan), 5)}</strong></div>
              </div>
              <p class="path">{html.escape(str(run["output_dir"]))}</p>
              <p class="setting">interpolated={html.escape(str(align.get("interpolated", "")))}
                {html.escape(str(align.get("interpolation_type", "")))} · decimation={html.escape(str(align.get("decimation", "")))}
                · init_steps={html.escape(str(align.get("init_steps", "")))}</p>
            </section>
            """
        )

    episode_sections = []
    for episode_ix in selected_episodes:
        run_blocks = []
        for run in runs:
            ep = run["per_episode"].get(episode_ix)
            if ep is None:
                run_blocks.append(f'<div class="run-block"><h4>{html.escape(run["label"])}</h4><div class="missing">missing episode</div></div>')
                continue
            label_safe = safe_name(run["label"])
            trace_png = output_dir / "assets" / "plots" / f"{label_safe}_episode_{episode_ix}_trace.png"
            heat_png = output_dir / "assets" / "plots" / f"{label_safe}_episode_{episode_ix}_tracking_error.png"
            if ep["trace_rows"]:
                plot_trace(
                    ep["trace_rows"],
                    f"{run['label']} episode {episode_ix}: policy target vs eval target vs eval state",
                    trace_png,
                )
                plot_error_heatmap(
                    ep["trace_rows"],
                    f"{run['label']} episode {episode_ix}: tracking error heatmap",
                    heat_png,
                )
            trace_rel = trace_png.relative_to(output_dir).as_posix()
            heat_rel = heat_png.relative_to(output_dir).as_posix()
            row = ep["row"]
            metrics = ep["metrics"]
            run_blocks.append(
                f"""
                <div class="run-block">
                  <div class="run-head">
                    <h4>{html.escape(run["label"])}</h4>
                    <div class="chips">
                      <span class="{ 'good' if to_bool(row.get('success')) else 'bad' }">success={html.escape(str(row.get('success')))}</span>
                      <span>open={html.escape(str(row.get('final_door_openness', '')))}</span>
                      <span>handle={html.escape(str(row.get('final_handle_distance', '')))}</span>
                    </div>
                  </div>
                  <div class="video-wrap">{video_tag(ep["video_rel"])}</div>
                  <div class="small-metrics">
                    <div><span>all mean</span><b>{fmt(metrics.get("mean_abs_track_error", np.nan), 5)}</b></div>
                    <div><span>all max</span><b>{fmt(metrics.get("max_abs_track_error", np.nan), 5)}</b></div>
                    <div><span>base mean</span><b>{fmt(metrics.get("base_mean_abs_track_error", np.nan), 5)}</b></div>
                    <div><span>arm mean</span><b>{fmt(metrics.get("arm_mean_abs_track_error", np.nan), 5)}</b></div>
                    <div><span>policy jump</span><b>{fmt(metrics.get("mean_abs_policy_jump", np.nan), 5)}</b></div>
                  </div>
                  {image_tag(trace_rel, f"{run['label']} episode {episode_ix} trace") if trace_png.exists() else '<div class="missing">no action trace plot</div>'}
                  {image_tag(heat_rel, f"{run['label']} episode {episode_ix} tracking error") if heat_png.exists() else ''}
                </div>
                """
            )
        episode_sections.append(
            f"""
            <section class="episode">
              <h3>Episode {episode_ix}</h3>
              <div class="runs">{''.join(run_blocks)}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Policy Eval Alignment Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --ink: #17202a;
      --muted: #657386;
      --line: #dce3ee;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --warn: #b42318;
      --ok: #047857;
      --shadow: 0 18px 45px rgba(31, 45, 61, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, #eef4fb 0, rgba(238, 244, 251, 0) 360px),
        var(--bg);
    }}
    header {{
      padding: 34px clamp(20px, 4vw, 58px) 24px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 10px; font-size: clamp(28px, 4vw, 44px); letter-spacing: 0; }}
    .lead {{ max-width: 1100px; color: var(--muted); font-size: 16px; line-height: 1.65; }}
    main {{ padding: 26px clamp(18px, 4vw, 58px) 60px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; margin-bottom: 26px; }}
    .card, .episode {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .card {{ padding: 20px; }}
    .card-title {{ font-weight: 800; font-size: 19px; margin-bottom: 14px; }}
    .metric-grid, .small-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }}
    .metric-grid div, .small-metrics div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfdff;
    }}
    span {{ color: var(--muted); font-size: 12px; display: block; }}
    strong, b {{ display: block; margin-top: 4px; font-size: 18px; }}
    .path, .setting {{ color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .episode {{ padding: 18px; margin: 24px 0; }}
    .episode h3 {{ margin: 0 0 16px; font-size: 22px; }}
    .runs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(620px, 100%), 1fr)); gap: 18px; }}
    .run-block {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfdff; }}
    .run-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 12px; }}
    h4 {{ margin: 0; font-size: 18px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 7px; justify-content: flex-end; }}
    .chips span {{
      display: inline-block;
      border: 1px solid var(--line);
      color: var(--ink);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      background: #fff;
    }}
    .chips .good {{ color: var(--ok); border-color: #9bd8bf; background: #edfbf4; }}
    .chips .bad {{ color: var(--warn); border-color: #f0b8b2; background: #fff4f2; }}
    .video-wrap {{ background: #101820; border-radius: 8px; overflow: hidden; margin-bottom: 12px; }}
    video {{ display: block; width: 100%; max-height: 420px; background: #101820; }}
    img {{ width: 100%; display: block; border: 1px solid var(--line); border-radius: 8px; margin-top: 14px; background: #fff; }}
    .missing {{ padding: 28px; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 8px; background: #fff; }}
    .note {{
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      padding: 12px 14px;
      border-radius: 6px;
      margin-bottom: 22px;
      color: #164e49;
      line-height: 1.6;
    }}
    .analysis {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 20px;
      margin-bottom: 26px;
    }}
    .analysis h2 {{ margin: 0 0 8px; font-size: 24px; }}
    .analysis-lead {{ margin: 0 0 16px; color: var(--muted); line-height: 1.65; }}
    .analysis-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 14px; }}
    .analysis-item {{ border: 1px solid var(--line); background: #fbfdff; border-radius: 8px; padding: 14px; }}
    .analysis-item h3 {{ margin: 0 0 8px; font-size: 18px; }}
    .analysis-item p {{ margin: 8px 0; color: #344256; line-height: 1.62; }}
  </style>
</head>
<body>
  <header>
    <h1>Policy Eval Alignment Report</h1>
    <p class="lead">这个报告只使用 policy eval 产物，不再依赖 record HDF5。曲线图里实线是执行后的 eval state，虚线是实际送进 simulator 的 eval target，叉号是 policy 原始 target 采样点；热力图显示每个关节的 |eval state - eval target|。</p>
  </header>
  <main>
    <div class="note">判读重点：如果虚线和叉号差距大，主要看 action postprocess / interpolation / chunk 展开；如果虚线和实线差距大，主要看 physics drive tracking；如果视频失败但 target/state tracking 很小，问题更可能在 policy 输出轨迹本身或观测分布。</div>
    {render_auto_analysis(runs)}
    <section class="cards">{''.join(cards)}</section>
    {''.join(episode_sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(label, path, output_dir) for label, path in (parse_run_descriptor(raw) for raw in args.run)]
    selected_episodes = choose_episodes(runs, args.n, args.seed)
    html_text = render_html(runs, selected_episodes, output_dir)
    report_path = output_dir / "index.html"
    report_path.write_text(html_text)
    print(report_path)


if __name__ == "__main__":
    main()
