#!/usr/bin/env python3
"""Create trajectory-driven poster sketches for mobile-vs-fixed workspace.

The Isaac renderer in this folder is useful for photoreal context, but this
script is intentionally data-first: it reads a planned AutoMoMa trajectory file
and produces clean top-down figures that make the mobile-base solution manifold
visible for poster layout iteration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch, Polygon


MOBILE = "#0B7A75"
MOBILE_DARK = "#064E4A"
MOBILE_LIGHT = "#7BDDD4"
FIXED = "#D45B2C"
FIXED_LIGHT = "#F3A76A"
INK = "#18201D"
MUTED = "#6E7770"
PAPER = "#F8F2E8"
GRID = "#D9D0C3"
OBJECT = "#E7C45C"
HIGHLIGHTS = ["#0B7A75", "#247BA0", "#70A288", "#C08497", "#F18F01", "#7A6F9B", "#5C946E", "#D1495B"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the 2D convex hull vertices using a monotonic chain."""

    if len(points) == 0:
        return points
    pts = sorted(set(map(tuple, np.round(points.astype(float), 5))))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_perimeter(poly: np.ndarray) -> float:
    if len(poly) < 2:
        return 0.0
    closed = np.vstack([poly, poly[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def buffered_hull_area(poly: np.ndarray, radius: float) -> float:
    """Approximate a convex polygon buffered by a circular arm reach."""

    return polygon_area(poly) + polygon_perimeter(poly) * radius + math.pi * radius * radius


def farthest_point_indices(features: np.ndarray, count: int) -> list[int]:
    """Pick representative trajectories by spreading samples in feature space."""

    if len(features) == 0:
        return []
    count = min(count, len(features))
    scaled = features.astype(float)
    span = np.ptp(scaled, axis=0)
    scaled = (scaled - scaled.mean(axis=0)) / np.maximum(span, 1e-6)

    center = np.median(scaled, axis=0)
    first = int(np.argmax(np.linalg.norm(scaled - center, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(scaled - scaled[first], axis=1)
    for _ in range(1, count):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(scaled - scaled[idx], axis=1))
    return selected


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    data = torch.load(path, map_location="cpu")
    traj = data["traj_robot"].float()
    success = data.get("traj_success")
    if success is not None:
        traj = traj[success.bool()]
    if traj.ndim != 3 or traj.shape[-1] < 3:
        raise ValueError(f"Expected traj_robot with shape [N, T, >=3], got {tuple(traj.shape)}")

    base = traj[:, :, :3].numpy()
    return {
        "base": base,
        "start": base[:, 0, :],
        "goal": base[:, -1, :],
        "all_xy": base[:, :, :2].reshape(-1, 2),
    }


def axis_limits(points: np.ndarray, reach_radius: float) -> tuple[float, float, float, float]:
    low = points.min(axis=0) - reach_radius - 0.25
    high = points.max(axis=0) + reach_radius + 0.25
    return (float(low[0]), float(high[0]), float(low[1]), float(high[1]))


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#AFA698",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
        }
    )


def setup_axis(ax: plt.Axes, limits: tuple[float, float, float, float], title: str) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_title(title, loc="left", fontsize=16, pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.65)
    ax.set_xlabel("base x / object-centered frame (m)")
    ax.set_ylabel("base y / object-centered frame (m)")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def add_object_glyph(ax: plt.Axes) -> None:
    body = FancyBboxPatch(
        (-0.31, -0.23),
        0.62,
        0.46,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        facecolor=OBJECT,
        edgecolor="#8D6E1D",
        linewidth=1.4,
        zorder=8,
    )
    handle = FancyBboxPatch(
        (-0.25, -0.04),
        0.08,
        0.08,
        boxstyle="round,pad=0.01,rounding_size=0.025",
        facecolor="#5A4112",
        edgecolor="none",
        zorder=9,
    )
    ax.add_patch(body)
    ax.add_patch(handle)
    ax.text(0.0, 0.34, "target object", ha="center", va="bottom", fontsize=9, color="#6E5414", zorder=10)


def add_hull(ax: plt.Axes, hull: np.ndarray, color: str, alpha: float, label: str | None = None) -> None:
    if len(hull) < 3:
        return
    ax.add_patch(Polygon(hull, closed=True, facecolor=color, edgecolor=color, linewidth=1.8, alpha=alpha, label=label))


def add_trajectory_lines(
    ax: plt.Axes,
    base: np.ndarray,
    indices: Iterable[int],
    *,
    color: str,
    alpha: float,
    linewidth: float,
) -> None:
    lines = [base[idx, :, :2] for idx in indices]
    if not lines:
        return
    collection = LineCollection(lines, colors=color, linewidths=linewidth, alpha=alpha, capstyle="round", zorder=4)
    ax.add_collection(collection)


def add_reach_circles(ax: plt.Axes, centers: np.ndarray, radius: float, color: str, alpha: float) -> None:
    for x, y in centers:
        ax.add_patch(Circle((float(x), float(y)), radius, facecolor=color, edgecolor="none", alpha=alpha, zorder=1))


def add_metric_card(ax: plt.Axes, text: str, xy: tuple[float, float] = (0.03, 0.97)) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.12", facecolor="#FFFDF8", edgecolor="#D8CBB8", alpha=0.95),
        zorder=20,
    )


def figure_mobile_trajectories(
    out_path: Path,
    base: np.ndarray,
    selected: list[int],
    hull: np.ndarray,
    metrics: dict[str, float | int],
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 8.0), dpi=220)
    setup_axis(ax, limits, "Mobile manipulation exposes many valid motion plans")
    add_hull(ax, hull, MOBILE_LIGHT, 0.28, "base-path envelope")
    add_trajectory_lines(ax, base, range(min(len(base), 800)), color=MOBILE_DARK, alpha=0.035, linewidth=0.75)

    for rank, idx in enumerate(selected):
        xy = base[idx, :, :2]
        color = HIGHLIGHTS[rank % len(HIGHLIGHTS)]
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.3, alpha=0.92, zorder=6)
        ax.scatter(xy[0, 0], xy[0, 1], s=42, color=color, marker="o", edgecolor="white", linewidth=0.8, zorder=7)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=54, color=color, marker="*", edgecolor="white", linewidth=0.8, zorder=7)

    add_object_glyph(ax)
    add_metric_card(
        ax,
        "\n".join(
            [
                f"{metrics['num_plans']:,} successful plans",
                f"{metrics['base_path_hull_m2']:.2f} m² base-path envelope",
                f"{metrics['max_base_motion_m']:.2f} m max base motion",
                "thin lines = feasible plans; bold = diverse exemplars",
            ]
        ),
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#FFFDF8", edgecolor="#D8CBB8")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_fixed_vs_mobile(
    out_path: Path,
    base: np.ndarray,
    selected: list[int],
    hull: np.ndarray,
    fixed_base: np.ndarray,
    metrics: dict[str, float | int],
    limits: tuple[float, float, float, float],
    reach_radius: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), dpi=220, sharex=True, sharey=True)
    setup_axis(axes[0], limits, "Fixed Franka: one base, local arm reach")
    setup_axis(axes[1], limits, "Summit-Franka: base motion sweeps the reach set")

    for ax in axes:
        add_object_glyph(ax)

    axes[0].add_patch(Circle(fixed_base, reach_radius, facecolor=FIXED_LIGHT, edgecolor=FIXED, linewidth=2.2, alpha=0.34))
    axes[0].scatter([fixed_base[0]], [fixed_base[1]], s=120, marker="s", color=FIXED, edgecolor="white", zorder=8)
    axes[0].annotate("fixed base", fixed_base, xytext=(18, -18), textcoords="offset points", color=FIXED, fontsize=11)
    add_metric_card(
        axes[0],
        "\n".join(
            [
                "Base root is a single point",
                f"local arm disk ≈ {metrics['fixed_reach_area_m2']:.2f} m²",
                "diversity must come from arm joints only",
            ]
        ),
    )

    add_reach_circles(axes[1], base[selected, 0, :2], reach_radius, MOBILE_LIGHT, 0.12)
    add_hull(axes[1], hull, MOBILE_LIGHT, 0.26)
    add_trajectory_lines(axes[1], base, range(min(len(base), 900)), color=MOBILE_DARK, alpha=0.035, linewidth=0.7)
    for rank, idx in enumerate(selected):
        xy = base[idx, :, :2]
        axes[1].plot(xy[:, 0], xy[:, 1], color=HIGHLIGHTS[rank % len(HIGHLIGHTS)], linewidth=2.0, alpha=0.9, zorder=6)
    axes[1].scatter(base[selected, 0, 0], base[selected, 0, 1], s=34, color=MOBILE, edgecolor="white", linewidth=0.6, zorder=8)
    add_metric_card(
        axes[1],
        "\n".join(
            [
                "Many base roots + many arm postures",
                f"swept reach envelope ≈ {metrics['mobile_swept_reach_m2']:.2f} m²",
                f"≈ {metrics['swept_vs_fixed_ratio']:.1f}× fixed local disk",
            ]
        ),
    )

    fig.suptitle("Same target, very different solution manifolds", fontsize=22, fontweight="bold", x=0.03, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)


def figure_triptych(
    out_path: Path,
    base: np.ndarray,
    selected: list[int],
    hull: np.ndarray,
    fixed_base: np.ndarray,
    metrics: dict[str, float | int],
    limits: tuple[float, float, float, float],
    reach_radius: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), dpi=220, sharex=True, sharey=True)
    titles = [
        "A. Fixed base",
        "B. Mobile base workspace",
        "C. Diverse planned trajectories",
    ]
    for ax, title in zip(axes, titles, strict=True):
        setup_axis(ax, limits, title)
        add_object_glyph(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")

    axes[0].add_patch(Circle(fixed_base, reach_radius, facecolor=FIXED_LIGHT, edgecolor=FIXED, linewidth=2.2, alpha=0.36))
    axes[0].scatter([fixed_base[0]], [fixed_base[1]], s=110, marker="s", color=FIXED, edgecolor="white", zorder=8)
    add_metric_card(axes[0], "single root pose\narm-only diversity", xy=(0.04, 0.94))

    add_hull(axes[1], hull, MOBILE_LIGHT, 0.35)
    axes[1].scatter(base[:, 0, 0], base[:, 0, 1], s=4, color=MOBILE_DARK, alpha=0.16, rasterized=True)
    axes[1].scatter(base[:, -1, 0], base[:, -1, 1], s=5, color=MOBILE, alpha=0.12, rasterized=True)
    add_metric_card(axes[1], f"{metrics['base_path_hull_m2']:.2f} m² base envelope\n{metrics['num_plans']:,} feasible plans", xy=(0.04, 0.94))

    add_reach_circles(axes[2], base[selected, 0, :2], reach_radius, MOBILE_LIGHT, 0.10)
    add_trajectory_lines(axes[2], base, range(min(len(base), 900)), color=MOBILE_DARK, alpha=0.03, linewidth=0.65)
    for rank, idx in enumerate(selected):
        xy = base[idx, :, :2]
        color = HIGHLIGHTS[rank % len(HIGHLIGHTS)]
        axes[2].plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2, alpha=0.94, zorder=6)
        axes[2].scatter(xy[0, 0], xy[0, 1], s=32, color=color, marker="o", edgecolor="white", linewidth=0.6, zorder=7)
        axes[2].scatter(xy[-1, 0], xy[-1, 1], s=44, color=color, marker="*", edgecolor="white", linewidth=0.6, zorder=7)
    add_metric_card(axes[2], f"{metrics['max_base_motion_m']:.2f} m max move\nreach swept by navigation", xy=(0.04, 0.94))

    fig.suptitle("Mobile manipulation turns one hard grasp into a large solution family", fontsize=23, fontweight="bold", x=0.03, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)


def compute_metrics(base: np.ndarray, hull: np.ndarray, reach_radius: float) -> dict[str, float | int]:
    motion = np.linalg.norm(base[:, -1, :2] - base[:, 0, :2], axis=1)
    fixed_area = math.pi * reach_radius * reach_radius
    swept_area = buffered_hull_area(hull, reach_radius)
    return {
        "num_plans": int(len(base)),
        "num_steps": int(base.shape[1]),
        "base_path_hull_m2": polygon_area(hull),
        "base_path_hull_perimeter_m": polygon_perimeter(hull),
        "mean_base_motion_m": float(motion.mean()),
        "max_base_motion_m": float(motion.max()),
        "fixed_reach_radius_m": float(reach_radius),
        "fixed_reach_area_m2": float(fixed_area),
        "mobile_swept_reach_m2": float(swept_area),
        "swept_vs_fixed_ratio": float(swept_area / fixed_area if fixed_area > 0 else float("nan")),
    }


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj_file",
        default=str(root / "data/trajs/summit_franka/microwave_7221/scene_0/train/traj_data_train.pt"),
        help="Converted 12D AutoMoMa trajectory file.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(root / "outputs/paper/poster/trajectory_workspace_7221"),
        help="Directory for generated poster sketches.",
    )
    parser.add_argument("--highlight_count", type=int, default=8, help="Number of diverse trajectories to highlight.")
    parser.add_argument(
        "--reach_radius",
        type=float,
        default=0.85,
        help="Approximate Franka local arm reach radius in meters for conceptual comparison.",
    )
    parser.add_argument(
        "--fixed_base",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Fixed-base location. Defaults to the median planned start base.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_style()

    traj_file = Path(args.traj_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_trajectory(traj_file)
    base = loaded["base"]
    hull = convex_hull(loaded["all_xy"])
    features = np.concatenate([base[:, 0, :3], base[:, -1, :3]], axis=1)
    selected = farthest_point_indices(features, args.highlight_count)
    fixed_base = np.asarray(args.fixed_base if args.fixed_base is not None else np.median(base[:, 0, :2], axis=0), dtype=float)
    metrics = compute_metrics(base, hull, args.reach_radius)
    limits = axis_limits(np.vstack([loaded["all_xy"], fixed_base.reshape(1, 2), np.zeros((1, 2))]), args.reach_radius)

    figure_mobile_trajectories(
        output_dir / "mobile_workspace_trajectories.png",
        base,
        selected,
        hull,
        metrics,
        limits,
    )
    figure_fixed_vs_mobile(
        output_dir / "fixed_vs_mobile_workspace.png",
        base,
        selected,
        hull,
        fixed_base,
        metrics,
        limits,
        args.reach_radius,
    )
    figure_triptych(
        output_dir / "poster_workspace_triptych.png",
        base,
        selected,
        hull,
        fixed_base,
        metrics,
        limits,
        args.reach_radius,
    )

    manifest = {
        "traj_file": str(traj_file),
        "output_dir": str(output_dir),
        "selected_indices": selected,
        "fixed_base_xy": fixed_base.tolist(),
        "metrics": metrics,
        "outputs": {
            "mobile_workspace_trajectories": str(output_dir / "mobile_workspace_trajectories.png"),
            "fixed_vs_mobile_workspace": str(output_dir / "fixed_vs_mobile_workspace.png"),
            "poster_workspace_triptych": str(output_dir / "poster_workspace_triptych.png"),
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[poster] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
