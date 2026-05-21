from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np


EE_TRACE_COLUMNS = [
    "target_ee_pos_x",
    "target_ee_pos_y",
    "target_ee_pos_z",
    "target_ee_quat_w",
    "target_ee_quat_x",
    "target_ee_quat_y",
    "target_ee_quat_z",
    "real_ee_pos_x",
    "real_ee_pos_y",
    "real_ee_pos_z",
    "real_ee_quat_w",
    "real_ee_quat_x",
    "real_ee_quat_y",
    "real_ee_quat_z",
    "ee_pos_error_m",
    "ee_rot_error_rad",
]


class SummitFrankaEeFk:
    def __init__(self, joint_names: list[str], link_name: str = "ee_link") -> None:
        import pinocchio as pin

        repo_root = Path(__file__).resolve().parents[1]
        robot_root = Path(os.environ.get("AUTOMOMA_ROBOT_ROOT", str(repo_root / "assets" / "robot")))
        urdf_path = robot_root / "summit_franka" / "summit_franka.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"Summit Franka URDF not found: {urdf_path}")

        self.pin = pin
        self.link_name = link_name
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId(link_name)
        if self.frame_id >= len(self.model.frames):
            raise ValueError(f"FK link '{link_name}' was not found in {urdf_path}")

        self.joint_q_indices: list[int] = []
        missing: list[str] = []
        for joint_name in joint_names:
            joint_id = self.model.getJointId(joint_name)
            if joint_id >= len(self.model.joints):
                missing.append(joint_name)
                continue
            joint_model = self.model.joints[joint_id]
            if joint_model.nq != 1:
                raise ValueError(f"FK only supports scalar joints; {joint_name} has nq={joint_model.nq}.")
            self.joint_q_indices.append(int(joint_model.idx_q))

        if missing:
            raise ValueError(f"FK model is missing action joints: {missing}. Known joints: {list(self.model.names)}")

    def pose(self, joint_values: Any) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(joint_values, dtype=np.float64).reshape(-1)
        q = np.zeros(self.model.nq, dtype=np.float64)
        for source_ix, q_ix in enumerate(self.joint_q_indices):
            if source_ix >= values.shape[0]:
                break
            q[q_ix] = values[source_ix]

        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        pos = np.asarray(placement.translation, dtype=np.float64).copy()
        quat = _quat_wxyz_from_rot(np.asarray(placement.rotation, dtype=np.float64))
        return pos, quat


def make_ee_fk(joint_names: list[str], link_name: str = "ee_link") -> SummitFrankaEeFk | None:
    try:
        return SummitFrankaEeFk(joint_names, link_name=link_name)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: EE FK trace disabled: {exc}", flush=True)
        return None


def ee_trace_values(
    fk: SummitFrankaEeFk | None,
    target_joint_pos: Any,
    real_joint_pos: Any,
) -> dict[str, float | str]:
    if fk is None:
        return {column: "" for column in EE_TRACE_COLUMNS}

    target_pos, target_quat = fk.pose(target_joint_pos)
    real_pos, real_quat = fk.pose(real_joint_pos)
    pos_error = float(np.linalg.norm(real_pos - target_pos))
    rot_error = _quat_angle(target_quat, real_quat)
    return {
        "target_ee_pos_x": float(target_pos[0]),
        "target_ee_pos_y": float(target_pos[1]),
        "target_ee_pos_z": float(target_pos[2]),
        "target_ee_quat_w": float(target_quat[0]),
        "target_ee_quat_x": float(target_quat[1]),
        "target_ee_quat_y": float(target_quat[2]),
        "target_ee_quat_z": float(target_quat[3]),
        "real_ee_pos_x": float(real_pos[0]),
        "real_ee_pos_y": float(real_pos[1]),
        "real_ee_pos_z": float(real_pos[2]),
        "real_ee_quat_w": float(real_quat[0]),
        "real_ee_quat_x": float(real_quat[1]),
        "real_ee_quat_y": float(real_quat[2]),
        "real_ee_quat_z": float(real_quat[3]),
        "ee_pos_error_m": pos_error,
        "ee_rot_error_rad": rot_error,
    }


def empty_ee_trace_values() -> dict[str, str]:
    return {column: "" for column in EE_TRACE_COLUMNS}


def _quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = float(abs(np.dot(q1, q2)))
    dot = max(min(dot, 1.0), -1.0)
    return float(2.0 * math.acos(dot))


def _quat_wxyz_from_rot(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s

    quat = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    return quat / max(np.linalg.norm(quat), 1e-12)
