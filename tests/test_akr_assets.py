from __future__ import annotations

import hashlib
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml
from yourdfpy import URDF

from automoma.assets.akr import (
    build_akr_urdf,
    generate_collision_spheres,
    load_grasp_pose,
    load_object_init,
    parse_grasp_ids,
)
from tools.assets.build_akr_assets import build

REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_DIR = REPO_ROOT / "assets/object/Microwave/7221"
ROBOT_DIR = REPO_ROOT / "assets/robot/summit_franka"


def _require_assets() -> None:
    if not (OBJECT_DIR / "7221_0_scaling.urdf").is_file():
        pytest.skip("7221 assets are not available")


def _args(output_dir: Path, *, generated_spheres: bool, grasp_ids: str) -> Namespace:
    return Namespace(
        object_urdf=OBJECT_DIR / "7221_0_scaling.urdf",
        object_init_state=OBJECT_DIR / "0/init_state.npz",
        grasp_dir=OBJECT_DIR / "grasp",
        grasp_ids=grasp_ids,
        object_base="link_1",
        handle_link="link_0",
        object_joint="joint_0",
        robot_urdf=ROBOT_DIR / "summit_franka.urdf",
        robot_yaml=ROBOT_DIR / "summit_franka.yml",
        robot_fixed_yaml=ROBOT_DIR / "summit_franka_fixed_base.yml",
        sphere_yaml=(None if generated_spheres else OBJECT_DIR / "summit_franka_7221_0_grasp_0000.yml"),
        generate_spheres=generated_spheres,
        asset_prefix="assets/object/Microwave/7221",
        output_dir=output_dir,
        force=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parse_grasp_ids() -> None:
    assert parse_grasp_ids("0-2,5,2") == [0, 1, 2, 5]
    with pytest.raises(ValueError):
        parse_grasp_ids("4-2")


def test_scale_grasp_and_quaternion_order() -> None:
    _require_assets()
    scale, qpos = load_object_init(OBJECT_DIR / "0/init_state.npz")
    assert scale == pytest.approx(0.3562990018302636)
    assert qpos == {0: 0.0, 1: 0.0}
    raw = np.load(OBJECT_DIR / "grasp/0000.npy")
    pose = load_grasp_pose(OBJECT_DIR / "grasp/0000.npy", scale)
    np.testing.assert_array_equal(pose[:3], (raw[:3] * scale).astype(np.float32).astype(np.float64))
    np.testing.assert_array_equal(pose[3:], raw[3:])


def test_inverse_topology_and_attachment_transform() -> None:
    _require_assets()
    scale, qpos = load_object_init(OBJECT_DIR / "0/init_state.npz")
    pose = load_grasp_pose(OBJECT_DIR / "grasp/0000.npy", scale)
    result = build_akr_urdf(
        OBJECT_DIR / "7221_0_scaling.urdf",
        ROBOT_DIR / "summit_franka.urdf",
        pose,
        object_base="link_1",
        handle_link="link_0",
        object_joint_positions=qpos,
        asset_prefix="assets/object/Microwave/7221",
        golden_compatibility=True,
    )
    joints = {joint.name: joint for joint in result.robot.joints}
    assert (joints["joint_0"].parent, joints["joint_0"].child) == (
        "link_0",
        "link_1",
    )
    assert (joints["joint_1"].parent, joints["joint_1"].child) == (
        "ee_link",
        "link_0",
    )
    golden = URDF.load(str(OBJECT_DIR / "summit_franka_7221_0_grasp_0000.urdf"))
    golden_joint = next(j for j in golden.robot.joints if j.name == "joint_1")
    np.testing.assert_allclose(joints["joint_1"].origin, golden_joint.origin)


def test_sphere_generation_is_cpu_only_and_cwd_independent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_assets()
    monkeypatch.chdir(tmp_path)
    spheres = generate_collision_spheres(
        OBJECT_DIR / "7221_0_scaling.urdf",
        handle_link="link_0",
        body_link="link_1",
    )
    assert set(spheres) == {"link_0", "link_1"}
    assert len(spheres["link_0"]) == 90
    assert len(spheres["link_1"]) == 80
    assert {item["radius"] for item in spheres["link_0"]} == {0.019}
    assert {item["radius"] for item in spheres["link_1"]} == {0.049}
    assert abs(len(spheres["link_0"]) - 89) / 89 <= 0.1
    assert abs(len(spheres["link_1"]) - 87) / 87 <= 0.1


def test_sphere_generation_rejects_geometry_free_links() -> None:
    _require_assets()
    with pytest.raises(ValueError, match="links without geometry"):
        generate_collision_spheres(
            OBJECT_DIR / "7221_0_scaling.urdf",
            handle_link="link_0",
            body_link="world",
        )


@pytest.mark.integration
def test_golden_63_file_regression(tmp_path: Path) -> None:
    _require_assets()
    outputs = build(_args(tmp_path, generated_spheres=False, grasp_ids="0-20"))
    formal_outputs = [path for path in outputs if path.name != "collision_spheres.yml"]
    assert len(formal_outputs) == 63
    for generated in formal_outputs:
        golden = OBJECT_DIR / generated.name
        assert golden.is_file(), generated.name
        assert _sha256(generated) == _sha256(golden), generated.name
        if generated.suffix == ".urdf":
            URDF.load(str(generated))
        else:
            config = yaml.safe_load(generated.read_text())
            kinematics = config["robot_cfg"]["kinematics"]
            cspace = kinematics["cspace"]
            lengths = {
                len(cspace["joint_names"]),
                len(cspace["retract_config"]),
                len(cspace["null_space_weight"]),
                len(cspace["cspace_distance_weight"]),
            }
            assert lengths == {13}
            assert kinematics["ee_link"] == "link_1"
            assert cspace["joint_names"][-1] == "joint_0"
            referenced_urdf = tmp_path / Path(kinematics["urdf_path"]).name
            model = URDF.load(str(referenced_urdf))
            link_names = {link.name for link in model.robot.links}
            joint_names = {joint.name for joint in model.robot.joints}
            assert set(kinematics["collision_link_names"]) <= link_names
            assert set(kinematics["collision_spheres"]) <= link_names
            assert set(cspace["joint_names"]) <= joint_names


@pytest.mark.integration
def test_generated_sphere_full_flow_matches_golden_semantics(
    tmp_path: Path,
) -> None:
    _require_assets()
    outputs = build(_args(tmp_path, generated_spheres=True, grasp_ids="0,20"))
    for grasp_id in (0, 20):
        stem = f"summit_franka_7221_0_grasp_{grasp_id:04d}"
        assert _sha256(tmp_path / f"{stem}.urdf") == _sha256(OBJECT_DIR / f"{stem}.urdf")
        generated = yaml.safe_load((tmp_path / f"{stem}.yml").read_text())
        golden = yaml.safe_load((OBJECT_DIR / f"{stem}.yml").read_text())
        del generated["robot_cfg"]["kinematics"]["collision_spheres"]
        del golden["robot_cfg"]["kinematics"]["collision_spheres"]
        assert generated == golden
    assert len(outputs) == 7


def test_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    _require_assets()
    (tmp_path / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError):
        build(_args(tmp_path, generated_spheres=False, grasp_ids="0"))
