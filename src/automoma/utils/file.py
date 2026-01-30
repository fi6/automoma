import json
import os
from typing import Dict


def load_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    return file_path


def get_project_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def process_robot_cfg(robot_cfg: Dict) -> Dict:
    if robot_cfg.get("kinematics", {}).get("urdf_path", ""):
        robot_cfg["kinematics"]["urdf_path"] = os.path.join(get_project_dir(), robot_cfg["kinematics"]["urdf_path"])
    if robot_cfg.get("kinematics", {}).get("external_asset_path", ""):
        robot_cfg["kinematics"]["external_asset_path"] = os.path.join(
            get_project_dir(), robot_cfg["kinematics"]["external_asset_path"]
        )
    if robot_cfg.get("kinematics", {}).get("asset_root_path", ""):
        robot_cfg["kinematics"]["asset_root_path"] = os.path.join(
            get_project_dir(), robot_cfg["kinematics"]["asset_root_path"]
        )
    return robot_cfg

if __name__ == "__main__":
    print(get_project_dir())
