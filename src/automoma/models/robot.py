from curobo.util_file import load_yaml
from automoma.utils.file import process_robot_cfg

class RobotDescription:
    def __init__(self, robot_name: str, curobo_yaml_path: str):
        self.robot_name = robot_name
        self.robot_cfg = process_robot_cfg(load_yaml(curobo_yaml_path)["robot_cfg"])


