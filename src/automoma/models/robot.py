from curobo.util_file import load_yaml
from automoma.utils.file import process_robot_cfg

class RobotDescription:
    def __init__(self, curobo_yaml_path: str):
        self.robot_cfg = process_robot_cfg(load_yaml(curobo_yaml_path)["robot_cfg"])


