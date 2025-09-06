from curobo.util_file import load_yaml

class RobotDescription:
    def __init__(self, curobo_yaml_path: str):
        self.robot_cfg = load_yaml(curobo_yaml_path)["robot_cfg"]
        

