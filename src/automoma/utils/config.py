# region Paths
PROJECT_ROOT = "/home/xinhai/Documents/automoma"


SCENE_GENERATION_VERSION = "v0_1"
SCENE_GENERATION_REPO = "scene/infinigen"
SCENE_OUTPUT_DIR = "scene/infinigen/output/kitchen"

# endregion Paths


import os
def abs_path(relative_path: str) -> str:
    if os.path.isabs(relative_path):
        return relative_path
    else:
        return os.path.join(PROJECT_ROOT, relative_path)
    
def make_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path