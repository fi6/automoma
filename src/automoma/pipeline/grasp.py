from automoma.models.object import ObjectDescription
import numpy as np


class GraspPipeline:
    # for a given object, output the grasp pose
    def __init__(self):
        raise NotImplementedError("GraspPipeline is not implemented yet.")

    def generate_grasps(self, object: ObjectDescription, count: int) -> list[np.ndarray]:
        # Generate a grasp pose for the given object description

        return grasps
