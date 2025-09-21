from automoma.models.object import ObjectDescription
import numpy as np
import os


class GraspPipeline:
    # for a given object, output the grasp pose
    def __init__(self):
        pass
    def grasp_scale(self, grasp: np.ndarray, scale: float | list[float]) -> np.ndarray:
        # scale the grasp pose according to the object scale
        scaled_grasp = np.copy(grasp)
        scaled_grasp[:3] *= scale
        return scaled_grasp
        
    def generate_grasps(self, object: ObjectDescription, count: int) -> list[np.ndarray]:
        # Generate a grasp pose for the given object description
        raise NotImplementedError("GraspPipeline is not implemented yet.")


class AOGraspPipeline(GraspPipeline):
    def __init__(self):
        super().__init__()
        
    def generate_grasps(self, object: ObjectDescription, count: int) -> list[np.ndarray]:
        # Read the grasp poses from a file
        urdf_path = object.urdf_path
        file_folder = urdf_path.replace(os.path.basename(urdf_path), "grasp")
        grasps = []
        for i in range(count):
            file_path = f"{file_folder}/{i:04d}.npy"
            grasp = np.load(file_path)
            grasp = self.grasp_scale(grasp, object.scale)
            grasps.append(grasp)
        return grasps
    
if __name__ == "__main__":
    from automoma.models.object import ObjectDescription
    object = ObjectDescription(
            asset_type="Dishwasher",
            asset_id="11622",
            scale=0.6,
            urdf_path="assets/object/Dishwasher/11622/mobility.urdf",
        )
    pipeline = AOGraspPipeline()
    grasps = pipeline.generate_grasps(object, 10)
    for grasp in grasps:
        print(grasp)