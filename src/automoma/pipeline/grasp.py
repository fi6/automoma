from automoma.models.object import ObjectDescription
import numpy as np


class GraspPipeline:
    # for a given object, output the grasp pose
    def __init__(self):
        raise NotImplementedError("GraspPipeline is not implemented yet.")
    
    
    def grasp_scale(self, grasp: np.ndarray, scale: float | list[float]) -> np.ndarray:
        # scale the grasp pose according to the object scale
        scaled_grasp = np.copy(grasp)
        scaled_grasp[:3, 3] *= scale
        return scaled_grasp
    
    def read_grasps_from_file(self, object: ObjectDescription, count: int) -> list[np.ndarray]:
        # read grasp poses from a file
        urdf_path = object.urdf_path
        object_id = urdf_path.split("/")[-2]  # assuming the folder name is the object id
        file_folder = f"assets/grasp/{object_id}/0/raw/pos"
        grasps = []
        for i in range(count):
            file_path = f"{file_folder}/{i:04d}.npy"
            grasp = np.load(file_path)
            grasp = self.grasp_scale(grasp, object.scale)
            grasps.append(grasp)
        return grasps
        

    def generate_grasps(self, object: ObjectDescription, count: int) -> list[np.ndarray]:
        # Generate a grasp pose for the given object description
        grasps = self.read_grasps_from_file(object, count)
        
        return grasps
