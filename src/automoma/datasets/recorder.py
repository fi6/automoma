from automoma.core.types import DatasetFormat
from automoma.simulation.env_wrapper import SimEnvWrapper
from automoma.datasets.dataset import LeRobotDatasetWrapper, HDF5DatasetWrapper, ZarrDatasetWrapper
from automoma.utils.math_utils import pose_multiply

from tqdm import tqdm

class Recorder:
    '''A class for recording and replaying datasets in various formats.'''
    
    def __init__(self, record_cfg):
        self.record_cfg = record_cfg
        self.env_warpper = SimEnvWrapper(record_cfg.get("env_cfg", {}))
        
    def setup_env(self):
        pass
        
    def replay_ik(self, iks, robot_type):
        '''Replay inverse kinematics data for a given robot type.'''
        for i, ik in enumerate(zip(iks)):
            print(f"Replaying IK {i+1}/{len(iks)}")
            robot_state, env_state = unpack_ik(ik)
            robot_state = self._adjust_pose_for_robot(robot_state, robot_type)
            self.env_warpper.set_state(robot_state, env_state)
            self.env_warpper.step()
    
    def replay_traj(self, trajs, robot_type):
        '''Replay a trajectory for a given robot type.'''
        for i, episode in enumerate(zip(trajs)):
            print(f"Replaying Traj {i+1}/{len(trajs)}")
            for i, ik in enumerate(zip(episode)):
                print(f"Replaying IK {i+1}/{len(episode)}")
                robot_state, env_state = unpack_ik(ik)
                robot_state = self._adjust_pose_for_robot(robot_state, robot_type)
                self.env_warpper.set_state(robot_state, env_state)
                self.env_warpper.step()    
            
    def record_traj(self, trajs, robot_type):
        format = self.record_cfg.get("format", DatasetFormat.LEROBOT)
        
        format_mapping = {
            DatasetFormat.LEROBOT: LeRobotDatasetWrapper,
            DatasetFormat.HDF5: HDF5DatasetWrapper,
            DatasetFormat.ZARR: ZarrDatasetWrapper,
        }
        dataset = format_mapping[format]()
        dataset.create(self.record_cfg.dataset)
        
        for episode in tqdm(trajs, desc="Recording Trajectories"):
            # Record frames
            for frame in episode:
                robot_state, env_state = unpack_ik(frame)
                robot_state = self._adjust_pose_for_robot(robot_state, robot_type)
                self.env_warpper.set_state(robot_state, env_state)
                obs = self.env_warpper.get_obs()
                dataset.add(obs)
                self.env_warpper.step()
            dataset.save()
            
        dataset.close()
        
        
        
        