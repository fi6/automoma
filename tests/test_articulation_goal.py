import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from curobo.types.math import Pose

from automoma.core.types import TrajResult
from automoma.planning.planner import CuroboPlanner
from automoma.utils.file_utils import articulation_goal_config


class ArticulationGoalTest(unittest.TestCase):
    def test_targets_moving_side_of_reversed_object_joint_without_mutating_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'akr.urdf'
            path.write_text('<robot name="akr"><joint name="door" type="revolute">'
                            '<parent link="moving_door"/><child link="fixed_body"/></joint></robot>')
            cfg = {'kinematics': {'urdf_path': str(path), 'ee_link': 'fixed_body',
                                  'link_names': ['fixed_body'], 'cspace': {'joint_names': ['arm', 'door']}}}
            updated, link = articulation_goal_config(cfg)
            self.assertEqual(link, 'moving_door')
            self.assertEqual(updated['kinematics']['link_names'], ['fixed_body', 'moving_door'])
            self.assertEqual(cfg['kinematics']['link_names'], ['fixed_body'])
            path.write_text(path.read_text().replace('type="revolute"', 'type="prismatic"'))
            self.assertEqual(articulation_goal_config(cfg)[1], 'moving_door')
            path.write_text(path.read_text().replace('type="prismatic"', 'type="fixed"'))
            with self.assertRaisesRegex(ValueError, 'movable final AKR'):
                articulation_goal_config(cfg)

    def test_filter_rejects_wrong_door_angle_even_with_perfect_anchor_fk(self):
        planner = CuroboPlanner.__new__(CuroboPlanner)
        planner.cfg = {'filter': {}}
        planner.tensor_args = SimpleNamespace(to_device=lambda value: value)
        pose = Pose(torch.zeros(1, 3), torch.tensor([[1., 0., 0., 0.]]))
        motion = SimpleNamespace(ik_solver=SimpleNamespace(fk=lambda value: SimpleNamespace(ee_pose=pose)))
        goal = torch.tensor([[0., 0., 0., -1.0471976]]).repeat(3, 1)
        paths = goal[:, None, :].repeat(1, 2, 1)
        paths[0, -1, -1] = -0.9461943  # 54.21 deg instead of 60 deg
        paths[1, -1, 0] = .4  # Robot may use a different IK solution at the same door goal.
        paths[2, -1, -1] = float('nan')
        result = TrajResult(start_states=torch.zeros(3, 4), goal_states=goal,
                            trajectories=paths, success=torch.ones(3, dtype=torch.bool))
        filtered = planner.filter_traj(result, {}, motion_gen=motion)
        self.assertEqual(filtered.num_samples, 1)
        self.assertEqual(float(filtered.trajectories[0, -1, 0]), float(paths[1, -1, 0]))


if __name__ == '__main__':
    unittest.main()
