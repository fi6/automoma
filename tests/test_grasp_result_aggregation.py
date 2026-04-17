import unittest

import torch

from automoma.core.types import IKResult, TrajResult, aggregate_grasp_goal_results


class AggregateGraspGoalResultsTest(unittest.TestCase):
    def test_concatenates_all_goal_angles(self):
        start_iks = [
            IKResult(
                target_poses=torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                iks=torch.tensor([[0.1, 0.2]]),
            ),
            IKResult(
                target_poses=torch.tensor([[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                iks=torch.tensor([[0.3, 0.4]]),
            ),
        ]
        goal_iks = [
            IKResult(
                target_poses=torch.tensor([[3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                iks=torch.tensor([[0.5, 0.6]]),
            ),
            IKResult(
                target_poses=torch.tensor([[4.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
                iks=torch.tensor([[0.7, 0.8]]),
            ),
        ]
        traj_results = [
            TrajResult(
                start_states=torch.tensor([[1.0, 1.1]]),
                goal_states=torch.tensor([[2.0, 2.1]]),
                trajectories=torch.tensor([[[1.0, 1.1], [2.0, 2.1]]]),
                success=torch.tensor([True]),
            ),
            TrajResult(
                start_states=torch.tensor([[3.0, 3.1]]),
                goal_states=torch.tensor([[4.0, 4.1]]),
                trajectories=torch.tensor([[[3.0, 3.1], [4.0, 4.1]]]),
                success=torch.tensor([False]),
            ),
        ]

        start_ik, goal_ik, traj = aggregate_grasp_goal_results(start_iks, goal_iks, traj_results)

        torch.testing.assert_close(start_ik.iks, torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        torch.testing.assert_close(goal_ik.iks, torch.tensor([[0.5, 0.6], [0.7, 0.8]]))
        torch.testing.assert_close(traj.start_states, torch.tensor([[1.0, 1.1], [3.0, 3.1]]))
        torch.testing.assert_close(traj.goal_states, torch.tensor([[2.0, 2.1], [4.0, 4.1]]))
        self.assertEqual(traj.success.tolist(), [True, False])
