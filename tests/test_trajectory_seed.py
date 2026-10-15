import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from curobo.types.math import Pose

from automoma.planning.planner import CuroboPlanner


class TrajectorySeedTest(unittest.TestCase):
    def setUp(self):
        self.planner = CuroboPlanner.__new__(CuroboPlanner)
        self.planner.cfg = {"traj": {"batch_size": 2}}
        self.planner.tensor_args = SimpleNamespace(to_device=lambda value: value)
        self.calls = []

        def fk(joints):
            pose = Pose(torch.zeros(len(joints), 3), torch.tensor([[1., 0., 0., 0.]]).repeat(len(joints), 1))
            return SimpleNamespace(ee_pose=pose, link_poses={"moving_door": pose})

        def solve(goal, **kwargs):
            self.calls.append((goal, kwargs))
            path = torch.stack([goal.current_state.position, goal.goal_state.position], dim=1)
            return SimpleNamespace(solution=SimpleNamespace(position=path), success=torch.ones(len(path), dtype=torch.bool))

        self.motion = SimpleNamespace(kinematics=SimpleNamespace(get_state=fk),
                                      trajopt_solver=SimpleNamespace(action_horizon=4, solve_batch=solve))
        self.start = torch.tensor([[0., 0.], [.1, 0.], [.2, 0.]])
        self.goal = torch.tensor([[.3, -1.], [.4, -1.2], [.5, -1.5]])
        u = torch.tensor([0., .2, .8, 1.])[None, :, None]
        self.seed = self.start[:, None] + u * (self.goal - self.start)[:, None]
        for target, kwargs in [
            ("torch.cuda.synchronize", {}),
            ("automoma.planning.planner.articulation_goal_config", {"return_value": ({}, "moving_door")}),
        ]:
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def plan(self, seed=None, **kwargs):
        return self.planner.plan_traj(self.start, self.goal, {}, motion_gen=self.motion, seed_traj=seed, **kwargs)

    def test_seed_follows_pairs_across_partial_batches_without_mutation(self):
        original = self.seed.clone()
        result = self.plan(self.seed)
        self.assertEqual(result.num_samples, 3)
        self.assertEqual(len(self.calls), 2)
        for (goal, kwargs), expected in zip(self.calls, [self.seed[:2], self.seed[2:]]):
            self.assertEqual(set(kwargs), {"seed_traj"})
            torch.testing.assert_close(kwargs["seed_traj"], expected.unsqueeze(0))
            torch.testing.assert_close(goal.current_state.position, expected[:, 0])
            torch.testing.assert_close(goal.goal_state.position, expected[:, -1])
            self.assertIn("moving_door", goal.links_goal_pose)
        torch.testing.assert_close(self.seed, original)

    def test_default_keeps_native_seed_generation(self):
        self.plan()
        self.assertTrue(all(kwargs == {} for _, kwargs in self.calls))

    def test_rejects_cartesian_expansion_with_paired_seeds(self):
        with self.assertRaisesRegex(ValueError, "explicit start/goal pairs"):
            self.plan(self.seed, plan_cfg={"expand_to_pairs": True})
        self.assertEqual(self.calls, [])

    def test_rejects_wrong_shape_and_nonfinite_values(self):
        for seed in [self.seed[:2], self.seed[:, :3], self.seed[:, :, :1], self.seed.tolist()]:
            with self.subTest(shape=getattr(seed, "shape", None)):
                with self.assertRaisesRegex(ValueError, "shape"):
                    self.plan(seed)
        for value in [float("nan"), float("inf")]:
            seed = self.seed.clone()
            seed[0, 1, 0] = value
            with self.assertRaisesRegex(ValueError, "finite"):
                self.plan(seed)
        self.assertEqual(self.calls, [])

    def test_rejects_mismatched_endpoints(self):
        for index in [0, -1]:
            seed = self.seed.clone()
            seed[1, index, 0] += .01
            with self.assertRaisesRegex(ValueError, "endpoints"):
                self.plan(seed)
        self.assertEqual(self.calls, [])

    def test_seed_uses_prepared_negative_object_coordinates(self):
        self.goal[:, -1] *= -1
        self.plan(self.seed)
        torch.testing.assert_close(self.calls[0][0].goal_state.position, self.seed[:2, -1])
        positive_seed = self.seed.clone()
        positive_seed[:, :, -1] *= -1
        with self.assertRaisesRegex(ValueError, "prepared AKR"):
            self.plan(positive_seed)


if __name__ == "__main__":
    unittest.main()
