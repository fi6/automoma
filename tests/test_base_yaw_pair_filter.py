import unittest

import torch

from automoma.utils.math_utils import (
    filter_pairs_by_joint_delta,
    filter_pairs_by_planar_delta,
    joint_excursion_mask,
    planar_excursion_mask,
)


class BaseYawPairFilterTest(unittest.TestCase):
    def test_filters_on_raw_bounded_joint_delta(self):
        start = torch.tensor(
            [
                [0.0, 0.0, -5.0, 0.1],
                [0.0, 0.0, -0.5, 0.2],
                [0.0, 0.0, 1.0, 0.3],
            ]
        )
        goal = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.1],
                [0.0, 0.0, 1.5, 1.2],
                [0.0, 0.0, 3.01, 1.3],
            ]
        )

        filtered_start, filtered_goal, mask = filter_pairs_by_joint_delta(
            start, goal, joint_index=2, max_delta=2.0
        )

        self.assertEqual(mask.tolist(), [False, True, False])
        torch.testing.assert_close(filtered_start, start[1:2])
        torch.testing.assert_close(filtered_goal, goal[1:2])

    def test_rejects_negative_limit(self):
        states = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            filter_pairs_by_joint_delta(states, states, joint_index=2, max_delta=-0.1)

    def test_rejects_intermediate_yaw_excursion(self):
        start = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, -0.2]])
        trajectories = torch.tensor(
            [
                [[0.0, 0.0, 0.1], [0.0, 0.0, 0.55], [0.0, 0.0, 0.2]],
                [[0.0, 0.0, -0.2], [0.0, 0.0, 0.31], [0.0, 0.0, -0.1]],
            ]
        )

        mask = joint_excursion_mask(
            start, trajectories, joint_index=2, max_excursion=0.5
        )

        self.assertEqual(mask.tolist(), [True, False])

    def test_filters_planar_delta_and_excursion(self):
        start = torch.zeros(2, 3)
        goal = torch.tensor([[0.6, 0.8, 0.0], [1.01, 0.0, 0.0]])
        filtered_start, filtered_goal, pair_mask = filter_pairs_by_planar_delta(
            start, goal, x_index=0, y_index=1, max_delta=1.0
        )
        self.assertEqual(pair_mask.tolist(), [True, False])
        torch.testing.assert_close(filtered_start, start[:1])
        torch.testing.assert_close(filtered_goal, goal[:1])

        trajectories = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
                [[0.0, 0.0, 0.0], [0.8, 0.7, 0.0]],
            ]
        )
        excursion_mask = planar_excursion_mask(
            start, trajectories, x_index=0, y_index=1, max_excursion=1.0
        )
        self.assertEqual(excursion_mask.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
