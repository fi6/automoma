import unittest

import torch

from automoma.utils.math_utils import filter_pairs_by_joint_delta


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


if __name__ == "__main__":
    unittest.main()
