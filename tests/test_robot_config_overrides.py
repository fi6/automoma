import unittest

from automoma.utils.file_utils import override_joint_distance_weight


class RobotConfigOverridesTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "kinematics": {
                "cspace": {
                    "joint_names": ["base_x", "base_y", "base_z", "arm"],
                    "cspace_distance_weight": [1.0, 1.0, 1.0, 1.0],
                }
            }
        }

    def test_overrides_only_named_weight_without_mutating_source(self):
        updated = override_joint_distance_weight(
            self.config, joint_name="base_z", weight=10.0
        )

        self.assertEqual(
            updated["kinematics"]["cspace"]["cspace_distance_weight"],
            [1.0, 1.0, 10.0, 1.0],
        )
        self.assertEqual(
            self.config["kinematics"]["cspace"]["cspace_distance_weight"],
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_rejects_unknown_joint(self):
        with self.assertRaisesRegex(ValueError, "Unknown joint"):
            override_joint_distance_weight(
                self.config, joint_name="missing", weight=10.0
            )


if __name__ == "__main__":
    unittest.main()
