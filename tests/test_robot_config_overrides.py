import unittest

from automoma.utils.file_utils import override_joint_distance_weights


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

    def test_overrides_named_weights_without_mutating_source(self):
        updated = override_joint_distance_weights(
            self.config, {"base_x": 2.0, "base_y": 2.0, "base_z": 10.0}
        )

        self.assertEqual(
            updated["kinematics"]["cspace"]["cspace_distance_weight"],
            [2.0, 2.0, 10.0, 1.0],
        )
        self.assertEqual(
            self.config["kinematics"]["cspace"]["cspace_distance_weight"],
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_rejects_unknown_joint(self):
        with self.assertRaisesRegex(ValueError, "Unknown joint"):
            override_joint_distance_weights(self.config, {"missing": 2.0})


if __name__ == "__main__":
    unittest.main()
