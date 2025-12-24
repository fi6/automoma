import torch
def adjust_pose_for_robot(robot_pose: torch.Tensor, robot_type, gripper_joint_value=0.02, *args):
    """Adjust the robot pose tensor based on the robot model."""
    if robot_type == "summit_franka":
        return torch.cat([
            robot_pose,
            torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
        ])
    elif robot_type == "tiago":
        return torch.cat([
            robot_pose,
            torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
        ])
    elif robot_type == "r1":
        return torch.cat([
            robot_pose[0:3],
            torch.tensor([2.1816, -2.6178, -0.4363, 0.0], device=robot_pose.device),
            robot_pose[3:],
            torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
        ])
    elif robot_type == "summit_franka_fixed_base":
        return torch.cat([
            torch.tensor([-0.9, 1.4, 0.0], device=robot_pose.device),
            robot_pose,
            torch.tensor([gripper_joint_value, gripper_joint_value], device=robot_pose.device),
        ])
    return robot_pose