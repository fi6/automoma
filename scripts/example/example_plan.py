from automoma.planning.planner import *

def test_planner():
    """
    Main pipeline function with trajectory filtering
    Complete AKR motion planning pipeline from scene to trajectory with filtering
    """
    print("=== AKR Motion Planning Pipeline with Filtering ===")
    
    # ===== CONFIGURATION =====
    # Scene configuration
    scene_cfg = {
        "path": "assets/scene/infinigen/kitchen_1130/scene_0_seed_0/export/export_scene.blend/export_scene.usdc",
        "pose": [0, 0, -0.13, 1, 0, 0, 0]
    }
    
    # Robot configuration  
    robot_cfg_path = "assets/robot/summit_franka/summit_franka.yml"
    robot_cfg = load_robot_cfg(robot_cfg_path)
    robot_cfg = process_robot_cfg(robot_cfg)
        
    # Object configuration (URDF path and basic info, dimensions loaded from metadata)
    object_cfg = {
        "path": "assets/object/Microwave/7221/7221_0_scaling.urdf",
        "asset_type": "Microwave",
        "asset_id": "7221"
    }
    metadata_path = "assets/scene/infinigen/kitchen_1130/scene_0_seed_0/info/metadata.json"
    object_cfg = load_object_from_metadata(metadata_path, object_cfg=object_cfg)
    
    # ===== INITIALIZATION =====
    print("\n1. Initializing Curobo Planner...")
    planner_cfg = {
        "voxel_dims": [5.0, 5.0, 5.0],
        "voxel_size": 0.02,
        "expanded_dims": [1.0, 0.2, 0.2],
        "collision_checker_type": CollisionCheckerType.VOXEL
    }
    planner = CuroboPlanner(planner_cfg)
    planner.setup_env(scene_cfg, object_cfg)
    
    
    # ===== LOAD GRASP POSES =====
    print("\n2. Loading grasp poses...")
    scaling_factor = 0.3562990018302636
    grasp_poses = get_grasp_poses(
        grasp_dir="assets/object/Microwave/7221/grasp",
        num_grasps=20,
        scaling_factor=scaling_factor
    )
    
    if not grasp_poses:
        print("ERROR: No grasp poses found!")
        return
        
    print(f"Loaded {len(grasp_poses)} grasp poses")
    
    motion_gen = planner.init_motion_gen(robot_cfg)
    
    clustering_params = {
        "ap_fallback_clusters": 30,
        "ap_clusters_upperbound": 80,
        "ap_clusters_lowerbound": 10
    }
    
    # ===== PROCESS EACH GRASP POSE =====
    for grasp_id, grasp_pose in enumerate(grasp_poses):
        print(f"\n{'='*80}")
        print(f"=== Processing Grasp Pose {grasp_id+1}/{len(grasp_poses)} ===")
        print(f"{'='*80}")
        
        # ===== IK PLANNING =====
        print("\n3. Planning IK solutions...")
        
        def plan_single_ik(angle):
            default_joint_cfg = {"joint_0": 0.0}
            joint_cfg={"joint_0": angle}
            target_Pose = get_open_ee_pose(
                object_pose=Pose.from_list(planner.object_pose),
                grasp_pose=Pose.from_list(grasp_pose),
                object_urdf=planner.object_urdf,
                handle="link_0",
                joint_cfg=joint_cfg,
                default_joint_cfg=default_joint_cfg
            )
            target_pose = torch.tensor(target_Pose.to_list())
            
            ik_result = planner.plan_ik(
                target_pose=target_pose,
                robot_cfg=robot_cfg,
                plan_cfg={
                    "joint_cfg": joint_cfg,
                    "enable_collision": True,
                },
                motion_gen=motion_gen,
            )
            # Stack angle
            ik_result.iks = stack_iks_angle(ik_result.iks, -angle) # TODO: negative
            
            print(f"  Angle {angle:.4f} rad: Found {ik_result.iks.shape[0]} IK solutions")
            
            # IK clustering
            ik_result = planner.ik_clustering(ik_result, **clustering_params)
            print(f"    After clustering: {ik_result.iks.shape[0]} IK solutions")
            return ik_result
        
        
        start_angles = [0.0]
        # goal_angles = [1.333088176515062, 1.2300752695249741, 1.0739553834158821, 0.9968029606353053]
        goal_angles = [0.9968029606353053]   
        
        # IK collection limits
        IK_LIMITS = {
            StageType.MOVE: [50, 50],
            StageType.MOVE_ARTICULATED: [30, 30],
        }
        
        start_ik_result = []
        goal_ik_result = []
        
        for _ in range(10):
            for angle in start_angles:
                ik_result = plan_single_ik(angle)
                start_ik_result.append(ik_result)
            if sum([r.iks.shape[0] for r in start_ik_result]) >= IK_LIMITS[StageType.MOVE_ARTICULATED][0]:
                break
            
        start_ik_result = IKResult.cat(start_ik_result)
    
        for _ in range(10):
            for angle in goal_angles:
                ik_result = plan_single_ik(angle)
                goal_ik_result.append(ik_result)
            if sum([r.iks.shape[0] for r in goal_ik_result]) >= IK_LIMITS[StageType.MOVE_ARTICULATED][1]:
                break
        
        goal_ik_result = IKResult.cat(goal_ik_result)
        
        print(f"IK Planning completed:")
        print(f"  Start IKs: {start_ik_result.iks.shape}")  
        print(f"  Goal IKs: {goal_ik_result.iks.shape}")
        
        # ===== SAVE IK RESULTS =====
        print("\n4. Saving IK results...")
        base_dir = f"data/run_1223/traj/summit_franka/scene_0_seed_0/7221/grasp_{grasp_id:04d}"
        os.makedirs(base_dir, exist_ok=True)
        save_ik(start_ik_result, f"{base_dir}/start_iks.pt")
        save_ik(goal_ik_result, f"{base_dir}/goal_iks.pt")
        
        if start_ik_result.iks.shape[0] == 0 or goal_ik_result.iks.shape[0] == 0:
            print("No IK solutions found for start or goal, skipping trajectory planning.")
            continue
        
        # ===== TRAJECTORY PLANNING =====
        print("\n5. Planning trajectories...")
        plan_cfg = {
            "stage_type": StageType.MOVE_ARTICULATED,
            "batch_size": 10,
            "expand_to_pairs": True,
        }
        
        akr_robot_cfg_path = f"assets/object/Microwave/7221/summit_franka_7221_0_grasp_{grasp_id:04d}.yml"
        
        akr_robot_cfg = load_robot_cfg(akr_robot_cfg_path)
        akr_robot_cfg = process_robot_cfg(akr_robot_cfg)
        
        motion_gen_akr = planner.init_motion_gen(akr_robot_cfg, fixed_base=True)
        
        traj_result = planner.plan_traj(
            start_iks=start_ik_result.iks,
            goal_iks=goal_ik_result.iks,
            robot_cfg=akr_robot_cfg,
            plan_cfg=plan_cfg,
            motion_gen=motion_gen_akr,
        )
        
        print(f"Trajectory Planning completed:")
        print(f"  Trajectories: {traj_result.trajectories.shape}")
        print(f"  Success rate: {traj_result.success.sum().item()}/{len(traj_result.success)}")
        
        # ===== SAVE TRAJECTORY RESULTS =====
        print("\n6. Saving trajectory results...")
        save_traj(traj_result, f"{base_dir}/traj_data.pt")
        
        # ===== TRAJECTORY FILTERING =====
        print("\n7. Filtering trajectories...")
        filter_cfg = {
            "stage_type": StageType.MOVE_ARTICULATED,
            "position_tolerance": 0.01,
            "rotation_tolerance": 0.05
        }
        filtered_result = planner.filter_traj(traj_result, robot_cfg=akr_robot_cfg, filter_cfg=filter_cfg, motion_gen=motion_gen_akr)
        
        print(f"Trajectory Filtering completed: {filtered_result.num_samples} trajectories")
        save_traj(filtered_result, f"{base_dir}/filtered_traj_data.pt")
        
        print(f"\n=== Processing for Grasp {grasp_id} Completed ===")

    print(f"\n=== ALL GRASP POSES PROCESSED ===")


if __name__ == "__main__":
    test_planner()