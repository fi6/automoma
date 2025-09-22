# Enhanced Replay Pipeline - Data Collection and Policy Evaluation

This document describes the new data collection and policy evaluation capabilities added to the Automoma replay pipeline.

## Overview

The enhanced replay pipeline now supports:

1. **Trajectory Recording**: Collect camera observations and robot state data during trajectory execution
2. **Policy Evaluation**: Test and evaluate trained policies against ground truth trajectories  
3. **Structured Data Storage**: Save data in HDF5 format compatible with existing training pipelines
4. **Fixed Camera Setup**: Automatically configure 3 cameras with consistent positioning

## New Components

### 1. Data Structures (`src/automoma/utils/data_structures.py`)

- **`CameraResult`**: Stores trajectory data with camera observations in the same format as `collect_data.py`
- **`TrajectoryEvaluationResult`**: Stores policy evaluation results with metrics

### 2. Enhanced Replayer (`src/automoma/utils/replayer.py`)

New methods added:
- `setup_fixed_cameras()`: Configure 3 fixed cameras (ego_topdown, ego_wrist, fix_local)
- `replay_traj_record()`: Record trajectory data with camera observations
- `replay_traj_evaluate()`: Evaluate policy models on trajectories
- Camera management and observation collection methods

### 3. Enhanced ReplayPipeline (`src/automoma/pipeline/replay.py`)

New methods added:
- `replay_traj_record()`: High-level interface for trajectory recording
- `replay_traj_evaluate()`: High-level interface for policy evaluation
- `save_evaluation_results()`: Save evaluation metrics and results

## Camera Configuration

The system sets up 3 fixed cameras automatically:

1. **ego_topdown**: Top-down view following the end-effector
2. **ego_wrist**: Wrist-mounted camera view  
3. **fix_local**: Fixed camera positioned at `[[-4.3, 4.7, 6.2], [-34.0, -26.0, -140.0]]` relative to object pose

## Data Storage Structure

Data is saved in the following directory structure:
```
output/
└── summit_franka/
    └── scene_1_seed_1/
        └── 7221/
            └── grasp_0001/
                ├── ik_data.pt          # Original IK data
                ├── traj_data.pt        # Original trajectory data
                ├── camera_data/        # NEW: Camera observations
                │   ├── episode000000.hdf5
                │   ├── episode000001.hdf5
                │   └── ...
                └── evaluation/         # NEW: Evaluation results
                    ├── eval_results_grasp_0001.pt
                    └── metrics_summary_grasp_0001.txt
```

## Usage Examples

### 1. Basic Trajectory Recording

```python
from automoma.pipeline import ReplayPipeline
from automoma.models.task import TaskDescription

# Create task description
task = TaskDescription(...)  # Your task setup

# Create replay pipeline  
replay_pipeline = ReplayPipeline(task, simulation_app)

# Record trajectory data
camera_results = replay_pipeline.replay_traj_record(
    grasp_id=0,
    num_episodes=10
)

print(f"Recorded {len(camera_results)} episodes")
```

### 2. Policy Evaluation

```python
# Your trained policy model
policy_model = load_your_policy_model()

# Evaluate policy
eval_results = replay_pipeline.replay_traj_evaluate(
    policy_model=policy_model,
    grasp_id=0, 
    num_episodes=5
)

# Save results
replay_pipeline.save_evaluation_results(
    eval_results, 
    output_dir="path/to/output",
    grasp_id=0
)
```

### 3. Data Format Validation

The recorded HDF5 files follow the same structure as `collect_data.py`:

```python
import h5py

with h5py.File("episode000000.hdf5", "r") as f:
    # Environment info
    env_info = f["env_info"]
    scene_id = env_info["scene_id"][()]
    robot_name = env_info["robot_name"][()]
    
    # Observations
    obs = f["obs"]
    joint_data = obs["joint"]    # Joint positions over time
    eef_data = obs["eef"]        # End-effector poses
    rgb_data = obs["rgb"]        # RGB images from cameras
    depth_data = obs["depth"]    # Depth images from cameras
    pc_data = obs["point_cloud"] # Point cloud data
```

## Integration with Existing Workflows

### With RoboTwin Data Collection

The new system produces data in the exact same format as the existing `collect_data.py`, allowing seamless integration:

- Same HDF5 structure
- Same observation keys and data types
- Compatible with existing training scripts

### With Policy Training

The evaluation system can work with various policy types by implementing the policy interface:

```python
class YourPolicy:
    def predict(self, observation):
        # Your policy logic here
        return action  # numpy array
```

### With Baseline Systems

The recording and evaluation systems use the same observation format as:
- `test_data.py`: For general policy testing
- `dp3.py`: For DP3 policy evaluation

## Testing and Validation

Run the example script to test the new functionality:

```bash
python examples/example_replay.py
```

This will demonstrate:
1. Standard replay functionality (unchanged)
2. New trajectory recording capabilities
3. Policy evaluation (with dummy policy)
4. Data format validation

## Future Enhancements

Potential improvements for future versions:

1. **Multiple Camera Views**: Support for additional camera angles
2. **Real-time Policy Inference**: Stream policy actions during recording
3. **Batch Processing**: Record/evaluate multiple grasps simultaneously
4. **Metrics Dashboard**: Visualization of evaluation metrics
5. **Data Augmentation**: Apply transforms during recording

## Troubleshooting

### Common Issues

1. **Camera Setup Fails**: Check Isaac Sim environment and camera prim paths
2. **Recording Empty Data**: Verify trajectory data exists and is accessible
3. **Policy Evaluation Errors**: Ensure policy model interface matches expected format
4. **HDF5 Save Errors**: Check write permissions and disk space

### Debug Tips

- Enable debug logging: `log_info()` messages show pipeline progress
- Check intermediate data: Inspect `CameraResult` objects before saving
- Validate file structure: Use HDF5 viewers to check saved data

## Dependencies

The enhanced system requires:
- Isaac Sim with Omniverse
- PyTorch
- h5py
- NumPy
- All existing Automoma dependencies

## Contact

For questions or issues with the enhanced replay pipeline, please create an issue or contact the development team.