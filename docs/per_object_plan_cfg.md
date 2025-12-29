# Per-Object Plan Configuration

## Overview

The AutoMoMa planning system now supports **per-object plan configuration overrides**. This allows you to customize planning parameters for specific objects without affecting the default configuration used by other objects.

## Motivation

Different objects may require different planning parameters:
- Complex objects may need more grasp samples (`num_grasps`)
- Large objects may need different collision voxel sizes
- Difficult-to-plan objects may benefit from more trajectory optimization seeds

Previously, you had to use the same planning parameters for all objects in an experiment. Now, you can override specific parameters on a per-object basis.

## How It Works

### 1. Configuration Structure

In your `plan.yaml`, you can now add a `plan_cfg` field inside any object definition:

```yaml
plan_cfg:
  num_grasps: 20  # Default for all objects
  num_trajopt_seeds: 12
  voxel_size: 0.02

env_cfg:
  object_cfg:
    "7221":
      path: assets/object/Microwave/7221/7221_0_scaling.urdf
      asset_id: "7221"
      # Override plan_cfg for this specific object
      plan_cfg:
        num_grasps: 50  # Use 50 grasps instead of default 20
        num_trajopt_seeds: 20  # Use 20 seeds instead of default 12
        # Other fields not specified will use defaults
    
    "7222":
      path: assets/object/Microwave/7222/7222_0_scaling.urdf
      asset_id: "7222"
      # No plan_cfg specified - uses all defaults
```

### 2. Merging Behavior

The system performs **deep merging** of object-specific configs with defaults:

- **Override**: Fields specified in object's `plan_cfg` override defaults
- **Inherit**: Fields not specified in object's `plan_cfg` use defaults
- **Recursive**: Nested dictionaries are merged recursively

Example:
```yaml
# Default
plan_cfg:
  num_grasps: 20
  filter:
    position_threshold: 0.01
    orientation_threshold: 0.05

# Object override
object_cfg:
  "7221":
    plan_cfg:
      num_grasps: 50
      filter:
        position_threshold: 0.02
        # orientation_threshold not specified

# Result for object 7221:
# num_grasps: 50 (overridden)
# filter.position_threshold: 0.02 (overridden)
# filter.orientation_threshold: 0.05 (inherited from default)
```

### 3. Usage in Code

The configuration system automatically preprocesses configs during loading:

```python
from automoma.core.config_loader import load_config
from automoma.tasks.factory import create_task

# Load config (preprocessing happens automatically)
cfg = load_config("my_experiment")

# Create task
task = create_task(cfg)

# Get plan_cfg for a specific object
object_id = "7221"
plan_cfg = task.get_plan_cfg(object_id)

# plan_cfg now contains merged config for object 7221
num_grasps = plan_cfg.num_grasps  # Returns 50 if overridden, else 20
```

### 4. Utility Functions

Two utility functions are provided:

#### `preprocess_object_plan_configs(cfg)`
Preprocesses the configuration to merge per-object plan configs with defaults.
Called automatically during config loading.

#### `get_object_plan_cfg(cfg, object_id)`
Retrieves the resolved plan_cfg for a specific object.

```python
from automoma.utils.config_utils import get_object_plan_cfg

plan_cfg = get_object_plan_cfg(cfg, "7221")
```

## Implementation Details

### Automatic Preprocessing

The config loader automatically calls `preprocess_object_plan_configs()` after loading:

```python
# In ConfigLoader.load()
cfg = Config(resolved_config)
preprocess_object_plan_configs(cfg)  # Automatic preprocessing
return cfg
```

### Storage

Resolved configs are stored in a special `_resolved_plan_cfg` attribute:

```python
cfg.env_cfg.object_cfg["7221"]._resolved_plan_cfg
```

This preserves the original object-specific config while providing easy access to the merged result.

### Task Integration

The `BaseTask` class provides a helper method:

```python
class BaseTask:
    def get_plan_cfg(self, object_id: Optional[str] = None):
        """
        Get the appropriate plan_cfg for an object.
        
        Args:
            object_id: Object ID. If None, returns default plan_cfg.
            
        Returns:
            Merged plan_cfg for the object, or default if object_id is None.
        """
        if object_id is None:
            return self.cfg.plan_cfg
        
        from automoma.utils.config_utils import get_object_plan_cfg
        return get_object_plan_cfg(self.cfg, object_id)
```

## Example: Customizing Planning for Difficult Objects

```yaml
plan_cfg:
  # Defaults
  num_grasps: 20
  num_trajopt_seeds: 12
  num_graph_seeds: 12
  plan_ik:
    limit: [100, 100]

env_cfg:
  object_cfg:
    # Easy object - use defaults
    "7221":
      path: assets/object/Microwave/7221/7221_0_scaling.urdf
      asset_id: "7221"
    
    # Difficult object - needs more samples
    "8888":
      path: assets/object/Oven/8888/8888_0_scaling.urdf
      asset_id: "8888"
      plan_cfg:
        num_grasps: 40  # More grasp samples
        num_trajopt_seeds: 20  # More optimization seeds
        plan_ik:
          limit: [200, 200]  # More IK attempts
    
    # Large object - needs different collision parameters
    "9999":
      path: assets/object/Refrigerator/9999/9999_0_scaling.urdf
      asset_id: "9999"
      plan_cfg:
        voxel_dims: [6.0, 6.0, 6.0]  # Larger collision world
        voxel_size: 0.03  # Coarser voxels for performance
```

## Testing

Run the test script to verify the feature:

```bash
cd /home/xinhai/Documents/AutoMoMa
conda activate refactor
python tests/debug/test_per_object_plan_cfg.py
```

The test script validates:
- Config loading
- Object config structure
- Utility function behavior
- Task integration
- Override and inheritance logic

## Migration Guide

### Existing Configs

Existing configs continue to work without modification. If an object doesn't have a `plan_cfg` field, it uses the default config.

### Updating Configs

To add per-object overrides:

1. Identify objects that need custom parameters
2. Add a `plan_cfg` field under those objects
3. Specify only the fields you want to override
4. All other fields will inherit from the default

### Code Changes

Task implementations have been updated to use `self.get_plan_cfg(object_id)` instead of `self.cfg.plan_cfg` in:
- `get_grasp_poses()` - for `num_grasps`
- `plan_ik_for_stage()` - for IK limits and clustering
- `_plan_trajectories()` - for trajectory planning parameters
- `_filter_trajectories()` - for filtering thresholds

## Files Modified

- `src/automoma/utils/config_utils.py` - New utility functions
- `src/automoma/core/config_loader.py` - Automatic preprocessing
- `src/automoma/tasks/base_task.py` - Helper method
- `src/automoma/tasks/open_task.py` - Use object-specific configs
- `configs/exps/single_object_open_test/plan.yaml` - Example usage

## Future Enhancements

Possible extensions:
- Per-object record_cfg (recording parameters)
- Per-object train_cfg (training hyperparameters)
- Per-scene configurations
- Template-based configurations for object categories
