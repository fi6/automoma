# Per-Object Plan Config Feature - Implementation Summary

## Overview
Implemented support for per-object `plan_cfg` overrides in the AutoMoMa planning system. Each object in `object_cfg` can now have its own `plan_cfg` that overrides the default configuration, allowing fine-tuned planning parameters for specific objects.

## Changes Made

### 1. Core Utilities (`src/automoma/utils/config_utils.py`)
**New file** containing:
- `preprocess_object_plan_configs(cfg)`: Preprocesses configs to merge object-specific plan_cfg with defaults
- `get_object_plan_cfg(cfg, object_id)`: Retrieves the resolved plan_cfg for a specific object
- `deep_merge_dicts(base, override)`: Deep merges two dictionaries recursively

### 2. Config Loader (`src/automoma/core/config_loader.py`)
**Modified** to automatically preprocess configs:
- `ConfigLoader.load()`: Added automatic call to `preprocess_object_plan_configs()`
- `ConfigLoader._load_single_config()`: Added preprocessing for plan.yaml files
- Preprocessing happens transparently during config loading

### 3. Base Task (`src/automoma/tasks/base_task.py`)
**Modified** to add helper method:
- `get_plan_cfg(object_id)`: Returns object-specific or default plan_cfg
  - If `object_id` is None, returns default plan_cfg
  - If object has custom config, returns merged config
  - Otherwise, returns default config

### 4. Open Task (`src/automoma/tasks/open_task.py`)
**Modified** to use object-specific configs:
- `get_grasp_poses()`: Uses `self.get_plan_cfg(object_id)` for `num_grasps`
- `plan_ik_for_stage()`: Uses object-specific IK limits and clustering params
- `_plan_trajectories()`: Uses object-specific trajectory planning params
  - Added `object_id` parameter
- `_filter_trajectories()`: Uses object-specific filter thresholds
  - Added `object_id` parameter
- `run_planning_pipeline()`: Passes `object_id` to planning methods

### 5. Configuration Example (`configs/exps/single_object_open_test/plan.yaml`)
**Modified** to show usage:
- Added header documentation explaining the feature
- Added commented example of per-object plan_cfg override
- Demonstrates inheritance and override behavior

### 6. Documentation (`docs/per_object_plan_cfg.md`)
**New file** containing:
- Feature overview and motivation
- Configuration structure and syntax
- Merging behavior explanation
- Usage examples in code
- Implementation details
- Migration guide
- Testing instructions

### 7. Tests
Created two test files:
- `tests/debug/test_per_object_plan_cfg.py`: Full integration test (requires torch)
- `tests/debug/test_per_object_plan_cfg_simple.py`: Simplified unit tests (no dependencies)

Both tests validate:
- Deep merging logic
- Config structure
- Preprocessing logic
- Override and inheritance behavior

## How It Works

### Configuration Flow
```
1. Load YAML config
   ↓
2. Merge with defaults (existing behavior)
   ↓
3. Create Config object
   ↓
4. preprocess_object_plan_configs() ← NEW
   ↓
5. For each object in object_cfg:
   - If object has plan_cfg: merge with default
   - Else: use default plan_cfg
   - Store result in object._resolved_plan_cfg
   ↓
6. Task uses get_plan_cfg(object_id) to access
```

### Merging Behavior
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

# Result for object 7221
_resolved_plan_cfg:
  num_grasps: 50                        # ← overridden
  filter:
    position_threshold: 0.02            # ← overridden
    orientation_threshold: 0.05         # ← inherited
```

## Usage Examples

### In YAML Config
```yaml
plan_cfg:
  num_grasps: 20  # Default
  num_trajopt_seeds: 12

env_cfg:
  object_cfg:
    "7221":
      path: assets/object/Microwave/7221/7221.urdf
      # Use custom config for this object
      plan_cfg:
        num_grasps: 50
        num_trajopt_seeds: 20
    
    "7222":
      path: assets/object/Microwave/7222/7222.urdf
      # No plan_cfg - uses defaults
```

### In Python Code
```python
from automoma.core.config_loader import load_config
from automoma.tasks.factory import create_task

# Load config (preprocessing is automatic)
cfg = load_config("my_experiment")
task = create_task(cfg)

# Get plan_cfg for specific object
plan_cfg = task.get_plan_cfg("7221")
num_grasps = plan_cfg.num_grasps  # Returns 50 if overridden

# Or use utility function directly
from automoma.utils.config_utils import get_object_plan_cfg
plan_cfg = get_object_plan_cfg(cfg, "7221")
```

## Benefits

1. **Flexibility**: Customize planning parameters per object without affecting others
2. **Maintainability**: Keep default values in one place, override only what's needed
3. **Backward Compatible**: Existing configs work without modification
4. **Type Safe**: Uses existing Config class with attribute access
5. **Automatic**: Preprocessing happens transparently during config loading
6. **Recursive**: Nested configs are merged properly

## Testing

Run tests to verify:
```bash
# Simplified test (no dependencies)
python tests/debug/test_per_object_plan_cfg_simple.py

# Full integration test (requires torch and full environment)
python tests/debug/test_per_object_plan_cfg.py
```

## Files Created/Modified

### Created (4 files)
- `src/automoma/utils/config_utils.py`
- `docs/per_object_plan_cfg.md`
- `tests/debug/test_per_object_plan_cfg.py`
- `tests/debug/test_per_object_plan_cfg_simple.py`

### Modified (4 files)
- `src/automoma/core/config_loader.py`
- `src/automoma/tasks/base_task.py`
- `src/automoma/tasks/open_task.py`
- `configs/exps/single_object_open_test/plan.yaml`

## Future Enhancements

Potential extensions of this feature:
- Per-object `record_cfg` (recording parameters)
- Per-object `train_cfg` (training hyperparameters)
- Per-scene configurations
- Template-based configurations for object categories
- Configuration validation and type checking
