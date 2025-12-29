# Per-Object Plan Config - Quick Reference

## TL;DR

Add a `plan_cfg` field inside any object to override planning parameters for that specific object.

```yaml
plan_cfg:
  num_grasps: 20  # Default for all objects

env_cfg:
  object_cfg:
    "7221":
      path: assets/object/Microwave/7221/7221.urdf
      plan_cfg:
        num_grasps: 50  # Override for this object only
```

## Common Use Cases

### 1. More grasp samples for complex objects
```yaml
object_cfg:
  "difficult_object":
    plan_cfg:
      num_grasps: 40  # vs default 20
```

### 2. More optimization seeds for hard-to-plan objects
```yaml
object_cfg:
  "tricky_object":
    plan_cfg:
      num_trajopt_seeds: 20  # vs default 12
      num_graph_seeds: 20
```

### 3. Larger collision world for big objects
```yaml
object_cfg:
  "large_object":
    plan_cfg:
      voxel_dims: [6.0, 6.0, 6.0]  # vs default [5.0, 5.0, 5.0]
      voxel_size: 0.03  # vs default 0.02
```

### 4. More IK attempts
```yaml
object_cfg:
  "constrained_object":
    plan_cfg:
      plan_ik:
        limit: [200, 200]  # vs default [100, 100]
```

### 5. Relaxed filtering for large motions
```yaml
object_cfg:
  "big_motion_object":
    plan_cfg:
      filter:
        position_threshold: 0.02  # vs default 0.01
        orientation_threshold: 0.08  # vs default 0.05
```

## Available Parameters

All fields from `plan_cfg` can be overridden:

```yaml
plan_cfg:
  output_dir: "data/output"
  num_grasps: 20
  voxel_dims: [5.0, 5.0, 5.0]
  voxel_size: 0.02
  expand_dims: [1.0, 0.2, 0.2]
  collision_checker_type: "VOXEL"
  num_trajopt_seeds: 12
  num_graph_seeds: 12
  interpolation_dt: 0.05
  
  cluster:
    ap_fallback_clusters: 30
    ap_clusters_upperbound: 80
    ap_cluster_lowerbound: 10
  
  plan_ik:
    limit: [100, 100]
  
  plan_traj:
    batch_size: 10
    expand_to_pairs: true
  
  filter:
    position_threshold: 0.01
    orientation_threshold: 0.05
  
  resume: true
```

## Code Usage

### In tasks
```python
# Get plan_cfg for specific object
plan_cfg = self.get_plan_cfg(object_id)
num_grasps = plan_cfg.num_grasps

# Get default plan_cfg
default_cfg = self.get_plan_cfg()
```

### Standalone
```python
from automoma.utils.config_utils import get_object_plan_cfg

plan_cfg = get_object_plan_cfg(cfg, object_id)
```

## Rules

1. **Unspecified fields inherit from defaults** - You only need to specify what you want to override
2. **Nested configs merge recursively** - Can override one field in a nested dict without affecting others
3. **Backward compatible** - Objects without `plan_cfg` use defaults
4. **Automatic preprocessing** - Happens during config loading, no manual steps needed

## Examples

See:
- `configs/exps/single_object_open_test/plan.yaml` - Commented example
- `configs/exps/single_object_open_test/plan_example_per_object.yaml` - Full example with multiple objects

## Testing

```bash
# Run simplified tests
python tests/debug/test_per_object_plan_cfg_simple.py

# Run full tests (requires environment)
python tests/debug/test_per_object_plan_cfg.py
```

## Documentation

- Full guide: `docs/per_object_plan_cfg.md`
- Implementation: `docs/per_object_plan_cfg_implementation.md`
