# Robot Configuration Design

## Overview

The robot configuration system supports both simple and articulated object manipulation tasks with a flexible, single motion generator approach.

## Configuration Structure

### Basic Configuration

Each task uses **one primary `motion_gen`** that can be replaced when needed for specific operations.

```yaml
# Main robot config (used for IK planning and simple tasks)
robot_cfg:
  robot_type: "summit_franka"
  path: "assets/robot/summit_franka/summit_franka.yml"

# Optional: Alternative config for IK planning (if different from robot_cfg)
ik_robot_cfg:
  robot_type: "summit_franka"
  path: "assets/robot/summit_franka/summit_franka.yml"
  fixed_base: false  # Mobile base for IK

# Template for articulated robot config (populated dynamically)
akr_robot_cfg:
  robot_type: "summit_franka"
  path: ""  # Filled as: assets/object/{asset_type}/{asset_id}/summit_franka_{asset_id}_0_grasp_{grasp_id:04d}.yml
  fixed_base: true  # Fixed base for articulated motion
```

## Task Types

### 1. Simple Tasks (e.g., Pick and Place)

**Configuration needed:**
- `robot_cfg`: Main robot configuration

**Usage:**
- Uses `self.motion_gen` initialized with `robot_cfg`
- No need for `akr_robot_cfg`

**Example:**
```python
class PickPlaceTask(BaseTask):
    def _plan_trajectories(self, start_iks, goal_iks, stage_type, grasp_id=None):
        # Uses self.motion_gen (from robot_cfg)
        return super()._plan_trajectories(start_iks, goal_iks, stage_type, grasp_id)
```

### 2. Articulated Object Tasks (e.g., Open Door/Microwave)

**Configuration needed:**
- `robot_cfg`: Main robot configuration
- `ik_robot_cfg` (optional): For IK planning with mobile base
- `akr_robot_cfg`: Template for articulated robot config

**Usage:**
- IK planning: Uses `self.motion_gen` (from `ik_robot_cfg` or `robot_cfg`)
- Trajectory planning: Temporarily replaces `self.motion_gen` with AKR motion gen
- AKR config path built as: `assets/object/{asset_type}/{asset_id}/summit_franka_{asset_id}_0_grasp_{grasp_id:04d}.yml`

**Example:**
```python
class OpenTask(BaseTask):
    def _plan_trajectories(self, start_iks, goal_iks, stage_type, grasp_id=None):
        if grasp_id is None:
            return super()._plan_trajectories(start_iks, goal_iks, stage_type)
        
        # Get AKR robot config and motion gen
        object_cfg = self.cfg.object_cfg[object_id]
        akr_robot_cfg, akr_motion_gen = self.get_akr_motion_gen(object_cfg, grasp_id)
        
        # Temporarily replace motion_gen
        original_motion_gen = self.motion_gen
        original_robot_cfg = self.robot_cfg
        
        self.motion_gen = akr_motion_gen
        self.robot_cfg = akr_robot_cfg
        
        try:
            result = super()._plan_trajectories(start_iks, goal_iks, stage_type)
        finally:
            # Restore originals
            self.motion_gen = original_motion_gen
            self.robot_cfg = original_robot_cfg
        
        return result
```

## AKR Robot Config Path Template

The AKR robot config path is constructed dynamically:

```
assets/object/{asset_type}/{asset_id}/{robot_type}_{asset_id}_0_grasp_{grasp_id:04d}.yml
```

**Components:**
- `asset_type`: Object type (e.g., "Microwave", "Refrigerator")
- `asset_id`: Object ID (e.g., "7221")
- `robot_type`: Robot type (e.g., "summit_franka")
- `grasp_id`: Grasp pose ID (zero-padded to 4 digits)

**Example:**
```
assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.yml
assets/object/Microwave/7221/summit_franka_7221_0_grasp_0001.yml
```

## Helper Method: `get_akr_motion_gen()`

The `BaseTask` class provides a helper method to create temporary AKR motion generators:

```python
def get_akr_motion_gen(self, object_cfg: Config, grasp_id: int):
    """
    Create a temporary motion generator with akr_robot_cfg.
    
    Returns:
        tuple: (akr_robot_cfg, akr_motion_gen)
    """
```

## Design Rationale

### Why Single Motion Generator?

1. **Simplicity**: Most tasks only need one motion generator
2. **Flexibility**: Easy to replace when needed for specific operations
3. **Clear ownership**: Each task controls its own motion planning strategy

### Why Not Multiple Motion Generators?

- **Overcomplicated**: Maintaining `motion_gen_ik` and `motion_gen_traj` adds unnecessary complexity
- **Unclear usage**: Which motion gen to use for each operation becomes confusing
- **Memory overhead**: Multiple motion generators consume more memory

### When to Use AKR Config?

Use AKR robot config when:
- Planning trajectories for **articulated objects** (doors, drawers, microwaves)
- Object is **attached to the robot** (fixed base planning)
- Need **collision checking** with the object during motion

Don't use AKR config for:
- Simple pick and place tasks
- Free-space motion planning
- IK planning (use `ik_robot_cfg` instead)

## Migration from Old Design

### Old Design (Multiple Motion Generators)
```python
# OLD - Don't use
self.motion_gen_ik = ...   # For IK planning
self.motion_gen_traj = ... # For trajectory planning
```

### New Design (Single Motion Generator)
```python
# NEW - Use this
self.motion_gen = ...  # Single motion gen, replace when needed

# For articulated tasks:
akr_robot_cfg, akr_motion_gen = self.get_akr_motion_gen(object_cfg, grasp_id)
# Temporarily replace self.motion_gen
# ... plan trajectories ...
# Restore original motion_gen
```

## Example Workflow

### Open Task Planning Pipeline

1. **Setup**: Initialize `motion_gen` with `ik_robot_cfg` (mobile base)
2. **IK Planning**: Use `self.motion_gen` to plan IK solutions
3. **Trajectory Planning**:
   - Get AKR robot config for specific grasp
   - Replace `self.motion_gen` with AKR motion gen
   - Plan trajectories with fixed base + attached object
   - Restore original motion gen
4. **Filtering**: Use same AKR motion gen as trajectory planning

```python
# 1. Setup
self.motion_gen = planner.init_motion_gen(ik_robot_cfg, fixed_base=False)

# 2. IK Planning
ik_result = planner.plan_ik(..., motion_gen=self.motion_gen)

# 3. Trajectory Planning
akr_robot_cfg, akr_motion_gen = self.get_akr_motion_gen(object_cfg, grasp_id)
self.motion_gen = akr_motion_gen
traj_result = planner.plan_traj(..., motion_gen=self.motion_gen)

# 4. Filtering  
filtered_result = planner.filter_traj(..., motion_gen=self.motion_gen)
```
