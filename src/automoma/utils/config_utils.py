"""Configuration utilities for preprocessing and merging configs."""

import copy
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def preprocess_object_plan_configs(cfg) -> None:
    """
    Preprocess object configurations to merge per-object plan_cfg with default plan_cfg.
    
    This function modifies the config in-place. For each object in object_cfg:
    - If the object has a 'plan_cfg' field, it overrides the default plan_cfg
    - If not specified, the object uses the default plan_cfg from cfg.plan_cfg
    - The merged plan_cfg is stored in a new field '_resolved_plan_cfg' for each object
    
    Args:
        cfg: Configuration object with plan_cfg and env_cfg.object_cfg
        
    Example:
        cfg.plan_cfg.num_grasps = 20  # default
        cfg.env_cfg.object_cfg["7221"].plan_cfg.num_grasps = 50  # object-specific
        
        After preprocessing:
        cfg.env_cfg.object_cfg["7221"]._resolved_plan_cfg.num_grasps = 50
        cfg.env_cfg.object_cfg["7221"]._resolved_plan_cfg.voxel_size = 0.02  # from default
    """
    if not hasattr(cfg, 'env_cfg') or not hasattr(cfg.env_cfg, 'object_cfg'):
        logger.debug("No object_cfg found, skipping object plan config preprocessing")
        return
    
    if not hasattr(cfg, 'plan_cfg'):
        logger.warning("No default plan_cfg found, cannot preprocess object plan configs")
        return
    
    # Get default plan_cfg as dict
    from automoma.core.config_loader import Config
    default_plan_cfg = cfg.plan_cfg.to_dict() if isinstance(cfg.plan_cfg, Config) else cfg.plan_cfg
    
    # Process each object
    for object_id, object_cfg in cfg.env_cfg.object_cfg.items():
        # Convert to Config if it's a dict
        if not isinstance(object_cfg, Config):
            object_cfg = Config(object_cfg)
            cfg.env_cfg.object_cfg[object_id] = object_cfg
        
        # Check if object has custom plan_cfg
        if hasattr(object_cfg, 'plan_cfg') and object_cfg.plan_cfg is not None:
            # Merge object-specific plan_cfg with default
            object_plan_cfg = object_cfg.plan_cfg.to_dict() if isinstance(object_cfg.plan_cfg, Config) else object_cfg.plan_cfg
            resolved_plan_cfg = deep_merge_dicts(default_plan_cfg, object_plan_cfg)
            logger.info(f"Object {object_id}: Using custom plan_cfg (merged with defaults)")
        else:
            # Use default plan_cfg
            resolved_plan_cfg = copy.deepcopy(default_plan_cfg)
            logger.debug(f"Object {object_id}: Using default plan_cfg")
        
        # Store resolved config
        object_cfg._resolved_plan_cfg = Config(resolved_plan_cfg)


def get_object_plan_cfg(cfg, object_id: str):
    """
    Get the resolved plan_cfg for a specific object.
    
    Args:
        cfg: Configuration object
        object_id: Object ID
        
    Returns:
        Config object with resolved plan_cfg for the object.
        Falls back to default plan_cfg if object not found or not preprocessed.
    """
    try:
        object_cfg = cfg.env_cfg.object_cfg[object_id]
        if hasattr(object_cfg, '_resolved_plan_cfg'):
            return object_cfg._resolved_plan_cfg
        else:
            logger.warning(f"Object {object_id} has no _resolved_plan_cfg, using default. "
                         f"Did you call preprocess_object_plan_configs()?")
            return cfg.plan_cfg
    except (AttributeError, KeyError) as e:
        logger.warning(f"Could not get plan_cfg for object {object_id}: {e}. Using default.")
        return cfg.plan_cfg


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Override values take precedence over base values.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base dictionary with default values
        override: Override dictionary with new values
        
    Returns:
        Merged dictionary with base values overridden by override
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result
