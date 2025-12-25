"""Advanced configuration system with hierarchical loading and attribute access."""

import os
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class Config:
    """
    Configuration object with attribute-style access.
    
    Supports:
    - Attribute access: cfg.plan_cfg.num_grasps
    - Dict-style access: cfg["plan_cfg"]["num_grasps"]
    - Hierarchical merging with defaults
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize config from dictionary."""
        if data is None:
            data = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name: str) -> Any:
        """Get attribute, return None if not found."""
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            return None
    
    def __getitem__(self, key: str) -> Any:
        """Dict-style access."""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style assignment."""
        if isinstance(value, dict):
            setattr(self, key, Config(value))
        else:
            setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return hasattr(self, key) and getattr(self, key) is not None
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.to_dict()})"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get with default value."""
        value = getattr(self, key, None)
        return value if value is not None else default
    
    def keys(self) -> List[str]:
        """Get all keys."""
        return [k for k in self.__dict__.keys() if not k.startswith('_')]
    
    def items(self):
        """Get all items."""
        for key in self.keys():
            yield key, getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key in self.keys():
            value = getattr(self, key)
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def update(self, other: Union[Dict[str, Any], "Config"]) -> None:
        """Update config with another dict or Config."""
        if isinstance(other, Config):
            other = other.to_dict()
        
        for key, value in other.items():
            if isinstance(value, dict):
                existing = getattr(self, key, None)
                if isinstance(existing, Config):
                    existing.update(value)
                else:
                    setattr(self, key, Config(value))
            else:
                setattr(self, key, value)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    Override values take precedence over base values.
    """
    result = copy.deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    
    return result


def load_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f) or {}


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for configs directory
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "configs").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


class ConfigLoader:
    """
    Configuration loader that supports hierarchical config loading.
    
    Usage:
        loader = ConfigLoader()
        cfg = loader.load("multi_object_open")
        
        # Access config with attribute syntax
        num_grasps = cfg.plan_cfg.num_grasps
        robot_type = cfg.robot_cfg.robot_type
    """
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize config loader.
        
        Args:
            project_root: Project root directory. If None, auto-detected.
        """
        if project_root is None:
            self.project_root = get_project_root()
        else:
            self.project_root = Path(project_root)
        
        self.configs_dir = self.project_root / "configs"
        self.exps_dir = self.configs_dir / "exps"
        self.default_config_path = self.configs_dir / "config.yaml"
    
    def load(self, exp_name: str) -> Config:
        """
        Load experiment configuration.
        
        Merges default config with experiment-specific configs.
        
        Args:
            exp_name: Experiment name (e.g., "multi_object_open")
            
        Returns:
            Config object with merged configuration
        """
        # Load default config
        if self.default_config_path.exists():
            default_config = load_yaml(self.default_config_path)
        else:
            default_config = {}
        
        # Load experiment-specific configs
        exp_dir = self.exps_dir / exp_name
        if not exp_dir.exists():
            raise ValueError(f"Experiment directory not found: {exp_dir}")
        
        # Merge all experiment YAML files
        merged_config = copy.deepcopy(default_config)
        
        # Load in specific order: plan -> record -> train -> eval
        config_files = ["plan.yaml", "record.yaml", "train.yaml", "eval.yaml"]
        
        for config_file in config_files:
            config_path = exp_dir / config_file
            if config_path.exists():
                exp_config = load_yaml(config_path)
                merged_config = deep_merge(merged_config, exp_config)
        
        # Add metadata
        merged_config["_exp_name"] = exp_name
        merged_config["_project_root"] = str(self.project_root)
        
        return Config(merged_config)
    
    def load_plan_config(self, exp_name: str) -> Config:
        """Load only planning configuration."""
        return self._load_single_config(exp_name, "plan.yaml")
    
    def load_record_config(self, exp_name: str) -> Config:
        """Load only recording configuration."""
        return self._load_single_config(exp_name, "record.yaml")
    
    def load_train_config(self, exp_name: str) -> Config:
        """Load only training configuration."""
        return self._load_single_config(exp_name, "train.yaml")
    
    def load_eval_config(self, exp_name: str) -> Config:
        """Load only evaluation configuration."""
        return self._load_single_config(exp_name, "eval.yaml")
    
    def _load_single_config(self, exp_name: str, config_file: str) -> Config:
        """Load a single config file with defaults merged."""
        # Load default config
        if self.default_config_path.exists():
            default_config = load_yaml(self.default_config_path)
        else:
            default_config = {}
        
        # Load specific config
        config_path = self.exps_dir / exp_name / config_file
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        
        exp_config = load_yaml(config_path)
        merged_config = deep_merge(default_config, exp_config)
        
        merged_config["_exp_name"] = exp_name
        merged_config["_project_root"] = str(self.project_root)
        
        return Config(merged_config)


# Convenience functions for module-level use
_default_loader: Optional[ConfigLoader] = None


def get_loader(project_root: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """Get or create the default config loader."""
    global _default_loader
    if _default_loader is None or project_root is not None:
        _default_loader = ConfigLoader(project_root)
    return _default_loader


def load_config(exp_name: str, project_root: Optional[Union[str, Path]] = None) -> Config:
    """
    Load experiment configuration.
    
    Args:
        exp_name: Experiment name (e.g., "multi_object_open")
        project_root: Optional project root directory
        
    Returns:
        Config object with merged configuration
    """
    loader = get_loader(project_root)
    return loader.load(exp_name)


def load_plan_config(exp_name: str, project_root: Optional[Union[str, Path]] = None) -> Config:
    """Load planning configuration for an experiment."""
    loader = get_loader(project_root)
    return loader.load_plan_config(exp_name)


def load_record_config(exp_name: str, project_root: Optional[Union[str, Path]] = None) -> Config:
    """Load recording configuration for an experiment."""
    loader = get_loader(project_root)
    return loader.load_record_config(exp_name)


def load_train_config(exp_name: str, project_root: Optional[Union[str, Path]] = None) -> Config:
    """Load training configuration for an experiment."""
    loader = get_loader(project_root)
    return loader.load_train_config(exp_name)


def load_eval_config(exp_name: str, project_root: Optional[Union[str, Path]] = None) -> Config:
    """Load evaluation configuration for an experiment."""
    loader = get_loader(project_root)
    return loader.load_eval_config(exp_name)
