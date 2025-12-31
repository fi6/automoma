"""Component registry for AutoMoMa framework."""

from typing import Dict, Any, Optional, Type, Callable
from functools import wraps


class Registry:
    """Central registry for framework components."""
    
    _instances: Dict[str, "Registry"] = {}
    
    def __init__(self, name: str):
        self.name = name
        self._components: Dict[str, Any] = {}
    
    @classmethod
    def get_registry(cls, name: str) -> "Registry":
        """Get or create a registry by name."""
        if name not in cls._instances:
            cls._instances[name] = cls(name)
        return cls._instances[name]
    
    def register(self, name: str, component: Any) -> None:
        """Register a component."""
        if name in self._components:
            raise ValueError(f"Component '{name}' already registered in '{self.name}'")
        self._components[name] = component
    
    def get(self, name: str) -> Any:
        """Get a registered component."""
        if name not in self._components:
            raise KeyError(f"Component '{name}' not found in '{self.name}'. Available: {list(self._components.keys())}")
        return self._components[name]
    
    def has(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components
    
    def list(self) -> list:
        """List all registered component names."""
        return list(self._components.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._components
    
    def __getitem__(self, name: str) -> Any:
        return self.get(name)


# Pre-defined registries
PLANNER_REGISTRY = Registry.get_registry("planners")
TASK_REGISTRY = Registry.get_registry("tasks")
POLICY_REGISTRY = Registry.get_registry("policies")
DATASET_REGISTRY = Registry.get_registry("datasets")
SIMULATOR_REGISTRY = Registry.get_registry("simulators")


def register_component(registry_name: str, component_name: str) -> Callable:
    """Decorator to register a component."""
    def decorator(cls: Type) -> Type:
        registry = Registry.get_registry(registry_name)
        registry.register(component_name, cls)
        return cls
    return decorator


def get_component(registry_name: str, component_name: str) -> Any:
    """Get a component from a registry."""
    registry = Registry.get_registry(registry_name)
    return registry.get(component_name)


def register_planner(name: str) -> Callable:
    """Register a planner component."""
    return register_component("planners", name)


def register_task(name: str) -> Callable:
    """Register a task component."""
    return register_component("tasks", name)


def register_policy(name: str) -> Callable:
    """Register a policy component."""
    return register_component("policies", name)
