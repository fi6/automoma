# Automoma

A Python package for robot trajectory generation using curobo as a motion planner. Automoma takes object, task, scene, and robot descriptions as input and generates optimized robot trajectories.

## Features

- **Multi-input Planning**: Accepts object, task, scene, and robot configurations
- **Curobo Integration**: Leverages NVIDIA's curobo for efficient motion planning
- **Trajectory Optimization**: Generates collision-free, optimized robot trajectories
- **Modular Design**: Clean separation of concerns with dedicated modules for each input type
- **Type Safety**: Full type annotations for better development experience
- **Extensible**: Easy to extend with custom objects, tasks, and robots

## Installation

### From PyPI (when available)

```bash
pip install automoma
```

### From Source

```bash
git clone https://github.com/yourusername/automoma.git
cd automoma
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/yourusername/automoma.git
cd automoma
pip install -e ".[dev]"
```

## Quick Start

```python
from automoma import AutomomaPlanner
from automoma.objects import ObjectDescription
from automoma.tasks import TaskDescription
from automoma.scenes import SceneDescription
from automoma.robots import RobotDescription

# Define your robot
robot = RobotDescription(
    name="ur5e",
    urdf_path="path/to/ur5e.urdf",
    # ... other robot parameters
)

# Define the scene
scene = SceneDescription(
    obstacles=[...],
    workspace_bounds=[...],
    # ... other scene parameters
)

# Define objects to manipulate
objects = [
    ObjectDescription(
        name="box",
        mesh_path="path/to/box.obj",
        pose=[0.5, 0.0, 0.1, 0, 0, 0, 1],  # [x, y, z, qx, qy, qz, qw]
    )
]

# Define the task
task = TaskDescription(
    task_type="pick_and_place",
    target_object="box",
    goal_pose=[0.3, 0.3, 0.1, 0, 0, 0, 1],
    # ... other task parameters
)

# Create planner and generate trajectory
planner = AutomomaPlanner()
trajectory = planner.plan(
    robot=robot,
    scene=scene,
    objects=objects,
    task=task
)

# Execute or visualize trajectory
print(f"Generated trajectory with {len(trajectory.waypoints)} waypoints")
```

## Project Organization

```
automoma/
├── src/
│   └── automoma/
│       ├── __init__.py          # Main package interface
│       ├── planner/             # Core planning module
│       │   ├── __init__.py
│       │   ├── automoma_planner.py
│       │   └── trajectory.py
│       ├── objects/             # Object description and handling
│       │   ├── __init__.py
│       │   ├── object_description.py
│       │   └── object_manager.py
│       ├── tasks/               # Task definition and processing
│       │   ├── __init__.py
│       │   ├── task_description.py
│       │   └── task_processor.py
│       ├── scenes/              # Scene representation and management
│       │   ├── __init__.py
│       │   ├── scene_description.py
│       │   └── scene_manager.py
│       ├── robots/              # Robot configuration and kinematics
│       │   ├── __init__.py
│       │   ├── robot_description.py
│       │   └── robot_manager.py
│       ├── curobo_interface/    # Curobo integration
│       │   ├── __init__.py
│       │   ├── planner_wrapper.py
│       │   └── config_converter.py
│       └── utils/               # Utility functions
│           ├── __init__.py
│           ├── transforms.py
│           ├── validation.py
│           └── visualization.py
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_planner/
│   ├── test_objects/
│   ├── test_tasks/
│   ├── test_scenes/
│   ├── test_robots/
│   ├── test_curobo_interface/
│   └── test_utils/
├── docs/                        # Documentation
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── api/
│   ├── Makefile
│   └── make.bat
├── examples/                    # Example scripts and notebooks
│   ├── basic_usage.py
│   ├── pick_and_place.py
│   └── multi_robot.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── .devcontainer/
│   ├── devcontainer.json
│   └── Dockerfile
├── .vscode/
│   └── settings.json
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Development

### Setting up Development Environment

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Install pre-commit hooks: `pre-commit install`
4. Run tests: `pytest`

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=automoma --cov-report=html

# Run specific test categories
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests
```

### Code Quality

This project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pylint**: Additional linting

Run all checks:

```bash
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Ensure all tests pass and code quality checks pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NVIDIA curobo](https://github.com/NVlabs/curobo) for the underlying motion planning
- The robotics community for inspiration and best practices

## Citation

If you use Automoma in your research, please cite:

```bibtex
@software{automoma,
  title={Automoma: Robot Trajectory Generation with Curobo},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/automoma}
}
```
