"""Dataset format converters."""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from automoma.core.types import DatasetFormat


class DatasetConverter:
    """
    Converter for different dataset formats.
    
    Supports conversion between:
    - LeRobot format
    - HDF5 format
    - Zarr format
    """
    
    def __init__(self, source_format: DatasetFormat, target_format: DatasetFormat):
        """
        Initialize the converter.
        
        Args:
            source_format: Source dataset format
            target_format: Target dataset format
        """
        self.source_format = source_format
        self.target_format = target_format
    
    def convert(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Convert dataset from source to target format.
        
        Args:
            source_path: Path to source dataset
            target_path: Path for target dataset
            **kwargs: Additional conversion options
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        
        # Load from source
        data = self._load_source(source_path, **kwargs)
        
        # Save to target
        self._save_target(data, target_path, **kwargs)
        
        print(f"Converted {source_path} ({self.source_format.name}) -> {target_path} ({self.target_format.name})")
    
    def _load_source(self, source_path: Path, **kwargs) -> Dict[str, Any]:
        """Load data from source format."""
        if self.source_format == DatasetFormat.LEROBOT:
            return self._load_lerobot(source_path, **kwargs)
        elif self.source_format == DatasetFormat.HDF5:
            return self._load_hdf5(source_path, **kwargs)
        elif self.source_format == DatasetFormat.ZARR:
            return self._load_zarr(source_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source format: {self.source_format}")
    
    def _save_target(self, data: Dict[str, Any], target_path: Path, **kwargs) -> None:
        """Save data to target format."""
        if self.target_format == DatasetFormat.LEROBOT:
            self._save_lerobot(data, target_path, **kwargs)
        elif self.target_format == DatasetFormat.HDF5:
            self._save_hdf5(data, target_path, **kwargs)
        elif self.target_format == DatasetFormat.ZARR:
            self._save_zarr(data, target_path, **kwargs)
        else:
            raise ValueError(f"Unsupported target format: {self.target_format}")
    
    def _load_lerobot(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Load LeRobot format dataset."""
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            dataset = LeRobotDataset(repo_id=str(path))
            
            data = {
                "episodes": [],
                "metadata": {
                    "fps": dataset.fps,
                    "robot_type": getattr(dataset, "robot_type", "unknown"),
                    "features": dataset.features if hasattr(dataset, "features") else {},
                },
            }
            
            # Extract episodes
            current_episode = {"frames": []}
            current_episode_idx = 0
            
            for i in range(len(dataset)):
                frame = dataset[i]
                episode_idx = frame.get("episode_index", 0)
                
                if episode_idx != current_episode_idx:
                    if current_episode["frames"]:
                        data["episodes"].append(current_episode)
                    current_episode = {"frames": []}
                    current_episode_idx = episode_idx
                
                frame_data = {}
                for key, value in frame.items():
                    if isinstance(value, torch.Tensor):
                        frame_data[key] = value.numpy()
                    else:
                        frame_data[key] = value
                
                current_episode["frames"].append(frame_data)
            
            if current_episode["frames"]:
                data["episodes"].append(current_episode)
            
            return data
            
        except Exception as e:
            print(f"Error loading LeRobot dataset: {e}")
            return {"episodes": [], "metadata": {}}
    
    def _load_hdf5(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Load HDF5 format dataset."""
        import h5py
        
        data = {"episodes": [], "metadata": {}}
        
        with h5py.File(path, "r") as f:
            # Load metadata
            if "metadata" in f:
                for key in f["metadata"].keys():
                    data["metadata"][key] = f["metadata"][key][()]
            
            # Load episodes
            if "episodes" in f:
                for ep_key in sorted(f["episodes"].keys()):
                    episode = {"frames": []}
                    ep_group = f["episodes"][ep_key]
                    
                    num_frames = len(ep_group.get("state", []))
                    for i in range(num_frames):
                        frame = {}
                        for data_key in ep_group.keys():
                            frame[data_key] = ep_group[data_key][i]
                        episode["frames"].append(frame)
                    
                    data["episodes"].append(episode)
        
        return data
    
    def _load_zarr(self, path: Path, **kwargs) -> Dict[str, Any]:
        """Load Zarr format dataset."""
        import zarr
        
        data = {"episodes": [], "metadata": {}}
        
        root = zarr.open(str(path), mode="r")
        
        # Load metadata
        if "metadata" in root:
            for key in root["metadata"].array_keys():
                data["metadata"][key] = root["metadata"][key][:]
        
        # Load episodes
        if "episodes" in root:
            for ep_key in sorted(root["episodes"].group_keys()):
                episode = {"frames": []}
                ep_group = root["episodes"][ep_key]
                
                num_frames = len(ep_group.get("state", []))
                for i in range(num_frames):
                    frame = {}
                    for data_key in ep_group.array_keys():
                        frame[data_key] = ep_group[data_key][i]
                    episode["frames"].append(frame)
                
                data["episodes"].append(episode)
        
        return data
    
    def _save_lerobot(self, data: Dict[str, Any], path: Path, **kwargs) -> None:
        """Save to LeRobot format."""
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            metadata = data.get("metadata", {})
            fps = metadata.get("fps", 15)
            robot_type = metadata.get("robot_type", "unknown")
            features = metadata.get("features", {})
            
            # Create dataset
            dataset = LeRobotDataset.create(
                repo_id=str(path.name),
                root=str(path.parent),
                fps=fps,
                features=features,
                robot_type=robot_type,
            )
            
            # Add episodes
            for episode in data.get("episodes", []):
                for frame in episode.get("frames", []):
                    dataset.add_frame(frame)
                dataset.save_episode()
            
            dataset.finalize()
            
        except Exception as e:
            print(f"Error saving LeRobot dataset: {e}")
    
    def _save_hdf5(self, data: Dict[str, Any], path: Path, **kwargs) -> None:
        """Save to HDF5 format."""
        import h5py
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(path, "w") as f:
            # Save metadata
            meta_group = f.create_group("metadata")
            for key, value in data.get("metadata", {}).items():
                if isinstance(value, str):
                    meta_group.create_dataset(key, data=value, dtype=h5py.string_dtype())
                else:
                    meta_group.create_dataset(key, data=value)
            
            # Save episodes
            ep_group = f.create_group("episodes")
            for ep_idx, episode in enumerate(data.get("episodes", [])):
                ep_subgroup = ep_group.create_group(f"episode_{ep_idx:06d}")
                
                frames = episode.get("frames", [])
                if not frames:
                    continue
                
                # Organize by keys
                frame_keys = frames[0].keys()
                for key in frame_keys:
                    values = [frame.get(key) for frame in frames]
                    if all(v is not None for v in values):
                        try:
                            ep_subgroup.create_dataset(key, data=np.array(values))
                        except (TypeError, ValueError) as e:
                            # Log specific conversion errors for debugging
                            print(f"Warning: Could not save key '{key}' in episode {ep_idx}: {e}")
    
    def _save_zarr(self, data: Dict[str, Any], path: Path, **kwargs) -> None:
        """Save to Zarr format."""
        import zarr
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        root = zarr.open(str(path), mode="w")
        
        # Save metadata
        meta_group = root.create_group("metadata")
        for key, value in data.get("metadata", {}).items():
            meta_group.create_dataset(key, data=np.array(value))
        
        # Save episodes
        ep_group = root.create_group("episodes")
        for ep_idx, episode in enumerate(data.get("episodes", [])):
            ep_subgroup = ep_group.create_group(f"episode_{ep_idx:06d}")
            
            frames = episode.get("frames", [])
            if not frames:
                continue
            
            # Organize by keys
            frame_keys = frames[0].keys()
            for key in frame_keys:
                values = [frame.get(key) for frame in frames]
                if all(v is not None for v in values):
                    try:
                        ep_subgroup.create_dataset(key, data=np.array(values))
                    except (TypeError, ValueError) as e:
                        # Log specific conversion errors for debugging
                        print(f"Warning: Could not save key '{key}' in episode {ep_idx}: {e}")


def convert_to_lerobot(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    source_format: str = "hdf5",
    **kwargs,
) -> None:
    """
    Convenience function to convert to LeRobot format.
    
    Args:
        source_path: Path to source dataset
        target_path: Path for LeRobot dataset
        source_format: Source format ("hdf5" or "zarr")
        **kwargs: Additional options
    """
    format_mapping = {
        "hdf5": DatasetFormat.HDF5,
        "zarr": DatasetFormat.ZARR,
        "lerobot": DatasetFormat.LEROBOT,
    }
    
    source_fmt = format_mapping.get(source_format.lower())
    if source_fmt is None:
        raise ValueError(f"Unknown source format: {source_format}")
    
    converter = DatasetConverter(source_fmt, DatasetFormat.LEROBOT)
    converter.convert(source_path, target_path, **kwargs)


def convert_from_lerobot(
    source_path: Union[str, Path],
    target_path: Union[str, Path],
    target_format: str = "hdf5",
    **kwargs,
) -> None:
    """
    Convenience function to convert from LeRobot format.
    
    Args:
        source_path: Path to LeRobot dataset
        target_path: Path for target dataset
        target_format: Target format ("hdf5" or "zarr")
        **kwargs: Additional options
    """
    format_mapping = {
        "hdf5": DatasetFormat.HDF5,
        "zarr": DatasetFormat.ZARR,
        "lerobot": DatasetFormat.LEROBOT,
    }
    
    target_fmt = format_mapping.get(target_format.lower())
    if target_fmt is None:
        raise ValueError(f"Unknown target format: {target_format}")
    
    converter = DatasetConverter(DatasetFormat.LEROBOT, target_fmt)
    converter.convert(source_path, target_path, **kwargs)
