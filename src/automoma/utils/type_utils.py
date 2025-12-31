"""
Type conversion utilities.

This module provides common type conversion functions to reduce code duplication
throughout the codebase. All functions handle multiple input types gracefully.
"""

from typing import Any, List, Optional, Union
import numpy as np
import torch


def to_list(data: Any) -> List[Any]:
    """
    Convert data to a Python list.
    
    Handles torch.Tensor, np.ndarray, and any iterable.
    
    Args:
        data: Input data to convert (tensor, array, list, tuple, etc.)
        
    Returns:
        Python list representation of the data
        
    Examples:
        >>> to_list(torch.tensor([1, 2, 3]))
        [1, 2, 3]
        >>> to_list(np.array([1.0, 2.0]))
        [1.0, 2.0]
        >>> to_list([1, 2, 3])
        [1, 2, 3]
    """
    if data is None:
        return []
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().tolist()
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (list, tuple)):
        return list(data)
    # Single value
    return [data]


def to_tensor(
    data: Any,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Convert data to a torch.Tensor.
    
    Args:
        data: Input data (list, np.ndarray, or existing tensor)
        dtype: Optional dtype for the tensor
        device: Optional device for the tensor
        
    Returns:
        torch.Tensor
        
    Examples:
        >>> to_tensor([1, 2, 3])
        tensor([1, 2, 3])
        >>> to_tensor(np.array([1.0, 2.0]), dtype=torch.float32)
        tensor([1.0, 2.0])
    """
    if isinstance(data, torch.Tensor):
        result = data
    elif isinstance(data, np.ndarray):
        result = torch.from_numpy(data)
    else:
        result = torch.tensor(data)
    
    if dtype is not None:
        result = result.to(dtype=dtype)
    if device is not None:
        result = result.to(device=device)
    
    return result


def to_numpy(data: Any) -> np.ndarray:
    """
    Convert data to a numpy array.
    
    Args:
        data: Input data (tensor, list, or existing array)
        
    Returns:
        numpy.ndarray
        
    Examples:
        >>> to_numpy(torch.tensor([1, 2, 3]))
        array([1, 2, 3])
    """
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)


def to_float(data: Any) -> float:
    """
    Convert data to a Python float.
    
    Handles tensors, arrays, and numeric types.
    
    Args:
        data: Input data
        
    Returns:
        Python float
        
    Raises:
        ValueError: If data is empty
        
    Examples:
        >>> to_float(torch.tensor(3.14))
        3.14
        >>> to_float(np.array(2.718))
        2.718
    """
    if isinstance(data, torch.Tensor):
        return data.item()
    if isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError("Cannot convert empty array to float")
        return float(data.flat[0])
    return float(data)


def ensure_non_negative(value: float) -> float:
    """
    Ensure a value is non-negative (>= 0).
    
    Args:
        value: Input value
        
    Returns:
        Absolute value of input (always >= 0)
    """
    return abs(float(value))


def ensure_tensor_shape(
    tensor: torch.Tensor,
    expected_ndim: int,
    expand_dim: int = 0
) -> torch.Tensor:
    """
    Ensure tensor has expected number of dimensions.
    
    Expands dimensions if necessary.
    
    Args:
        tensor: Input tensor
        expected_ndim: Expected number of dimensions
        expand_dim: Dimension to expand if needed
        
    Returns:
        Tensor with correct number of dimensions
        
    Examples:
        >>> t = torch.tensor([1, 2, 3])  # shape (3,)
        >>> ensure_tensor_shape(t, 2, expand_dim=0).shape
        torch.Size([1, 3])
    """
    while tensor.ndim < expected_ndim:
        tensor = tensor.unsqueeze(expand_dim)
    return tensor


def safe_bool_tensor(data: Any) -> torch.Tensor:
    """
    Convert data to a boolean tensor.
    
    Args:
        data: Input data
        
    Returns:
        Boolean tensor
    """
    tensor = to_tensor(data)
    return tensor.to(torch.bool)
