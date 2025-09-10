import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Literal, Dict, Any, Union
import yaml
import trimesh
import os, sys

# CuRobo imports for Sphere type
from curobo.geom.types import Sphere
from curobo.types.base import TensorDeviceType

# URDF processing imports
try:
    from yourdfpy import URDF
    URDF_AVAILABLE = True
except ImportError:
    print("Warning: yourdfpy not available. URDF processing functions will not work.")
    URDF_AVAILABLE = False


def _calculate_sphere_grid(
    box_dims: Tuple[float, float, float],
    sphere_radius: float,
    spacing_factor: float,
    sample_type: str
) -> Tuple[int, int, int, int]:
    """
    Calculate optimal sphere grid distribution based on spacing factor.
    
    Args:
        box_dims: Dimensions of the box (width, height, depth)
        sphere_radius: Radius of each sphere
        spacing_factor: Controls sphere spacing (2.0 = tight, 2.5 = moderate, 3.0 = loose)
        sample_type: "surface" or "volume"
    
    Returns:
        Tuple of (nx, ny, nz, estimated_total) - grid dimensions and estimated sphere count
    """
    w, h, d = box_dims
    
    # Calculate optimal spacing based on sphere radius and spacing factor
    sphere_spacing = sphere_radius * spacing_factor
    
    if sample_type == "volume":
        # For volume sampling: grid throughout the volume
        nx = max(1, int(w / sphere_spacing))
        ny = max(1, int(h / sphere_spacing))
        nz = max(1, int(d / sphere_spacing))
        
        estimated_total = nx * ny * nz
        
    else:  # surface sampling
        # For surface sampling: grid on the 6 faces
        nx = max(2, int(w / sphere_spacing) + 1)
        ny = max(2, int(h / sphere_spacing) + 1)
        nz = max(2, int(d / sphere_spacing) + 1)
        
        # Calculate surface sphere count with overlap handling
        face_xy = nx * ny * 2  # Front and back faces
        face_yz = (ny - 2) * nz * 2 if ny > 2 else 0  # Left and right (avoid double counting edges)
        face_xz = (nx - 2) * (nz - 2) * 2 if nx > 2 and nz > 2 else 0  # Top and bottom (avoid edges)
        
        estimated_total = face_xy + face_yz + face_xz
    
    return nx, ny, nz, estimated_total

def generate_box_spheres(
    box_center: Tuple[float, float, float] = (0, 0, 0),
    box_dims: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sphere_radius: float = 0.05,
    spacing_factor: float = 2.5,
    sample_type: Literal["surface", "volume"] = "surface"
) -> List[Tuple[float, float, float, float]]:
    """
    Generate spheres on/in a box using grid sampling with automatic optimal distribution.
    
    Args:
        box_center: Center position of the box (x, y, z)
        box_dims: Dimensions of the box (width, height, depth)
        sphere_radius: Radius of each sphere
        spacing_factor: Spacing between sphere centers (as multiple of radius)
        sample_type: "surface" for surface sampling, "volume" for volume sampling
    
    Returns:
        List of spheres as (x, y, z, radius) tuples
    """
    cx, cy, cz = box_center
    w, h, d = box_dims
    
    # Calculate optimal grid distribution
    nx, ny, nz, estimated_total = _calculate_sphere_grid(box_dims, sphere_radius, spacing_factor, sample_type)
    
    print(f"Grid distribution: nx={nx}, ny={ny}, nz={nz} (estimated: {estimated_total} spheres)")
    
    spheres = []
    
    if sample_type == "volume":
        # Generate grid points throughout the volume
        x_coords = np.linspace(cx - w/2, cx + w/2, nx)
        y_coords = np.linspace(cy - h/2, cy + h/2, ny)
        z_coords = np.linspace(cz - d/2, cz + d/2, nz)
        
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    spheres.append((x, y, z, sphere_radius))
    
    elif sample_type == "surface":
        # Generate points on the 6 faces of the box
        half_w, half_h, half_d = w/2, h/2, d/2
        
        # Front and back faces (z = ±half_d)
        x_coords = np.linspace(cx - half_w, cx + half_w, nx)
        y_coords = np.linspace(cy - half_h, cy + half_h, ny)
        for x in x_coords:
            for y in y_coords:
                spheres.append((x, y, cz + half_d, sphere_radius))  # Front face
                spheres.append((x, y, cz - half_d, sphere_radius))  # Back face
        
        # Left and right faces (x = ±half_w)
        y_coords = np.linspace(cy - half_h, cy + half_h, ny)
        z_coords = np.linspace(cz - half_d, cz + half_d, nz)
        for y in y_coords:
            for z in z_coords:
                if not (abs(z - (cz + half_d)) < 1e-6 or abs(z - (cz - half_d)) < 1e-6):  # Avoid corners
                    spheres.append((cx + half_w, y, z, sphere_radius))  # Right face
                    spheres.append((cx - half_w, y, z, sphere_radius))  # Left face
        
        # Top and bottom faces (y = ±half_h)
        x_coords = np.linspace(cx - half_w, cx + half_w, nx)
        z_coords = np.linspace(cz - half_d, cz + half_d, nz)
        for x in x_coords:
            for z in z_coords:
                if not (abs(z - (cz + half_d)) < 1e-6 or abs(z - (cz - half_d)) < 1e-6 or 
                       abs(x - (cx + half_w)) < 1e-6 or abs(x - (cx - half_w)) < 1e-6):  # Avoid edges
                    spheres.append((x, cy + half_h, z, sphere_radius))  # Top face
                    spheres.append((x, cy - half_h, z, sphere_radius))  # Bottom face
    
    return spheres


def convert_to_curobo_spheres(
    spheres: List[Tuple[float, float, float, float]],
    tensor_args: TensorDeviceType = None
) -> List[Sphere]:
    """
    Convert simple sphere tuples to CuRobo Sphere objects.
    
    Args:
        spheres: List of spheres as (x, y, z, radius) tuples
        tensor_args: Device and precision settings for CuRobo
    
    Returns:
        List of CuRobo Sphere objects
    """
    if tensor_args is None:
        tensor_args = TensorDeviceType()
    
    curobo_spheres = []
    
    for i, (x, y, z, radius) in enumerate(spheres):
        # Create pose: [x, y, z, qw, qx, qy, qz] (identity quaternion)
        pose = [x, y, z, 1.0, 0.0, 0.0, 0.0]
        
        # Create CuRobo Sphere object
        sphere = Sphere(
            name=f"box_sphere_{i}",
            pose=pose,
            radius=radius,
            tensor_args=tensor_args
        )
        
        curobo_spheres.append(sphere)
    
    print(f"Converted {len(spheres)} simple spheres to CuRobo Sphere objects")
    
    return curobo_spheres



def extract_bounding_box(
    urdf: URDF, 
    process_links: List[str], 
    root_link: str,
    type: str = "union"
) -> Union[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Extract bounding box(es) from URDF with coordinate transformation relative to root link.
    
    Args:
        urdf: URDF object
        process_links: List of link names to process
        root_link: Name of the root link to use as coordinate origin (0,0,0).
                  All poses will be relative to this link.
        type: "separate" for individual bounding boxes per link, "union" for combined bounding box
    
    Returns:
        If type="separate": List of tuples (center, dimensions) for each link
        If type="union": Single tuple (center, dimensions) for combined bounding box
    """
    if not URDF_AVAILABLE:
        raise ImportError("yourdfpy is required for URDF processing")
    
    if not process_links:
        raise ValueError("process_links must be provided and non-empty")
    
    # Get transformation matrix from root link to world
    try:
        # Get the transformation matrix from root link to world coordinate
        root_transform = urdf.get_transform(root_link, "world")
        # Invert to get world to root transform
        world_to_root = np.linalg.inv(root_transform)
    except:
        # If transformation fails, use identity (assume root is already at origin)
        world_to_root = np.eye(4)
        print(f"Warning: Could not get transformation for root link '{root_link}', using identity")
    
    if type == "separate":
        # Process each link separately and return list of bounding boxes
        results = []
        
        for link_name in process_links:
            try:
                # Get bounding box for the specific link in world coordinates
                link_center_world, link_dims = _extract_single_link_bbox(urdf, link_name)
                
                # Transform center to root link coordinates
                center_world_homo = np.array([*link_center_world, 1.0])
                center_root_homo = world_to_root @ center_world_homo
                center = center_root_homo[:3].tolist()
                dimensions = list(link_dims)
                
                results.append((tuple(center), tuple(dimensions)))
                
            except Exception as e:
                print(f"Warning: Could not process link '{link_name}': {e}")
                continue
        
        return results
    
    elif type == "union":
        # Process all links and create union bounding box
        all_bounds = []
        
        for link_name in process_links:
            try:
                # Get bounding box for this link in world coordinates
                link_center_world, link_dims = _extract_single_link_bbox(urdf, link_name)
                
                # Transform center to root link coordinates
                center_world_homo = np.array([*link_center_world, 1.0])
                center_root_homo = world_to_root @ center_world_homo
                center_root = center_root_homo[:3]
                
                # Calculate bounding box corners in root coordinates
                min_corner = center_root - np.array(link_dims) / 2
                max_corner = center_root + np.array(link_dims) / 2
                all_bounds.extend([min_corner, max_corner])
                
            except Exception as e:
                print(f"Warning: Could not process link '{link_name}': {e}")
                continue
        
        if all_bounds:
            all_bounds = np.array(all_bounds)
            min_bound = all_bounds.min(axis=0)
            max_bound = all_bounds.max(axis=0)
            center = ((min_bound + max_bound) / 2).tolist()
            dimensions = (max_bound - min_bound).tolist()
        else:
            # Ultimate fallback
            center = [0.0, 0.0, 0.0]
            dimensions = [1.0, 1.0, 1.0]
        
        return tuple(center), tuple(dimensions)
    
    else:
        raise ValueError(f"Invalid type '{type}'. Must be 'separate' or 'union'")


def _extract_single_link_bbox(urdf: URDF, link_name: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Extract bounding box of a single link in world coordinates.
    
    Args:
        urdf: URDF object
        link_name: Name of the link
    
    Returns:
        Tuple of (center, dimensions) of the bounding box in world coordinates
    """
    # Find the link
    link = None
    for l in urdf.robot.links:
        if l.name == link_name:
            link = l
            break
    
    if link is None:
        raise ValueError(f"Link '{link_name}' not found in URDF")
    
    # Try to get collision mesh first, then visual
    mesh_element = None
    if link.collisions:
        mesh_element = link.collisions[0]
    elif link.visuals:
        mesh_element = link.visuals[0]
    else:
        raise ValueError(f"No visual or collision geometry found for link '{link_name}'")
    
    # Get the transformation for this link
    try:
        link_transform = urdf.get_transform(link_name, "world")
    except:
        link_transform = np.eye(4)
    
    # Extract geometry and get local bounding box
    if hasattr(mesh_element.geometry, 'mesh') and mesh_element.geometry.mesh:
        mesh_path = mesh_element.geometry.mesh.filename
        try:
            mesh = trimesh.load(mesh_path)
            bounds = mesh.bounds
            local_center = (bounds[0] + bounds[1]) / 2
            dimensions = (bounds[1] - bounds[0]).tolist()
        except:
            # Fallback to primitive geometry
            local_center, dimensions = _extract_primitive_geometry(mesh_element)
    else:
        # Handle primitive shapes
        local_center, dimensions = _extract_primitive_geometry(mesh_element)
    
    # Transform local center to world coordinates
    local_center_homo = np.array([*local_center, 1.0])
    world_center_homo = link_transform @ local_center_homo
    world_center = world_center_homo[:3].tolist()
    
    return tuple(world_center), tuple(dimensions)


def _extract_primitive_geometry(mesh_element) -> Tuple[List[float], List[float]]:
    """
    Extract geometry from primitive shapes.
    
    Args:
        mesh_element: Geometry element from URDF
    
    Returns:
        Tuple of (local_center, dimensions)
    """
    if hasattr(mesh_element.geometry, 'box') and mesh_element.geometry.box:
        dimensions = list(mesh_element.geometry.box.size)
        local_center = [0.0, 0.0, 0.0]
    elif hasattr(mesh_element.geometry, 'cylinder') and mesh_element.geometry.cylinder:
        r = mesh_element.geometry.cylinder.radius
        h = mesh_element.geometry.cylinder.length
        dimensions = [r*2, r*2, h]
        local_center = [0.0, 0.0, 0.0]
    elif hasattr(mesh_element.geometry, 'sphere') and mesh_element.geometry.sphere:
        r = mesh_element.geometry.sphere.radius
        dimensions = [r*2, r*2, r*2]
        local_center = [0.0, 0.0, 0.0]
    else:
        # Default fallback
        local_center = [0.0, 0.0, 0.0]
        dimensions = [0.1, 0.1, 0.1]
        print(f"Warning: Could not process link '{mesh_element.name}': Invalid geometry type. Using default small box.")
    
    return local_center, dimensions


def generate_urdf_collision_spheres(
    urdf_path: str,
    handle_links: List[str],
    sphere_radius: Tuple[float, float] = (0.05, 0.05),
    spacing_factors: Tuple[float, float] = (2.5, 2.5),
    sample_type: Literal["surface", "volume"] = "surface"
) -> Dict[str, Any]:
    """
    Generate collision spheres for URDF object with coordinate transformation.
    
    Args:
        urdf_path: Path to URDF file
        handle_links: List of handle link names
        sphere_radius: Tuple of (handle_sphere_radius, other_sphere_radius)
        spacing_factors: Tuple of (handle_spacing_factor, other_spacing_factor)
        sample_type: "surface" or "volume" sampling
    
    Returns:
        Dictionary with collision spheres in the required format
    """
    if not URDF_AVAILABLE:
        raise ImportError("yourdfpy is required for URDF processing")
    
    # Load URDF
    urdf = URDF.load(urdf_path)
    
    # Get all link names
    all_link_names = [link.name for link in urdf.robot.links]
    
    # Separate handle links and other links
    other_links = [link for link in all_link_names if link not in handle_links]
    
    collision_spheres = {}
    
    # Generate spheres for handle links separately
    if handle_links:
        try:
            handle_bboxes = extract_bounding_box(urdf, handle_links, handle_links[0], type="separate")
            
            for i, (handle_center, handle_dims) in enumerate(handle_bboxes):
                handle_link_name = handle_links[i]
                
                handle_simple_spheres = generate_box_spheres(
                    box_center=handle_center,
                    box_dims=handle_dims,
                    sphere_radius=sphere_radius[0],
                    sample_type=sample_type,
                    spacing_factor=spacing_factors[0]
                )
                
                # Convert to required format
                collision_spheres[handle_link_name] = []
                for x, y, z, r in handle_simple_spheres:
                    sphere_data = {
                        "center": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
                        "radius": round(float(r), 3)
                    }
                    collision_spheres[handle_link_name].append(sphere_data)
                
                print(f"Generated {len(handle_simple_spheres)} spheres for handle link '{handle_link_name}'")
                
        except Exception as e:
            print(f"Warning: Could not process handle links: {e}")
            for handle_link_name in handle_links:
                collision_spheres[handle_link_name] = []
    
    # Generate spheres for other links as union
    if other_links:
        try:
            overall_center, overall_dims = extract_bounding_box(urdf, other_links, other_links[0], type="union")
            
            overall_simple_spheres = generate_box_spheres(
                box_center=overall_center,
                box_dims=overall_dims,
                sphere_radius=sphere_radius[1],
                sample_type=sample_type,
                spacing_factor=spacing_factors[1]
            )
            
            # Convert to required format - use the first other link as the key
            other_link_key = other_links[0] if other_links else "other_links"
            collision_spheres[other_link_key] = []
            for x, y, z, r in overall_simple_spheres:
                sphere_data = {
                    "center": [round(float(x), 3), round(float(y), 3), round(float(z), 3)],
                    "radius": round(float(r), 3)
                }
                collision_spheres[other_link_key].append(sphere_data)
            
            print(f"Generated {len(overall_simple_spheres)} spheres for other links union")
            
        except Exception as e:
            print(f"Warning: Could not process other links: {e}")
            if other_links:
                collision_spheres[other_links[0]] = []
    
    return {"collision_spheres": collision_spheres}


def save_collision_spheres_yaml(collision_data: Dict[str, Any], output_path: str):
    """
    Save collision spheres data to YAML file.
    
    Args:
        collision_data: Dictionary with collision spheres
        output_path: Path to save YAML file
    """
    with open(output_path, 'w') as f:
        yaml.dump(collision_data, f, default_flow_style=False, sort_keys=False)


def visualize_collision_spheres_from_yaml(yaml_file_path: str):
    """
    Visualize collision spheres from a YAML file with each link in a separate subplot.
    
    Args:
        yaml_file_path: Path to the YAML file containing collision spheres
    """
    # Load YAML file
    try:
        with open(yaml_file_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return
    
    # Extract collision spheres
    if 'collision_spheres' not in data:
        print("No 'collision_spheres' key found in YAML file")
        return
    
    collision_spheres = data['collision_spheres']
    link_names = list(collision_spheres.keys())
    
    if not link_names:
        print("No links found in collision_spheres")
        return
    
    # Calculate subplot grid (2 columns)
    columns = 2
    rows = (len(link_names) + columns - 1) // columns
    
    # Set up colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(link_names))))
    
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Collision Spheres Visualization - {yaml_file_path.split("/")[-1]}', fontsize=16)
    
    for idx, link_name in enumerate(link_names):
        spheres_data = collision_spheres[link_name]
        
        if not spheres_data:
            continue
        
        # Create subplot
        ax = fig.add_subplot(rows, columns, idx + 1, projection='3d')
        
        # Extract sphere data
        sphere_centers = []
        sphere_radii = []
        
        for sphere in spheres_data:
            center = sphere.get('center', [0, 0, 0])
            radius = sphere.get('radius', 0.05)
            sphere_centers.append(center)
            sphere_radii.append(radius)
        
        if not sphere_centers:
            ax.set_title(f'{link_name} (No spheres)')
            continue
        
        # Convert to numpy arrays for easier handling
        centers = np.array(sphere_centers)
        radii = np.array(sphere_radii)
        
        # Plot spheres
        color = colors[idx % len(colors)]
        for i, (center, radius) in enumerate(zip(centers, radii)):
            x, y, z = center
            
            # Create sphere surface
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            sphere_x = radius * np.outer(np.cos(u), np.sin(v)) + x
            sphere_y = radius * np.outer(np.sin(u), np.sin(v)) + y
            sphere_z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            
            ax.plot_surface(sphere_x, sphere_y, sphere_z, 
                          alpha=0.7, color=color)
        
        # Calculate bounding box for proper axis scaling
        min_coords = np.min(centers - radii[:, np.newaxis], axis=0)
        max_coords = np.max(centers + radii[:, np.newaxis], axis=0)
        
        # Add some padding
        padding = np.max(radii) * 0.1
        min_coords -= padding
        max_coords += padding
        
        # Calculate center and maximum range for equal scaling
        center_coords = (min_coords + max_coords) / 2
        ranges = max_coords - min_coords
        max_range = np.max(ranges) / 2
        
        # Set equal axis limits to ensure spheres appear as spheres
        ax.set_xlim(center_coords[0] - max_range, center_coords[0] + max_range)
        ax.set_ylim(center_coords[1] - max_range, center_coords[1] + max_range)
        ax.set_zlim(center_coords[2] - max_range, center_coords[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{link_name} ({len(spheres_data)} spheres)')
        
        # Make axes equal aspect ratio
        ax.set_box_aspect([1,1,1])
    
    # Remove empty subplots
    total_subplots = rows * columns
    for idx in range(len(link_names), total_subplots):
        ax = fig.add_subplot(rows, columns, idx + 1)
        ax.remove()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nVisualization Summary:")
    print(f"Total links: {len(link_names)}")
    for link_name in link_names:
        sphere_count = len(collision_spheres[link_name])
        print(f"  {link_name}: {sphere_count} spheres")


def visualize_box_spheres(
    box_center: Tuple[float, float, float],
    box_dims: Tuple[float, float, float],
    spheres: List[Tuple[float, float, float, float]],
    show_box: bool = True,
    alpha_spheres: float = 0.6,
    alpha_box: float = 0.3
):
    """
    Visualize the box and generated spheres in 3D.
    
    Args:
        box_center: Center of the box
        box_dims: Dimensions of the box
        spheres: List of spheres as (x, y, z, radius) tuples
        show_box: Whether to show the box wireframe
        alpha_spheres: Transparency of spheres
        alpha_box: Transparency of box
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw spheres
    for x, y, z, r in spheres:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = r * np.outer(np.cos(u), np.sin(v)) + x
        sphere_y = r * np.outer(np.sin(u), np.sin(v)) + y
        sphere_z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=alpha_spheres, color='red')
    
    # Draw box wireframe
    if show_box:
        cx, cy, cz = box_center
        w, h, d = box_dims
        
        # Define the 8 vertices of the box
        vertices = [
            [cx - w/2, cy - h/2, cz - d/2],
            [cx + w/2, cy - h/2, cz - d/2],
            [cx + w/2, cy + h/2, cz - d/2],
            [cx - w/2, cy + h/2, cz - d/2],
            [cx - w/2, cy - h/2, cz + d/2],
            [cx + w/2, cy - h/2, cz + d/2],
            [cx + w/2, cy + h/2, cz + d/2],
            [cx - w/2, cy + h/2, cz + d/2]
        ]
        
        # Define the 12 edges of the box
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        for edge in edges:
            points = np.array([vertices[edge[0]], vertices[edge[1]]])
            ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2, alpha=alpha_box)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Box to Spheres Visualization ({len(spheres)} spheres)')
    
    # Set equal aspect ratio
    max_range = max(box_dims) / 2
    cx, cy, cz = box_center
    ax.set_xlim([cx - max_range, cx + max_range])
    ax.set_ylim([cy - max_range, cy + max_range])
    ax.set_zlim([cz - max_range, cz + max_range])
    
    plt.show()


def main():
    """Example usage of the box to sphere functions."""
    
    EXAMPLE_1 = False
    EXAMPLE_2 = False
    EXAMPLE_3 = False
    EXAMPLE_4 = False
    EXAMPLE_5 = True
    EXAMPLE_6 = False
    
    box_center = (0, 0, 0)
    box_dims = (0.5, 0.3, 0.1)
    sphere_radius = 0.02
    spacing_factor = 2.5
    # Example 1: Surface sampling
    print("Example 1: Surface sampling")
    if EXAMPLE_1:
        spheres_surface = generate_box_spheres(
            box_center=box_center,
            box_dims=box_dims,
            sphere_radius=sphere_radius,
            spacing_factor=spacing_factor,
            sample_type="surface"
        )

        print(f"Generated {len(spheres_surface)} spheres on surface")
        visualize_box_spheres(box_center, box_dims, spheres_surface)
    
    # Example 2: Volume sampling
    print("\nExample 2: Volume sampling")
    if EXAMPLE_2:
        spheres_volume = generate_box_spheres(
            box_center=box_center,
            box_dims=box_dims,
            sphere_radius=sphere_radius,
            spacing_factor=spacing_factor,
            sample_type="volume"
        )
        
        print(f"Generated {len(spheres_volume)} spheres in volume")
        visualize_box_spheres(box_center, box_dims, spheres_volume)
    
    # Example 3: Convert to CuRobo Sphere objects
    print("\nExample 3: Convert to CuRobo Sphere objects")
    if EXAMPLE_3:
        # Generate spheres first for this example
        spheres_surface = generate_box_spheres(
            box_center=box_center,
            box_dims=box_dims,
            sphere_radius=sphere_radius,
            spacing_factor=spacing_factor,
            sample_type="surface"
        )
        
        curobo_spheres = convert_to_curobo_spheres(spheres_surface)

        # Print some information about the converted spheres
        print(f"First few CuRobo spheres:")
        for i, sphere in enumerate(curobo_spheres[:3]):  # Show first 3
            x, y, z = sphere.pose[:3]
            print(f"  {sphere.name}: center=({x:.3f}, {y:.3f}, {z:.3f}), radius={sphere.radius:.3f}")
        
        print(f"Total CuRobo spheres created: {len(curobo_spheres)}")
        print(f"Type of first sphere: {type(curobo_spheres[0])}")
    
    # Example 4: Generate spheres with different spacing factors
    print("\nExample 4: Sphere generation with different spacing factors")
    if EXAMPLE_4:
        spacing_factors = [2.0, 2.5, 3.0]
        for factor in spacing_factors:
            print(f"\nSpacing factor: {factor}")
            spheres = generate_box_spheres(
                box_center=box_center,
                box_dims=box_dims,
                sphere_radius=sphere_radius,
                spacing_factor=factor,
                sample_type="surface"
            )
            print(f"Generated {len(spheres)} spheres")
            visualize_box_spheres(box_center, box_dims, spheres)
        
    # Example 5: URDF processing (if available)
    if URDF_AVAILABLE and EXAMPLE_5:
        print("\nExample 5: URDF collision sphere generation")
        # Note: Replace with actual URDF path and link names
        urdf_path = "assets/robot/7221/7221_0_scaling.urdf"  # Example path
        yml_path = "output/robot/7221/7221_0_scaling_collision_spheres.yml"
        handle_links = ["link_0"]  # Example handle links (can be multiple)
        sphere_radius = (0.02, 0.08)
        spacing_factors = (2.5, 2.0)
        
        try:
            collision_data = generate_urdf_collision_spheres(
                urdf_path=urdf_path,
                handle_links=handle_links,
                sphere_radius=sphere_radius,
                spacing_factors=spacing_factors,
                sample_type="surface"
            )
            
            # Print summary
            print("Generated collision spheres:")
            for link_name, spheres in collision_data["collision_spheres"].items():
                print(f"  {link_name}: {len(spheres)} spheres")
            
            # Save to YAML file
            if yml_path:
                os.makedirs(os.path.dirname(yml_path), exist_ok=True)
            save_collision_spheres_yaml(collision_data, yml_path)
            print(f"Saved collision spheres to '{yml_path}'")

            # Visualize the collision spheres
            visualize_collision_spheres_from_yaml(yml_path)
            print("Visualized collision spheres")

        except Exception as e:
            print(f"URDF processing example failed: {e}")
    
    # Example 6: Visualize collision spheres from YAML file
    print("\nExample 6: Visualize collision spheres from YAML file")
    
    if EXAMPLE_6:
        """Test the collision sphere visualization function."""
    
        print("Testing collision sphere visualization...")
        
        # Test with the available YAML files
        yaml_files = [
            "output_collision_spheres.yml",
            "assets/robot/7221/7221_0_col.yml"
        ]
        
        for yaml_file in yaml_files:
            if os.path.exists(yaml_file):
                print(f"\n=== Visualizing: {yaml_file} ===")
                try:
                    visualize_collision_spheres_from_yaml(yaml_file)
                    print(f"Successfully visualized {yaml_file}")
                    
                except Exception as e:
                    print(f"Error visualizing {yaml_file}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"File not found: {yaml_file}")
        
        print("\nVisualization test complete!")


if __name__ == "__main__":
    main()
