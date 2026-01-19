#!/usr/bin/env python3
"""
URDF Scaling Utility

This script loads a URDF file, scales all geometric elements by a given factor,
and saves the scaled URDF to a new file.
"""

import xml.etree.ElementTree as ET
import argparse
import os
from typing import List, Tuple


def scale_vector(vector_str: str, scale_factor: float) -> str:
    """Scale a space-separated vector string by the given factor."""
    if not vector_str:
        return vector_str

    values = vector_str.split()
    scaled_values = [str(float(val) * scale_factor) for val in values]
    return " ".join(scaled_values)


def scale_geometry_elements(element: ET.Element, scaling_factor: float) -> None:
    """Scale geometry elements within the provided XML element."""
    geometry_elements = [
        "box",  # size attribute
        "cylinder",  # radius and length attributes
        "sphere",  # radius attribute
        "mesh",  # scale attribute (if present)
    ]

    for elem_name in geometry_elements:
        elements = element.findall(f".//{elem_name}")
        for geom_elem in elements:
            if elem_name == "box" and "size" in geom_elem.attrib:
                geom_elem.attrib["size"] = scale_vector(geom_elem.attrib["size"], scaling_factor)

            elif elem_name == "cylinder":
                if "radius" in geom_elem.attrib:
                    geom_elem.attrib["radius"] = str(float(geom_elem.attrib["radius"]) * scaling_factor)
                if "length" in geom_elem.attrib:
                    geom_elem.attrib["length"] = str(float(geom_elem.attrib["length"]) * scaling_factor)

            elif elem_name == "sphere" and "radius" in geom_elem.attrib:
                geom_elem.attrib["radius"] = str(float(geom_elem.attrib["radius"]) * scaling_factor)

            elif elem_name == "mesh":
                if "scale" in geom_elem.attrib:
                    current_scale = geom_elem.attrib["scale"]
                    geom_elem.attrib["scale"] = scale_vector(current_scale, scaling_factor)
                else:
                    scale_str = f"{scaling_factor} {scaling_factor} {scaling_factor}"
                    geom_elem.attrib["scale"] = scale_str


def scale_origin_elements(element: ET.Element, scaling_factor: float) -> None:
    """Scale origin xyz attributes within the provided XML element."""
    origin_elements = element.findall(".//origin")
    for origin in origin_elements:
        if "xyz" in origin.attrib:
            origin.attrib["xyz"] = scale_vector(origin.attrib["xyz"], scaling_factor)


def scale_inertial_elements(element: ET.Element, scaling_factor: float) -> None:
    """Scale inertial mass and inertia tensor elements within the provided XML element."""
    mass_scale_factor = scaling_factor**3
    inertia_scale_factor = scaling_factor**5

    mass_elements = element.findall(".//mass")
    for mass_elem in mass_elements:
        if "value" in mass_elem.attrib:
            current_mass = float(mass_elem.attrib["value"])
            mass_elem.attrib["value"] = str(current_mass * mass_scale_factor)

    inertia_elements = element.findall(".//inertia")
    for inertia in inertia_elements:
        inertia_attrs = ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]
        for attr in inertia_attrs:
            if attr in inertia.attrib:
                current_value = float(inertia.attrib[attr])
                inertia.attrib[attr] = str(current_value * inertia_scale_factor)


def scale_urdf(
    input_path: str, output_path: str, scaling_factor: float, mode: str = "urdf", object_link_prefix: str = "link_"
) -> None:
    """
    Scale a URDF file by the given scaling factor.

    Args:
        input_path: Path to the input URDF file
        output_path: Path where the scaled URDF will be saved
        scaling_factor: Factor by which to scale the URDF
    """

    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input URDF file not found: {input_path}")

    # Parse the URDF XML
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse URDF file: {e}")

    if mode == "urdf":
        scale_geometry_elements(root, scaling_factor)
        scale_origin_elements(root, scaling_factor)
        scale_inertial_elements(root, scaling_factor)
    elif mode == "akr_urdf":
        object_links = set()
        for link in root.findall(".//link"):
            link_name = link.attrib.get("name", "")
            if link_name.startswith(object_link_prefix):
                object_links.add(link_name)
                scale_geometry_elements(link, scaling_factor)
                scale_origin_elements(link, scaling_factor)
                scale_inertial_elements(link, scaling_factor)

        for joint in root.findall(".//joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_link = parent.attrib.get("link")
            child_link = child.attrib.get("link")
            if parent_link in object_links or child_link in object_links:
                origin = joint.find("origin")
                if origin is not None and "xyz" in origin.attrib:
                    origin.attrib["xyz"] = scale_vector(origin.attrib["xyz"], scaling_factor)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the scaled URDF
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Scaled URDF saved to: {output_path}")
    print(f"Scaling factor applied: {scaling_factor}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scale a URDF file by a given factor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python scale_urdf.py robot.urdf scaled_robot.urdf 0.5
            python scale_urdf.py input/robot.urdf output/small_robot.urdf 0.1
            python scale_urdf.py robot.urdf big_robot.urdf 2.0
        """,
    )

    parser.add_argument("input_path", help="Path to the input URDF file")
    parser.add_argument("output_path", help="Path for the output scaled URDF file")
    parser.add_argument(
        "scaling_factor", type=float, help="Scaling factor (e.g., 0.5 for half size, 2.0 for double size)"
    )
    parser.add_argument(
        "--mode",
        choices=["urdf", "akr_urdf"],
        default="urdf",
        help="Scaling mode: urdf scales all; akr_urdf scales only object links (default: urdf)",
    )
    parser.add_argument(
        "--object-link-prefix", default="link_", help="Object link name prefix used in akr_urdf mode (default: link_)"
    )

    args = parser.parse_args()

    # Validate scaling factor
    if args.scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive")

    try:
        scale_urdf(
            args.input_path,
            args.output_path,
            args.scaling_factor,
            mode=args.mode,
            object_link_prefix=args.object_link_prefix,
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

# python scripts/assets/scale_urdf.py assets/object/Microwave/7221/mobility.urdf assets/object/Microwave/7221/mobility_05.urdf 0.5
'''
python scripts/assets/scale_urdf.py \
    assets/object/Microwave/7221/7221_0_scaling.urdf \
    assets/object/Microwave/7221/7221_0_scaling_05.urdf \
    0.5 \
    --mode urdf
    
python scripts/assets/scale_urdf.py \
    assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.urdf \
    assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000_05.urdf \
    0.5 \
    --mode akr_urdf 
'''