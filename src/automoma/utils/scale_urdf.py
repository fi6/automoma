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
    return ' '.join(scaled_values)


def scale_urdf(input_path: str, output_path: str, scaling_factor: float) -> None:
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
    
    # Elements that need scaling
    geometry_elements = [
        'box',      # size attribute
        'cylinder', # radius and length attributes
        'sphere',   # radius attribute
        'mesh'      # scale attribute (if present)
    ]
    
    position_elements = [
        'origin'    # xyz attribute
    ]
    
    # Scale geometry elements
    for elem_name in geometry_elements:
        elements = root.findall(f".//{elem_name}")
        for element in elements:
            if elem_name == 'box' and 'size' in element.attrib:
                element.attrib['size'] = scale_vector(element.attrib['size'], scaling_factor)
            
            elif elem_name == 'cylinder':
                if 'radius' in element.attrib:
                    element.attrib['radius'] = str(float(element.attrib['radius']) * scaling_factor)
                if 'length' in element.attrib:
                    element.attrib['length'] = str(float(element.attrib['length']) * scaling_factor)
            
            elif elem_name == 'sphere' and 'radius' in element.attrib:
                element.attrib['radius'] = str(float(element.attrib['radius']) * scaling_factor)
            
            elif elem_name == 'mesh':
                if 'scale' in element.attrib:
                    # If mesh already has scale, multiply it
                    current_scale = element.attrib['scale']
                    element.attrib['scale'] = scale_vector(current_scale, scaling_factor)
                else:
                    # Add scale attribute
                    scale_str = f"{scaling_factor} {scaling_factor} {scaling_factor}"
                    element.attrib['scale'] = scale_str
    
    # Scale origin elements (positions)
    origin_elements = root.findall('.//origin')
    for origin in origin_elements:
        if 'xyz' in origin.attrib:
            origin.attrib['xyz'] = scale_vector(origin.attrib['xyz'], scaling_factor)
    
    # Scale inertial properties (mass scales with volume ~ scale^3)
    mass_scale_factor = scaling_factor ** 3
    inertia_scale_factor = scaling_factor ** 5  # inertia scales with mass * length^2
    
    # Scale mass
    mass_elements = root.findall('.//mass')
    for mass_elem in mass_elements:
        if 'value' in mass_elem.attrib:
            current_mass = float(mass_elem.attrib['value'])
            mass_elem.attrib['value'] = str(current_mass * mass_scale_factor)
    
    # Scale inertia tensors
    inertia_elements = root.findall('.//inertia')
    for inertia in inertia_elements:
        inertia_attrs = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
        for attr in inertia_attrs:
            if attr in inertia.attrib:
                current_value = float(inertia.attrib[attr])
                inertia.attrib[attr] = str(current_value * inertia_scale_factor)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the scaled URDF
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Scaled URDF saved to: {output_path}")
    print(f"Scaling factor applied: {scaling_factor}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description='Scale a URDF file by a given factor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scale_urdf.py robot.urdf scaled_robot.urdf 0.5
  python scale_urdf.py input/robot.urdf output/small_robot.urdf 0.1
  python scale_urdf.py robot.urdf big_robot.urdf 2.0
        """
    )
    
    parser.add_argument('input_path', help='Path to the input URDF file')
    parser.add_argument('output_path', help='Path for the output scaled URDF file')
    parser.add_argument('scaling_factor', type=float, help='Scaling factor (e.g., 0.5 for half size, 2.0 for double size)')
    
    args = parser.parse_args()
    
    # Validate scaling factor
    if args.scaling_factor <= 0:
        raise ValueError("Scaling factor must be positive")
    
    try:
        scale_urdf(args.input_path, args.output_path, args.scaling_factor)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
    
# python tests/assets/scale_urdf.py assets/object/Microwave/7221/mobility.urdf output/test/kitchen_7221/mobility.urdf 0.5