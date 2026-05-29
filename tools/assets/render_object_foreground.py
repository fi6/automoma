#!/usr/bin/env python3
"""Render a single articulated object foreground image with transparent background.

Examples:
    python tools/assets/render_object_foreground.py --category Microwave --object-id 7221
    python tools/assets/render_object_foreground.py --category Oven --object-id 101773 --output outputs/renders/oven.png
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import bpy  # type: ignore
    from mathutils import Matrix, Vector  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - executed by normal Python wrapper.
    bpy = None  # type: ignore
    Matrix = None  # type: ignore
    Vector = None  # type: ignore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True, help="Object category directory, e.g. Microwave, Dishwasher, Oven, TrashCan.")
    parser.add_argument("--object-id", "--id", dest="object_id", required=True, help="Object id under assets/object/<category>.")
    parser.add_argument("--output", type=Path, help="Output PNG path. Defaults to outputs/renders/<category>_<id>_urdf_1024.png.")
    parser.add_argument("--resolution", type=int, default=1024, help="Square image resolution in pixels.")
    parser.add_argument("--urdf", type=Path, help="URDF path. Defaults to assets/object/<category>/<id>/mobility.urdf.")
    parser.add_argument("--joint-value", type=float, default=0.0, help="Value applied to all non-fixed joints, in radians/meters.")
    parser.add_argument("--heading-deg", type=float, default=-35.0, help="Yaw applied before rendering for a three-quarter view.")
    parser.add_argument(
        "--view-dir",
        nargs=3,
        type=float,
        default=(-0.35, -6.0, 2.4),
        metavar=("X", "Y", "Z"),
        help="Camera direction vector from object center. Default shows the front/right side for common PartNet assets.",
    )
    parser.add_argument("--ortho-margin", type=float, default=1.45, help="Orthographic framing margin multiplier.")
    parser.add_argument(
        "--apply-fixed-root",
        action="store_true",
        help="Apply the fixed joint from base/world to the root visual link. Default skips it for cleaner PartNet foreground renders.",
    )
    parser.add_argument("--blender", type=Path, help="Blender executable for wrapper mode. Defaults to $BLENDER, PATH, or ~/blender/blender.")
    return parser.parse_args(argv)


def run_in_blender(args: argparse.Namespace) -> None:
    blender = args.blender or os.environ.get("BLENDER") or shutil.which("blender")
    if not blender:
        local = Path.home() / "blender" / "blender"
        blender = str(local) if local.exists() else None
    if not blender:
        raise SystemExit("Blender not found. Set --blender or BLENDER=/path/to/blender.")

    forwarded = []
    skip_next = False
    for item in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if item == "--blender":
            skip_next = True
            continue
        if item.startswith("--blender="):
            continue
        forwarded.append(item)
    cmd = [str(blender), "--background", "--python", str(Path(__file__).resolve()), "--", *forwarded]
    subprocess.run(cmd, check=True)


def xyz_attr(node: ET.Element | None, name: str, default: tuple[float, float, float] = (0, 0, 0)):
    if node is None:
        return Vector(default)
    raw = node.attrib.get(name)
    if not raw:
        return Vector(default)
    return Vector(tuple(float(x) for x in raw.split()))


def transform_from_origin(origin: ET.Element | None):
    xyz = xyz_attr(origin, "xyz")
    rpy = xyz_attr(origin, "rpy")
    return (
        Matrix.Translation(xyz)
        @ Matrix.Rotation(rpy[2], 4, "Z")
        @ Matrix.Rotation(rpy[1], 4, "Y")
        @ Matrix.Rotation(rpy[0], 4, "X")
    )


def axis_angle(axis, q: float):
    vec = Vector(axis)
    if vec.length < 1e-9:
        return Matrix.Identity(4)
    return Matrix.Rotation(q, 4, vec.normalized())


def parse_urdf_visuals(urdf_path: Path, joint_value: float, apply_fixed_root: bool):
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    visuals = []
    for link in robot.findall("link"):
        link_name = link.attrib["name"]
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            visuals.append((link_name, mesh.attrib["filename"], transform_from_origin(visual.find("origin"))))

    children: dict[str, list[tuple[str, object]]] = {}
    for joint in robot.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        joint_type = joint.attrib.get("type", "fixed")
        origin_tf = transform_from_origin(joint.find("origin"))
        axis = xyz_attr(joint.find("axis"), "xyz", (1, 0, 0))

        if joint_type in {"revolute", "continuous"}:
            motion_tf = axis_angle(axis, joint_value)
        elif joint_type == "prismatic":
            motion_tf = Matrix.Translation(axis.normalized() * joint_value)
        else:
            motion_tf = Matrix.Identity(4)

        edge_tf = origin_tf @ motion_tf
        if joint_type == "fixed" and not apply_fixed_root and parent in {"base", "world"}:
            edge_tf = Matrix.Identity(4)
        children.setdefault(parent, []).append((child, edge_tf))

    root_names = [link.attrib["name"] for link in robot.findall("link") if link.attrib.get("name") in {"base", "world"}]
    root_name = root_names[0] if root_names else robot.find("link").attrib["name"]
    link_tf = {root_name: Matrix.Identity(4)}
    stack = [root_name]
    while stack:
        parent = stack.pop()
        for child, tf in children.get(parent, []):
            link_tf[child] = link_tf[parent] @ tf
            stack.append(child)

    return [(link, filename, link_tf.get(link, Matrix.Identity(4)) @ visual_tf) for link, filename, visual_tf in visuals]


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_obj(path: Path):
    before = set(bpy.data.objects)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path), use_split_objects=False, use_split_groups=False)
    else:
        bpy.ops.import_scene.obj(filepath=str(path), use_split_objects=False, use_split_groups=False)
    return [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]


def object_bounds(objects):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        eval_obj = obj.evaluated_get(depsgraph)
        points.extend(eval_obj.matrix_world @ Vector(corner) for corner in eval_obj.bound_box)
    if not points:
        raise RuntimeError("No mesh objects were imported from the URDF visuals.")
    return (
        Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points))),
        Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points))),
    )


def resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path
    if filename.startswith("assets/"):
        return REPO_ROOT / path
    return urdf_path.parent / path


def configure_scene(resolution: int) -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3
        scene.eevee.gtao_factor = 1.6


def add_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(-2.8, -4.8, 4.8), rotation=(math.radians(60), 0, math.radians(-20)))
    key = bpy.context.object
    key.data.energy = 750
    key.data.size = 5.0
    bpy.ops.object.light_add(type="AREA", location=(3.0, -2.0, 3.0), rotation=(math.radians(60), 0, math.radians(40)))
    fill = bpy.context.object
    fill.data.energy = 120
    fill.data.size = 4.0


def render_object(args: argparse.Namespace) -> Path:
    category = args.category.strip("/")
    object_id = str(args.object_id)
    asset_dir = REPO_ROOT / "assets" / "object" / category / object_id
    urdf_path = args.urdf or asset_dir / "mobility.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output = args.output or REPO_ROOT / "outputs" / "renders" / f"{category.lower()}_{object_id}_urdf_{args.resolution}.png"
    if not output.is_absolute():
        output = REPO_ROOT / output

    clear_scene()
    configure_scene(args.resolution)

    meshes = []
    for _link, filename, matrix_world in parse_urdf_visuals(urdf_path, args.joint_value, args.apply_fixed_root):
        mesh_path = resolve_mesh_path(urdf_path, filename)
        imported = import_obj(mesh_path)
        for obj in imported:
            obj.matrix_world = matrix_world @ obj.matrix_world
            meshes.append(obj)

    heading_tf = Matrix.Rotation(math.radians(args.heading_deg), 4, "Z")
    for obj in meshes:
        obj.matrix_world = heading_tf @ obj.matrix_world
    bpy.context.view_layer.update()

    min_corner, max_corner = object_bounds(meshes)
    center = (min_corner + max_corner) / 2
    move_tf = Matrix.Translation(Vector((-center.x, -center.y, -min_corner.z)))
    for obj in meshes:
        obj.matrix_world = move_tf @ obj.matrix_world
    bpy.context.view_layer.update()

    min_corner, max_corner = object_bounds(meshes)
    center = (min_corner + max_corner) / 2
    add_lighting()

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    view_dir = Vector(tuple(args.view_dir))
    camera.location = center + view_dir
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    span_x = max_corner.x - min_corner.x
    span_y = max_corner.y - min_corner.y
    span_z = max_corner.z - min_corner.z
    camera.data.ortho_scale = max(span_x, span_y * 0.6, span_z) * args.ortho_margin
    camera.data.clip_end = 1000
    bpy.context.scene.camera = camera

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)
    print(f"WROTE {output}")
    return output


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    args = parse_args(argv)
    if bpy is None:
        run_in_blender(args)
        return
    render_object(args)


if __name__ == "__main__":
    main()
