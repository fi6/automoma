from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, List

import networkx as nx
import numpy as np
import trimesh.transformations as tra

from numpy.linalg import inv
from yourdfpy.urdf import (
    URDF,
    Collision,
    Color,
    Joint,
    Link,
    Material,
    Robot,
    Visual,
)


class BaseObject(Robot):
    link_map: Dict[str, Link]
    joint_map: Dict[str, Joint]
    material_map: Dict[str, Material]

    def __init__(
        self, name, link_map={}, joint_map={}, material_map={}, attach_location=None
    ):
        Robot.__init__(self, name)
        self.logger = logging.getLogger(__name__)
        self.link_map = link_map
        self.joint_map = joint_map
        self.material_map = material_map
        self._attach_location = attach_location

        self.add_materials(
            Material(self.name + "_white", color=Color(np.array([1.0, 1.0, 1.0, 1.0])))
        )

    @staticmethod
    def load(fname, **kwargs):
        """
        do not use filename_handler here, it is used when actual mesh is loaded, and will not modify the link.
        """
        load_kwargs = deepcopy(kwargs)
        load_kwargs.pop("prefix", None)
        urdf = URDF.load(fname, **load_kwargs)
        return BaseObject.from_robot(urdf.robot, **kwargs)

    @staticmethod
    def from_robot(robot: Robot, create_world_joint=False, **kwargs):
        obj = BaseObject(robot.name)
        if "prefix" in kwargs:
            prefix = kwargs["prefix"]
            for link in robot.links:
                link.name = prefix + link.name
            for joint in robot.joints:
                joint.parent = prefix + joint.parent
                joint.child = prefix + joint.child
                joint.name = prefix + joint.name
        obj.links = robot.links
        obj.joints = robot.joints
        obj.materials = robot.materials
        obj._build_graph()
        if create_world_joint:
            obj.create_world_joint()
        return obj

    @property
    def world_joint(self):
        world_joints = []
        for j in self.joints:
            if j.parent == "world":
                world_joints.append(j)
        if len(world_joints) == 0:
            raise ValueError("No world joint found")
        if len(world_joints) == 1:
            return world_joints[0]
        else:
            print("Multiple world joints found")
            return world_joints[0]

    # joint
    @property
    def joints(self):
        return self.joint_map.values()

    @joints.setter
    def joints(self, value):
        self.joint_map = {}
        for j in value:
            self.joint_map[j.name] = j

    def add_joints(self, *joints: Joint):
        """
        will replace joint if existed
        """
        for j in joints:
            if j.name in self.joint_map.keys():
                self.logger.warning("joint %s already existed, replacing", j.name)
            self.joint_map[j.name] = j

    # link
    @property
    def links(self):
        return self.link_map.values()

    @links.setter
    def links(self, value):
        self.link_map = {}
        for l in value:
            self.link_map[l.name] = l

    def add_links(self, *links: Link):
        """
        will replace link if existed
        """
        for l in links:
            if self.link_map.get(l.name):
                self.logger.warning("link %s already existed, replacing", l.name)
            self.link_map[l.name] = l

    # material
    @property
    def materials(self):
        return self.material_map.values()

    @materials.setter
    def materials(self, value):
        self.material_map = {}
        for m in value:
            self.material_map[m.name] = m

    def add_materials(self, *materials: Material):
        """
        will replace material if existed
        """
        for m in materials:
            if not m.name:
                raise ValueError("Material name is empty")
            if self.material_map.get(m.name):
                self.logger.warning("material %s already existed, replacing", m.name)
            self.material_map[m.name] = m

    def add_object_child(
        self,
        obj: BaseObject,
        parent_link: str,
        child_link: str,
        xyz: List[float],
        rpy: List[float],
    ):
        """
        add object as a child of this object
        xyz and rpy are origin
        """
        self._merge_robot(obj)
        transformation = self._make_transformation(xyz, rpy)
        joint = Joint(
            self.name + "_" + obj.name + "_attach_joint",
            type="fixed",
            parent=parent_link,
            child=child_link,
            origin=transformation,
        )
        parent_joints = []
        for j in self.joints:
            if j.child == joint.child:
                self.logger.warning(
                    "parent of link %s already existed when adding object as child: %s\t replacing joint.",
                    j.child,
                    j.name,
                )
                parent_joints.append(j.name)
        for j in parent_joints:
            self.joint_map.pop(j)

        self.add_joints(joint)

    # other
    def create_world_joint(
        self,
        xyz=[0, 0, 0],
        yaw=0.0,
        rpy: List[float] | None = None,
        create_world_link=True
    ):
        if self.joint_map.get(self.name + "_world_joint"):
            self.logger.warning("world joint already existed, replacing")
        if rpy is None:
            rpy = [0, 0, yaw]
        world_joint = Joint(
            self.name + "_world_joint",
            type="fixed",
            parent="world",
            child="corpus",
            origin=self._make_transformation(xyz, rpy),
        )
        self.add_joints(world_joint)
        if create_world_link and not self.link_map.get("world"):
            world_link = Link("world")
            self.add_links(world_link)
        if not self.link_map.get("corpus"):
            corpus_link = Link("corpus")
            self.add_links(corpus_link)

    def inverse_root_tip(self, root_link: str, tip_link: str):
        """
        inverse root and tip link
        """
        if not self.world_joint:
            raise ValueError("No world joint found")
        self._build_graph()
        path = self._get_path(root_link, tip_link)

        tip_root_transform = tra.compose_matrix()
        prev_old_joint_tf = tra.compose_matrix()
        prev_old_joint_tf_inv = tra.compose_matrix()
        for joint in path[::-1]:
            current_joint_origin = deepcopy(joint.origin)
            if joint.parent == "world":  # if world joint
                self.logger.debug("inverting world joint: %s", joint.name)
                tip_root_transform = self.world_joint.origin @ tip_root_transform
                childs = self.G.out_edges(joint.child)
                if not len(childs):
                    continue
                # reverse branch
                for child in childs:
                    joint_name = self.find_joint(
                        child[0], child[1]
                    ).name  # get joint name from edge property
                    if joint_name not in map(lambda j: j.name, path):
                        self.joint_map[joint_name].origin = (
                            inv(prev_old_joint_tf) @ self.joint_map[joint_name].origin
                        )
                continue
            self.logger.debug("inverting: %s", joint.name)
            joint.parent, joint.child = joint.child, joint.parent
            tip_root_transform = joint.origin @ tip_root_transform

            if joint.parent == tip_link:
                joint.origin = tra.compose_matrix()
            else:
                joint.origin = prev_old_joint_tf_inv

            # reverse branch
            childs = self.G.out_edges(joint.parent)
            if len(childs):
                for child in childs:
                    joint_name = self.find_joint(child[0], child[1]).name
                    if joint_name not in map(lambda j: j.name, path):
                        self.joint_map[joint_name].origin = inv(prev_old_joint_tf) @ (
                            self.joint_map[joint_name].origin
                            if self.joint_map[joint_name].origin is not None
                            else tra.compose_matrix()
                        )

            prev_old_joint_tf = current_joint_origin
            prev_old_joint_tf_inv = inv(prev_old_joint_tf)

            # get the link to be modified
            link = self.link_map[joint.child]

            if link.inertial:
                link.inertial.origin = prev_old_joint_tf_inv @ (
                    link.inertial.origin
                    if link.inertial.origin is not None
                    else tra.compose_matrix()
                )
            if len(link.visuals):
                for visual in link.visuals:
                    visual.origin = prev_old_joint_tf_inv @ (
                        visual.origin
                        if visual.origin is not None
                        else tra.compose_matrix()
                    )
            if len(link.collisions):
                for collision in link.collisions:
                    collision.origin = prev_old_joint_tf_inv @ (
                        collision.origin
                        if collision.origin is not None
                        else tra.compose_matrix()
                    )
            if joint.type in ["revolute", "prismatic"]:
                if joint.limit:
                    joint.limit.lower, joint.limit.upper = (
                        -joint.limit.upper,
                        -joint.limit.lower,
                    )  # nolint
        self.world_joint.parent = "world"
        self.world_joint.child = tip_link
        self.world_joint.type = "fixed"
        self.world_joint.origin = tip_root_transform
        self._build_graph()

    def _get_path(self, root_link: str, tip_link: str) -> List[Joint]:
        """
        use bfs to find joint path from root link to tip link
        """
        queue = [(root_link, [])]
        while queue:
            link, path = queue.pop(0)
            if link == tip_link:
                return path
            for joint in self.joints:
                if joint.parent == link:
                    queue.append((joint.child, path + [joint]))
        raise ValueError("No path found")

    def _make_inertia(self, ixx, iyy, izz):
        inertia = np.array(
            [
                [ixx, 0.0, 0.0],
                [0.0, iyy, 0.0],
                [0.0, 0.0, izz],
            ],
            dtype=np.float64,
        )
        return inertia

    @staticmethod
    def _make_transformation(xyz=[0, 0, 0], rpy=[0, 0, 0]):
        return tra.compose_matrix(translate=np.array(xyz), angles=np.array(rpy))

    def _make_collision(self, visuals: List[Visual]):
        v = visuals[0]
        if not v.geometry:
            raise ValueError("Visual must have geometry")

        collision = Collision(
            v.name if v.name else self.name + "_collision",
            origin=v.origin,
            geometry=v.geometry,
        )
        return collision

    def _merge_robot(self, *others: Robot):
        for other in others:
            self.add_joints(*other.joints)
            self.add_links(*other.links)
            self.add_materials(*other.materials)
        self._build_graph()

    def _build_graph(self):
        self.G = nx.DiGraph()
        for j in self.joints:
            self.G.add_edge(j.parent, j.child, name=j.name)

    def find_joint(self, parent_link: str, child_link: str) -> Joint:
        return self.joint_map[self.G[parent_link][child_link]["name"]]
