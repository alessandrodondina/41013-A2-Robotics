"""MediAssist - Doosan M0609 with official CAD meshes."""

import time
from pathlib import Path

import numpy as np
import swift
from spatialmath import SE3
from spatialmath.base import rpy2r
from spatialgeometry import Mesh, Cuboid

from environment.ward import build_ward
from robot_models.doosan_m0609 import create_doosan, JOINT_XYZ, JOINT_RPY, swift_mesh_path

env = swift.Swift()
env.launch(realtime=True)
build_ward(env)

robot = create_doosan()

MESH_DIR = Path(__file__).resolve().parent / "meshes" / "m0609_white"

RAIL_Y = 0.75
BASE_Z = 0.96
CARRIAGE_Z = 0.90

carriage = Cuboid(
    scale=[0.38, 0.40, 0.12],
    pose=SE3(1.65, RAIL_Y, CARRIAGE_Z),
    color=[0.20, 0.25, 0.32, 1]
)
env.add(carriage)

MESH_FILES = [
    ["MF0609_0_0.dae"],
    ["MF0609_1_0.dae"],
    ["MF0609_2_0.dae", "MF0609_2_1.dae", "MF0609_2_2.dae"],
    ["MF0609_3_0.dae"],
    ["MF0609_4_0.dae", "MF0609_4_1.dae"],
    ["MF0609_5_0.dae"],
    ["MF0609_6_0.dae"],
]

mesh_groups = []

for group in MESH_FILES:
    pieces = []
    for filename in group:
        mesh = Mesh(
            filename=str(MESH_DIR / filename),
            scale=[0.001, 0.001, 0.001],
        )
        mesh.filename = swift_mesh_path(mesh.filename)
        env.add(mesh)
        pieces.append(mesh)
    mesh_groups.append(pieces)


def update_robot(rail_x, q):
    carriage.T = SE3(rail_x, RAIL_Y, CARRIAGE_Z)

    base = SE3(rail_x, RAIL_Y, BASE_Z)
    robot.base = base
    robot.q = q

    for mesh in mesh_groups[0]:
        mesh.T = base

    T = base

    for i in range(6):
        # URDF RPY: Rz(yaw) @ Ry(pitch) @ Rx(roll)
        T = T * SE3.Rt(
            rpy2r(JOINT_RPY[i], order="zyx"),
            JOINT_XYZ[i],
        )
        T = T * SE3.Rz(q[i])

        for mesh in mesh_groups[i + 1]:
            mesh.T = T


print("Doosan M0609 real-mesh animation running.")

try:
    start = time.monotonic()

    while True:
        t = time.monotonic() - start

        rail_x = 1.65 + 1.2 * np.sin(t * 0.25)

        q = np.array([
            0.15 * np.sin(t * 0.5),
            -0.6 + 0.15 * np.sin(t * 0.4),
            0.9 + 0.15 * np.sin(t * 0.3),
            0.0,
            0.25,
            0.0,
        ])

        update_robot(rail_x, q)
        env.step(0.05)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Simulation stopped.")

finally:
    env.close()