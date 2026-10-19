"""
Doosan M0609 - custom 6-DOF kinematic model.

Joint origins, orientations and limits are based on the
official Doosan M0609 URDF.

The robot is constructed using Robotics Toolbox ETS.
Visual geometry is generated separately using simple
shapes, rather than importing a prebuilt robot model.
"""

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base import rpy2r


# Joint origins: XYZ coordinates in metres
JOINT_XYZ = [
    (0.0, 0.0, 0.1345),
    (0.0, 0.0062, 0.0),
    (0.411, 0.0, 0.0),
    (0.0, -0.368, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, -0.121, 0.0),
]

# Joint orientations: RPY in radians
JOINT_RPY = [
    (0.0, 0.0, 0.0),
    (0.0, -1.571, -1.571),
    (0.0, 0.0, 1.571),
    (1.571, 0.0, 0.0),
    (-1.571, 0.0, 0.0),
    (1.571, 0.0, 0.0),
]

JOINT_LIMITS = np.array([
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
    [-2.618, 2.618],
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
    [-6.2832, 6.2832],
])


def create_doosan():
    """Create a six-revolute-joint Doosan M0609 ETS model."""

    chain = rtb.ETS()

    for i, (xyz, rpy) in enumerate(zip(JOINT_XYZ, JOINT_RPY)):
        # Fixed transform from the previous joint
        R = rpy2r(rpy, order="xyz")
        T = SE3.Rt(R, xyz)

        chain *= rtb.ETS([rtb.ET.SE3(T.A)])

        # Each revolute joint rotates about its local Z axis
        chain *= rtb.ETS([rtb.ET.Rz(jindex=i)])

    robot = rtb.ERobot(chain, name="Doosan M0609")

    for i, link in enumerate(robot.links):
        if link.isjoint:
            link.qlim = JOINT_LIMITS[i]

    return robot


if __name__ == "__main__":
    robot = create_doosan()

    print("Robot:", robot.name)
    print("Degrees of freedom:", robot.n)
    print("Home position:")
    print(robot.fkine(np.zeros(6)))
