"""
simple_gripper.py
=================
"""

import numpy as np
from spatialmath import SE3
import spatialgeometry as sg


class SimpleGripper:
    PALM_SIZE = (0.06, 0.15, 0.05)      # x, y, z of the palm block
    FINGER_SIZE = (0.03, 0.012, 0.09)   # x, y, z of one finger
    MAX_OPENING = 0.12                  # gap between fingers, fully open

    # Flange to the point between the fingertips where an object is held
    TCP_OFFSET = PALM_SIZE[2] + FINGER_SIZE[2] - 0.02

    def __init__(self):
        self.opening = self.MAX_OPENING
        self._T_flange = SE3()
        self._palm = None
        self._fingers = []

    def add_to_env(self, env, T_flange):
        grey = (0.30, 0.32, 0.35, 1.0)
        dark = (0.15, 0.15, 0.17, 1.0)
        self._palm = sg.Cuboid(scale=list(self.PALM_SIZE), color=grey)
        env.add(self._palm)
        for _ in range(2):
            f = sg.Cuboid(scale=list(self.FINGER_SIZE), color=dark)
            env.add(f)
            self._fingers.append(f)
        self.update(T_flange)
        return self

    def tcp(self, T_flange):
        """Tool centre point: where the held object's grasp point sits."""
        return SE3(T_flange) * SE3(0, 0, self.TCP_OFFSET)

    def set_opening(self, gap):
        self.opening = float(np.clip(gap, 0.0, self.MAX_OPENING))
        self.update()

    def update(self, T_flange=None):
        """Re-pose the palm and fingers on the flange."""
        if T_flange is not None:
            self._T_flange = SE3(T_flange)
        if self._palm is None:
            return
        T = self._T_flange
        pz = self.PALM_SIZE[2]
        fz = self.FINGER_SIZE[2]
        fy = self.FINGER_SIZE[1]

        self._palm.T = T * SE3(0, 0, pz / 2)
        # Fingers slide apart along the gripper's y axis
        half = self.opening / 2 + fy / 2
        for side, f in zip((-1, 1), self._fingers):
            f.T = T * SE3(0, side * half, pz + fz / 2)
