"""
sensors.py
==========
Placeholder models of the two sensors in the Kawasaki half of the cell.
Visual only for now - they don't read any real data yet.

RGBDCamera
    Hangs on a boom above the medication shelf, looking down at the
    containers. identify() shows a scan ray and a highlight box around
    the container it has "recognised".

PhotoelectricSensor
    Sits beside the transfer plate with a reflector on the other side.
    The beam is red while the plate is empty and turns green once a
    container breaks it.

Both are positioned from layout.py so they move if the shelf or plate do.
"""

import numpy as np
from spatialmath import SE3
import spatialgeometry as sg

import layout

HIDDEN = SE3(0, 0, -5)      # parked under the floor when not shown

BLACK = (0.08, 0.08, 0.09, 1.0)
DARK = (0.20, 0.21, 0.23, 1.0)
STEEL = (0.62, 0.65, 0.69, 1.0)
LENS = (0.10, 0.25, 0.55, 1.0)
RED_BEAM = (1.0, 0.10, 0.10, 0.65)
GREEN_BEAM = (0.10, 1.0, 0.25, 0.65)


def look_at(eye, target):
    """Pose at `eye` whose z axis points at `target` (x stays level)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return SE3.Rt(np.column_stack((x, y, z)), eye)


def segment(p0, p1):
    """Pose + length of a cylinder running from p0 to p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    T = look_at(p0, p1)
    L = float(np.linalg.norm(p1 - p0))
    return (SE3.Rt(T.R, (p0 + p1) / 2) if L > 0 else T), L


# ======================================================================
# RGB-D CAMERA
# ======================================================================
class RGBDCamera:
    # Boom mounted behind the shelf at the far end, well above the
    # highest point the Kawasaki reaches over the shelf (~1.53 m)
    POST_XY = (2.20, -0.86)
    BOOM_HEIGHT = 1.82
    LENS_POS = (1.65, -0.86, 1.74)
    LOOK_AT = (1.65, -0.72, layout.SHELF_TOP)   # middle of the shelf

    def __init__(self, env):
        self.env = env
        x, y = self.POST_XY
        base_z = layout.WORKSTATION_TOP
        post_len = self.BOOM_HEIGHT - base_z

        # Upright post and horizontal boom
        env.add(sg.Cuboid(scale=[0.05, 0.05, post_len], color=STEEL,
                          pose=SE3(x, y, base_z + post_len / 2)))
        env.add(sg.Cuboid(scale=[0.14, 0.14, 0.015], color=DARK,
                          pose=SE3(x, y, base_z + 0.0075)))
        boom_len = x - self.LENS_POS[0] + 0.05
        env.add(sg.Cuboid(scale=[boom_len, 0.04, 0.04], color=STEEL,
                          pose=SE3(x - boom_len / 2 + 0.025, y,
                                   self.BOOM_HEIGHT)))
        # Short drop bracket from boom to camera
        env.add(sg.Cuboid(scale=[0.03, 0.03, 0.06], color=DARK,
                          pose=SE3(self.LENS_POS[0], y,
                                   self.BOOM_HEIGHT - 0.04)))

        # Camera body tilted to look at the shelf. Camera z = view axis.
        self.T = look_at(self.LENS_POS, self.LOOK_AT)
        env.add(sg.Cuboid(scale=[0.18, 0.03, 0.035], color=BLACK,
                          pose=self.T))
        # Three lenses on the front face (IR projector, RGB, IR camera)
        for dx in (-0.055, 0.0, 0.055):
            env.add(sg.Cylinder(radius=0.009, length=0.006, color=LENS,
                                pose=self.T * SE3(dx, 0, 0.019)))

        # Scan ray + highlight box, hidden until identify() is called
        self._ray = sg.Cylinder(radius=0.006, length=1.0,
                                color=(0.3, 0.9, 1.0, 0.45), pose=HIDDEN)
        self._box = sg.Cuboid(
            scale=[2 * layout.CONTAINER_RADIUS + 0.03,
                   2 * layout.CONTAINER_RADIUS + 0.03,
                   layout.CONTAINER_HEIGHT + 0.03],
            color=(0.3, 0.9, 1.0, 0.30), pose=HIDDEN)
        env.add(self._ray)
        env.add(self._box)

    def identify(self, container_point):
        """Show the camera 'locking on' to the container at this point."""
        target = SE3(container_point).t
        lens = (self.T * SE3(0, 0, 0.02)).t
        pose, L = segment(lens, target)
        # Swift can't resize a cylinder after it's added, so make a new one
        self._ray.T = HIDDEN.A
        self._ray = sg.Cylinder(radius=0.006, length=L,
                                color=(0.3, 0.9, 1.0, 0.45), pose=pose)
        self.env.add(self._ray)
        self._box.T = SE3(target).A

    def clear(self):
        self._ray.T = HIDDEN.A
        self._box.T = HIDDEN.A


# ======================================================================
# PHOTOELECTRIC SENSOR
# ======================================================================
class PhotoelectricSensor:
    # Beam runs along x across the transfer plate, 2 cm above it -
    # low enough that any container placed there breaks it, and below
    # the gripper fingertips so the fingers never do.
    BEAM_HEIGHT = layout.TRANSFER_PLATE_TOP + 0.02
    SENSOR_X = layout.TRANSFER_POINT.t[0] + 0.27
    REFLECTOR_X = layout.TRANSFER_POINT.t[0] - 0.27
    Y = layout.TRANSFER_POINT.t[1]

    def __init__(self, env):
        self.env = env
        self.detected = False
        bench = layout.WORKSTATION_TOP
        stand_h = self.BEAM_HEIGHT - bench

        for x in (self.SENSOR_X, self.REFLECTOR_X):
            env.add(sg.Cuboid(scale=[0.03, 0.03, stand_h], color=STEEL,
                              pose=SE3(x, self.Y, bench + stand_h / 2)))

        # Sensor head: housing, lens facing the plate, status LED on top
        env.add(sg.Cuboid(scale=[0.035, 0.05, 0.06], color=BLACK,
                          pose=SE3(self.SENSOR_X, self.Y, self.BEAM_HEIGHT)))
        env.add(sg.Cylinder(radius=0.011, length=0.006, color=LENS,
                            pose=SE3(self.SENSOR_X - 0.019, self.Y,
                                     self.BEAM_HEIGHT) * SE3.Ry(np.pi / 2)))
        # Swift 1.1.0 can't change a shape's colour once it's added (it
        # freezes waiting for a reply), so there's a red AND a green
        # version of the LED and beam, and we swap which one is visible.
        self._led_pose = SE3(self.SENSOR_X, self.Y, self.BEAM_HEIGHT + 0.034)
        self._led = {}
        for state, colour in ((False, RED_BEAM), (True, GREEN_BEAM)):
            self._led[state] = sg.Sphere(radius=0.007,
                                         color=colour[:3] + (1.0,),
                                         pose=HIDDEN)
            env.add(self._led[state])

        # Reflector: flat red/silver plate facing the sensor
        env.add(sg.Cuboid(scale=[0.008, 0.045, 0.045], color=(0.8, 0.1, 0.1, 1.0),
                          pose=SE3(self.REFLECTOR_X, self.Y,
                                   self.BEAM_HEIGHT)))

        # The beam itself
        p0 = (self.SENSOR_X - 0.02, self.Y, self.BEAM_HEIGHT)
        p1 = (self.REFLECTOR_X + 0.005, self.Y, self.BEAM_HEIGHT)
        self._beam_pose, L = segment(p0, p1)
        self._beam = {}
        for state, colour in ((False, RED_BEAM), (True, GREEN_BEAM)):
            self._beam[state] = sg.Cylinder(radius=0.003, length=L,
                                            color=colour, pose=HIDDEN)
            env.add(self._beam[state])

        self.set_detected(False)

    def set_detected(self, detected):
        """Red = plate empty, green = container detected."""
        self.detected = bool(detected)
        for state in (False, True):
            on = state == self.detected
            self._beam[state].T = (self._beam_pose if on else HIDDEN).A
            self._led[state].T = (self._led_pose if on else HIDDEN).A