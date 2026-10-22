"""
kawasaki_pick_place.py
======================
Robot 1 task: the Kawasaki RS007N picks a medication container off the
storage shelf and places it on the transfer plate.

*Notes for my VIVA* - Motion choices:
    Big moves between areas  -> joint space trajectory (jtraj).
        Smooth, fast, and every joint stays inside its limits. The exact
        path of the tool doesn't matter while it's high in free space.
    Going down to / up from a container -> RMRC straight line.
        Near the shelf the tool must go STRAIGHT down and up so the
        fingers don't clip the neighbouring containers. jtraj can't
        promise a straight line, RMRC can.

The RGB-D camera finds the requested colour and its 3D position; the
robot picks at the DETECTED position, not a hard coded one.

Each run saves what the camera saw to captures/latest_detection.png
"""

import os
import sys
import time

import numpy as np
from spatialmath import SE3
from roboticstoolbox import jtraj

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout
from robot_models.kawasaki_rs007n import KawasakiRS007N
from robot_models.simple_gripper import SimpleGripper

DT = 0.03                # seconds per animation frame
APPROACH_HEIGHT = 0.15   # how far above a container the tool hovers
TOOL_DOWN = SE3.Rx(np.pi)  # tool z pointing straight down at the floor


class KawasakiCell:
    """The Kawasaki, its gripper and whatever it is holding."""

    def __init__(self, env):
        self.env = env
        self.bot = KawasakiRS007N(base=layout.KAWASAKI_BASE)
        self.gripper = SimpleGripper()
        self.held = None            # object currently in the gripper
        self._T_held = None         # flange -> object, fixed while held

        self.bot.add_to_env(env)
        self.gripper.add_to_env(env, self.bot.fk())

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def show(self, q):
        """Move the robot to q and drag the gripper / held object along."""
        self.bot.update(q)
        T_flange = self.bot.fk()
        self.gripper.update(T_flange)
        if self.held is not None:
            self.held.T = T_flange * self._T_held
        self.env.step(DT)
        time.sleep(DT)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    def flange_for(self, point, height=0.0):
        """Flange pose that puts the gripper's grasp point at `point`
        (+ height above it), with the tool pointing straight down."""
        p = SE3(point).t + [0, 0, height + self.gripper.TCP_OFFSET]
        return SE3(p) * TOOL_DOWN

    # ------------------------------------------------------------------
    # Motions
    # ------------------------------------------------------------------
    def move_joint(self, q_goal, steps=70):
        """Joint space move from the current pose to q_goal."""
        for q in jtraj(self.bot.q, q_goal, steps).q:
            self.show(q)

    def move_to(self, T_goal, steps=70):
        """IK for T_goal, then a joint space move there."""
        q_goal = self.bot.solve_ik(T_goal, q0=self.bot.q)
        self.move_joint(q_goal, steps)

    def move_line(self, T_goal, steps=50, damping=0.01):
        """RMRC: move the flange in a straight line to T_goal.

        Each frame:
          1. work out where the flange SHOULD be on the straight line
          2. compare to where it actually is -> desired velocity xdot
          3. joint velocities from the Jacobian:  qdot = J^-1 * xdot
             (damped least squares, so it stays well behaved near a
             singularity instead of blowing up)
          4. q = q + qdot * dt
        Because step 2 measures the real position every frame, small
        errors get corrected instead of building up.
        """
        T0 = self.bot.fk()
        p0, p1 = T0.t, SE3(T_goal).t
        R_goal = SE3(T_goal).R
        q = self.bot.q.copy()
        lo, hi = self.bot.qlim

        for k in range(1, steps + 1):
            # Smooth 0 -> 1 profile: starts and ends at zero speed
            s = (1 - np.cos(np.pi * k / steps)) / 2
            p_des = p0 + s * (p1 - p0)

            T = self.bot.fkine(q)
            v = (p_des - T.t) / DT
            # Orientation error (rotation that takes current R to R_goal)
            Re = R_goal @ T.R.T
            w = 0.5 * np.array([Re[2, 1] - Re[1, 2],
                                Re[0, 2] - Re[2, 0],
                                Re[1, 0] - Re[0, 1]]) / DT
            xdot = np.hstack((v, w))

            J = self.bot.jacob0(q)
            qdot = J.T @ np.linalg.solve(
                J @ J.T + damping**2 * np.eye(6), xdot)
            q = np.clip(q + qdot * DT, lo, hi)
            self.show(q)

        err = np.linalg.norm(self.bot.fk().t - p1)
        print(f"    straight line finished, position error {err*1000:.2f} mm")

    def gripper_to(self, gap, steps=20):
        """Animate the fingers opening or closing."""
        for g in np.linspace(self.gripper.opening, gap, steps):
            self.gripper.set_opening(g)
            self.show(self.bot.q)

    # ------------------------------------------------------------------
    # Grasping
    # ------------------------------------------------------------------
    def grab(self, obj):
        """Lock obj to the flange where it is right now."""
        self.held = obj
        self._T_held = self.bot.fk().inv() * SE3(obj.T)

    def release(self):
        self.held = None
        self._T_held = None

    # ------------------------------------------------------------------
    # The task
    # ------------------------------------------------------------------
    def pause(self, seconds):
        """Hold still for a moment (keeps Swift updating)."""
        for _ in range(int(seconds / DT)):
            self.show(self.bot.q)

    def pick_and_place(self, obj, pick_point, place_point,
                       camera=None, sensor=None):
        """camera / sensor are optional placeholders (environment/sensors.py)
        that only change what is shown, not how the robot moves."""
        container_gap = 2 * layout.CONTAINER_RADIUS
        if sensor is not None:
            sensor.set_detected(False)

        print("  -> home")
        self.move_joint(self.bot.qhome)

        if camera is not None:
            print("  -> RGB-D camera identifying container")
            camera.identify(pick_point)
            self.pause(1.5)
            camera.clear()

        print("  -> above container")
        self.move_to(self.flange_for(pick_point, APPROACH_HEIGHT))
        self.gripper_to(self.gripper.MAX_OPENING)

        print("  -> down to container (RMRC)")
        self.move_line(self.flange_for(pick_point))
        self.gripper_to(container_gap)
        self.grab(obj)

        print("  -> lift (RMRC)")
        self.move_line(self.flange_for(pick_point, APPROACH_HEIGHT))

        print("  -> carry to transfer plate")
        self.move_joint(self.bot.qhome, steps=60)
        self.move_to(self.flange_for(place_point, APPROACH_HEIGHT))

        print("  -> lower onto plate (RMRC)")
        self.move_line(self.flange_for(place_point))
        self.release()
        if sensor is not None:
            print("  -> photoelectric sensor: container detected")
            sensor.set_detected(True)
        self.gripper_to(self.gripper.MAX_OPENING)

        print("  -> retreat (RMRC) and home")
        self.move_line(self.flange_for(place_point, APPROACH_HEIGHT))
        self.move_joint(self.bot.qhome)
        print("  done.")


def find_object_at(containers, point, tol=0.03):
    """The Swift container sitting at a detected position (the gripper
    grabs whatever is physically there)."""
    for obj in containers:
        if np.linalg.norm(SE3(obj.T).t - point) < tol:
            return obj
    return None


def parse_order(text):
    """'bed_2 green', 'bed_2=green' or just 'green' -> (bed, colour)."""
    parts = text.replace("=", " ").lower().split()
    colour = next((p for p in parts if p in layout.MEDICATION_SLOTS), None)
    bed = next((p for p in parts if p in layout.BEDS), None)
    return bed, colour


def main():
    import swift
    from environment.ward import build_ward
    from environment.sensors import RGBDCamera, PhotoelectricSensor
    from perception.rgbd_detection import (PinholeCamera, SimulatedRGBD,
                                           detect_containers,
                                           save_debug_image)

    env = swift.Swift()
    env.launch(realtime=True)
    containers = build_ward(env)
    camera = RGBDCamera(env)
    sensor = PhotoelectricSensor(env)

    # Perception: the camera model sits exactly where the camera prop is
    pinhole = PinholeCamera(camera.T)
    rgbd = SimulatedRGBD(pinhole, containers)
    capture_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "captures")
    os.makedirs(capture_dir, exist_ok=True)

    cell = KawasakiCell(env)
    cell.pause(1.0)     # let Swift finish loading before moving

    # Orders from the command line, e.g.  bed_2=green bed_1=red
    # or typed in one at a time if none are given.
    queue = [parse_order(a) for a in sys.argv[1:]]
    interactive = not queue
    on_plate = None     # container currently sitting on the transfer plate

    try:
        while True:
            if interactive:
                text = input("\nOrder (e.g. 'bed_2 green', q to quit): ")
                if text.strip().lower() in ("q", "quit", "exit"):
                    break
                bed, colour = parse_order(text)
            elif queue:
                bed, colour = queue.pop(0)
            else:
                break

            if colour is None:
                print("  Didn't recognise a colour (red / green / blue).")
                continue
            bed_label = bed or "no bed given"
            print(f"\nOrder: {colour} medication for {bed_label}")

            # Plate must be clear before the next container goes on it.
            # (Placeholder for the Doosan: it takes it to the bed table.)
            if on_plate is not None:
                obj, dest = on_plate
                print(f"  Transfer plate cleared (Doosan placeholder -> {dest})")
                if dest in layout.BEDS:
                    obj.T = layout.BEDS[dest].A
                else:
                    obj.T = SE3(0, 0, -5).A
                sensor.set_detected(False)
                cell.pause(0.5)
                on_plate = None

            # 1. Look: capture an RGB-D frame and find the containers
            rgb, depth = rgbd.capture()
            found = detect_containers(rgb, depth, pinhole)
            shelf = [d for d in found if d.position[1] < -0.55]
            save_debug_image(os.path.join(capture_dir, "latest_detection.png"),
                             rgb, depth, shelf)
            print(f"  RGB-D camera sees {len(shelf)} container(s): "
                  + ", ".join(d.colour for d in shelf))

            # 2. Choose: first container of the requested colour
            wanted = [d for d in shelf if d.colour == colour]
            if not wanted:
                print(f"  No {colour} container left on the shelf - "
                      "order skipped, robot stays home.")
                continue
            det = wanted[0]
            print(f"  Target: {det.colour} at {np.round(det.position, 3)}")
            obj = find_object_at(containers, det.position)
            if obj is None:
                print("  Detection doesn't match anything graspable - skipped.")
                continue

            # 3. Act: pick at the DETECTED position, place on the plate
            cell.pick_and_place(obj, SE3(det.position), layout.TRANSFER_POINT,
                                camera=camera, sensor=sensor)
            on_plate = (obj, bed)

        print("\nAll orders done. Ctrl+C to close.")
        while True:
            env.step(0.05)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Simulation stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()