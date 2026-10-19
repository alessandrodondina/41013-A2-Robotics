"""
kawasaki_pick_place.py
======================
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
    def pick_and_place(self, obj, pick_point, place_point):
        container_gap = 2 * layout.CONTAINER_RADIUS

        print("  -> home")
        self.move_joint(self.bot.qhome)

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
        self.gripper_to(self.gripper.MAX_OPENING)

        print("  -> retreat (RMRC) and home")
        self.move_line(self.flange_for(place_point, APPROACH_HEIGHT))
        self.move_joint(self.bot.qhome)
        print("  done.")


def main():
    import swift
    from environment.ward import build_ward

    colour = sys.argv[1] if len(sys.argv) > 1 else "red"
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    env = swift.Swift()
    env.launch(realtime=True)
    containers = build_ward(env)

    # Match the requested colour/index to the actual cylinder object.
    # build_ward adds the containers in layout.SHELF_ORDER order.
    matches = [i for i, c in enumerate(layout.SHELF_ORDER) if c == colour]
    shelf_index = matches[index]
    obj = containers[shelf_index]
    pick_point = layout.MEDICATION_SLOTS[colour][index]

    cell = KawasakiCell(env)
    print(f"Picking {colour} container {index}")
    cell.pick_and_place(obj, pick_point, layout.TRANSFER_POINT)

    try:
        while True:
            env.step(0.05)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Simulation stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
