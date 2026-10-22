"""Teach / jog panel: joint, Cartesian and rail jogging with fault checks."""

import numpy as np
from spatialmath import SE3

import layout
from robot_models.doosan_delivery_safe_v7 import (
    base_pose, rail_clearance, arm_points, TRANSFER_X)

# Light-curtain posts beside the transfer plate (same as the delivery check)
CURTAIN_POSTS = ((0.94, 0.39), (2.36, 0.39))
POST_HALF = 0.12
POST_Z = (0.70, 1.72)


def post_clearance(x, joints):
    pts = arm_points(x, joints)
    segs = [(pts[i - 1], pts[i]) for i in range(2, 7)]
    segs.append((pts[6], pts[6] + np.array([0, 0, -0.17])))
    for a, b in segs:
        for t in np.linspace(0, 1, 13):
            p = a + (b - a) * t
            if POST_Z[0] <= p[2] <= POST_Z[1]:
                for px, py in CURTAIN_POSTS:
                    if abs(p[0] - px) <= POST_HALF and abs(p[1] - py) <= POST_HALF:
                        return False
    return True

STEP_OPTIONS = [("5 mm", 0.005), ("1 cm", 0.01), ("5 cm", 0.05)]
JOINT_SPEED = np.deg2rad(1.5)      # max change per frame (~30 deg/s)
CART_FRAMES = 15                   # frames per Cartesian jog step
SINGULAR_LIMIT = 0.004             # manipulability below this = refuse


# ======================================================================
# Robot adapters: same interface for both arms
# ======================================================================
class KawasakiJog:
    name = "Kawasaki RS007N"
    has_rail = False

    def __init__(self, cell):
        self.cell = cell
        self.bot = cell.bot
        self.qlim = np.asarray(self.bot.qlim, dtype=float)

    def get_q(self):
        return self.bot.q.copy()

    def apply(self, q):
        self.bot.update(q)
        self.cell.gripper.update(self.bot.fk())

    def fk(self, q):
        return self.bot.fkine(q)

    def jacob(self, q):
        return self.bot.jacob0(q)

    def check(self, q):
        """Keep every joint and the fingertips above the bench / shelf."""
        frames = self.bot.link_frames(q)
        tips = (frames[-1] * SE3(0, 0, 0.14)).t
        for p in [f.t for f in frames[2:]] + [tips]:
            on_bench = (0.875 <= p[0] <= 2.425) and (-1.065 <= p[1] <= 0.485)
            if on_bench and p[2] < layout.WORKSTATION_TOP + 0.03:
                return False, "would hit the workstation surface"
            on_shelf = (1.20 <= p[0] <= 2.10) and (-0.84 <= p[1] <= -0.60)
            if on_shelf and p[2] < layout.SHELF_TOP + 0.02:
                return False, "would hit the medication shelf"
        return True, ""


class DoosanJog:
    name = "Doosan M0609"
    has_rail = True
    RAIL_LIMITS = (0.15, 3.20)

    def __init__(self, cell):
        self.cell = cell
        self.robot = cell.robot
        self.qlim = np.asarray(self.robot.qlim, dtype=float)

    def get_q(self):
        return self.cell.q.copy()

    def apply(self, q, x=None):
        if x is not None:
            self.cell.x = float(x)
        self.cell.q = np.asarray(q, dtype=float)
        self.cell.redraw()

    def fk(self, q):
        self.robot.base = base_pose(self.cell.x)
        return self.robot.fkine(q)

    def jacob(self, q):
        self.robot.base = base_pose(self.cell.x)
        return self.robot.jacob0(q)

    def check(self, q, x=None):
        x = self.cell.x if x is None else x
        ok, msg = rail_clearance(x, q)
        if not ok:
            return False, "would hit the rail: " + msg
        if not post_clearance(x, q):
            return False, "would hit a light-curtain post"
        return True, ""


# ======================================================================
# Jog controller (the web control panel calls these)
# ======================================================================
class JogController:
    def __init__(self, kawasaki_cell, doosan_cell):
        self.robots = [KawasakiJog(kawasaki_cell), DoosanJog(doosan_cell)]
        self.sel = 0
        self.teach = False
        self.step_idx = 1
        self.joint_target = None
        self.rail_target = None
        self.cart = None
        self.fault = "none"

    @property
    def robot(self):
        return self.robots[self.sel]

    @property
    def step(self):
        return STEP_OPTIONS[self.step_idx][1]

    # --- requests ------------------------------------------------------
    def set_mode(self, teach):
        self.cancel()
        self.teach = bool(teach)
        self.fault = "none"

    def set_robot(self, idx):
        self.cancel()
        self.sel = int(idx) % len(self.robots)
        self.fault = "none"

    def set_step(self, idx):
        self.step_idx = int(idx) % len(STEP_OPTIONS)

    def _blocked(self):
        if not self.teach:
            self.fault = "switch to Teach mode to move robots by hand"
            return True
        return False

    def set_joint(self, j, value_deg):
        if self._blocked():
            return
        self.fault = "none"
        if self.joint_target is None:
            self.joint_target = self.robot.get_q()
        lo, hi = self.robot.qlim[:, j]
        self.joint_target[j] = float(np.clip(np.deg2rad(value_deg), lo, hi))
        self.cart = None

    def set_rail(self, value):
        if self._blocked():
            return
        if not self.robot.has_rail:
            self.fault = "the rail only moves the Doosan"
            return
        self.fault = "none"
        self.rail_target = float(np.clip(value, *DoosanJog.RAIL_LIMITS))
        self.cart = None

    def cart_jog(self, axis, sign):
        if self._blocked():
            return
        T0 = self.robot.fk(self.robot.get_q())
        goal = T0.t.copy()
        goal[axis] += sign * self.step
        self.fault = "none"
        self.joint_target = None
        self.rail_target = None
        self.cart = {"p0": T0.t, "p1": goal, "R": T0.R, "k": 0}

    def home(self):
        if self._blocked():
            return
        self.cart = None
        self.fault = "none"
        if self.sel == 0:
            self.joint_target = self.robot.bot.qhome.copy()
        else:
            # Doosan home is its park pose (all-zeros is a singularity)
            self.joint_target = self.robot.cell.park_q()
            self.rail_target = TRANSFER_X

    def cancel(self):
        self.joint_target = None
        self.rail_target = None
        self.cart = None

    def busy(self):
        return any(v is not None for v in
                   (self.joint_target, self.rail_target, self.cart))

    # --- one frame of motion -------------------------------------------
    def frame(self):
        """Advance any active jog by one frame. Returns True if moved."""
        if not self.teach or not self.busy():
            return False
        r = self.robot
        q = r.get_q()
        if self.cart is not None:
            return self._cart_frame(r, q)

        q_new = q.copy()
        if self.joint_target is not None:
            q_new = q + np.clip(self.joint_target - q,
                                -JOINT_SPEED, JOINT_SPEED)
        x_new = None
        if self.rail_target is not None and r.has_rail:
            x = r.cell.x
            x_new = x + np.clip(self.rail_target - x, -0.01, 0.01)

        ok, msg = r.check(q_new, x_new) if r.has_rail else r.check(q_new)
        if not ok:
            self._fault(msg)
            return False
        if r.has_rail:
            r.apply(q_new, x_new)
        else:
            r.apply(q_new)

        done_q = self.joint_target is None or \
            np.allclose(q_new, self.joint_target, atol=1e-4)
        done_x = self.rail_target is None or x_new is None or \
            abs(x_new - self.rail_target) < 1e-4
        if done_q and done_x:
            self.joint_target = None
            self.rail_target = None
        return True

    def _cart_frame(self, r, q):
        c = self.cart
        c["k"] += 1
        n = CART_FRAMES
        s = (1 - np.cos(np.pi * min(c["k"], n) / n)) / 2
        p_des = c["p0"] + s * (c["p1"] - c["p0"])

        # RMRC step: velocity towards the line, hold orientation
        T = r.fk(q)
        v = p_des - T.t
        Re = c["R"] @ T.R.T
        w = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0],
                            Re[1, 0] - Re[0, 1]])
        J = r.jacob(q)
        manip = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
        if manip < SINGULAR_LIMIT:
            self._fault(f"too close to a singularity (manipulability "
                        f"{manip:.4f}) - press Home or jog a joint first")
            return False
        dq = J.T @ np.linalg.solve(J @ J.T + 0.01**2 * np.eye(6),
                                   np.hstack((v, w)))
        q_new = q + dq
        lo, hi = r.qlim
        if np.any(q_new < lo) or np.any(q_new > hi):
            j = int(np.argmax((q_new < lo) | (q_new > hi)))
            self._fault(f"J{j + 1} would pass its joint limit")
            return False
        ok, msg = r.check(q_new)
        if not ok:
            self._fault(msg)
            return False
        r.apply(q_new)

        if c["k"] >= n + 5:
            err = np.linalg.norm(r.fk(r.get_q()).t - c["p1"])
            self.cart = None
            if err > 0.002:
                self._fault(f"could only get within {err*1000:.0f} mm - "
                            "target at the edge of the workspace")
        return True

    def _fault(self, msg):
        self.cancel()
        self.fault = msg
        print("[JOG] Refused:", msg, flush=True)

    # --- what the web page shows ----------------------------------------
    def snapshot(self):
        r = self.robot
        q = r.get_q()
        lo, hi = r.qlim
        p = r.fk(q).t
        return {
            "teach": self.teach,
            "robots": [rb.name for rb in self.robots],
            "sel": self.sel,
            "steps": [n for n, _ in STEP_OPTIONS],
            "step": self.step_idx,
            "q": [round(float(v), 1) for v in np.rad2deg(q)],
            "qmin": [round(float(v)) for v in np.rad2deg(lo)],
            "qmax": [round(float(v)) for v in np.rad2deg(hi)],
            "has_rail": r.has_rail,
            "rail": round(float(r.cell.x), 3) if r.has_rail else None,
            "rail_limits": list(DoosanJog.RAIL_LIMITS),
            "tool": [round(float(v), 3) for v in p],
            "busy": self.busy(),
            "fault": self.fault,
        }
