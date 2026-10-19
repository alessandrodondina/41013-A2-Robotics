"""
kawasaki_rs007n.py
==================
Kawasaki RS007N (Robot 1) - parameter based DH model.

Job in MediAssist: pick the requested colour coded medication container
off the storage shelf and place it on the transfer plate for the Doosan.

WHERE THE NUMBERS COME FROM
    Link lengths:   Kawasaki's own ROS description of the RS007N
                    (khi_robot package, rs007n_macro.xacro)
    Joint limits:   Kawasaki RS007N standard specification sheet
                    (90151-0313DEC)
    Cross check:    shoulder-to-wrist reach = 0.355 + 0.375 = 0.730 m,
                    which matches the 730 mm max reach on the datasheet.

Stage 1: kinematics + simple cylinder graphics (A1 style).
         Realistic visuals come in the next stage.

Check the model:          python robots/kawasaki_rs007n.py
Show it in the ward:      python robots/kawasaki_rs007n.py --swift
"""

import numpy as np
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH

try:
    import spatialgeometry as sg
except ImportError:
    # Graphics are optional, the kinematics work without them
    sg = None


class KawasakiRS007N(DHRobot):
    """6-DOF Kawasaki RS007N, 7 kg payload, 730 mm reach."""

    # ------------------------------------------------------------------
    # DIMENSIONS (metres)
    # ------------------------------------------------------------------
    D1 = 0.360     # base to shoulder (J2) height
    A2 = 0.355     # upper arm, shoulder (J2) to elbow (J3)
    D4 = 0.375     # forearm, elbow (J3) to wrist centre (J5)
    D6 = 0.078     # wrist centre to tool flange face

    # ------------------------------------------------------------------
    # JOINT LIMITS (degrees) - straight from the Kawasaki spec sheet
    # ------------------------------------------------------------------
    QLIM_DEG = [
        (-180, 180),   # JT1 arm rotation
        (-135, 135),   # JT2 arm out-in
        (-155, 155),   # JT3 arm up-down
        (-200, 200),   # JT4 wrist swivel
        (-125, 125),   # JT5 wrist bend
        (-360, 360),   # JT6 wrist twist
    ]

    def __init__(self, base=SE3(), name="KawasakiRS007N"):
        deg = np.pi / 180.0
        lim = [np.deg2rad(l) for l in self.QLIM_DEG]

        # --------------------------------------------------------------
        # THE DH TABLE (standard DH)
        #
        #  i | theta      |   d   |   a   | alpha
        # ---+------------+-------+-------+-------
        #  1 | q1         | 0.360 | 0     |  -90
        #  2 | q2 - 90    | 0     | 0.355 |    0
        #  3 | q3 + 90    | 0     | 0     |  +90
        #  4 | q4         | 0.375 | 0     |  -90
        #  5 | q5         | 0     | 0     |  +90
        #  6 | q6         | 0.078 | 0     |    0
        #
        #   J1      Spins the whole arm about the vertical axis. The -90
        #           twist tips the next axis horizontal so J2 can lean.
        #   J2, J3  Shoulder and elbow. Their axes are parallel, so they
        #           form the planar "arm" that sets reach and height.
        #           The -90 / +90 offsets make q = 0 the Kawasaki zero:
        #           arm pointing straight up, like the datasheet drawing.
        #   J3      Unlike a UR, the forearm comes off the elbow ALONG z
        #           (d4 = 0.375) and the +90 twist turns it that way.
        #   J4-J6   Spherical wrist: all three axes meet at one point
        #           (the wrist centre). Alternating -90 / +90 twists keep
        #           them perpendicular so the tool can point anywhere.
        #
        # Positive q2 and q3 both lean the arm forward (+x of the base),
        # positive q5 bends the wrist down.
        # --------------------------------------------------------------
        links = [
            RevoluteDH(d=self.D1, a=0.0,     alpha=-90 * deg,
                       offset=0.0,        qlim=lim[0]),   # JT1
            RevoluteDH(d=0.0,     a=self.A2, alpha=0.0,
                       offset=-90 * deg,  qlim=lim[1]),   # JT2
            RevoluteDH(d=0.0,     a=0.0,     alpha=90 * deg,
                       offset=90 * deg,   qlim=lim[2]),   # JT3
            RevoluteDH(d=self.D4, a=0.0,     alpha=-90 * deg,
                       offset=0.0,        qlim=lim[3]),   # JT4
            RevoluteDH(d=0.0,     a=0.0,     alpha=90 * deg,
                       offset=0.0,        qlim=lim[4]),   # JT5
            RevoluteDH(d=self.D6, a=0.0,     alpha=0.0,
                       offset=0.0,        qlim=lim[5]),   # JT6
        ]

        super().__init__(links, name=name, base=base,
                         manufacturer="Kawasaki")

        # Named poses
        #   qzero  - arm straight up (Kawasaki's mechanical zero)
        #   qhome  - "ready" pose: elbow bent, tool pointing straight
        #            down in front of the robot, clear of the bench
        self.qzero = np.zeros(6)
        self.qhome = np.deg2rad([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])
        self.q = self.qhome.copy()

        self._shapes = []
        self._env = None

    # ==================================================================
    # FORWARD KINEMATICS
    # ==================================================================
    def link_frames(self, q=None):
        """Every frame along the arm: [base, T1, T2, ... T6].

        Multiplying the DH transforms from the base outwards walks up the
        arm one joint at a time:  base x A1 x A2 x ... x A6 = flange pose
        """
        q = self.q if q is None else np.asarray(q, dtype=float)
        T = SE3(self.base)
        frames = [T]
        for link, qi in zip(self.links, q):
            T = T * link.A(qi)
            frames.append(T)
        return frames

    def fk(self, q=None):
        """Flange pose only. Same as link_frames()[-1]."""
        return self.fkine(self.q if q is None else q)

    # ==================================================================
    # INVERSE KINEMATICS
    # ==================================================================
    def solve_ik(self, Tgoal, q0=None, position_only=False,
                 attempts=25, tol=1e-6):
        """Joint angles that put the flange at Tgoal.

        position_only=False by default because for pick and place the
        tool has to point DOWN at the container, so orientation matters.
        Retries from random seeds so the solver can't get stuck.
        """
        mask = [1, 1, 1, 0, 0, 0] if position_only else [1, 1, 1, 1, 1, 1]
        q0 = np.asarray(self.q if q0 is None else q0, dtype=float)
        rng = np.random.default_rng(0)      # fixed seed = repeatable demo
        best = None

        for k in range(attempts):
            seed = q0 if k == 0 else rng.uniform(*self.qlim)
            try:
                sol = self.ikine_LM(Tgoal, q0=seed, mask=mask,
                                    joint_limits=True, tol=tol)
            except TypeError:
                sol = self.ikine_LM(Tgoal, q0=seed, mask=mask)

            if not getattr(sol, "success", False):
                continue
            q = np.asarray(sol.q, dtype=float)
            if not self.within_limits(q):
                continue

            err = np.linalg.norm(self.fkine(q).t - SE3(Tgoal).t)
            # Prefer the answer closest to where we started, so the arm
            # doesn't do a wild swing to reach a nearby target
            dist = np.linalg.norm(q - q0)
            if best is None or (err < 1e-4 and dist < best[2]):
                best = (q, err, dist)

        if best is None:
            raise RuntimeError(
                f"IK failed for target\n{SE3(Tgoal)}\n"
                "Target is probably outside the RS007N workspace.")
        return best[0]

    def within_limits(self, q):
        lo, hi = self.qlim
        return bool(np.all(q >= lo - 1e-6) and np.all(q <= hi + 1e-6))

    def reach(self):
        """Shoulder to wrist centre, arm dead straight (datasheet: 730 mm)."""
        return self.A2 + self.D4

    # ==================================================================
    # GRAPHICS (stage 1: simple cylinders, replaced in stage 2)
    # ==================================================================
    @staticmethod
    def _segment_pose(p0, p1):
        """Pose + length of a cylinder running from point p0 to p1."""
        v = np.asarray(p1) - np.asarray(p0)
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            return SE3(p0), 0.0
        z = v / L
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(helper, z))) > 0.99:
            helper = np.array([1.0, 0.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.column_stack((x, y, z))
        return SE3.Rt(R, (np.asarray(p0) + np.asarray(p1)) / 2.0), L

    def add_to_env(self, env):
        if sg is None:
            raise ImportError("spatialgeometry is required for graphics")
        self._env = env
        frames = self.link_frames()
        white = (0.93, 0.93, 0.91, 1.0)
        grey = (0.35, 0.37, 0.40, 1.0)

        # Base housing below J1
        self._base_shape = sg.Cylinder(radius=0.09, length=0.20,
                                       color=grey)
        env.add(self._base_shape)

        # Link cylinders between consecutive frame origins
        self._shapes = []
        for i in range(6):
            T, L = self._segment_pose(frames[i].t, frames[i + 1].t)
            if L < 1e-6:
                continue        # frames on top of each other (wrist)
            cyl = sg.Cylinder(radius=0.045, length=L, pose=T, color=white)
            env.add(cyl)
            self._shapes.append(("link", i, cyl))

        # Joint markers along each joint axis (z of the frame before it)
        for i in range(6):
            jc = sg.Cylinder(radius=0.055, length=0.11,
                             pose=frames[i], color=grey)
            env.add(jc)
            self._shapes.append(("joint", i, jc))

        self._tool = sg.Sphere(radius=0.02, color=(0.9, 0.1, 0.1, 1.0))
        env.add(self._tool)

        self.update()
        return self

    def update(self, q=None):
        """Re-pose every shape from the current joint angles."""
        if q is not None:
            self.q = np.asarray(q, dtype=float)
        if not self._shapes:
            return
        frames = self.link_frames()
        self._base_shape.T = frames[0] * SE3(0, 0, 0.10)
        for kind, i, shape in self._shapes:
            if kind == "link":
                shape.T, _ = self._segment_pose(frames[i].t,
                                                frames[i + 1].t)
            else:
                shape.T = frames[i]
        self._tool.T = frames[6]


# ======================================================================
# SELF TEST
# ======================================================================
def _self_test():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import layout

    bot = KawasakiRS007N(base=layout.KAWASAKI_BASE)
    print(bot)

    # 1. Zero pose: arm straight up, flange should sit
    #    0.360 + 0.355 + 0.375 + 0.078 = 1.168 m above the base
    h = (bot.fk(bot.qzero).t - bot.base.t)[2]
    print(f"Flange height at zero pose: {h:.3f} m (expected 1.168)")

    # 2. Reach vs datasheet
    print(f"Shoulder to wrist reach:    {bot.reach():.3f} m "
          "(datasheet 0.730)")

    # 3. IK round trip at home: FK -> IK -> FK should land on itself
    Tgoal = bot.fk(bot.qhome)
    q_ik = bot.solve_ik(Tgoal, q0=bot.qhome)
    err = np.linalg.norm(bot.fk(q_ik).t - Tgoal.t)
    print(f"IK round-trip error:        {err:.2e} m")

    # 4. Can it reach every container and the transfer plate with the
    #    tool pointing straight down? Flange held 0.10 m above the
    #    container centre to leave room for a gripper later.
    print("\nReachability (tool pointing down, 0.10 m above target):")
    down = SE3.Rx(np.pi)
    targets = []
    for colour, slots in layout.MEDICATION_SLOTS.items():
        for k, T in enumerate(slots):
            targets.append((f"{colour} {k}", T))
    targets.append(("transfer plate", layout.TRANSFER_POINT))

    q_prev = bot.qhome
    for label, T in targets:
        goal = SE3(T.t) * SE3(0, 0, 0.10) * down
        try:
            q = bot.solve_ik(goal, q0=q_prev)
            e = np.linalg.norm(bot.fk(q).t - goal.t)
            print(f"  {label:15s} OK   err {e:.1e} m   "
                  f"q = {np.round(np.rad2deg(q), 1)}")
        except RuntimeError:
            print(f"  {label:15s} UNREACHABLE")


def _swift_demo():
    """Mount the robot in the ward and wave it through a few poses."""
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    import swift
    from roboticstoolbox import jtraj
    import layout
    from environment.ward import build_ward

    env = swift.Swift()
    env.launch(realtime=True)
    build_ward(env)

    bot = KawasakiRS007N(base=layout.KAWASAKI_BASE)
    bot.q = bot.qzero.copy()
    bot.add_to_env(env)

    poses = [bot.qzero, bot.qhome,
             np.deg2rad([-90, 20, 70, 0, 90, 0]),   # towards the shelf
             np.deg2rad([90, 20, 70, 0, 90, 0]),    # towards the plate
             bot.qhome]

    try:
        while True:
            for q_from, q_to in zip(poses[:-1], poses[1:]):
                for q in jtraj(q_from, q_to, 60).q:
                    bot.update(q)
                    env.step(0.03)
                    time.sleep(0.03)
            poses = poses[::-1]
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    import sys
    if "--swift" in sys.argv:
        _swift_demo()
    else:
        _self_test()
