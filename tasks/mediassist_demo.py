"""
mediassist_demo.py
==================
    python -m tasks.mediassist_demo

"""

import os
import sys
import time

import numpy as np
import swift
from spatialmath import SE3
from spatialmath.base import rpy2r
from spatialgeometry import Mesh, Cuboid, Cylinder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import layout
from environment.ward import build_ward
from environment.sensors import RGBDCamera, PhotoelectricSensor
from perception.rgbd_detection import (PinholeCamera, SimulatedRGBD,
                                       detect_containers, save_debug_image)
from tasks.kawasaki_pick_place import KawasakiCell, find_object_at
from tasks.safety_equipment import SafetyEquipment
from robot_models.doosan_m0609 import create_doosan, JOINT_XYZ, JOINT_RPY
from robot_models.doosan_delivery_safe_v7 import (
    RAIL_Y, CARRIAGE_Z, TRANSFER_X, TRANSFER_Y, TRANSFER_Z, TABLES,
    BOTTLE_H, MESH_DIR, MESH_FILES, base_pose, arm_points, solve_ik,
    plan_rail_safe_path)

DT = 0.05
COLOUR_OPTIONS = ["none", "red", "green", "blue"]
BED_NUMBERS = [1, 2, 3]

# The ward and V7 number the bedside tables along +x (table 1 at
# x = 0.86), but in our system diagram Bed 1 is at the OTHER end of the
# rail. This maps the bed the operator picks to the right table.
BED_TABLE = {1: 3, 2: 2, 3: 1}


class OrderCancelled(Exception):
    """Raised inside a motion when the e-stop says stop."""


# ======================================================================
# DOOSAN (Alessandro's delivery logic, wrapped so it can run per order)
# ======================================================================
def hide_ward_carriage(env):
    """ward.py still has the old static 'Doosan carriage' placeholder at
    the transfer position. DoosanCell draws its own carriage that moves
    with the robot, so park the static one under the floor or the two
    flicker through each other."""
    for obj in getattr(env, "swift_objects", []):
        if not isinstance(obj, Cuboid):
            continue
        p = SE3(obj.T).t
        if np.allclose(p, [TRANSFER_X, RAIL_Y, CARRIAGE_Z], atol=0.01):
            obj.T = SE3(0, 0, -5).A


class DoosanCell:
    def __init__(self, env, tick):
        """tick() is called every frame: safety check + Swift step."""
        self.env = env
        self.tick = tick
        self.robot = create_doosan()
        self.x = TRANSFER_X
        self.q = np.zeros(6)
        self.bottle = None
        self.grasp = SE3()

        hide_ward_carriage(env)
        self.carriage = Cuboid(scale=[.38, .40, .12],
                               pose=SE3(self.x, RAIL_Y, CARRIAGE_Z),
                               color=[.2, .25, .32, 1])
        env.add(self.carriage)

        self.groups = []
        self.simple = []
        if os.path.isdir(MESH_DIR):
            for filenames in MESH_FILES:
                group = []
                for filename in filenames:
                    m = Mesh(filename=str(MESH_DIR / filename),
                             scale=[.001] * 3)
                    env.add(m)
                    group.append(m)
                self.groups.append(group)
        else:
            print(f"[DOOSAN] Mesh folder not found ({MESH_DIR}). "
                  "Drawing a simple stand-in arm instead.", flush=True)
            for _ in range(6):
                c = Cylinder(radius=0.05, length=0.1,
                             color=[.92, .92, .92, 1])
                env.add(c)
                self.simple.append(c)
        self.redraw()

    # --- drawing ------------------------------------------------------
    def redraw(self):
        r = self.robot
        r.base = base_pose(self.x)
        r.q = self.q
        self.carriage.T = SE3(self.x, RAIL_Y, CARRIAGE_Z)
        if self.groups:
            T = base_pose(self.x)
            for m in self.groups[0]:
                m.T = T
            for i in range(6):
                T = T * SE3.Rt(rpy2r(JOINT_RPY[i], order='zyx'),
                               JOINT_XYZ[i]) * SE3.Rz(self.q[i])
                for m in self.groups[i + 1]:
                    m.T = T
        else:
            pts = arm_points(self.x, self.q)
            for c, a, b in zip(self.simple, pts[:-1], pts[1:]):
                self._place_segment(c, a, b)
        if self.bottle is not None:
            # Same as V7: follow the grasp but keep the container upright
            p = (r.fkine(self.q) * self.grasp).t
            self.bottle.T = SE3(*[float(v) for v in p]).A

    @staticmethod
    def _place_segment(cyl, a, b):
        v = np.asarray(b) - np.asarray(a)
        L = np.linalg.norm(v)
        if L < 1e-6:
            cyl.T = SE3(a).A
            return
        z = v / L
        h = np.array([0, 0, 1.0]) if abs(z[2]) < .99 else np.array([1.0, 0, 0])
        x = np.cross(h, z); x /= np.linalg.norm(x)
        R = np.column_stack((x, np.cross(z, x), z))
        # Cylinders can't be resized after adding, so scale isn't changed;
        # the stand-in is only a rough visual
        cyl.T = SE3.Rt(R, (np.asarray(a) + np.asarray(b)) / 2).A

    # --- motion (V7 move()) --------------------------
    def move(self, new_x, new_q, seconds):
        start = np.r_[self.x, self.q]
        target = np.r_[new_x, np.asarray(new_q, dtype=float)]
        path = plan_rail_safe_path(start, target)
        w = np.array([2, 1, 1, 1, .7, .7, .7])
        lengths = [np.linalg.norm((b - a) * w)
                   for a, b in zip(path[:-1], path[1:])]
        total = max(sum(lengths), 1e-9)
        for (a, b), distance in zip(zip(path[:-1], path[1:]), lengths):
            duration = max(.4, seconds * distance / total)
            n = max(2, int(duration / DT))
            for i in range(n + 1):
                u = i / n
                s = 10 * u**3 - 15 * u**4 + 6 * u**5
                state = a + (b - a) * s
                self.x = float(state[0])
                self.q = state[1:].copy()
                self.redraw()
                self.tick()

    # --- one delivery (V7 main() sequence) ----------------------------
    def deliver(self, bottle, bed, on_grasp=None):
        tx, ty, tz = TABLES[BED_TABLE[bed]]
        drop_x = float(np.clip(tx, .15, 3.20))
        pickup_z = TRANSFER_Z + BOTTLE_H
        drop_z = tz + BOTTLE_H

        qa = solve_ik(self.robot, TRANSFER_X,
                      SE3(TRANSFER_X, TRANSFER_Y, pickup_z + .15), self.q)
        qb = solve_ik(self.robot, TRANSFER_X,
                      SE3(TRANSFER_X, TRANSFER_Y, pickup_z), qa)
        qc = solve_ik(self.robot, drop_x, SE3(tx, ty, drop_z + .15), qb)
        qd = solve_ik(self.robot, drop_x, SE3(tx, ty, drop_z), qc)

        self.move(TRANSFER_X, qa, 2)
        self.move(TRANSFER_X, qb, 1.5)
        self.robot.base = base_pose(self.x)
        self.grasp = self.robot.fkine(self.q).inv() * SE3(bottle.T)
        self.bottle = bottle
        if on_grasp:
            on_grasp()
        self.move(TRANSFER_X, qa, 1.5)
        self.move(drop_x, qc, 3)
        self.move(drop_x, qd, 1.5)

        # V7's visual placement correction onto the table top
        start = SE3(bottle.T).t.copy()
        end = np.array([tx, ty, tz + BOTTLE_H / 2])
        self.bottle = None
        for k in range(31):
            u = k / 30
            s = 10 * u**3 - 15 * u**4 + 6 * u**5
            bottle.T = SE3(*(start + (end - start) * s)).A
            self.tick()
        self.move(drop_x, qc, 1.5)

    def return_to_transfer(self):
        q_park = solve_ik(self.robot, TRANSFER_X,
                          SE3(TRANSFER_X, TRANSFER_Y,
                              TRANSFER_Z + BOTTLE_H + .25), self.q)
        self.move(TRANSFER_X, q_park, 2.5)


# ======================================================================
# ORDER PANEL (Swift HTML elements)
# ======================================================================
class OrderPanel:
    def __init__(self, env):
        self.choice = {bed: 0 for bed in BED_NUMBERS}
        self.confirmed = False

        env.add(swift.Label("MediAssist | Medication orders"))
        for bed in BED_NUMBERS:
            env.add(swift.Select(
                cb=lambda idx, b=bed: self._choose(b, idx),
                desc=f"Bed {bed}", options=COLOUR_OPTIONS, value=0))
        env.add(swift.Button(cb=lambda _: self._confirm(),
                             desc="CONFIRM ORDERS"))
        self.status = swift.Label("Status: choose a colour for each bed")
        env.add(self.status)
        self.progress = swift.Label("")
        env.add(self.progress)

    def _choose(self, bed, idx):
        self.choice[bed] = int(idx)

    def _confirm(self):
        self.confirmed = True

    def orders(self):
        """[(bed, colour), ...] in bed order, skipping 'none'."""
        return [(b, COLOUR_OPTIONS[self.choice[b]]) for b in BED_NUMBERS
                if self.choice[b] != 0]

    def say(self, status=None, progress=None):
        if status is not None:
            self.status.desc = "Status: " + status
            print("[ORDERS]", status, flush=True)
        if progress is not None:
            self.progress.desc = progress


# ======================================================================
# THE DEMO
# ======================================================================
def main(worker=False):
    safety = None
    if worker:
        from tasks.safety_control import SafetyControl, SimulationStopped
        safety = SafetyControl()
    else:
        SimulationStopped = OrderCancelled

    env = swift.Swift()
    env.launch(realtime=True)
    try:
        env.set_camera_pose([1.65, -2.3, 2.9], [1.65, 0.55, 0.8])
    except Exception:
        pass

    containers = build_ward(env)
    camera = RGBDCamera(env)
    sensor = PhotoelectricSensor(env)
    equipment = SafetyEquipment(env, layout.KAWASAKI_BASE)

    pinhole = PinholeCamera(camera.T)
    rgbd = SimulatedRGBD(pinhole, containers)
    capture_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "captures")
    os.makedirs(capture_dir, exist_ok=True)

    def tick():
        if safety is not None:
            state = safety.status()
            equipment.update(state)
            if state != "RUNNING":
                raise SimulationStopped("Stop requested")
        env.step(DT)
        time.sleep(DT)

    kawasaki = KawasakiCell(env)
    if safety is not None:
        kawasaki.safety = safety
    doosan = DoosanCell(env, tick)
    panel = OrderPanel(env)
    kawasaki.pause(1.0)
    equipment.update("READY")

    def run_order(bed, colour, n, total):
        tag = f"Bed {bed} ({n}/{total})"

        # 1. Camera finds the colour on the shelf
        panel.say(f"{tag}: RGB-D camera looking for {colour}")
        rgb, depth = rgbd.capture()
        shelf = [d for d in detect_containers(rgb, depth, pinhole)
                 if d.position[1] < -0.55]
        save_debug_image(os.path.join(capture_dir, "latest_detection.png"),
                         rgb, depth, shelf)
        wanted = [d for d in shelf if d.colour == colour]
        if not wanted:
            panel.say(f"{tag}: no {colour} left on the shelf - skipped")
            return False
        det = wanted[0]
        obj = find_object_at(containers, det.position)

        # 2. Kawasaki: shelf -> transfer plate
        panel.say(f"{tag}: Kawasaki fetching {colour}")
        kawasaki.pick_and_place(obj, SE3(det.position),
                                layout.TRANSFER_POINT,
                                camera=camera, sensor=sensor)

        # 3. Doosan: transfer plate -> bedside table
        panel.say(f"{tag}: container on plate, Doosan collecting")
        doosan.deliver(obj, bed,
                       on_grasp=lambda: sensor.set_detected(False))
        panel.say(f"{tag}: {colour} delivered")
        return True

    print("\nMediAssist demo ready. Choose colours in the Swift page "
          "and press CONFIRM ORDERS.", flush=True)
    try:
        while True:
            # Idle: keep Swift alive so the panel's buttons respond
            if not panel.confirmed:
                if safety is not None:
                    state = safety.status()
                    # launcher says RUNNING while we wait for orders, but
                    # the cell is idle, so show green unless stopped
                    equipment.update("READY" if state == "RUNNING" else state)
                env.step(DT)
                time.sleep(DT)
                continue
            panel.confirmed = False

            orders = panel.orders()
            if not orders:
                panel.say("no beds have a colour selected")
                continue

            # Check the shelf has enough of each colour before starting
            rgb, depth = rgbd.capture()
            shelf = [d.colour for d in detect_containers(rgb, depth, pinhole)
                     if d.position[1] < -0.55]
            short = [c for c in set(c for _, c in orders)
                     if sum(1 for _, oc in orders if oc == c) > shelf.count(c)]
            if short:
                panel.say("not enough " + ", ".join(short) +
                          " on the shelf for these orders")
                continue

            plan = ", ".join(f"Bed {b}: {c}" for b, c in orders)
            panel.say("orders confirmed", progress=plan)
            if safety is not None and safety.status() != "RUNNING":
                panel.say("safety panel is not RUNNING - press START there")
                continue
            equipment.update("RUNNING")

            try:
                for n, (bed, colour) in enumerate(orders, start=1):
                    run_order(bed, colour, n, len(orders))
                doosan.return_to_transfer()
                kawasaki.move_joint(kawasaki.bot.qhome)
                panel.say("all deliveries complete - ready for new orders")
                equipment.update("READY")
            except SimulationStopped:
                equipment.update("E-STOPPED")
                panel.say("STOPPED - motion cancelled. Reset, then "
                          "START on the safety panel.")
                if safety is not None:
                    # Hold everything still (no motion) until the safety
                    # panel's START asks for a fresh scene
                    while safety.status() != "RESTARTING":
                        env.step(DT)
                        time.sleep(DT)
                    return env
            except RuntimeError as exc:
                equipment.update("FAULT")
                panel.say(f"planning fault: {exc}")
    except KeyboardInterrupt:
        print("Simulation stopped.", flush=True)
    finally:
        env.close()
    return env


if __name__ == "__main__":
    main(worker="--worker" in sys.argv)