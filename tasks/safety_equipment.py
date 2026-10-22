"""Clearly positioned visual workcell safety equipment (simulation only).

These shapes are illustrative; they are not physical collision guards.
"""
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder


class SafetyEquipment:
    def __init__(self, env, kawasaki_base):
        self.env = env
        self.objects = []

        def box(size, xyz, color):
            obj = Cuboid(scale=size, pose=SE3(*xyz), color=color)
            env.add(obj)
            self.objects.append(obj)
            return obj

        yellow = [0.97, 0.76, 0.08, 1]
        charcoal = [0.11, 0.13, 0.15, 1]
        steel = [0.37, 0.43, 0.47, 1]
        amber = [1.0, 0.63, 0.03, 1]
        red = [0.92, 0.08, 0.09, 1]
        green = [0.06, 0.85, 0.20, 1]
        dark_amber = [0.28, 0.16, 0.04, 1]
        dark_red = [0.32, 0.075, 0.075, 1]
        dark_green = [0.055, 0.26, 0.10, 1]
        floor_z = 0.008

        # Restrained rail boundary: continuous thin lines, not 44 alternating
        # blocks. The inner rail and the medication transfer remain clear.
        for y in (0.41, 1.11):
            box([3.72, 0.038, 0.007], (1.70, y, floor_z), yellow)
        # Only four high-contrast black end caps mark the boundary endpoints.
        for x in (-0.15, 3.55):
            for y in (0.41, 1.11):
                box([0.11, 0.055, 0.008], (x, y, floor_z + 0.002), charcoal)

        # Kawasaki: three-sided U boundary (open towards handoff/rail).
        # Avoid the checkerboard around the entire workbench.
        kx, ky = float(kawasaki_base.t[0]), float(kawasaki_base.t[1])
        box([1.55, 0.038, 0.008], (kx, ky - 0.79, floor_z), yellow)
        for x in (kx - 0.78, kx + 0.78):
            box([0.038, 0.91, 0.008], (x, ky - 0.34, floor_z), yellow)
            box([0.085, 0.08, 0.009], (x, ky - 0.79, floor_z + 0.002), charcoal)

        # Compact two-post rail-end barriers, with dark frames and a single
        # high-visibility yellow crossbar; keep clear of the rail endpoints.
        for bx in (-0.43, 3.86):
            for by in (0.40, 1.10):
                box([0.08, 0.08, 0.73], (bx, by, 0.365), charcoal)
                box([0.16, 0.16, 0.025], (bx, by, 0.014), steel)
                box([0.09, 0.09, 0.075], (bx, by, 0.695), yellow)
            box([0.055, 0.74, 0.075], (bx, 0.75, 0.54), yellow)

        # Recognisable stack-light tower: housing + RED upper lens and GREEN
        # lower lens. Both lenses remain visible; only the active one is bright.
        bx, by = 3.95, 0.25
        box([0.13, 0.13, 0.035], (bx, by, 0.018), charcoal)
        box([0.048, 0.048, 0.96], (bx, by, 0.48), steel)
        box([0.15, 0.15, 0.035], (bx, by, 0.98), charcoal)
        for z in (1.06, 1.21, 1.36):
            box([0.16, 0.16, 0.025], (bx, by, z - 0.075), charcoal)
            box([0.16, 0.16, 0.025], (bx, by, z + 0.075), charcoal)

        self._lamps = {}
        for name, z, bright, dim in (
            ('green', 1.06, green, dark_green),
            ('amber', 1.21, amber, dark_amber),
            ('red', 1.36, red, dark_red),
        ):
            self._lamps[name] = (
                Cylinder(radius=0.069, length=0.125,
                         pose=SE3(bx, by, -3), color=bright),
                Cylinder(radius=0.069, length=0.125,
                         pose=SE3(bx, by, -3), color=dim),
                SE3(bx, by, z),
            )
            for lamp in self._lamps[name][:2]:
                env.add(lamp)
                self.objects.append(lamp)

        self._hidden = SE3(bx, by, -3)
        self._last = None
        self.update('READY')


    def update(self, state):
        """Green=safe, amber=operating, red=stop/fault."""
        if state in ("E-STOPPED", "FAULT"):
            active = "red"
        elif state == "RUNNING":
            active = "amber"
        else:
            active = "green"

        if active == self._last:
            return

        self._last = active

        for name, (bright, dim, pose) in self._lamps.items():
            bright.T = pose if name == active else self._hidden
            dim.T = self._hidden if name == active else pose
