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
        red = [0.92, 0.08, 0.09, 1]
        green = [0.06, 0.85, 0.20, 1]
        dark_red = [0.32, 0.075, 0.075, 1]
        dark_green = [0.055, 0.26, 0.10, 1]
        floor_z = 0.008

        # Two simulated scanner-monitored rail-end access boundaries.
        # Yellow tape indicates where people must not cross.
        # These are visual boundaries, not physical safety devices.

        scanner_positions = (
            (-0.18, -0.43),
            (3.58, 3.86),
        )

        for sx, boundary_x in scanner_positions:
            # Single yellow no-crossing line at each rail end.
            box(
                [0.025, 1.30, 0.008],
                (boundary_x, 0.75, floor_z),
                yellow
            )

            # Scanner mounted close to the rail base.
            box(
                [0.15, 0.16, 0.06],
                (sx, 0.75, 0.035),
                charcoal
            )
            box(
                [0.12, 0.13, 0.12],
                (sx, 0.75, 0.12),
                yellow
            )

            # Outward-facing sensor window.
            lens_x = sx + (-0.065 if sx < 0 else 0.065)

            box(
                [0.028, 0.10, 0.055],
                (lens_x, 0.75, 0.14),
                [0.10, 0.42, 0.65, 1]
            )

        # Three-colour warning beacon, positioned away from rail scanners.
        bx, by = 4.25, -0.25

        box([0.13, 0.13, 0.035], (bx, by, 0.018), charcoal)
        box([0.048, 0.048, 0.96], (bx, by, 0.48), steel)
        box([0.15, 0.15, 0.035], (bx, by, 0.98), charcoal)

        lamp_positions = (
            ('green', 1.06, green, dark_green),
            ('amber', 1.21, [1.0, 0.65, 0.05, 1],
             [0.30, 0.20, 0.04, 1]),
            ('red', 1.36, red, dark_red),
        )

        self._lamps = {}

        for name, z, bright, dim in lamp_positions:
            box([0.16, 0.16, 0.025],
                (bx, by, z - 0.075), charcoal)
            box([0.16, 0.16, 0.025],
                (bx, by, z + 0.075), charcoal)

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
        """Update the beacon according to the safety controller state."""
        if state in ('E-STOPPED', 'FAULT', 'ZONE BREACH',
                     'SCANNER BREACH'):
            active = 'red'
        elif state == 'RUNNING':
            active = 'amber'
        else:
            active = 'green'

        if active == self._last:
            return

        self._last = active

        for name, (bright, dim, pose) in self._lamps.items():
            bright.T = pose if name == active else self._hidden
            dim.T = self._hidden if name == active else pose
