"""
layout.py

Where everything in the MediAssist cell is.

Everything is a PLACEHOLDER for now (metres, world frame, z up).
Will tweak once the scene is built and I can run a reach check.
"""

from spatialmath import SE3

# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
BENCH_HEIGHT = 0.75     # top of the storage bench and transfer table

# ----------------------------------------------------------------------
# Robot 1 - Kawasaki RS007N, bolted to the storage bench
# ----------------------------------------------------------------------
KAWASAKI_BASE = SE3(0.0, 0.0, BENCH_HEIGHT)

# One slot per colour. Dict so the GUI can look a slot up by name
# e.g. MEDICATION_SLOTS["red"]
MEDICATION_SLOTS = {
    "red":   SE3(0.45, -0.20, BENCH_HEIGHT),
    "green": SE3(0.45,  0.00, BENCH_HEIGHT),
    "blue":  SE3(0.45,  0.20, BENCH_HEIGHT),
}

# ----------------------------------------------------------------------
# Transfer point - BOTH robots have to reach this
# ----------------------------------------------------------------------
TRANSFER_POINT = SE3(0.0, 0.55, BENCH_HEIGHT)

# ----------------------------------------------------------------------
# Robot 2 - Doosan M0609 on a linear rail running along the ward (+x)
# ----------------------------------------------------------------------
RAIL_START = SE3(0.0, 1.10, 0.0)
RAIL_LENGTH = 3.0       # how far along x the Doosan can slide

# One drop off point per bed, spaced out along the rail
BEDS = {
    "bed_1": SE3(0.5, 1.80, 0.70),
    "bed_2": SE3(1.5, 1.80, 0.70),
    "bed_3": SE3(2.5, 1.80, 0.70),
}