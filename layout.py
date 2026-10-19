"""
layout.py

Where everything in the MediAssist cell is.

All values match environment/ward.py (metres, world frame, z up).
Coordinate system:
    X = left/right along the rail
    Y = from the medication station towards the beds
    Z = height
If something moves in ward.py, update it here too.
"""

from spatialmath import SE3

# ----------------------------------------------------------------------
# Medication containers (dummy cylinders)
# ----------------------------------------------------------------------
CONTAINER_RADIUS = 0.045
CONTAINER_HEIGHT = 0.11
HALF_CONTAINER = CONTAINER_HEIGHT / 2

# ----------------------------------------------------------------------
# Surfaces (top face heights)
# ----------------------------------------------------------------------
WORKSTATION_TOP = 0.77      # grey medication workstation
SHELF_TOP = 0.87            # medication storage shelf
MOUNT_PLATE_TOP = 0.81      # Kawasaki mounting plate
TRANSFER_PLATE_TOP = 0.83   # transfer plate
BEDSIDE_TABLE_TOP = 0.72    # all three bedside tables

# ----------------------------------------------------------------------
# Robot 1 - Kawasaki RS007N, bolted to the mounting plate
# ----------------------------------------------------------------------
KAWASAKI_BASE = SE3(1.65, -0.27, MOUNT_PLATE_TOP)

# Container centre poses on the shelf, grouped by colour.
# Shelf order left to right: blue, green, red, blue, green, red
# e.g. MEDICATION_SLOTS["red"][0] is the first red container
SHELF_Y = -0.72
SHELF_X_START = 1.30
SHELF_SPACING = 0.14
SHELF_ORDER = ["blue", "green", "red", "blue", "green", "red"]

MEDICATION_SLOTS = {"blue": [], "green": [], "red": []}
for i, colour in enumerate(SHELF_ORDER):
    x = SHELF_X_START + i * SHELF_SPACING
    MEDICATION_SLOTS[colour].append(
        SE3(x, SHELF_Y, SHELF_TOP + HALF_CONTAINER)
    )

# ----------------------------------------------------------------------
# Transfer point - BOTH robots have to reach this
# Container centre pose when it is sitting on the transfer plate
# ----------------------------------------------------------------------
TRANSFER_POINT = SE3(1.65, 0.12, TRANSFER_PLATE_TOP + HALF_CONTAINER)

# ----------------------------------------------------------------------
# Robot 2 - Doosan M0609 on a linear rail running along the ward (+x)
# ----------------------------------------------------------------------
RAIL_START_X = -0.05
RAIL_LENGTH = 3.50
RAIL_Y = 0.75
RAIL_HEIGHT = 0.65

# Carriage centre (Doosan will sit on top of this)
DOOSAN_CARRIAGE = SE3(1.65, RAIL_Y, RAIL_HEIGHT + 0.25)

# ----------------------------------------------------------------------
# Beds and bedside drop off points
# Container centre pose when placed on each bedside table
# ----------------------------------------------------------------------
BED_X_POSITIONS = [0.25, 1.65, 3.05]
BED_Y = 1.95

BEDS = {}
for i, bed_x in enumerate(BED_X_POSITIONS, start=1):
    BEDS[f"bed_{i}"] = SE3(
        bed_x + 0.61,
        BED_Y - 0.48,
        BEDSIDE_TABLE_TOP + HALF_CONTAINER,
    )