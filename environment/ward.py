"""MediAssist ward - environment only, no robots yet."""

import time
import swift
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder


def box(env, name, pos, size, colour):
    obj = Cuboid(
        pose=SE3(*pos),
        scale=list(size),
        color=list(colour),
    )
    env.add(obj)
    print("Added:", name)
    return obj


def cylinder(env, name, pos, colour):
    obj = Cylinder(
        radius=0.045,
        length=0.11,
        pose=SE3(*pos),
        color=list(colour),
    )
    env.add(obj)
    print("Added:", name)
    return obj


def build_ward(env):
    # Coordinate system:
    # X = left/right along rail
    # Y = from medication station towards beds
    # Z = height

    # Floor
    box(env, "Ward floor",
        (1.55, 1.20, -0.06),
        (5.1, 4.8, 0.10),
        (0.65, 0.82, 0.91, 1))

    # MEDICATION WORKSTATION - top of diagram
    # Large grey work surface
    box(env, "Medication workstation",
        (1.65, -0.29, 0.72),
        (1.25, 1.22, 0.10),
        (0.70, 0.73, 0.77, 1))

    # Workstation support
    box(env, "Workstation pedestal",
        (1.65, -0.35, 0.33),
        (0.95, 0.85, 0.66),
        (0.43, 0.47, 0.53, 1))

    # Medication storage shelf at far end
    box(env, "Medication storage shelf",
        (1.65, -0.72, 0.82),
        (0.90, 0.24, 0.10),
        (0.68, 0.40, 0.18, 1))

    colours = [
        (0.10, 0.40, 0.95, 1),
        (0.10, 0.75, 0.25, 1),
        (0.95, 0.12, 0.12, 1),
        (0.10, 0.40, 0.95, 1),
        (0.10, 0.75, 0.25, 1),
        (0.95, 0.12, 0.12, 1),
    ]

    containers = []
    for i, colour in enumerate(colours):
        x = 1.30 + i * 0.14
        containers.append(cylinder(
            env, f"Dummy container {i+1}",
            (x, -0.72, 0.925),
            colour
        ))

    # Kawasaki mounting area - empty for now
    box(env, "Kawasaki mounting plate",
        (1.65, -0.27, 0.79),
        (0.27, 0.27, 0.04),
        (0.32, 0.36, 0.41, 1))

    # Transfer plate - closest to Doosan rail
    box(env, "Transfer plate",
        (1.65, 0.12, 0.81),
        (0.42, 0.27, 0.04),
        (0.91, 0.67, 0.25, 1))

    # DOOSAN LINEAR RAIL - middle of diagram
    rail_start = -0.05
    rail_length = 3.50
    rail_y = 0.75
    rail_height = 0.65

    box(env, "Rail body",
        (rail_start + rail_length / 2, rail_y, rail_height + 0.09),
        (rail_length, 0.28, 0.18),
        (0.20, 0.25, 0.32, 1))

    for offset in (-0.09, 0.09):
        box(env, "Silver rail guide",
            (rail_start + rail_length / 2,
             rail_y + offset, rail_height + 0.195),
            (rail_length, 0.035, 0.025),
            (0.78, 0.82, 0.87, 1))


    # Elevated support wall beneath the rail
    box(env, "Rail support wall",
        (rail_start + rail_length / 2, rail_y, rail_height / 2),
        (rail_length - 0.15, 0.22, rail_height),
        (0.52, 0.56, 0.62, 1))

    # Empty carriage for future Doosan
    box(env, "Doosan carriage",
        (1.65, rail_y, rail_height + 0.25),
        (0.38, 0.40, 0.12),
        (0.48, 0.53, 0.60, 1))

    # THREE BEDS - bottom of diagram
    # Spacing gives each bed its own bedside table.
    bed_x_positions = [0.25, 1.65, 3.05]
    bed_y = 1.95

    for i, x in enumerate(bed_x_positions, start=1):
        # Frame
        box(env, f"Bed {i} frame",
            (x, bed_y, 0.31),
            (0.78, 1.55, 0.13),
            (0.38, 0.43, 0.49, 1))

        # Four legs supporting each bed
        for dx in (-0.30, 0.30):
            for dy in (-0.62, 0.62):
                box(env, f"Bed {i} leg",
                    (x + dx, bed_y + dy, 0.125),
                    (0.07, 0.07, 0.25),
                    (0.42, 0.47, 0.53, 1))

        # Mattress
        box(env, f"Bed {i} mattress",
            (x, bed_y, 0.43),
            (0.73, 1.48, 0.13),
            (0.86, 0.93, 0.98, 1))

        # Pillow at the head, nearest the rail
        box(env, f"Bed {i} pillow",
            (x, bed_y - 0.52, 0.52),
            (0.51, 0.24, 0.07),
            (0.98, 0.98, 0.98, 1))

        # Bedside table to LEFT of bed, near pillow
        # Exactly as shown in the concept image
        table_x = x + 0.61
        table_y = bed_y - 0.48

        box(env, f"Bed {i} bedside table",
            (table_x, table_y, 0.69),
            (0.37, 0.38, 0.06),
            (0.65, 0.40, 0.20, 1))

        box(env, f"Bed {i} table support",
            (table_x, table_y, 0.33),
            (0.13, 0.13, 0.66),
            (0.48, 0.50, 0.54, 1))

    print("MediAssist ward environment ready.")
    return containers


def main():
    env = swift.Swift()
    env.launch(realtime=True)
    build_ward(env)

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