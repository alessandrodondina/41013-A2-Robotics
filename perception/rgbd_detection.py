"""
rgbd_detection.py
=================
Colour + 3D position detection of medication containers from an RGB-D
frame, and a simulated RGB-D camera to feed it inside the Swift ward.
"""

import numpy as np
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage
from spatialmath import SE3

import layout

# Hue windows (0-1) for each medication colour. Red wraps around 0.
COLOUR_HUES = {
    "red":   [(0.00, 0.03), (0.95, 1.00)],
    "green": [(0.25, 0.45)],
    "blue":  [(0.55, 0.72)],
}
MIN_SATURATION = 0.5    # rejects the grey bench and pale floor
MIN_VALUE = 0.25        # rejects pixels in deep shadow
MIN_BLOB_PIXELS = 40    # anything smaller is noise, not a container
TOP_FACE_BAND = 0.012   # points within 12 mm of the highest = lid


class Detection:
    def __init__(self, colour, position, pixels, centroid_px):
        self.colour = colour
        self.position = position          # container centre, world (m)
        self.pixels = pixels
        self.centroid_px = centroid_px    # (u, v) in the image

    def __repr__(self):
        p = np.round(self.position, 3)
        return f"Detection({self.colour}, centre={p}, {self.pixels} px)"


# ======================================================================
# CAMERA MODEL
# ======================================================================
class PinholeCamera:
    """Intrinsics + extrinsics. Camera frame: x right, y down, z forward."""

    def __init__(self, T_world_cam, width=320, height=240, hfov_deg=55):
        self.T = SE3(T_world_cam)
        self.width, self.height = width, height
        self.fx = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
        self.fy = self.fx
        self.cx, self.cy = width / 2, height / 2

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]])

    def pixel_to_camera(self, u, v, depth):
        """Back-project: pixel + depth -> 3D point(s) in the camera frame."""
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.stack([x, y, depth], axis=-1)

    def camera_to_world(self, pts):
        """Nx3 camera-frame points -> Nx3 world-frame points."""
        R, t = self.T.R, self.T.t
        return pts @ R.T + t


# ======================================================================
# DETECTOR
# ======================================================================
def detect_containers(rgb, depth, camera, colours=COLOUR_HUES):
    """Find every coloured container in an RGB-D frame.

    rgb   HxWx3 floats 0-1     depth  HxW metres (0 or nan = no reading)
    Returns a list of Detection, left to right along world x.
    """
    hsv = rgb_to_hsv(np.clip(rgb, 0, 1))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s > MIN_SATURATION) & (v > MIN_VALUE) & np.isfinite(depth) \
        & (depth > 0)

    detections = []
    for name, windows in colours.items():
        mask = np.zeros_like(valid)
        for lo, hi in windows:
            mask |= (h >= lo) & (h <= hi)
        mask &= valid
        # Clean up speckle, then label each separate blob
        mask = ndimage.binary_opening(mask, iterations=1)
        labels, n = ndimage.label(mask)

        for k in range(1, n + 1):
            vs, us = np.nonzero(labels == k)
            if len(us) < MIN_BLOB_PIXELS:
                continue
            pts_cam = camera.pixel_to_camera(us, vs, depth[vs, us])
            pts = camera.camera_to_world(pts_cam)

            # The lid is the flat top face: take the highest points.
            # Its centroid is directly above the container's centre.
            top = pts[pts[:, 2] > pts[:, 2].max() - TOP_FACE_BAND]
            lid_z = np.median(top[:, 2])
            centre = np.array([top[:, 0].mean(), top[:, 1].mean(),
                               lid_z - layout.CONTAINER_HEIGHT / 2])
            detections.append(Detection(name, centre, len(us),
                                        (us.mean(), vs.mean())))

    detections.sort(key=lambda d: d.position[0])
    return detections


# ======================================================================
# SIMULATED RGB-D CAMERA (stands in for the real one in simulation)
# ======================================================================
class SimulatedRGBD:
    """Ray casts the shelf area from the camera pose to make RGB + depth."""

    DEPTH_NOISE = 0.002      # 2 mm, roughly a RealSense at this range

    def __init__(self, camera, containers, seed=0):
        """containers: list of Swift cylinder objects (their live poses
        and colours are read on every capture, so moved ones disappear
        from the shelf view)."""
        self.camera = camera
        self.containers = containers
        self.rng = np.random.default_rng(seed)

        # Static boxes the camera can see: (min corner, max corner, colour)
        st, ws = layout.SHELF_TOP, layout.WORKSTATION_TOP
        self.boxes = [
            ((1.20, -0.84, st - 0.10), (2.10, -0.60, st),
             (0.68, 0.40, 0.18)),                       # shelf
            ((1.025, -0.90, ws - 0.10), (2.275, 0.32, ws),
             (0.70, 0.73, 0.77)),                       # bench
        ]
        self.background = np.array([0.65, 0.82, 0.91])  # floor / far

    def _rays(self):
        cam = self.camera
        u, v = np.meshgrid(np.arange(cam.width) + 0.5,
                           np.arange(cam.height) + 0.5)
        d_cam = np.stack([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy,
                          np.ones_like(u)], axis=-1)        # z = 1
        d_world = d_cam @ cam.T.R.T
        return d_world, d_cam

    def capture(self):
        cam = self.camera
        o = cam.T.t
        d, d_cam = self._rays()
        H, W = cam.height, cam.width
        t_best = np.full((H, W), np.inf)
        rgb = np.tile(self.background, (H, W, 1)).astype(float)
        light = np.array([0.3, -0.4, 0.87])
        light /= np.linalg.norm(light)

        def shade(colour, normal_dot_light):
            return np.asarray(colour)[None, :] * \
                (0.55 + 0.45 * np.clip(normal_dot_light, 0, 1))[:, None]

        # --- boxes (slab method), only their top faces matter here
        for lo, hi, col in self.boxes:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (np.array(lo) - o) / d
                t2 = (np.array(hi) - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
            hit = (tmax >= tmin) & (tmin > 0) & (tmin < t_best)
            t_best[hit] = tmin[hit]
            rgb[hit] = shade(col, np.full(hit.sum(), light[2]))

        # --- vertical cylinders: side wall + lid
        r = layout.CONTAINER_RADIUS
        hh = layout.CONTAINER_HEIGHT / 2
        for obj in self.containers:
            c = SE3(obj.T).t
            col = np.asarray(obj.color[:3])
            # side wall: |(o + t d - c)_xy| = r
            ox, oy = o[0] - c[0], o[1] - c[1]
            dx, dy = d[..., 0], d[..., 1]
            a = dx**2 + dy**2
            b = 2 * (ox * dx + oy * dy)
            cc = ox**2 + oy**2 - r**2
            disc = b**2 - 4 * a * cc
            with np.errstate(invalid="ignore", divide="ignore"):
                ts = (-b - np.sqrt(disc)) / (2 * a)
            zs = o[2] + ts * d[..., 2]
            side = (disc > 0) & (ts > 0) & (np.abs(zs - c[2]) <= hh) \
                & (ts < t_best)
            if side.any():
                px = o[0] + ts[side] * dx[side] - c[0]
                py = o[1] + ts[side] * dy[side] - c[1]
                n = np.stack([px, py, np.zeros_like(px)], -1) / r
                t_best[side] = ts[side]
                rgb[side] = shade(col, n @ light)
            # lid: plane z = c_z + hh, inside radius
            with np.errstate(divide="ignore", invalid="ignore"):
                tl = (c[2] + hh - o[2]) / d[..., 2]
            lx = o[0] + tl * dx - c[0]
            ly = o[1] + tl * dy - c[1]
            lid = (tl > 0) & (lx**2 + ly**2 <= r**2) & (tl < t_best)
            t_best[lid] = tl[lid]
            rgb[lid] = shade(col, np.full(lid.sum(), light[2]))

        # Depth = distance along the camera's z axis (what RGB-D reports)
        depth = t_best * d_cam[..., 2]
        depth[~np.isfinite(depth)] = 0.0
        hitmask = depth > 0
        depth[hitmask] += self.rng.normal(0, self.DEPTH_NOISE,
                                          hitmask.sum())
        return rgb, depth


def save_debug_image(path, rgb, depth, detections):
    """RGB with detections marked, next to the depth image - for the
    demo / video, so people can see what the camera 'saw'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].imshow(rgb)
    ax[0].set_title("RGB + detections")
    for i, det in enumerate(detections):
        u, v = det.centroid_px
        x, y, _ = det.position
        ax[0].plot(u, v, "w+", ms=12, mew=2)
        up = i % 2 == 0          # alternate labels above / below
        ax[0].text(u, v + (-22 if up else 22),
                   f"{det.colour}\n({x:.3f}, {y:.3f})",
                   color="white", fontsize=7, ha="center",
                   va="bottom" if up else "top",
                   bbox=dict(facecolor="black", alpha=0.6, pad=1))
    d = np.where(depth > 0, depth, np.nan)
    im = ax[1].imshow(d, cmap="viridis")
    ax[1].set_title("Depth (m)")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
