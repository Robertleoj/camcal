"""Generate the pinhole vs real lens projection diagram for the calibration guide."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

bg = "#111111"
fg = "white"
cmap = plt.colormaps["jet"]


def make_grid_lines(n_lines: int = 9, n_pts: int = 200) -> list[np.ndarray]:
    """Return list of (n_pts, 2) arrays, one per grid line."""
    t = np.linspace(-1, 1, n_lines)
    fine = np.linspace(-1, 1, n_pts)
    lines = []
    for y in t:
        lines.append(np.column_stack([fine, np.full(n_pts, y)]))
    for x in t:
        lines.append(np.column_stack([np.full(n_pts, x), fine]))
    return lines


def barrel_distort(
    xs: np.ndarray, ys: np.ndarray, k1: float = -0.3, k2: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    r2 = xs**2 + ys**2
    factor = 1 + k1 * r2 + k2 * r2**2
    return xs * factor, ys * factor


def colored_segments_2d(lines: list[np.ndarray], lines_for_color: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    all_segs = []
    all_colors = []
    if lines_for_color is None:
        lines_for_color = lines
    for line, color_line in zip(lines, lines_for_color):
        segs = np.stack([line[:-1], line[1:]], axis=1)
        diag = (color_line[:-1, 0] + color_line[:-1, 1] + 2) / 4 * 0.85 + 0.05
        all_segs.append(segs)
        all_colors.append(cmap(diag))
    return np.concatenate(all_segs), np.concatenate(all_colors)


def colored_segments_3d(lines_3d: list[np.ndarray], lines_2d_for_color: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    all_segs = []
    all_colors = []
    for line3d, line2d in zip(lines_3d, lines_2d_for_color):
        segs = np.stack([line3d[:-1], line3d[1:]], axis=1)
        diag = (line2d[:-1, 0] + line2d[:-1, 1] + 2) / 4 * 0.85 + 0.05
        all_segs.append(segs)
        all_colors.append(cmap(diag))
    return np.concatenate(all_segs), np.concatenate(all_colors)


# --- Build figure ---
fig = plt.figure(figsize=(12, 5))
fig.patch.set_facecolor(bg)

# Left: 3D view
ax3d = fig.add_subplot(121, projection="3d")
ax3d.set_facecolor(bg)
for axis in [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]:
    axis.pane.set_facecolor(bg)
    axis.pane.set_edgecolor(bg)
    axis.set_ticks([])
    axis.line.set_color(bg)
ax3d.set_axis_off()

# Camera looking along +Y, pulled back so it doesn't overlap the grid
cam_y = -2.0
cam_size = 0.5
cam_corners = np.array([
    [-cam_size, cam_y + cam_size * 2.5, -cam_size * 0.7],
    [cam_size, cam_y + cam_size * 2.5, -cam_size * 0.7],
    [cam_size, cam_y + cam_size * 2.5, cam_size * 0.7],
    [-cam_size, cam_y + cam_size * 2.5, cam_size * 0.7],
])
cam_apex = np.array([0, cam_y, 0])

# Camera body edges
for i in range(4):
    j = (i + 1) % 4
    ax3d.plot3D(*zip(cam_corners[i], cam_corners[j]), color=fg, linewidth=1.5)
    ax3d.plot3D(*zip(cam_apex, cam_corners[i]), color=fg, linewidth=1.0, alpha=0.6)

# Grid in 3D at y=grid_dist, in the XZ plane
grid_dist = 2.5
grid_scale = 1.3
grid_lines = make_grid_lines()
grid_lines_3d = []
for line in grid_lines:
    x3 = line[:, 0] * grid_scale
    y3 = np.full_like(x3, grid_dist)
    z3 = line[:, 1] * grid_scale
    grid_lines_3d.append(np.column_stack([x3, y3, z3]))

segs3d, colors3d = colored_segments_3d(grid_lines_3d, grid_lines)
ax3d.add_collection(Line3DCollection(segs3d, colors=colors3d, linewidths=1.2))

# A few faint "ray" lines from camera apex to grid corners
ray_corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
for rx, ry in ray_corners:
    ax3d.plot3D(
        [0, rx * grid_scale], [cam_y, grid_dist], [0, ry * grid_scale],
        color=fg, linewidth=0.5, alpha=0.25, linestyle="--",
    )

ax3d.set_xlim(-grid_scale * 1.1, grid_scale * 1.1)
ax3d.set_ylim(cam_y - 0.5, grid_dist + 0.5)
ax3d.set_zlim(-grid_scale * 1.1, grid_scale * 1.1)
ax3d.view_init(elev=30, azim=-50)
ax3d.set_proj_type("persp", focal_length=0.35)

# Right: distorted projection (what the camera sees)
ax2d = fig.add_subplot(122)
ax2d.set_facecolor(bg)
ax2d.set_aspect("equal")
ax2d.set_xlim(-1.15, 1.15)
ax2d.set_ylim(-1.15, 1.15)
ax2d.set_xticks([])
ax2d.set_yticks([])
for spine in ax2d.spines.values():
    spine.set_color(fg)
    spine.set_linewidth(1.5)

distorted_lines = []
for line in grid_lines:
    dx, dy = barrel_distort(line[:, 0], line[:, 1])
    distorted_lines.append(np.column_stack([dx, dy]))

segs2d, colors2d = colored_segments_2d(distorted_lines, grid_lines)
ax2d.add_collection(LineCollection(segs2d, colors=colors2d, linewidths=1.5))

# Arrow between panels
fig.text(0.50, 0.50, "→", fontsize=36, color=fg, ha="center", va="center", fontweight="bold")

ax3d.set_position([-0.15, -0.15, 0.7, 1.25])
ax2d.set_position([0.55, 0.1, 0.42, 0.78])
fig.savefig("docs/media/calibration_docs/projection_diagram.png", dpi=200, facecolor=bg)
print("Saved projection_diagram.png")
