from __future__ import annotations

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar

import lensboy as lb
from lensboy.analysis.image import to_color


class Color:
    """Container for some common colors."""

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)


def _draw_points(
    img: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = Color.green,
    r: int = 4,
    thickness: int = -1,
) -> np.ndarray:
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), r, color, thickness)
    return img


def draw_points_in_image(
    points_in_image: np.ndarray,
    *,
    image: np.ndarray | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    color: tuple[int, int, int] = Color.green,
    r: int = 4,
) -> np.ndarray:
    """Draw 2D points onto an image.

    If no image is provided, draws on a blank (black) canvas whose size
    is given by ``image_width`` and ``image_height``.

    Args:
        points_in_image: Pixel coordinates to draw, shape (N, 2).
        image: Optional BGR image to draw on (will be copied).
        image_width: Canvas width when ``image`` is None.
        image_height: Canvas height when ``image`` is None.
        color: BGR circle colour.
        r: Circle radius in pixels.

    Returns:
        BGR image with points drawn, shape (H, W, 3).
    """
    if image is not None:
        canvas = to_color(image.copy())
    else:
        if image_width is None or image_height is None:
            raise ValueError(
                "image_width and image_height are required when image is None"
            )
        canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    return _draw_points(canvas, points_in_image, color=color, r=r)


def plot_detection_coverage(
    detections: list[lb.Frame],
    *,
    image_width: int,
    image_height: int,
    title: str = "Coverage",
    s: float = 6.0,
    grid_cells: int = 30,
) -> None:
    """Scatter-plot all detected points with a smoothed coverage heatmap.

    Shows a density heatmap behind the scatter points so that regions with
    poor coverage are immediately visible as dark patches.

    Args:
        detections: Frames containing detected calibration points.
        image_width: Sensor width in pixels, sets the x-axis limit.
        image_height: Sensor height in pixels, sets the y-axis limit.
        title: Plot title.
        s: Marker size passed to ``ax.scatter``.
        grid_cells: Number of grid cells along the longer image axis for the heatmap.
    """
    pts_list = []
    for d in detections:
        if d.detected_points_in_image is None:
            continue
        p = np.asarray(d.detected_points_in_image, dtype=float)
        if p.size == 0:
            continue
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError(f"detected_points_in_image must be (K,2), got {p.shape}")
        pts_list.append(p)

    pts = np.concatenate(pts_list, axis=0) if pts_list else np.empty((0, 2), dtype=float)

    bg = "#111111"
    fg = "white"
    accent = "#00d4ff"

    fig, ax = plt.subplots(figsize=(20, 15))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    if pts.shape[0] > 0:
        aspect = image_width / image_height
        if aspect >= 1:
            nx = grid_cells
            ny = max(1, int(round(grid_cells / aspect)))
        else:
            ny = grid_cells
            nx = max(1, int(round(grid_cells * aspect)))

        counts, _, _ = np.histogram2d(
            pts[:, 0],
            pts[:, 1],
            bins=[nx, ny],
            range=[[0, image_width], [0, image_height]],
        )
        # counts is (nx, ny) with x along axis 0 — transpose so y is rows
        ax.imshow(
            counts.T,
            extent=[0, image_width, image_height, 0],  # type: ignore
            cmap="inferno",
            interpolation="gaussian",
            aspect="auto",
        )
        ax.scatter(pts[:, 0], pts[:, 1], s=s, color=accent)

    ax.set_title(f"{title}  (N={pts.shape[0]})")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)

    ax.set_aspect("equal", adjustable="box")
    plt.show()


def plot_distortion_grid(
    model: lb.OpenCV | lb.PinholeSplined,
    *,
    grid_step_norm: float = 0.05,
    fov_fraction: float | None = None,
    ux_max: float | None = None,
    uy_max: float | None = None,
    cmap_name: str = "jet",
    show_spline_knots: bool = False,
) -> None:
    """Project a regular grid through a camera model to visualize distortion.

    Builds a grid in normalized (tan-angle) space from the model's FOV, projects
    it, and clips to the image bounds.

    Args:
        model: Camera model instance.
        grid_step_norm: Spacing between grid lines in normalized coordinates.
        fov_fraction: Fraction of the full FOV to sample (0, 1].
        ux_max: Upper bound in normalized x, mirrored to negative.
        uy_max: Upper bound in normalized y, mirrored to negative.
        cmap_name: Matplotlib colormap name.
        show_spline_knots: When True and the model is a PinholeSplined,
            overlay the spline control points on both panels.
    """

    W = int(model.image_width)
    H = int(model.image_height)

    cx = model.cx
    cy = model.cy

    fov_x_half = np.tan(np.deg2rad(model.fov_deg_x) / 2.0)
    fov_y_half = np.tan(np.deg2rad(model.fov_deg_y) / 2.0)

    if fov_fraction is not None:
        x_half = fov_x_half * fov_fraction
        y_half = fov_y_half * fov_fraction
    elif ux_max is not None or uy_max is not None:
        x_half = ux_max if ux_max is not None else fov_x_half
        y_half = uy_max if uy_max is not None else fov_y_half
    else:
        x_half = fov_x_half
        y_half = fov_y_half

    if ux_max is not None and fov_fraction is not None:
        x_half = min(x_half, ux_max)
    if uy_max is not None and fov_fraction is not None:
        y_half = min(y_half, uy_max)

    x_min, x_max = -x_half, +x_half
    y_min, y_max = -y_half, +y_half

    cmap = plt.colormaps[cmap_name]

    def nice_ticks(lo, hi, step):
        start = np.floor(lo / step) * step
        end = np.ceil(hi / step) * step
        return np.arange(start, end + step * 0.5, step)

    x_lines = nice_ticks(x_min, x_max, grid_step_norm)
    y_lines = nice_ticks(y_min, y_max, grid_step_norm)

    def project_polyline(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
        pts = np.stack([xn, yn, np.ones_like(xn)], axis=1)  # (N,3)
        uv = model.project_points(pts)
        return np.asarray(uv, dtype=float)

    x_range = x_max - x_min
    y_range = y_max - y_min

    def make_segments(
        xs: np.ndarray,
        ys: np.ndarray,
        src_xs: np.ndarray,
        src_ys: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build LineCollection segments with diagonal-gradient colors.

        Args:
            xs: x coordinates to draw, shape (N,).
            ys: y coordinates to draw, shape (N,).
            src_xs: Source normalized x for coloring, shape (N,).
            src_ys: Source normalized y for coloring, shape (N,).

        Returns:
            Segments array of shape (N-1, 2, 2) and colors array of shape (N-1, 4).
        """
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        mid_sx = (src_xs[:-1] + src_xs[1:]) / 2.0
        mid_sy = (src_ys[:-1] + src_ys[1:]) / 2.0
        diag = ((mid_sx - x_min) / x_range + (mid_sy - y_min) / y_range) / 2.0
        colors = cmap(diag)
        return segs, colors

    panel_w = 10
    panel_h = panel_w * (H / W)
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(2 * panel_w, panel_h), constrained_layout=True
    )

    fig.patch.set_facecolor("#111111")

    for ax in [ax0, ax1]:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # --- Left panel: grid in normalized space ---
    lw = 1.0
    n_subdivide = 64
    for x0 in x_lines:
        ys = np.linspace(y_min, y_max, n_subdivide)
        xs = np.full_like(ys, x0)
        segs, colors = make_segments(xs, ys, xs, ys)
        ax0.add_collection(LineCollection(segs, colors=colors, linewidths=lw))  # type: ignore[reportArgumentType]
    for y0 in y_lines:
        xs = np.linspace(x_min, x_max, n_subdivide)
        ys = np.full_like(xs, y0)
        segs, colors = make_segments(xs, ys, xs, ys)
        ax0.add_collection(LineCollection(segs, colors=colors, linewidths=lw))  # type: ignore[reportArgumentType]

    # Rectangle corner markers in normalized space
    rect_step = grid_step_norm * 4
    corner_len = grid_step_norm * 2
    norm_aspect = x_half / y_half
    n_rects = max(1, int(min(x_half, y_half) / rect_step))
    rect_lw = 2.5
    rect_border_lw = rect_lw * 3
    for i in range(1, n_rects + 1):
        rh = i * rect_step
        rw = round(rh * norm_aspect / grid_step_norm) * grid_step_norm
        rx1, ry1 = -rw, -rh
        rx2, ry2 = rw, rh
        for rcx, rcy, sx, sy in [
            (rx1, ry1, 1, 1),
            (rx2, ry1, -1, 1),
            (rx1, ry2, 1, -1),
            (rx2, ry2, -1, -1),
        ]:
            for color, lw_r in [("black", rect_border_lw), ("white", rect_lw)]:
                ax0.plot(
                    [rcx, rcx + sx * corner_len],
                    [rcy, rcy],
                    color=color,
                    linewidth=lw_r,
                    solid_capstyle="butt",
                    zorder=10,
                )
                ax0.plot(
                    [rcx, rcx],
                    [rcy, rcy + sy * corner_len],
                    color=color,
                    linewidth=lw_r,
                    solid_capstyle="butt",
                    zorder=10,
                )

    ax0.scatter(
        0.0, 0.0, s=80, color="white", edgecolor="black", linewidth=1.5, zorder=11
    )
    ax0.set_title("Grid in normalized space")
    ax0.set_xlabel("x_n")
    ax0.set_ylabel("y_n")
    ax0.set_aspect("equal")
    ax0.set_xlim(x_min, x_max)
    ax0.set_ylim(y_max, y_min)

    # --- Right panel: projected grid in pixel space ---
    yn_dense = np.linspace(y_min, y_max, 1000)
    for x0 in x_lines:
        xn = np.full_like(yn_dense, x0)
        uv = project_polyline(xn, yn_dense)
        u, v = uv[:, 0], uv[:, 1]
        valid = np.isfinite(u) & np.isfinite(v)
        valid &= (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if not np.any(valid):
            continue
        idx = np.flatnonzero(valid)
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if run.size >= 2:
                segs, colors = make_segments(u[run], v[run], xn[run], yn_dense[run])
                ax1.add_collection(LineCollection(segs, colors=colors, linewidths=lw))  # type: ignore[reportArgumentType]

    xn_dense = np.linspace(x_min, x_max, 1000)
    for y0 in y_lines:
        yn = np.full_like(xn_dense, y0)
        uv = project_polyline(xn_dense, yn)
        u, v = uv[:, 0], uv[:, 1]
        valid = np.isfinite(u) & np.isfinite(v)
        valid &= (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if not np.any(valid):
            continue
        idx = np.flatnonzero(valid)
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            if run.size >= 2:
                segs, colors = make_segments(u[run], v[run], xn_dense[run], yn[run])
                ax1.add_collection(LineCollection(segs, colors=colors, linewidths=lw))  # type: ignore[reportArgumentType]

    # Rectangle corner markers projected to pixel space
    n_marker_pts = 32
    for i in range(1, n_rects + 1):
        rh = i * rect_step
        rw = round(rh * norm_aspect / grid_step_norm) * grid_step_norm
        rx1, ry1 = -rw, -rh
        rx2, ry2 = rw, rh
        for rcx, rcy, sx, sy in [
            (rx1, ry1, 1, 1),
            (rx2, ry1, -1, 1),
            (rx1, ry2, 1, -1),
            (rx2, ry2, -1, -1),
        ]:
            ts = np.linspace(0, 1, n_marker_pts)
            h_xn = rcx + ts * sx * corner_len
            h_yn = np.full_like(ts, rcy)
            h_uv = project_polyline(h_xn, h_yn)
            v_yn = rcy + ts * sy * corner_len
            v_xn = np.full_like(ts, rcx)
            v_uv = project_polyline(v_xn, v_yn)
            for color, lw_r in [("black", rect_border_lw), ("white", rect_lw)]:
                ax1.plot(
                    h_uv[:, 0],
                    h_uv[:, 1],
                    color=color,
                    linewidth=lw_r,
                    solid_capstyle="butt",
                    zorder=10,
                )
                ax1.plot(
                    v_uv[:, 0],
                    v_uv[:, 1],
                    color=color,
                    linewidth=lw_r,
                    solid_capstyle="butt",
                    zorder=10,
                )

    if show_spline_knots and isinstance(model, lb.PinholeSplined):
        Nx = model.num_knots_x
        Ny = model.num_knots_y
        kx_indices = np.arange(Nx)
        ky_indices = np.arange(Ny)
        knot_xn = -fov_x_half + (kx_indices - 1) * 2 * fov_x_half / (Nx - 3)
        knot_yn = -fov_y_half + (ky_indices - 1) * 2 * fov_y_half / (Ny - 3)
        knot_xn_grid, knot_yn_grid = np.meshgrid(knot_xn, knot_yn)
        knot_xn_flat = knot_xn_grid.ravel()
        knot_yn_flat = knot_yn_grid.ravel()

        ax0.scatter(
            knot_xn_flat,
            knot_yn_flat,
            s=30,
            color="deeppink",
            edgecolor="black",
            linewidth=0.5,
            zorder=9,
        )

        knot_pts_3d = np.stack(
            [knot_xn_flat, knot_yn_flat, np.ones_like(knot_xn_flat)], axis=1
        )
        knot_uv = model.project_points(knot_pts_3d)
        ax1.scatter(
            knot_uv[:, 0],
            knot_uv[:, 1],
            s=30,
            color="deeppink",
            edgecolor="black",
            linewidth=0.5,
            zorder=9,
        )

    ax1.scatter(cx, cy, s=80, color="white", edgecolor="black", linewidth=1.5, zorder=11)
    ax1.set_title("Grid in pixel space")
    ax1.set_xlabel("u (px)")
    ax1.set_ylabel("v (px)")
    ax1.set_xlim(0, W - 1)
    ax1.set_ylim(H - 1, 0)
    ax1.set_aspect("equal")

    plt.show()


def plot_residuals(
    frame_infos: list[lb.FrameInfo],
    *,
    bins: int = 100,
    n_sigma: float = 6.0,
    axis_range: float | None = None,
    title: str = "Reprojection residuals",
) -> None:
    """Per-component histogram and 2D scatter of reprojection residuals.

    Left panel: stacked histogram (inliers blue, outliers red) with a 1D
    Gaussian fit overlaid. Right panel: 2D scatter of residuals with fitted
    2D Gaussian contours.  Both axes are trimmed to ±``n_sigma`` standard
    deviations.

    Args:
        frame_infos: Per-frame reprojection diagnostics.
        bins: Number of histogram bins.
        n_sigma: Number of fitted-Gaussian standard deviations for axis limits.
        axis_range: Fixed symmetric axis limit (±value) for the histogram and
            2D scatter plots. The full-range plot is unaffected. Auto-scaled
            from n_sigma if None.
        title: Overall figure title.
    """
    inlier_2d: list[np.ndarray] = []
    outlier_2d: list[np.ndarray] = []
    for fi in frame_infos:
        inlier_2d.append(fi.residuals[fi.inlier_mask])
        outlier_2d.append(fi.residuals[~fi.inlier_mask])

    inlier_pts = np.concatenate(inlier_2d) if inlier_2d else np.empty((0, 2))
    outlier_pts = np.concatenate(outlier_2d) if outlier_2d else np.empty((0, 2))
    all_pts = np.concatenate([inlier_pts, outlier_pts])  # (N, 2)

    if all_pts.shape[0] == 0:
        return

    # --- 1D stats (robust fit on inliers only) ---
    inlier_vals = inlier_pts.ravel()

    mu_1d = float(np.median(inlier_vals))
    mad = float(np.median(np.abs(inlier_vals - mu_1d)))
    sigma_1d = 1.4826 * mad

    half = axis_range if axis_range is not None else n_sigma * sigma_1d
    lo = mu_1d - half
    hi = mu_1d + half
    bin_edges = np.linspace(lo, hi, bins + 1)

    # --- 2D stats (robust fit on inliers only) ---
    mu_2d = np.median(inlier_pts, axis=0)  # (2,)
    mad_x = float(np.median(np.abs(inlier_pts[:, 0] - mu_2d[0])))
    mad_y = float(np.median(np.abs(inlier_pts[:, 1] - mu_2d[1])))
    sigma_x = 1.4826 * mad_x
    sigma_y = 1.4826 * mad_y
    # Keep sample correlation for the off-diagonal
    sample_cov = np.cov(inlier_pts, rowvar=False)
    rho = sample_cov[0, 1] / np.sqrt(sample_cov[0, 0] * sample_cov[1, 1])
    cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
    )
    cov_inv = np.linalg.inv(cov)

    bg = "#111111"
    fg = "white"
    accent = "#00d4ff"

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(bg)
    fig.suptitle(title, color=fg, fontsize=14)

    gs = GridSpec(2, 2, figure=fig)
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_2d = fig.add_subplot(gs[1, 0])
    ax_full = fig.add_subplot(gs[:, 1])

    for ax in (ax_hist, ax_2d, ax_full):
        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

    # --- Top-left: histogram (inliers only) ---
    ax_hist.hist(
        inlier_vals,
        bins=bin_edges,  # type: ignore
        color=accent,
    )

    x = np.linspace(lo, hi, 500)
    bin_width = (hi - lo) / bins
    scale = inlier_vals.size * bin_width
    pdf = (
        scale
        / (sigma_1d * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mu_1d) / sigma_1d) ** 2)
    )
    ax_hist.plot(
        x, pdf, color="white", linewidth=1.5, label=f"Gaussian (MAD σ={sigma_1d:.3f} px)"
    )

    ax_hist.set_xlim(lo, hi)
    ax_hist.set_title("Per-component histogram")
    ax_hist.set_xlabel("residual [px]")
    ax_hist.set_ylabel("count")
    ax_hist.legend(facecolor=bg, edgecolor=fg, labelcolor=fg, loc="upper right")
    ax_hist.grid(True, linewidth=0.5, alpha=0.15, color=fg)

    # --- Right: 2D scatter + Gaussian contours ---
    sigma_max = max(sigma_x, sigma_y)
    grid_half = axis_range if axis_range is not None else n_sigma * sigma_max
    gx = np.linspace(mu_2d[0] - grid_half, mu_2d[0] + grid_half, 400)
    gy = np.linspace(mu_2d[1] - grid_half, mu_2d[1] + grid_half, 400)
    GX, GY = np.meshgrid(gx, gy)
    diff = np.stack([GX - mu_2d[0], GY - mu_2d[1]], axis=-1)  # (400, 400, 2)
    maha2 = np.einsum("...i,ij,...j", diff, cov_inv, diff)

    # --- Bottom: scatter + contour lines ---
    contour_levels = [1.0, 4.0, 9.0]
    cs = ax_2d.contour(
        GX, GY, maha2, levels=contour_levels, colors=accent, linewidths=1.2
    )
    labels = ax_2d.clabel(
        cs,
        fmt={1.0: "1σ", 4.0: "2σ", 9.0: "3σ"},
        fontsize=8,
    )
    for lbl in labels:
        lbl.set_color(bg)
        lbl.set_bbox({"facecolor": accent, "pad": 1.5, "edgecolor": "none"})

    if inlier_pts.shape[0] > 0:
        ax_2d.scatter(
            inlier_pts[:, 0],
            inlier_pts[:, 1],
            s=3,
            alpha=0.15,
            color="white",
            edgecolors="none",
        )

    lim = axis_range if axis_range is not None else n_sigma * sigma_max
    ax_2d.set_xlim(mu_2d[0] - lim, mu_2d[0] + lim)
    ax_2d.set_ylim(mu_2d[1] - lim, mu_2d[1] + lim)
    ax_2d.set_aspect("equal", adjustable="box")
    ax_2d.set_xlabel("x residual [px]")
    ax_2d.set_ylabel("y residual [px]")
    ax_2d.set_title(f"2D residuals (σx={sigma_x:.3f}, σy={sigma_y:.3f} px)")

    # --- Right column: full-range scatter highlighting outliers ---
    if inlier_pts.shape[0] > 0:
        ax_full.scatter(
            inlier_pts[:, 0],
            inlier_pts[:, 1],
            s=3,
            alpha=0.15,
            color="white",
            edgecolors="none",
        )
    if outlier_pts.shape[0] > 0:
        ax_full.scatter(
            outlier_pts[:, 0],
            outlier_pts[:, 1],
            s=20,
            alpha=0.9,
            color="#ff4444",
            edgecolors="white",
            linewidths=0.5,
            zorder=10,
        )

    ax_full.set_aspect("equal", adjustable="box")
    ax_full.set_xlabel("x residual [px]")
    ax_full.set_ylabel("y residual [px]")
    n_outliers = outlier_pts.shape[0] // 2  # 2 components per point
    ax_full.set_title(f"Full range ({n_outliers} outlier points)")

    plt.tight_layout()
    plt.show()


def plot_residual_vectors(
    frames: list[lb.Frame],
    frame_infos: list[lb.FrameInfo],
    *,
    image_width: int,
    image_height: int,
    title: str = "Residual vectors",
    scale: float = 10.0,
    scale_by_magnitude: bool = True,
    color_by: str = "magnitude",
) -> None:
    """Quiver plot of reprojection residual vectors over the image plane.

    Each arrow is placed at the detected point location with direction and
    length given by the residual.

    Args:
        frames: Detected calibration frames.
        frame_infos: Matching per-frame reprojection diagnostics.
        image_width: Sensor width in pixels, sets the x-axis limit.
        image_height: Sensor height in pixels, sets the y-axis limit.
        title: Plot title.
        scale: Multiplier applied to arrow lengths for visibility.
        scale_by_magnitude: When False, all arrows are drawn with uniform
            length (direction only).
        color_by: ``"magnitude"`` colours by residual norm, ``"angle"``
            colours by residual direction using a cyclic colormap.
    """
    positions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for frame, fi in zip(frames, frame_infos):
        positions.append(frame.detected_points_in_image)
        residuals.append(fi.residuals)

    pos = np.concatenate(positions) if positions else np.empty((0, 2))
    res = np.concatenate(residuals) if residuals else np.empty((0, 2))

    if pos.shape[0] == 0:
        return

    magnitudes = np.linalg.norm(res, axis=1)

    if scale_by_magnitude:
        arrows = res * scale
    else:
        safe_mag = np.where(magnitudes > 0, magnitudes, 1.0)
        arrows = res / safe_mag[:, None] * scale

    bg = "#111111"
    fg = "white"

    fig, ax = plt.subplots(figsize=(20, 15))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    if color_by == "angle":
        angles = np.arctan2(res[:, 1], res[:, 0])
        color_values = angles
        norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
        cmap = plt.colormaps["hsv"]
        cbar_label = "residual angle [rad]"
    elif color_by == "magnitude":
        color_values = magnitudes
        norm = mcolors.Normalize(vmin=0, vmax=np.percentile(magnitudes, 95))
        cmap = plt.colormaps["plasma"]
        cbar_label = "residual magnitude [px]"
    else:
        raise ValueError(f"color_by must be 'magnitude' or 'angle', got {color_by!r}")

    q = ax.quiver(
        pos[:, 0],
        pos[:, 1],
        arrows[:, 0],
        arrows[:, 1],
        color_values,
        cmap=cmap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.001,
        headwidth=1.5,
        headlength=1.5,
        headaxislength=1.5,
    )

    cbar: Colorbar = fig.colorbar(q, ax=ax, shrink=0.6)
    cbar.set_label(cbar_label, color=fg)
    cbar.ax.tick_params(colors=fg)

    ax.set_title(f"{title}  (N={pos.shape[0]}, scale={scale}x)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def plot_residual_grid(
    frames: list[lb.Frame],
    frame_infos: list[lb.FrameInfo],
    *,
    image_width: int,
    image_height: int,
    grid_cells: int = 50,
    arrow_scale: float = 100.0,
    heatmap_max: float | None = None,
    title: str = "Residual grid",
) -> None:
    """Binned residual summary showing per-cell magnitude and mean direction.

    The image plane is divided into a grid. Each cell is coloured by the mean
    inlier residual magnitude and has an arrow showing the mean residual
    vector, revealing spatial bias patterns.

    Args:
        frames: Detected calibration frames.
        frame_infos: Matching per-frame reprojection diagnostics.
        image_width: Sensor width in pixels, sets the x-axis limit.
        image_height: Sensor height in pixels, sets the y-axis limit.
        grid_cells: Number of grid cells along the longer image axis.
        arrow_scale: Multiplier applied to the mean-residual arrows.
        heatmap_max: Upper limit for the colour scale. Auto-scaled if None.
        title: Plot title.
    """
    positions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for frame, fi in zip(frames, frame_infos):
        positions.append(frame.detected_points_in_image[fi.inlier_mask])
        residuals.append(fi.residuals[fi.inlier_mask])

    pos = np.concatenate(positions) if positions else np.empty((0, 2))
    res = np.concatenate(residuals) if residuals else np.empty((0, 2))

    if pos.shape[0] == 0:
        return

    aspect = image_width / image_height
    if aspect >= 1:
        nx = grid_cells
        ny = max(1, int(round(grid_cells / aspect)))
    else:
        ny = grid_cells
        nx = max(1, int(round(grid_cells * aspect)))

    cell_w = image_width / nx
    cell_h = image_height / ny

    ix = np.clip((pos[:, 0] / cell_w).astype(int), 0, nx - 1)
    iy = np.clip((pos[:, 1] / cell_h).astype(int), 0, ny - 1)

    mean_mag = np.full((ny, nx), np.nan)
    mean_dx = np.zeros((ny, nx))
    mean_dy = np.zeros((ny, nx))
    counts = np.zeros((ny, nx), dtype=int)

    norms = np.linalg.norm(res, axis=1)
    sum_norms = np.zeros((ny, nx))

    for i in range(pos.shape[0]):
        cx, cy = ix[i], iy[i]
        counts[cy, cx] += 1
        mean_dx[cy, cx] += res[i, 0]
        mean_dy[cy, cx] += res[i, 1]
        sum_norms[cy, cx] += norms[i]

    mask = counts > 0
    mean_dx[mask] /= counts[mask]
    mean_dy[mask] /= counts[mask]
    mean_mag[mask] = sum_norms[mask] / counts[mask]

    bg = "#111111"
    fg = "white"

    fig, ax = plt.subplots(figsize=(20, 15))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    im = ax.imshow(
        mean_mag,
        extent=[0, image_width, image_height, 0],  # type: ignore[arg-type]
        cmap="plasma",
        interpolation="nearest",
        aspect="auto",
        vmax=heatmap_max,
    )

    cbar: Colorbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("mean residual magnitude [px]", color=fg)
    cbar.ax.tick_params(colors=fg)

    cx_arr = (np.arange(nx) + 0.5) * cell_w
    cy_arr = (np.arange(ny) + 0.5) * cell_h
    CX, CY = np.meshgrid(cx_arr, cy_arr)

    arrow_mask = mask & (mean_mag > 0)
    ax.quiver(
        CX[arrow_mask],
        CY[arrow_mask],
        mean_dx[arrow_mask] * arrow_scale,
        mean_dy[arrow_mask] * arrow_scale,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        headwidth=3,
        headlength=3,
        headaxislength=2,
    )

    ax.set_title(f"{title}  ({nx}x{ny} grid, arrow_scale={arrow_scale}x)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def plot_target_and_poses(
    target_points: np.ndarray,
    cameras_T_target: list[lb.Pose],
    *,
    triad_scale: float = 20.0,
    title: str = "Target and camera poses",
) -> None:
    """3D scatter of the calibration target with camera poses shown as triads.

    Each camera is drawn as a coordinate-frame triad (X=red, Y=green, Z=blue)
    at the camera position in the target reference frame.

    Args:
        target_points: Calibration target 3D coordinates, shape (N, 3).
        cameras_T_target: Camera-from-target poses, one per image.
        triad_scale: Length of each triad axis arrow in target units.
        title: Plot title.
    """
    bg = "#111111"
    fg = "white"

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)  # type: ignore

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):  # type: ignore
        axis.pane.set_facecolor(bg)  # type: ignore[union-attr]
        axis.pane.set_edgecolor(fg)  # type: ignore[union-attr]
        axis.label.set_color(fg)
    ax.tick_params(colors=fg)

    # Target points
    ax.scatter(
        target_points[:, 0],
        target_points[:, 1],
        target_points[:, 2],  # type: ignore[arg-type]
        c="red",
        s=8,
        depthshade=True,
        label="target points",
    )

    # Camera triads
    axis_colors = ["red", "green", "blue"]
    camera_origins = []
    for pose in cameras_T_target:
        target_T_camera = pose.inverse()
        origin = target_T_camera.translation
        rotmat = target_T_camera.rotmat
        camera_origins.append(origin)

        for axis_idx, color in enumerate(axis_colors):
            direction = rotmat[:, axis_idx] * triad_scale
            ax.quiver(
                origin[0],
                origin[1],
                origin[2],
                direction[0],
                direction[1],
                direction[2],
                color=color,
                arrow_length_ratio=0.1,
                linewidth=1.5,
            )

    camera_origins_arr = np.array(camera_origins)

    # Equal aspect ratio
    all_pts = np.vstack([target_points, camera_origins_arr])
    ranges = all_pts.max(axis=0) - all_pts.min(axis=0)
    max_range = ranges.max() / 2 * 1.1
    mid = np.mean(all_pts, axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)  # type: ignore
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)  # type: ignore
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)  # type: ignore
    ax.set_box_aspect([1, 1, 1])  # type: ignore

    ax.set_xlabel("X", color=fg)
    ax.set_ylabel("Y", color=fg)
    ax.set_zlabel("Z", color=fg)  # type: ignore
    ax.set_title(title, color=fg)

    plt.tight_layout()
    plt.show()


def plot_target_warp(
    target_points: np.ndarray,
    target_warp: lb.TargetWarp,
    *,
    grid_res: int = 300,
    contour_levels: int = 15,
    title: str = "Target warp",
) -> None:
    """Contour plot of the target warp z-displacement viewed from above.

    Evaluates the warp function over a dense grid in the warp frame's xy plane
    and shows filled contours of the z height, with target point positions
    scattered on top.

    Args:
        target_points: Calibration target 3D coordinates, shape (N, 3).
        target_warp: Estimated target warp.
        grid_res: Number of grid samples along each axis.
        contour_levels: Number of contour lines.
        title: Plot title.
    """
    wc = target_warp.warp_coordinates
    a, b, c, d, e = target_warp.object_warp

    s = np.linspace(-1, 1, grid_res)
    SX, SY = np.meshgrid(s, s)

    # P2(t) = 0.5*(3t²-1), P4(t) = 0.125*(35t⁴-30t²+3)
    P2X = 0.5 * (3 * SX**2 - 1)
    P2Y = 0.5 * (3 * SY**2 - 1)
    P4X = 0.125 * (35 * SX**4 - 30 * SX**2 + 3)
    P4Y = 0.125 * (35 * SY**4 - 30 * SY**2 + 3)
    Z = a * P2X + b * P2Y + c * P2X * P2Y + d * P4X + e * P4Y

    gx = SX * wc.x_scale
    gy = SY * wc.y_scale

    pts_in_warp = wc.target_from_warp_frame.inverse().apply(target_points)

    bg = "#111111"
    fg = "white"

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=fg)
    ax.xaxis.label.set_color(fg)
    ax.yaxis.label.set_color(fg)
    ax.title.set_color(fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    im = ax.imshow(
        Z,
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],  # type: ignore[arg-type]
        cmap="plasma",
        aspect="auto",
        interpolation="bilinear",
        origin="lower",
    )
    cs = ax.contour(gx, gy, Z, levels=contour_levels, colors="black", linewidths=0.4)
    ax.clabel(cs, fontsize=7, colors="black", fmt="%.4f")

    cbar: Colorbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("z displacement [target units]", color=fg)
    cbar.ax.tick_params(colors=fg)

    ax.scatter(
        pts_in_warp[:, 0],
        pts_in_warp[:, 1],
        s=25,
        color="red",
        edgecolors="black",
        linewidths=0.5,
        zorder=10,
        label="target points",
    )
    ax.legend(facecolor=bg, edgecolor=fg, labelcolor=fg, loc="upper right")

    margin = 0.1
    x_extent = gx.max() - gx.min()
    y_extent = gy.max() - gy.min()
    ax.set_xlim(gx.min() - margin * x_extent, gx.max() + margin * x_extent)
    ax.set_ylim(gy.min() - margin * y_extent, gy.max() + margin * y_extent)

    coeff_str = ", ".join(f"{v:.5f}" for v in (a, b, c, d, e))
    ax.set_title(f"{title}  ({coeff_str})")
    ax.set_xlabel("warp x [target units]")
    ax.set_ylabel("warp y [target units]")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()


def _remap_point(model: lb.PinholeRemapped, x: float, y: float) -> tuple[float, float]:
    """Look up undistorted pixel in the remap tables with bilinear interpolation."""
    ix, iy = int(x), int(y)
    ix = min(max(ix, 0), model.image_width - 2)
    iy = min(max(iy, 0), model.image_height - 2)
    fx, fy = x - ix, y - iy
    fx = min(max(fx, 0.0), 1.0)
    fy = min(max(fy, 0.0), 1.0)
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    dx = float(
        w00 * model.map_x[iy, ix]
        + w10 * model.map_x[iy, ix + 1]
        + w01 * model.map_x[iy + 1, ix]
        + w11 * model.map_x[iy + 1, ix + 1]
    )
    dy = float(
        w00 * model.map_y[iy, ix]
        + w10 * model.map_y[iy, ix + 1]
        + w01 * model.map_y[iy + 1, ix]
        + w11 * model.map_y[iy + 1, ix + 1]
    )
    return dx, dy


def plot_undistortion(
    model: lb.PinholeRemapped,
    *,
    image: np.ndarray | None = None,
    grid_step_px: int = 50,
    line_thickness: int | None = None,
) -> None:
    """Visualize distortion and undistortion with grid warping.

    Row 1: regular grid in distorted pixel space (left) remapped to
    undistorted pixel space (right).  Row 2: regular grid in undistorted
    pixel space (right) mapped back to distorted pixel space (left).
    When an image is provided, a third row shows the original and
    undistorted image.

    Args:
        model: Undistorted pinhole model with remap tables.
        image: Optional distorted input image, shape (H, W) or (H, W, 3).
        grid_step_px: Spacing between grid lines in pixels.
        line_thickness: Line width in pixels.  When None, chosen automatically
            based on the image diagonal.
    """
    W_in = model.input_image_width
    H_in = model.input_image_height
    W_out = model.image_width
    H_out = model.image_height

    if line_thickness is None:
        diag = np.hypot(W_in, H_in)
        line_thickness = max(1, int(round(diag / 800)))

    cmap = plt.colormaps["jet"]
    x_lines = np.arange(grid_step_px, W_in, grid_step_px)
    y_lines = np.arange(grid_step_px, H_in, grid_step_px)

    # Smooth diagonal color gradient masked to grid lines
    yy, xx = np.mgrid[0:H_in, 0:W_in]
    diag_pos = (xx / W_in + yy / H_in) / 2.0
    gradient_img = (cmap(diag_pos)[:, :, :3] * 255).astype(np.uint8)

    grid_mask = np.zeros((H_in, W_in), dtype=np.uint8)
    for x0 in x_lines:
        cv2.line(
            grid_mask, (int(x0), 0), (int(x0), H_in - 1), 255, thickness=line_thickness
        )
    for y0 in y_lines:
        cv2.line(
            grid_mask, (0, int(y0)), (W_in - 1, int(y0)), 255, thickness=line_thickness
        )

    grid_img = gradient_img * (grid_mask[:, :, None] > 0)

    # White concentric rectangles snapped to grid intersections
    rect_step = grid_step_px * 4
    rect_thickness = line_thickness * 2
    rect_cx = round(W_in / 2.0 / grid_step_px) * grid_step_px
    rect_cy = round(H_in / 2.0 / grid_step_px) * grid_step_px
    aspect = W_in / H_in
    corner_len = grid_step_px
    n_rects = max(1, int(min(rect_cx, rect_cy) / rect_step))
    for i in range(1, n_rects + 1):
        half_h = i * rect_step
        half_w = round(half_h * aspect / grid_step_px) * grid_step_px
        x1, y1 = int(rect_cx - half_w), int(rect_cy - half_h)
        x2, y2 = int(rect_cx + half_w), int(rect_cy + half_h)
        for cx, cy, sx, sy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            for color, thickness in [
                ((0, 0, 0), rect_thickness * 3),
                ((255, 255, 255), rect_thickness),
            ]:
                cv2.line(grid_img, (cx, cy), (cx + sx * corner_len, cy), color, thickness)
                cv2.line(grid_img, (cx, cy), (cx, cy + sy * corner_len), color, thickness)

    undistorted_img = model.undistort(grid_img, interpolation=cv2.INTER_LANCZOS4)

    # --- Row 2: regular grid in undistorted space mapped to distorted space ---
    x_lines_out = np.arange(grid_step_px, W_out, grid_step_px)
    y_lines_out = np.arange(grid_step_px, H_out, grid_step_px)

    def _diag_colored_segments(
        xs: np.ndarray,
        ys: np.ndarray,
        w: float,
        h: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build LineCollection segments with diagonal-gradient jet colors.

        Args:
            xs: x coordinates of the polyline, shape (N,).
            ys: y coordinates of the polyline, shape (N,).
            w: Width used to normalize x for the diagonal gradient.
            h: Height used to normalize y for the diagonal gradient.

        Returns:
            Segments array of shape (N-1, 2, 2) and colors array of shape (N-1, 4).
        """
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        mid_x = (xs[:-1] + xs[1:]) / 2.0
        mid_y = (ys[:-1] + ys[1:]) / 2.0
        diag = (mid_x / w + mid_y / h) / 2.0
        colors = cmap(diag)
        return segs, colors

    n_rows = 3 if image is not None else 2
    panel_w = 10
    panel_h = panel_w * (H_in / W_in)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(2 * panel_w, panel_h * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    fig.patch.set_facecolor("#111111")

    for ax in axes.flat:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # Row 1: distorted grid → undistorted (rasterized)
    axes[0, 0].imshow(grid_img)
    axes[0, 0].set_title("Grid in distorted pixel space")
    axes[0, 0].set_xlabel("u (px)")
    axes[0, 0].set_ylabel("v (px)")
    axes[0, 0].set_aspect("equal")

    axes[0, 1].imshow(undistorted_img)
    axes[0, 1].set_title("Grid in undistorted pixel space")
    axes[0, 1].set_xlabel("u (px)")
    axes[0, 1].set_ylabel("v (px)")
    axes[0, 1].set_aspect("equal")

    # Row 2 right: regular grid in undistorted space (diagonal gradient)
    # Convert cv2 pixel line width to matplotlib points to match row 1
    mpl_lw = line_thickness * panel_w * 72 / W_out
    n_subdivide = 64
    for x0 in x_lines_out:
        ys = np.linspace(0, H_out, n_subdivide)
        xs = np.full_like(ys, float(x0))
        segs, colors = _diag_colored_segments(xs, ys, W_out, H_out)
        axes[1, 1].add_collection(LineCollection(segs, colors=colors, linewidths=mpl_lw))  # type: ignore[reportArgumentType]
    for y0 in y_lines_out:
        xs = np.linspace(0, W_out, n_subdivide)
        ys = np.full_like(xs, float(y0))
        segs, colors = _diag_colored_segments(xs, ys, W_out, H_out)
        axes[1, 1].add_collection(LineCollection(segs, colors=colors, linewidths=mpl_lw))  # type: ignore[reportArgumentType]

    # Row 2 right: corner markers in undistorted space
    px_to_pt = panel_w * 72 / W_out
    marker_lw_inner = rect_thickness * px_to_pt
    marker_lw_border = rect_thickness * 3 * px_to_pt
    rect_cx_out = round(W_out / 2.0 / grid_step_px) * grid_step_px
    rect_cy_out = round(H_out / 2.0 / grid_step_px) * grid_step_px
    aspect_out = W_out / H_out
    n_rects_out = max(1, int(min(rect_cx_out, rect_cy_out) / rect_step))
    for i in range(1, n_rects_out + 1):
        half_h = i * rect_step
        half_w = round(half_h * aspect_out / grid_step_px) * grid_step_px
        x1, y1 = rect_cx_out - half_w, rect_cy_out - half_h
        x2, y2 = rect_cx_out + half_w, rect_cy_out + half_h
        for cx, cy, sx, sy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            for color, lw in [("black", marker_lw_border), ("white", marker_lw_inner)]:
                axes[1, 1].plot(
                    [cx, cx + sx * corner_len],
                    [cy, cy],
                    color=color,
                    linewidth=lw,
                    solid_capstyle="butt",
                )
                axes[1, 1].plot(
                    [cx, cx],
                    [cy, cy + sy * corner_len],
                    color=color,
                    linewidth=lw,
                    solid_capstyle="butt",
                )

    axes[1, 1].set_xlim(0, W_out)
    axes[1, 1].set_ylim(H_out, 0)
    axes[1, 1].set_title("Grid in undistorted pixel space")
    axes[1, 1].set_xlabel("u (px)")
    axes[1, 1].set_ylabel("v (px)")
    axes[1, 1].set_aspect("equal")

    # Row 2 left: same grid traced through remap tables to distorted space
    # Color each segment by its undistorted-space position for consistency
    for x0 in x_lines_out:
        u = int(round(x0))
        if 0 <= u < W_out:
            dx = model.map_x[:, u]
            dy = model.map_y[:, u]
            # Color by undistorted position: x=x0, y varies over rows
            src_ys = np.arange(H_out, dtype=np.float64)
            segs, colors = _diag_colored_segments(dx, dy, W_out, H_out)
            # Recompute colors from undistorted coordinates
            mid_y = (src_ys[:-1] + src_ys[1:]) / 2.0
            diag_vals = (x0 / W_out + mid_y / H_out) / 2.0
            colors = cmap(diag_vals)
            axes[1, 0].add_collection(
                LineCollection(segs, colors=colors, linewidths=mpl_lw)  # type: ignore[reportArgumentType]
            )
    for y0 in y_lines_out:
        v = int(round(y0))
        if 0 <= v < H_out:
            dx = model.map_x[v, :]
            dy = model.map_y[v, :]
            src_xs = np.arange(W_out, dtype=np.float64)
            segs, colors = _diag_colored_segments(dx, dy, W_out, H_out)
            # Recompute colors from undistorted coordinates
            mid_x = (src_xs[:-1] + src_xs[1:]) / 2.0
            diag_vals = (mid_x / W_out + y0 / H_out) / 2.0
            colors = cmap(diag_vals)
            axes[1, 0].add_collection(
                LineCollection(segs, colors=colors, linewidths=mpl_lw)  # type: ignore[reportArgumentType]
            )

    # Row 2 left: corner markers traced through remap tables
    n_marker_pts = 16
    for i in range(1, n_rects_out + 1):
        half_h = i * rect_step
        half_w = round(half_h * aspect_out / grid_step_px) * grid_step_px
        x1, y1 = rect_cx_out - half_w, rect_cy_out - half_h
        x2, y2 = rect_cx_out + half_w, rect_cy_out + half_h
        for cx, cy, sx, sy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            # Sample along each arm and trace through remap
            h_ts = np.linspace(0, 1, n_marker_pts)
            h_xs = cx + h_ts * sx * corner_len
            h_ys = np.full_like(h_ts, cy)
            h_d = np.array([_remap_point(model, px, py) for px, py in zip(h_xs, h_ys)])
            v_ys = cy + h_ts * sy * corner_len
            v_xs = np.full_like(h_ts, cx)
            v_d = np.array([_remap_point(model, px, py) for px, py in zip(v_xs, v_ys)])
            for color, lw in [("black", marker_lw_border), ("white", marker_lw_inner)]:
                axes[1, 0].plot(
                    h_d[:, 0], h_d[:, 1], color=color, linewidth=lw, solid_capstyle="butt"
                )
                axes[1, 0].plot(
                    v_d[:, 0], v_d[:, 1], color=color, linewidth=lw, solid_capstyle="butt"
                )

    axes[1, 0].set_xlim(0, W_in)
    axes[1, 0].set_ylim(H_in, 0)
    axes[1, 0].set_title("Grid in distorted pixel space")
    axes[1, 0].set_xlabel("u (px)")
    axes[1, 0].set_ylabel("v (px)")
    axes[1, 0].set_aspect("equal")

    # Row 3: original and undistorted image
    if image is not None:
        img_rgb = cv2.cvtColor(to_color(image), cv2.COLOR_BGR2RGB)
        undistorted_rgb = cv2.cvtColor(
            to_color(model.undistort(image, interpolation=cv2.INTER_LANCZOS4)),
            cv2.COLOR_BGR2RGB,
        )

        axes[2, 0].imshow(img_rgb)
        axes[2, 0].set_title("Distorted image")
        axes[2, 0].set_xlabel("u (px)")
        axes[2, 0].set_ylabel("v (px)")
        axes[2, 0].set_aspect("equal")

        axes[2, 1].imshow(undistorted_rgb)
        axes[2, 1].set_title("Undistorted image")
        axes[2, 1].set_xlabel("u (px)")
        axes[2, 1].set_ylabel("v (px)")
        axes[2, 1].set_aspect("equal")

    plt.show()


def plot_worst_residual_frames(
    frame_infos: list[lb.FrameInfo],
    frames: list[lb.Frame],
    images: list[np.ndarray],
    *,
    n: int = 5,
    scale: float = 10.0,
    title: str = "Worst residual frames",
) -> None:
    """Show the frames with the largest residuals, with residual vectors overlaid.

    Frames are ranked by their single worst (max-magnitude) residual and the
    top ``n`` are displayed in a single-column figure.  Each subplot shows the
    image with quiver arrows from detected points in the direction and
    magnitude of the residual, coloured by magnitude.

    Args:
        frame_infos: Per-frame reprojection diagnostics.
        frames: Detected calibration frames (same order as ``frame_infos``).
        images: Source images corresponding to each frame, shape (H, W) or (H, W, 3).
        n: Number of worst frames to display.
        scale: Multiplier applied to arrow lengths for visibility.
        title: Overall figure title.
    """
    max_mags = [float(np.max(np.linalg.norm(fi.residuals, axis=1))) for fi in frame_infos]
    ranked = sorted(range(len(max_mags)), key=lambda i: max_mags[i], reverse=True)
    selected = ranked[:n]

    bg = "#111111"
    fg = "white"
    cmap = plt.colormaps["plasma"]

    h0, w0 = images[selected[0]].shape[:2]
    panel_w = 14
    panel_h = panel_w * (h0 / w0)
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(panel_w, panel_h * len(selected)),
        squeeze=False,
    )
    fig.patch.set_facecolor(bg)
    fig.suptitle(title, color=fg, fontsize=14)

    for ax_row, idx in zip(axes, selected):
        ax = ax_row[0]
        fi = frame_infos[idx]
        frame = frames[idx]
        img = to_color(images[idx])

        pos = frame.detected_points_in_image
        res = fi.residuals
        mags = np.linalg.norm(res, axis=1)
        frame_norm = mcolors.Normalize(vmin=0, vmax=float(np.max(mags)))

        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # type: ignore[arg-type]

        ax.quiver(
            pos[:, 0],
            pos[:, 1],
            res[:, 0] * scale,
            res[:, 1] * scale,
            mags,
            cmap=cmap,
            norm=frame_norm,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.002,
            headwidth=2,
            headlength=2,
            headaxislength=1.5,
        )

        ax.set_facecolor(bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

        worst = float(max_mags[idx])
        mean = float(np.mean(mags))
        ax.set_title(f"Frame {idx}  (max={worst:.2f} px, mean={mean:.2f} px)")
        ax.set_aspect("equal", adjustable="box")

        cbar: Colorbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=frame_norm, cmap=cmap),  # type: ignore[arg-type]
            ax=ax,
            shrink=0.6,
        )
        cbar.set_label("residual [px]", color=fg)
        cbar.ax.tick_params(colors=fg)

    plt.tight_layout()
    plt.show()
