import lensboy as lb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_detection_coverage(
    detections: list[lb.Frame],
    *,
    image_width: int,
    image_height: int,
    title: str = "Coverage",
    s: float = 6.0,
    alpha: float = 0.35,
) -> None:
    """Scatter-plot all detected points over the image extent.

    Useful for checking that calibration frames adequately cover the sensor area.

    Args:
        detections: Frames containing detected calibration points.
        image_width: Sensor width in pixels, sets the x-axis limit.
        image_height: Sensor height in pixels, sets the y-axis limit.
        title: Plot title.
        s: Marker size passed to ``ax.scatter``.
        alpha: Marker opacity passed to ``ax.scatter``.
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

    fig, ax = plt.subplots(figsize=(20, 15))

    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=alpha)

    ax.set_title(f"{title}  (N={pts.shape[0]})")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    ax.set_xlim(0, image_width)
    ax.set_ylim(image_height, 0)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.3)

    plt.show()


def plot_normalized_grid_deformation_fov_clipped(
    model,
    *,
    grid_step_norm: float = 0.1,
    samples_per_line: int = 500,
    margin_px: float = 0.0,
    jump_thresh_px: float = 80.0,  # max allowed step between adjacent samples
    disp_mad_k: float = 6.0,  # robust gate on displacement vs pinhole
    hard_disp_cap_px: float | None = None,  # optional absolute cap, e.g. 400
    cmap_name: str = "viridis",
    # if True: vertical lines colored by x_n, horizontal by y_n
    # if False: everything colored by radius
    title: str | None = None,
    show_normalized_panel: bool = True,
):
    """
    Uses fov_deg_x/y for normalized domain, projects grid, and applies robust clipping:
      - in-image (with margin)
      - finite outputs
      - teleport/jump threshold between adjacent samples
      - robust displacement gate vs ideal pinhole projection (MAD-based)

    Plus: smooth colormap coloring (no matplotlib rainbow cycle).

    model must have:
      image_width, image_height, fx, fy, cx, cy, fov_deg_x, fov_deg_y
      project_points(points_in_cam: (N,3)) -> (N,2)
    """

    W = int(model.image_width)
    H = int(model.image_height)

    fx = float(getattr(model, "fx"))
    fy = float(getattr(model, "fy"))
    cx = float(getattr(model, "cx"))
    cy = float(getattr(model, "cy"))

    fov_x = np.deg2rad(float(getattr(model, "fov_deg_x")))
    fov_y = np.deg2rad(float(getattr(model, "fov_deg_y")))

    x_half = np.tan(fov_x / 2.0)
    y_half = np.tan(fov_y / 2.0)
    x_min, x_max = -x_half, +x_half
    y_min, y_max = -y_half, +y_half

    # --- colormap setup (smooth gradient, no rainbow cycling) ---
    cmap = cm.get_cmap(cmap_name)
    norm_x = mcolors.Normalize(vmin=x_min, vmax=x_max)
    norm_y = mcolors.Normalize(vmin=y_min, vmax=y_max)

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

    def pinhole_uv(xn: np.ndarray, yn: np.ndarray) -> np.ndarray:
        # ideal pinhole (no distortion): u = fx*x + cx, v = fy*y + cy
        u = fx * xn + cx
        v = fy * yn + cy
        return np.stack([u, v], axis=1)

    def mad(x: np.ndarray) -> float:
        med = np.median(x)
        return 1.4826 * np.median(np.abs(x - med))  # robust sigma-ish

    def plot_segments(
        ax,
        xn: np.ndarray,
        yn: np.ndarray,
        uv: np.ndarray,
        *,
        lw: float = 1.0,
        color=None,
    ):
        u = uv[:, 0]
        v = uv[:, 1]

        # Base validity: finite + inside image
        valid = np.isfinite(u) & np.isfinite(v)
        valid &= (u >= -margin_px) & (u <= (W - 1) + margin_px)
        valid &= (v >= -margin_px) & (v <= (H - 1) + margin_px)

        if not np.any(valid):
            return

        # Teleport / jump filter: invalidate points where step from prev is huge
        du = np.diff(u)
        dv = np.diff(v)
        step = np.sqrt(du * du + dv * dv)

        jump_ok = np.ones_like(u, dtype=bool)
        # Mark the *second* point as invalid when the step into it is insane.
        jump_ok[1:] = step <= jump_thresh_px
        valid &= jump_ok

        if not np.any(valid):
            return

        # Robust displacement filter vs ideal pinhole
        uv0 = pinhole_uv(xn, yn)
        disp = np.sqrt((u - uv0[:, 0]) ** 2 + (v - uv0[:, 1]) ** 2)

        disp_valid = valid & np.isfinite(disp)
        if np.any(disp_valid):
            sigma = mad(disp[disp_valid])
            if sigma < 1e-9:
                gate = np.full_like(disp, True, dtype=bool)
            else:
                gate = disp <= (np.median(disp[disp_valid]) + disp_mad_k * sigma)
            valid &= gate

        if hard_disp_cap_px is not None:
            valid &= disp <= float(hard_disp_cap_px)

        if not np.any(valid):
            return

        # Split into contiguous valid runs
        idx = np.flatnonzero(valid)
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)

        for run in runs:
            if run.size >= 2:
                seg = uv[run]
                ax.plot(seg[:, 0], seg[:, 1], linewidth=lw, color=color)

    if show_normalized_panel:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)
        ax0 = None

    fig.patch.set_facecolor("#111111")

    for ax in [ax0, ax1] if ax0 is not None else [ax1]:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    # Left: normalized grid (optionally colored too)
    if ax0 is not None:
        for x0 in x_lines:
            c = cmap(norm_x(x0))
            ax0.plot([x0, x0], [y_min, y_max], linewidth=1, color=c)
        for y0 in y_lines:
            c = cmap(norm_y(y0))
            ax0.plot([x_min, x_max], [y0, y0], linewidth=1, color=c)

        ax0.scatter(
            0.0,
            0.0,
            s=80,
            color="white",
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )
        ax0.set_title("Grid in normalized space (FOV domain)")
        ax0.set_xlabel("x_n")
        ax0.set_ylabel("y_n")
        ax0.set_aspect("equal")
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_max, y_min)

    # Right: projected + clipped (smooth color gradient)
    yn = np.linspace(y_min, y_max, samples_per_line)
    for x0 in x_lines:
        xn = np.full_like(yn, x0)
        uv = project_polyline(xn, yn)

        color = cmap(norm_x(x0))
        plot_segments(ax1, xn, yn, uv, lw=1.0, color=color)

    xn = np.linspace(x_min, x_max, samples_per_line)
    for y0 in y_lines:
        yn = np.full_like(xn, y0)
        uv = project_polyline(xn, yn)

        color = cmap(norm_y(y0))
        plot_segments(ax1, xn, yn, uv, lw=1.0, color=color)

    # Bright marker with black edge so it works on any background
    ax1.scatter(
        cx,
        cy,
        s=80,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )

    ax1.set_title(title or "Warped grid after projection (robust-clipped)")
    ax1.set_xlabel("u (px)")
    ax1.set_ylabel("v (px)")
    ax1.set_xlim(0, W - 1)
    ax1.set_ylim(H - 1, 0)
    ax1.set_aspect("equal")

    plt.show()
