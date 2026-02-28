import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import lensboy as lb


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
        canvas = image.copy()
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
) -> None:
    """Scatter-plot all detected points over the image extent.

    Useful for checking that calibration frames adequately cover the sensor area.

    Args:
        detections: Frames containing detected calibration points.
        image_width: Sensor width in pixels, sets the x-axis limit.
        image_height: Sensor height in pixels, sets the y-axis limit.
        title: Plot title.
        s: Marker size passed to ``ax.scatter``.
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
    fov_fraction: float = 1.0,
    cmap_name: str = "jet",
) -> None:
    """Project a regular grid through a camera model to visualize distortion.

    Builds a grid in normalized (tan-angle) space from the model's FOV, projects
    it, and clips to the image bounds.

    Args:
        model: Camera model instance.
        grid_step_norm: Spacing between grid lines in normalized coordinates.
        fov_fraction: Fraction of the full FOV to sample (0, 1].
        cmap_name: Matplotlib colormap name.
    """

    W = int(model.image_width)
    H = int(model.image_height)

    cx = model.cx
    cy = model.cy

    fov_x = np.deg2rad(model.fov_deg_x)
    fov_y = np.deg2rad(model.fov_deg_y)

    x_half = np.tan(fov_x / 2.0) * fov_fraction
    y_half = np.tan(fov_y / 2.0) * fov_fraction
    x_min, x_max = -x_half, +x_half
    y_min, y_max = -y_half, +y_half

    cmap = plt.colormaps[cmap_name]
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

    def plot_segments(
        ax,
        uv: np.ndarray,
        *,
        lw: float = 1.0,
        color=None,
    ):
        u = uv[:, 0]
        v = uv[:, 1]

        valid = np.isfinite(u) & np.isfinite(v)
        valid &= (u >= 0) & (u <= W - 1)
        valid &= (v >= 0) & (v <= H - 1)

        if not np.any(valid):
            return

        idx = np.flatnonzero(valid)
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)

        for run in runs:
            if run.size >= 2:
                seg = uv[run]
                ax.plot(seg[:, 0], seg[:, 1], linewidth=lw, color=color)

    fig_w = 12
    panel_h = fig_w * (H / W)
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(fig_w, 2 * panel_h), constrained_layout=True
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
        ax0.set_title("Grid in normalized space")
        ax0.set_xlabel("x_n")
        ax0.set_ylabel("y_n")
        ax0.set_aspect("equal")
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_max, y_min)

    yn = np.linspace(y_min, y_max, 1000)
    for x0 in x_lines:
        xn = np.full_like(yn, x0)
        uv = project_polyline(xn, yn)

        color = cmap(norm_x(x0))
        plot_segments(ax1, uv, lw=1.0, color=color)

    xn = np.linspace(x_min, x_max, 1000)
    for y0 in y_lines:
        yn = np.full_like(xn, y0)
        uv = project_polyline(xn, yn)

        color = cmap(norm_y(y0))
        plot_segments(ax1, uv, lw=1.0, color=color)

    ax1.scatter(
        cx,
        cy,
        s=80,
        color="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=10,
    )

    ax1.set_title("Grid in pixel space")
    ax1.set_xlabel("u (px)")
    ax1.set_ylabel("v (px)")
    ax1.set_xlim(0, W - 1)
    ax1.set_ylim(H - 1, 0)
    ax1.set_aspect("equal")

    plt.show()


def plot_residual_histogram(
    frame_infos: list[lb.FrameInfo],
    *,
    bins: int = 100,
    n_sigma: float = 6.0,
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

    lo = mu_1d - n_sigma * sigma_1d
    hi = mu_1d + n_sigma * sigma_1d
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
        bins=bin_edges,
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
    gx = np.linspace(mu_2d[0] - n_sigma * sigma_max, mu_2d[0] + n_sigma * sigma_max, 400)
    gy = np.linspace(mu_2d[1] - n_sigma * sigma_max, mu_2d[1] + n_sigma * sigma_max, 400)
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

    lim = n_sigma * sigma_max
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
