import lensboy as lb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2


class Color:
    """Container for some common colors."""

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)


def _draw_points(img, points, color=Color.green, r=4, thickness=-1):
    for x, y in points:
        cv2.circle(img, (int(x), int(y)), r, color, thickness)
    return img


def draw_frame_detections(
    frame: lb.Frame,
    *,
    image: np.ndarray | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    color: tuple[int, int, int] = Color.green,
    r: int = 4,
) -> np.ndarray:
    """Draw detected points onto an image.

    If no image is provided, draws on a blank (black) canvas whose size
    is given by ``image_width`` and ``image_height``.

    Args:
        frame: Frame containing detected calibration points.
        image: Optional BGR image to draw on (will be copied).
        image_width: Canvas width when ``image`` is None.
        image_height: Canvas height when ``image`` is None.
        color: BGR circle colour.
        r: Circle radius in pixels.

    Returns:
        BGR image with detections drawn, shape (H, W, 3).
    """
    if image is not None:
        canvas = image.copy()
    else:
        if image_width is None or image_height is None:
            raise ValueError(
                "image_width and image_height are required when image is None"
            )
        canvas = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    return _draw_points(canvas, frame.detected_points_in_image, color=color, r=r)


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
