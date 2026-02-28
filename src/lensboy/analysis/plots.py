import lensboy as lb
import numpy as np
import matplotlib.pyplot as plt


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
