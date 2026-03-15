try:
    from lensboy.analysis.plots import (
        draw_points,
        plot_detection_coverage,
        plot_distortion_grid,
        plot_projection_diff,
        plot_undistortion,
    )
except ImportError as e:
    raise ImportError(
        "The analysis module requires extra dependencies. "
        "Install them with: pip install lensboy[analysis]"
    ) from e

__all__ = [
    "draw_points",
    "plot_detection_coverage",
    "plot_distortion_grid",
    "plot_projection_diff",
    "plot_undistortion",
]
