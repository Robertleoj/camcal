try:
    from lensboy.analysis.plots import (
        draw_points_in_image,
        plot_detection_coverage,
        plot_distortion_grid,
        plot_residual_grid,
        plot_residual_histogram,
        plot_residual_vectors,
        plot_target_and_poses,
        plot_target_warp,
        plot_worst_residual_frames,
    )
except ImportError as e:
    raise ImportError(
        "The analysis module requires extra dependencies. "
        "Install them with: pip install lensboy[analysis]"
    ) from e

__all__ = [
    "draw_points_in_image",
    "plot_detection_coverage",
    "plot_distortion_grid",
    "plot_residual_grid",
    "plot_residual_histogram",
    "plot_residual_vectors",
    "plot_target_and_poses",
    "plot_target_warp",
    "plot_worst_residual_frames",
]
