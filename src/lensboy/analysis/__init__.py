try:
    from lensboy.analysis.plots import (
        draw_points,
        plot_detection_coverage,
        plot_distortion_grid,
        plot_frame_residuals,
        plot_per_image_rms,
        plot_projection_diff,
        plot_residual_grid,
        plot_residual_vectors,
        plot_residuals,
        plot_target_and_poses,
        plot_target_warp,
        plot_undistortion,
        plot_worst_residual_frames,
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
    "plot_frame_residuals",
    "plot_per_image_rms",
    "plot_residual_grid",
    "plot_residuals",
    "plot_projection_diff",
    "plot_residual_vectors",
    "plot_target_and_poses",
    "plot_target_warp",
    "plot_undistortion",
    "plot_worst_residual_frames",
]
