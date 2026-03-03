"""Integration tests using a real charuco dataset."""

from pathlib import Path

import numpy as np

import lensboy as lb

DATASET_PATH = Path(__file__).parent.parent / "data/test_datasets/wide_angle_charuco.npz"


def load_test_dataset() -> tuple[np.ndarray, list[lb.Frame], int, int]:
    """Load the pre-extracted charuco test dataset.

    Returns:
        target_points: 3D target coordinates, shape (N, 3).
        frames: Per-image detection frames.
        image_height: Image height in pixels.
        image_width: Image width in pixels.
    """
    data = np.load(DATASET_PATH)
    target_points = data["target_points"]
    image_height = int(data["image_height"])
    image_width = int(data["image_width"])
    num_frames = int(data["num_frames"])

    frames = [
        lb.Frame(
            target_point_indices=data[f"frame_{i}_indices"],
            detected_points_in_image=data[f"frame_{i}_detections"],
        )
        for i in range(num_frames)
    ]

    return target_points, frames, image_height, image_width


def test_opencv_full14() -> None:
    """Calibrate an OpenCV model with all 14 distortion coefficients."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        initial_focal_length=1000,
        included_distoriton_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = (result.num_outliers() / result.num_detections()) * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 0.4, f"Too many outliers: {outlier_pct:.1f}%"

    _check_first_frame_projection(result, target_points, frames[0])


def test_spline_30x20() -> None:
    """Calibrate a 30x20 spline model."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.PinholeSplinedConfig(
        img_h,
        img_w,
        initial_focal_length=1000,
        num_knots_x=30,
        num_knots_y=20,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100

    assert sigma < 0.09, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 0.3, f"Too many outliers: {outlier_pct:.1f}%"

    _check_first_frame_projection(result, target_points, frames[0])


def _check_first_frame_projection(
    result: lb.CalibrationResult,
    target_points: np.ndarray,
    frame: lb.Frame,
) -> None:
    """Verify that manual projection matches the stored FrameInfo for frame 0."""
    model = result.optimized_camera_model
    pose = result.optimized_cameras_T_target[0]
    fi = result.frame_infos[0]

    points_in_target = target_points[frame.target_point_indices]
    if result.warp_info is not None:
        points_in_target = result.warp_info.warp_target(points_in_target)

    points_in_cam = pose.apply(points_in_target)
    projected = model.project_points(points_in_cam)

    np.testing.assert_allclose(projected, fi.projected_points, atol=1e-6)

    expected_residuals = frame.detected_points_in_image - projected
    np.testing.assert_allclose(expected_residuals, fi.residuals, atol=1e-6)
