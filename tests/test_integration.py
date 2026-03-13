"""Integration tests using a real charuco dataset and synthetic data."""

from pathlib import Path

import pytest
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
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = (result.num_outliers() / result.num_detections()) * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.2, f"Too many outliers: {outlier_pct:.1f}%"

    _check_frame_projections(result, target_points, frames)


def test_opencv_full14_explicit_focal_length() -> None:
    """Calibrate with an explicit initial focal length guess."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        initial_focal_length=1000,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = (result.num_outliers() / result.num_detections()) * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.2, f"Too many outliers: {outlier_pct:.1f}%"


def test_spline_30x20() -> None:
    """Calibrate a 30x20 spline model."""
    target_points, frames, img_h, img_w = load_test_dataset()

    config = lb.PinholeSplinedConfig(
        img_h,
        img_w,
        num_knots_x=30,
        num_knots_y=20,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100

    assert sigma < 0.09, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.6, f"Too many outliers: {outlier_pct:.1f}%"

    _check_frame_projections(result, target_points, frames)


def test_opencv_all_outliers_in_one_frame() -> None:
    """Calibration succeeds when one frame has all its points corrupted."""
    target_points, frames, img_h, img_w = load_test_dataset()

    # Corrupt frame 0 by shifting every detection by a large random offset
    rng = np.random.default_rng(42)
    n_corrupted = len(frames[0])
    r = rng.uniform(25, 40, size=n_corrupted)
    theta = rng.uniform(0, 2 * np.pi, size=n_corrupted)
    offsets = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    corrupted_frame = lb.Frame(
        target_point_indices=frames[0].target_point_indices,
        detected_points_in_image=frames[0].detected_points_in_image + offsets,
    )
    frames_with_corruption = [corrupted_frame] + frames[1:]

    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(
        target_points, frames_with_corruption, camera_model_config=config
    )

    # The corrupted frame should have all points marked as outliers
    assert not result.frame_diagnostics[0].inlier_mask.any(), (
        "Expected all points in corrupted frame to be outliers"
    )

    sigma = result.residual_sigma_map()
    outlier_pct = result.num_outliers() / result.num_detections() * 100
    extra_outlier_pct = n_corrupted / result.num_detections() * 100

    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
    assert outlier_pct < 1.2 + extra_outlier_pct, f"Too many outliers: {outlier_pct:.1f}%"


def test_opencv_distortion_mask() -> None:
    """Unselected distortion coefficients remain zero after calibration."""
    target_points, frames, img_h, img_w = load_test_dataset()

    masks = {
        "NONE": lb.OpenCVConfig.NONE,
        "STANDARD": lb.OpenCVConfig.STANDARD,
        "RADIAL_6": lb.OpenCVConfig.RADIAL_6,
        "TANGENTIAL": lb.OpenCVConfig.TANGENTIAL,
        "THIN_PRISM": lb.OpenCVConfig.THIN_PRISM,
    }

    for name, mask in masks.items():
        config = lb.OpenCVConfig(
            image_height=img_h,
            image_width=img_w,
            included_distortion_coefficients=mask,
        )
        result = lb.calibrate_camera(target_points, frames, camera_model_config=config)
        coeffs = result.camera_model.distortion_coeffs

        disabled = ~mask
        assert np.all(coeffs[disabled] == 0), (
            f"{name}: expected zeros at disabled indices {np.where(disabled)[0]}, "
            f"got {coeffs[disabled]}"
        )


def _check_frame_projections(
    result: lb.CalibrationResult,
    target_points: np.ndarray,
    frames: list[lb.Frame],
) -> None:
    model = result.camera_model

    for i, frame in enumerate(frames):
        pose = result.cameras_from_target[i]
        fi = result.frame_diagnostics[i]

        points_in_target = target_points[frame.target_point_indices]
        if result.target_warp is not None:
            points_in_target = result.target_warp.warp_target(points_in_target)

        points_in_cam = pose.apply(points_in_target)
        projected = model.project_points(points_in_cam)

        np.testing.assert_allclose(
            projected, fi.projected_points, atol=1e-6, err_msg=f"Frame {i}"
        )

        expected_residuals = projected - frame.detected_points_in_image
        np.testing.assert_allclose(
            expected_residuals, fi.residuals, atol=1e-6, err_msg=f"Frame {i}"
        )


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_planar_grid(
    cols: int = 12, rows: int = 9, spacing: float = 30.0
) -> np.ndarray:
    """Create a planar calibration grid centred at the origin.

    Returns:
        Grid points, shape (cols * rows, 3) with z=0.
    """
    xs = np.arange(cols) * spacing - (cols - 1) * spacing / 2
    ys = np.arange(rows) * spacing - (rows - 1) * spacing / 2
    gx, gy = np.meshgrid(xs, ys)
    gz = np.zeros_like(gx)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _generate_synthetic_frames(
    rng: np.random.Generator,
    model: lb.OpenCV,
    target_points: np.ndarray,
    num_frames: int = 40,
    noise_sigma: float = 0.1,
) -> list[lb.Frame]:
    """Project a target grid through a known camera model to create frames.

    Args:
        rng: Numpy random generator.
        model: Ground-truth camera model.
        target_points: 3D target coordinates, shape (N, 3).
        num_frames: Number of synthetic views to generate.
        noise_sigma: Gaussian noise added to pixel detections in pixels.

    Returns:
        Synthetic detection frames.
    """
    all_indices = np.arange(len(target_points))
    frames: list[lb.Frame] = []

    for _ in range(num_frames):
        rotvec = rng.normal(scale=0.2, size=3)
        tz = rng.uniform(300, 900)
        tx = rng.normal(scale=40)
        ty = rng.normal(scale=40)
        pose = lb.Pose.from_rotvec_trans(rotvec=rotvec, trans=np.array([tx, ty, tz]))

        points_in_cam = pose.apply(target_points)
        projected = model.project_points(points_in_cam)

        margin = 20
        in_bounds = (
            (projected[:, 0] >= margin)
            & (projected[:, 0] < model.image_width - margin)
            & (projected[:, 1] >= margin)
            & (projected[:, 1] < model.image_height - margin)
            & (points_in_cam[:, 2] > 0)
        )
        if in_bounds.sum() < 10:
            continue

        noise = rng.normal(scale=noise_sigma, size=(in_bounds.sum(), 2))
        detected = projected[in_bounds] + noise
        frames.append(
            lb.Frame(
                target_point_indices=all_indices[in_bounds],
                detected_points_in_image=detected,
            )
        )

    return frames


# ---------------------------------------------------------------------------
# Synthetic focal-length estimation stress tests
# ---------------------------------------------------------------------------

_SYNTHETIC_MODELS = [
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=500.0,
            fy=500.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([-0.3, 0.1, 0.0, 0.0, 0.0]),
        ),
        id="wide_fov_1920x1080_f500",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=1800.0,
            fy=1800.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([0.1, -0.05, 0.001, -0.001, 0.0]),
        ),
        id="narrow_fov_1920x1080_f1800",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=4000,
            image_height=3000,
            fx=3500.0,
            fy=3500.0,
            cx=2000.0,
            cy=1500.0,
            distortion_coeffs=np.array([0.05, 0.01, 0.0, 0.0, -0.002]),
        ),
        id="telephoto_4000x3000_f3500",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=640,
            image_height=480,
            fx=300.0,
            fy=300.0,
            cx=320.0,
            cy=240.0,
            distortion_coeffs=np.array([-0.4, 0.15, 0.0, 0.0, -0.02]),
        ),
        id="low_res_fisheye_640x480_f300",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=3088,
            image_height=2064,
            fx=1350.0,
            fy=1350.0,
            cx=1544.0,
            cy=1032.0,
            distortion_coeffs=np.array([1.5, 0.4, -0.0001, 0.0, 0.008,
                                        1.8, 0.78, 0.06, 0.0, 0.0,
                                        0.0002, 0.0, 0.0005, -0.0003]),
        ),
        id="full14_distortion_3088x2064_f1350",
    ),
]


@pytest.mark.parametrize("ground_truth", _SYNTHETIC_MODELS)
def test_synthetic_auto_focal_length(ground_truth: lb.OpenCV) -> None:
    """Calibrate synthetic data without an initial focal length guess."""
    rng = np.random.default_rng(123)
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points)

    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=lb.OpenCVConfig.STANDARD,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    assert f_err_pct < 2, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.5, f"Residual sigma too high: {sigma:.3f}px"
