"""Integration tests using a real charuco dataset and synthetic data."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

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
    fi0 = result.frame_diagnostics[0]
    assert fi0 is not None, "Expected diagnostics for corrupted frame"
    assert not fi0.inlier_mask.any(), (
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
        if pose is None or fi is None:
            continue

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


def _make_planar_grid(cols: int = 12, rows: int = 9, spacing: float = 30.0) -> np.ndarray:
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

    # Scale camera distances to the target's spatial extent
    centroid = target_points.mean(axis=0)
    target_radius = np.linalg.norm(target_points - centroid, axis=1).max()
    base_dist = max(
        target_radius * model.fx / (min(model.image_width, model.image_height) / 2), 100.0
    )

    for _ in range(num_frames):
        rotvec = rng.normal(scale=0.2, size=3)
        tz = rng.uniform(base_dist * 0.8, base_dist * 2.5)
        tx = rng.normal(scale=base_dist * 0.1)
        ty = rng.normal(scale=base_dist * 0.1)
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
            distortion_coeffs=np.array(
                [
                    1.5,
                    0.4,
                    -0.0001,
                    0.0,
                    0.008,
                    1.8,
                    0.78,
                    0.06,
                    0.0,
                    0.0,
                    0.0002,
                    0.0,
                    0.0005,
                    -0.0003,
                ]
            ),
        ),
        id="full14_distortion_3088x2064_f1350",
    ),
]


@pytest.mark.parametrize("ground_truth", _SYNTHETIC_MODELS)
def test_synthetic_auto_focal_length(ground_truth: lb.OpenCV) -> None:
    """Calibrate synthetic data without an initial focal length guess."""
    rng = np.random.default_rng(123)
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=80)

    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=lb.OpenCVConfig.STANDARD,
    )
    result = lb.calibrate_camera(target_points, frames, camera_model_config=config)

    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    print(f"focal error pct {f_err_pct}")
    assert f_err_pct < 0.05, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"


# ---------------------------------------------------------------------------
# Exotic target generators
# ---------------------------------------------------------------------------


def _apply_random_similarity_transform(
    rng: np.random.Generator, points: np.ndarray
) -> np.ndarray:
    """Apply a random rotation, translation, and uniform scale to target points.

    Args:
        rng: Numpy random generator.
        points: 3D points, shape (N, 3).

    Returns:
        Transformed points, shape (N, 3).
    """
    scale = rng.uniform(0.01, 1000.0)
    rotvec = rng.normal(scale=3, size=3)
    # Keep translation modest so the target stays visible from the synthetic cameras
    trans = rng.normal(scale=30.0, size=3)
    pose = lb.Pose.from_rotvec_trans(rotvec=rotvec, trans=trans)
    return pose.apply(points * scale)


def _make_random_planar(
    rng: np.random.Generator, n_points: int = 100, extent: float = 150.0
) -> np.ndarray:
    """Create randomly scattered points on the z=0 plane.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        extent: Half-width of the square region.

    Returns:
        Points, shape (n_points, 3) with z=0.
    """
    xy = rng.uniform(-extent, extent, size=(n_points, 2))
    z = np.zeros((n_points, 1))
    return np.hstack([xy, z])


def _make_ball(
    rng: np.random.Generator, n_points: int = 100, radius: float = 80.0
) -> np.ndarray:
    """Create points uniformly distributed inside a 3D ball.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        radius: Ball radius.

    Returns:
        Points, shape (n_points, 3).
    """
    # Rejection sampling for uniform distribution in a ball
    points = []
    while len(points) < n_points:
        candidates = rng.uniform(-radius, radius, size=(n_points * 2, 3))
        inside = np.linalg.norm(candidates, axis=1) <= radius
        points.extend(candidates[inside].tolist())
    return np.array(points[:n_points])


def _make_two_intersecting_planes(
    rng: np.random.Generator,
    n_per_plane: int = 50,
    extent: float = 120.0,
    angle_deg: float = 30.0,
) -> np.ndarray:
    """Create points on two planes intersecting along the y-axis.

    Args:
        rng: Numpy random generator.
        n_per_plane: Number of points per plane.
        extent: Half-width of each plane.
        angle_deg: Half-angle between the two planes.

    Returns:
        Points, shape (2 * n_per_plane, 3).
    """
    angle = np.radians(angle_deg)

    # Plane 1: tilted by +angle around y-axis
    xy1 = rng.uniform(-extent, extent, size=(n_per_plane, 2))
    plane1 = np.column_stack(
        [xy1[:, 0] * np.cos(angle), xy1[:, 1], xy1[:, 0] * np.sin(angle)]
    )

    # Plane 2: tilted by -angle around y-axis
    xy2 = rng.uniform(-extent, extent, size=(n_per_plane, 2))
    plane2 = np.column_stack(
        [xy2[:, 0] * np.cos(-angle), xy2[:, 1], xy2[:, 0] * np.sin(-angle)]
    )

    return np.vstack([plane1, plane2])


def _make_hemisphere(
    rng: np.random.Generator, n_points: int = 100, radius: float = 100.0
) -> np.ndarray:
    """Create points on the surface of a hemisphere (z >= 0).

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        radius: Hemisphere radius.

    Returns:
        Points, shape (n_points, 3).
    """
    # Uniform sampling on a sphere via normal distribution, then take z >= 0
    points = []
    while len(points) < n_points:
        raw = rng.normal(size=(n_points * 3, 3))
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        raw *= radius
        upper = raw[raw[:, 2] >= 0]
        points.extend(upper.tolist())
    return np.array(points[:n_points])


def _make_cylinder(
    rng: np.random.Generator,
    n_points: int = 100,
    radius: float = 60.0,
    height: float = 200.0,
) -> np.ndarray:
    """Create points on the surface of a cylinder aligned with the y-axis.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        radius: Cylinder radius.
        height: Cylinder height.

    Returns:
        Points, shape (n_points, 3).
    """
    theta = rng.uniform(0, 2 * np.pi, size=n_points)
    y = rng.uniform(-height / 2, height / 2, size=n_points)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def _make_random_3d_cluster(
    rng: np.random.Generator, n_points: int = 100, extent: float = 100.0
) -> np.ndarray:
    """Create randomly scattered points in a 3D box.

    Args:
        rng: Numpy random generator.
        n_points: Number of points to generate.
        extent: Half-width of the box along each axis.

    Returns:
        Points, shape (n_points, 3).
    """
    return rng.uniform(-extent, extent, size=(n_points, 3))


# ---------------------------------------------------------------------------
# Synthetic exotic-target tests
# ---------------------------------------------------------------------------

_EXOTIC_TARGET_MODELS = [
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([0.02, -0.005, 0.0, 0.0, 0.0]),
        ),
        id="minimal_distortion",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=1920,
            image_height=1080,
            fx=800.0,
            fy=800.0,
            cx=960.0,
            cy=540.0,
            distortion_coeffs=np.array([-0.3, 0.12, 0.001, -0.002, -0.04]),
        ),
        id="medium_distortion",
    ),
    pytest.param(
        lb.OpenCV(
            image_width=3088,
            image_height=2064,
            fx=1354.5124985080904,
            fy=1354.3181984440832,
            cx=1514.104403863959,
            cy=1076.8896015546975,
            distortion_coeffs=np.array(
                [
                    1.721749851751697,
                    0.4929049527387092,
                    -0.00012249059620334055,
                    6.571195104303754e-05,
                    0.010826498817585985,
                    2.040924560626435,
                    0.9497700975338902,
                    0.0744348380132027,
                    -6.852182482012876e-05,
                    -8.155006688534965e-06,
                    0.00021009345380118726,
                    -4.392347849675705e-06,
                    0.0005388341393511124,
                    -0.0003861499673898091,
                ]
            ),
        ),
        id="extreme_distortion",
    ),
]

_EXOTIC_TARGETS = [
    pytest.param(_make_random_planar, id="random_planar"),
    pytest.param(_make_ball, id="ball"),
    pytest.param(_make_two_intersecting_planes, id="two_intersecting_planes"),
    pytest.param(_make_hemisphere, id="hemisphere"),
    pytest.param(_make_cylinder, id="cylinder"),
    pytest.param(_make_random_3d_cluster, id="random_3d_cluster"),
]


@pytest.mark.parametrize("ground_truth", _EXOTIC_TARGET_MODELS)
@pytest.mark.parametrize("make_target", _EXOTIC_TARGETS)
def test_synthetic_exotic_targets(
    ground_truth: lb.OpenCV,
    make_target: Callable[[np.random.Generator], np.ndarray],
) -> None:
    """Calibrate with exotic (non-grid) calibration targets."""
    rng = np.random.default_rng(777)
    target_points = make_target(rng)
    target_points = _apply_random_similarity_transform(rng, target_points)

    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=40)
    assert len(frames) >= 10, f"Too few valid frames ({len(frames)})"

    # Fit the same distortion terms the ground truth uses
    dist_mask = ground_truth.distortion_coeffs != 0
    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=dist_mask,
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
    )

    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    print(f"focal error pct: {f_err_pct}")
    assert f_err_pct < 0.05, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    print(f"sigma: {sigma}")
    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"

    _check_frame_projections(result, target_points, frames)


def test_pnp_failure() -> None:
    """Verify that frames failing initial PnP are recovered after optimization.

    Uses a wrong initial focal length so PnP produces bad poses initially.
    After the first optimization refines intrinsics, the retry should recover them.
    """
    ground_truth = lb.OpenCV(
        image_width=3088,
        image_height=2064,
        fx=1354.5124985080904,
        fy=1354.3181984440832,
        cx=1514.104403863959,
        cy=1076.8896015546975,
        distortion_coeffs=np.array(
            [
                1.721749851751697,
                0.4929049527387092,
                -0.00012249059620334055,
                6.571195104303754e-05,
                0.010826498817585985,
                2.040924560626435,
                0.9497700975338902,
                0.0744348380132027,
                -6.852182482012876e-05,
                -8.155006688534965e-06,
                0.00021009345380118726,
                -4.392347849675705e-06,
                0.0005388341393511124,
                -0.0003861499673898091,
            ]
        ),
    )

    rng = np.random.default_rng(999)
    target_points = _make_planar_grid()
    frames = _generate_synthetic_frames(rng, ground_truth, target_points, num_frames=60)
    assert len(frames) >= 20, f"Too few valid frames ({len(frames)})"

    # Inject some frames with only 3 detections — too few for PnP (needs >= 4).
    # These should fail initially but be recovered after the first optimization
    # refines intrinsics and retries with distortion-aware PnP.
    sparse_frames = []
    for f in frames[:5]:
        sparse_frames.append(
            lb.Frame(
                target_point_indices=f.target_point_indices[:3],
                detected_points_in_image=f.detected_points_in_image[:3],
            )
        )
    frames_with_sparse = sparse_frames + frames

    dist_mask = ground_truth.distortion_coeffs != 0
    config = lb.OpenCVConfig(
        image_height=ground_truth.image_height,
        image_width=ground_truth.image_width,
        included_distortion_coefficients=dist_mask,
    )
    result = lb.calibrate_camera(
        target_points,
        frames_with_sparse,
        camera_model_config=config,
    )

    # Output lists must match input length
    n_total = len(frames_with_sparse)
    assert len(result.cameras_from_target) == n_total
    assert len(result.frame_diagnostics) == n_total
    assert len(result.frames) == n_total

    # The first 5 frames (sparse, <4 points) should have None pose/diagnostics
    for i in range(5):
        assert result.cameras_from_target[i] is None, (
            f"Expected None pose for sparse frame {i}"
        )
        assert result.frame_diagnostics[i] is None, (
            f"Expected None diagnostics for sparse frame {i}"
        )

    # The remaining frames (full detections) should all have valid pose/diagnostics
    for i in range(5, n_total):
        assert result.cameras_from_target[i] is not None, (
            f"Expected valid pose for frame {i}"
        )
        assert result.frame_diagnostics[i] is not None, (
            f"Expected valid diagnostics for frame {i}"
        )

    # Should still converge — sparse frames are just ignored
    recovered = result.camera_model
    f_err_pct = abs(recovered.fx - ground_truth.fx) / ground_truth.fx * 100
    assert f_err_pct < 0.05, (
        f"Focal length off by {f_err_pct:.1f}%: "
        f"recovered {recovered.fx:.1f} vs ground truth {ground_truth.fx:.1f}"
    )

    sigma = result.residual_sigma_map()
    assert sigma < 0.11, f"Residual sigma too high: {sigma:.3f}px"
