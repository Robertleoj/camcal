"""Integration test for target warp recovery on a synthetic dataset."""

from pathlib import Path

import numpy as np

import lensboy as lb
from lensboy.calibration.calibrate import _make_warp_coordinates

OPENCV_MODEL_PATH = Path(__file__).parent.parent / "data/camera_models/opencv.json"


def _make_synthetic_warp_dataset(
    rng: np.random.Generator,
    model: lb.OpenCV,
    target_warp: lb.TargetWarp,
) -> tuple[np.ndarray, list[lb.Frame]]:
    """Build a synthetic calibration dataset with known target warp.

    Creates an almost-planar grid, applies the given warp to generate pixel
    detections, and returns the *unwarped* target points together with the
    frames (so calibration must recover the warp).

    Args:
        rng: Numpy random generator.
        model: Ground-truth camera model used for projection.
        target_warp: The warp to bake into the synthetic detections.

    Returns:
        target_points: Unwarped (almost-planar) target grid, shape (N, 3).
        frames: Synthetic detection frames.
    """
    target_points = _make_grid(rng)
    warped_target = target_warp.warp_target(target_points)

    all_indices = np.arange(len(target_points))

    frames: list[lb.Frame] = []
    for _ in range(30):
        rotvec = rng.normal(scale=0.15, size=3)
        tz = rng.uniform(400, 800)
        tx = rng.normal(scale=30)
        ty = rng.normal(scale=30)
        pose = lb.Pose.from_rotvec_trans(rotvec=rotvec, trans=np.array([tx, ty, tz]))

        points_in_cam = pose.apply(warped_target)
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

        frames.append(
            lb.Frame(
                target_point_indices=all_indices[in_bounds],
                detected_points_in_image=projected[in_bounds],
            )
        )

    return target_points, frames


def _make_grid(rng: np.random.Generator) -> np.ndarray:
    """Create an almost-planar 12x9 grid with small z perturbations.

    Args:
        rng: Numpy random generator.

    Returns:
        Grid points, shape (N, 3).
    """
    cols, rows = 12, 9
    spacing = 30.0
    xs = np.arange(cols) * spacing - (cols - 1) * spacing / 2
    ys = np.arange(rows) * spacing - (rows - 1) * spacing / 2
    gx, gy = np.meshgrid(xs, ys)
    gz = rng.normal(scale=5.0, size=gx.shape)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def test_warp_recovery_opencv() -> None:
    """Calibrate with known warp and verify warped geometry is recovered."""
    model = lb.OpenCV.load(OPENCV_MODEL_PATH)
    rng = np.random.default_rng(42)

    grid = _make_grid(np.random.default_rng(42))
    warp_coords = _make_warp_coordinates(grid)
    assert warp_coords is not None
    ground_truth_coeffs = (3.0, -1.3, 0.1, 0.15, -0.2)
    ground_truth_warp = lb.TargetWarp(
        warp_coordinates=warp_coords, object_warp=ground_truth_coeffs
    )

    target_points, frames = _make_synthetic_warp_dataset(rng, model, ground_truth_warp)

    config = lb.OpenCVConfig(
        image_height=model.image_height,
        image_width=model.image_width,
        initial_focal_length=float(model.fx),
        included_distoriton_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(
        target_points,
        frames,
        camera_model_config=config,
        estimate_target_warp=True,
    )

    assert result.warp_info is not None, "Warp should have been estimated"

    gt_warped = ground_truth_warp.warp_target(target_points)
    recovered_warped = result.warp_info.warp_target(target_points)
    np.testing.assert_allclose(
        recovered_warped,
        gt_warped,
        atol=0.02,
        err_msg="Recovered warped target doesn't match ground truth",
    )
