"""Test that get_pinhole_model produces a consistent undistorted view."""

import os
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

import lensboy as lb
from lensboy.common_targets.charuco import _detect_charuco

SHOW_DEBUG = os.environ.get("DEBUG_VIS", "") == "1"

DATA_DIR = Path(__file__).parent.parent / "data/test_datasets"
BOARD = cv2.aruco.CharucoBoard(
    (14, 9),
    40,
    30,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
)


def test_get_pinhole_model() -> None:
    """Undistorted pinhole model is geometrically consistent with the spline model.

    Detects a charuco board in the original distorted image and in the
    undistorted image, then cross-projects between the two models and checks
    that the reprojections match the detections.
    """
    spline_model = lb.PinholeSplined.load(DATA_DIR / "spline.json")
    pinhole_model = spline_model.get_pinhole_model()

    distorted_img = iio.imread(DATA_DIR / "charuco.png")

    undistorted_img = pinhole_model.undistort(
        distorted_img, interpolation=cv2.INTER_CUBIC
    )

    frame_distorted = _detect_charuco(distorted_img, BOARD)
    frame_undistorted = _detect_charuco(undistorted_img, BOARD)

    assert frame_distorted is not None, "Failed to detect charuco in distorted image"
    assert frame_undistorted is not None, "Failed to detect charuco in undistorted image"

    # Match detections by corner ID
    ids_dist = frame_distorted.target_point_indices
    ids_undist = frame_undistorted.target_point_indices
    common_ids = np.intersect1d(ids_dist, ids_undist)
    assert len(common_ids) > 20, f"Only {len(common_ids)} common detections"

    dist_mask = np.isin(ids_dist, common_ids)
    undist_mask = np.isin(ids_undist, common_ids)

    # Sort by ID so points correspond
    dist_order = np.argsort(ids_dist[dist_mask])
    undist_order = np.argsort(ids_undist[undist_mask])

    pts_distorted = frame_distorted.detected_points_in_image[dist_mask][dist_order]
    pts_undistorted = frame_undistorted.detected_points_in_image[undist_mask][
        undist_order
    ]

    # Distorted → unproject with spline → project with pinhole → compare to undistorted
    bearing_from_dist = spline_model.normalize_points(pts_distorted)
    reprojected_to_undist = pinhole_model.project_points(bearing_from_dist)

    # Undistorted → unproject with pinhole → project with spline → compare to distorted
    bearing_from_undist = pinhole_model.normalize_points(pts_undistorted)
    reprojected_to_dist = spline_model.project_points(bearing_from_undist)

    err_to_undist = np.linalg.norm(reprojected_to_undist - pts_undistorted, axis=1)
    err_to_dist = np.linalg.norm(reprojected_to_dist - pts_distorted, axis=1)

    if SHOW_DEBUG:

        def _draw(img, detected, reprojected):
            vis = img.copy()
            if vis.ndim == 2:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            for pt in detected:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), 2)
            for pt in reprojected:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), 2)
            return vis

        vis_dist = _draw(distorted_img, pts_distorted, reprojected_to_dist)
        vis_undist = _draw(undistorted_img, pts_undistorted, reprojected_to_undist)

        cv2.imshow("Distorted (green=detected, red=reprojected)", vis_dist)
        cv2.imshow("Undistorted (green=detected, red=reprojected)", vis_undist)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(err_to_undist, bins=30)
        axes[0].set_title("Distorted → Undistorted")
        axes[0].set_xlabel("Reprojection error (px)")
        axes[0].set_ylabel("Count")
        axes[1].hist(err_to_dist, bins=30)
        axes[1].set_title("Undistorted → Distorted")
        axes[1].set_xlabel("Reprojection error (px)")
        plt.tight_layout()
        plt.show()

    assert err_to_undist.mean() < 0.3, (
        f"Distorted→undistorted mean error too high: {err_to_undist.mean():.3f} px"
    )
    assert err_to_dist.mean() < 0.3, (
        f"Undistorted→distorted mean error too high: {err_to_dist.mean():.3f} px"
    )

    np.testing.assert_allclose(
        reprojected_to_undist,
        pts_undistorted,
        atol=1.0,
        err_msg="Distorted→undistorted cross-projection mismatch",
    )
    np.testing.assert_allclose(
        reprojected_to_dist,
        pts_distorted,
        atol=1.0,
        err_msg="Undistorted→distorted cross-projection mismatch",
    )


def test_remap_table_consistency() -> None:
    """Remap tables match spline.project(pinhole.normalize(pt)) at every pixel."""
    spline_model = lb.PinholeSplined.load(DATA_DIR / "spline.json")
    pinhole_model = spline_model.get_pinhole_model()

    # Sample a grid of integer undistorted pixel coordinates
    xs = np.linspace(0, pinhole_model.image_width - 1, 50, dtype=int)
    ys = np.linspace(0, pinhole_model.image_height - 1, 50, dtype=int)
    grid_x, grid_y = np.meshgrid(xs, ys)
    cols = grid_x.ravel()
    rows = grid_y.ravel()
    undist_pts = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])

    # What the remap tables say: direct lookup at integer pixel positions
    from_map = np.column_stack(
        [
            pinhole_model.map_x[rows, cols],
            pinhole_model.map_y[rows, cols],
        ]
    )

    # What the models say: unproject through pinhole, project through spline
    bearings = pinhole_model.normalize_points(undist_pts)
    from_models = spline_model.project_points(bearings)

    err = np.linalg.norm(from_map - from_models, axis=1)

    if SHOW_DEBUG:
        plt.figure(figsize=(6, 4))
        plt.hist(err, bins=30)
        plt.title("Remap table vs model projection")
        plt.xlabel("Error (px)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    np.testing.assert_allclose(
        from_map,
        from_models,
        atol=0.0005,
        err_msg="Remap table does not match spline.project(pinhole.normalize(pt))",
    )
