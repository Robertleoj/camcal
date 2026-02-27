import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from random import seed
from timeit import default_timer
from typing import Generic, TypeVar, overload

import cv2
import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig
from lensboy.camera_models.opencv import OpenCV, OpenCVConfig
from lensboy.camera_models.pinhole_splined import (
    PinholeSplined,
    PinholeSplinedCalibrationMetadata,
    PinholeSplinedConfig,
)
from lensboy.geometry.pose import Pose

LOG = logging.getLogger(__name__)

DEFAULT_OUTLIER_THRESHOLD = 3.0
MAX_OUTLIER_FILTER_PASSES = 2


@dataclass
class Detection:
    target_point_indices: np.ndarray
    detected_points_in_image: np.ndarray

    def __post_init__(self):
        assert (
            self.target_point_indices.shape[0] == self.detected_points_in_image.shape[0]
        ), (
            "Expected target_point_indices to have the "
            "length shape as detected_points_in_image"
        )

        assert self.target_point_indices.ndim == 1, (
            f"Expected 1D target_point_indices, got {self.target_point_indices.ndim}D"
        )
        assert np.issubdtype(self.target_point_indices.dtype, np.integer), (
            "Expected integer dtype for target_point_indices, ",
            f"got {self.target_point_indices.dtype}",
        )

        assert (
            self.detected_points_in_image.ndim == 2
            and self.detected_points_in_image.shape[1] == 2
        ), f"Expected (N, 2) points in image, got {self.detected_points_in_image.shape}"
        assert np.issubdtype(self.detected_points_in_image.dtype, np.floating), (
            "Expected floating dtype for points, "
            f"got {self.detected_points_in_image.dtype}"
        )

    def to_cpp(self) -> tuple[list[int], list[np.ndarray]]:
        return (self.target_point_indices.tolist(), list(self.detected_points_in_image))

    def __len__(self):
        return self.target_point_indices.shape[0]


T = TypeVar("T", bound=CameraModel)


@dataclass
class DetectionInfo:
    # N x 2
    projected_points: np.ndarray

    # N x 2
    residuals: np.ndarray

    # N
    inlier_mask: np.ndarray


@dataclass
class CalibrationResult(Generic[T]):
    optimized_camera_model: T
    optimized_cameras_T_target: list[Pose]
    detection_infos: list[DetectionInfo]


def _project_and_calculate_residuals(
    target_points: np.ndarray,
    camera_from_target: Pose,
    detection: Detection,
    model: OpenCV | PinholeSplined,
) -> tuple[np.ndarray, np.ndarray]:
    point_indices = detection.target_point_indices

    points_in_target = target_points[point_indices]
    points_in_camera = camera_from_target.apply(points_in_target)

    projected_points_in_image = model.project_points(points_in_camera)

    residuals = detection.detected_points_in_image - projected_points_in_image

    return projected_points_in_image, residuals


def _mad_sigma_1d(x: np.ndarray) -> float:
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # 1.4826 = 1 / Phi^{-1}(0.75)  (MAD->sigma for 1D normal)
    return 1.4826 * mad


def _robust_sigma_xy(residuals: list[np.ndarray]) -> float:
    R = np.concatenate(residuals, axis=0)  # (M,2)
    sx = _mad_sigma_1d(R[:, 0])
    sy = _mad_sigma_1d(R[:, 1])
    # combine to one sigma (assume roughly same scale)
    return float(np.sqrt(0.5 * (sx * sx + sy * sy)))


def _radius_threshold_from_k(k: float) -> float:
    """
    If x,y ~ N(0, sigma^2), then r^2/sigma^2 ~ chi2(df=2).
    A 1D ±kσ "inlier probability" is p = 2*Phi(k)-1.
    For df=2: chi2 CDF is 1-exp(-x/2), so quantile is x = -2 ln(1-p).
    Threshold radius = sqrt(x) * sigma.
    """
    # 1D normal CDF via erf
    p1 = 0.5 * (1.0 + math.erf(k / math.sqrt(2.0)))
    p = 2.0 * p1 - 1.0
    x = -2.0 * np.log(1.0 - p)
    return float(np.sqrt(x))


def _filter_outliers(
    detections: list,
    residuals: list[np.ndarray],
    k: float = 3.5,
    sigma_floor_px: float = 0.25,  # prevents collapse
) -> list:
    sigma = max(_robust_sigma_xy(residuals), sigma_floor_px)
    gate = _radius_threshold_from_k(k) * sigma  # radius in pixels

    filtered = []
    for det, r in zip(detections, residuals):
        inlier_mask = np.linalg.norm(r, axis=1) <= gate
        filtered.append(
            type(det)(
                det.target_point_indices[inlier_mask],
                det.detected_points_in_image[inlier_mask],
            )
        )
    return filtered


def _opencv_calibrate_inner(
    curr_intrinsics: OpenCV,
    config: OpenCVConfig,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    detections: list[Detection],
) -> tuple[OpenCV, list[Pose]]:
    params = curr_intrinsics._params()
    mask = config.optimize_mask()
    intrinsics_param_optimize_mask = mask.tolist()

    cameras_from_target_in = [p.to_cpp() for p in curr_cameras_from_target]

    LOG.info("Running full optimization...")
    start_time = default_timer()
    result = lbb.calibrate_opencv(
        intrinsics_initial_value=params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_target=cameras_from_target_in,
        target_points=list(target_points),
        detections=[d.to_cpp() for d in detections],
    )
    end_time = default_timer()
    LOG.info(f"Ran optimizer in {end_time - start_time:.2f}s")

    optimized_intrinsics = curr_intrinsics._with_params(result["intrinsics"])

    optimized_cameras_from_target: list[Pose] = [
        Pose.from_cpp(np.array(a)) for a in result["cameras_from_target"]
    ]

    return optimized_intrinsics, optimized_cameras_from_target


def _compute_detection_infos(
    intrinsics: OpenCV | PinholeSplined,
    cameras_from_target: list[Pose],
    original_detections: list[Detection],
    filtered_detections: list[Detection] | None,
    target_points: np.ndarray,
) -> list[DetectionInfo]:
    detection_infos: list[DetectionInfo] = []
    for i in range(len(cameras_from_target)):
        projected, residuals = _project_and_calculate_residuals(
            target_points,
            cameras_from_target[i],
            original_detections[i],
            intrinsics,
        )

        if filtered_detections is not None:
            inlier_mask = np.isin(
                original_detections[i].target_point_indices,
                filtered_detections[i].target_point_indices,
            )
        else:
            inlier_mask = np.ones(len(original_detections[i]), dtype=bool)

        detection_infos.append(DetectionInfo(projected, residuals, inlier_mask))

    return detection_infos


def _log_residual_stats(detection_infos: list[DetectionInfo]) -> None:
    inlier_norms = np.concatenate(
        [
            np.linalg.norm(di.residuals[di.inlier_mask], axis=1)
            for di in detection_infos
            if di.inlier_mask.any()
        ]
    )
    LOG.info(
        f"Residuals (inliers): mean={np.mean(inlier_norms):.3f}px, "
        f"worst={np.max(inlier_norms):.3f}px"
    )


def _run_with_outlier_filtering(
    optimize_fn: Callable,
    initial_intrinsics,
    initial_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    original_detections: list[Detection],
    num_stddevs_outlier_threshold: float | None,
):
    curr_intrinsics = initial_intrinsics
    curr_cameras_from_target = initial_cameras_from_target
    curr_detections = original_detections
    total_observations = sum(len(d) for d in original_detections)

    for i in range(MAX_OUTLIER_FILTER_PASSES + 1):
        curr_intrinsics, curr_cameras_from_target = optimize_fn(
            curr_intrinsics, curr_cameras_from_target, target_points, curr_detections
        )

        if num_stddevs_outlier_threshold is None or i == MAX_OUTLIER_FILTER_PASSES:
            break

        curr_residuals = [
            _project_and_calculate_residuals(target_points, cam, det, curr_intrinsics)[1]
            for cam, det in zip(curr_cameras_from_target, curr_detections)
        ]

        new_detections = _filter_outliers(
            curr_detections, curr_residuals, num_stddevs_outlier_threshold
        )

        if all(
            len(new_det) == len(old_det)
            for new_det, old_det in zip(new_detections, curr_detections)
        ):
            break

        curr_detections = new_detections

        total_remaining = sum(len(d) for d in curr_detections)
        total_outliers = total_observations - total_remaining
        LOG.info(
            f"Threw out some outliers, now have {total_outliers}/{total_observations}"
            f" ({total_outliers / total_observations * 100:.1f}%) - going again..."
        )

    return curr_intrinsics, curr_cameras_from_target, curr_detections


def _initialize_poses_with_pnp(
    initial_intrinsics: OpenCV,
    target_points: np.ndarray,
    detections: list[Detection],
) -> list[Pose]:
    K = initial_intrinsics.K()
    dist_coeffs = np.zeros(5, dtype=np.float64)

    poses = []
    for det in detections:
        obj_pts = target_points[det.target_point_indices].astype(np.float64)
        img_pts = det.detected_points_in_image.astype(np.float64)

        if len(obj_pts) >= 4:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs)
            if success:
                poses.append(
                    Pose.from_rotvec_trans(
                        rotvec=rvec.flatten(), trans=tvec.flatten()
                    )
                )
                continue

        LOG.warning("PnP init failed for detection, using fallback pose")
        poses.append(Pose.from_tz(100))

    return poses


def _opencv_calibrate(
    target_points: np.ndarray,
    detections: list[Detection],
    config: OpenCVConfig,
    num_stddevs_outlier_threshold: float | None,
) -> CalibrationResult[OpenCV]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    initial_intrinsics = config.get_initial_value()
    LOG.info("Computing initial poses with PnP...")
    initial_cameras_from_target = _initialize_poses_with_pnp(
        initial_intrinsics, target_points, detections
    )

    def optimize_fn(intrinsics, cameras, tp, dets):
        return _opencv_calibrate_inner(intrinsics, config, cameras, tp, dets)

    curr_intrinsics, curr_cameras_from_target, curr_detections = (
        _run_with_outlier_filtering(
            optimize_fn,
            initial_intrinsics,
            initial_cameras_from_target,
            target_points,
            detections,
            num_stddevs_outlier_threshold,
        )
    )

    detection_infos = _compute_detection_infos(
        curr_intrinsics,
        curr_cameras_from_target,
        detections,
        curr_detections if curr_detections is not detections else None,
        target_points,
    )

    _log_residual_stats(detection_infos)
    return CalibrationResult(
        optimized_camera_model=curr_intrinsics,
        optimized_cameras_T_target=curr_cameras_from_target,
        detection_infos=detection_infos,
    )


def _pinhole_splined_refine_inner(
    curr_intrinsics: PinholeSplined,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    detections: list[Detection],
) -> tuple[PinholeSplined, list[Pose]]:
    fine_tune_result = lbb.fine_tune_pinhole_splined(
        model_config=curr_intrinsics._cpp_config(),
        intrinsics_parameters=curr_intrinsics._cpp_params(),
        cameras_from_target=[pose.to_cpp() for pose in curr_cameras_from_target],
        target_points=list(target_points),
        detections=[d.to_cpp() for d in detections],
    )

    optimized_cameras_from_target = [
        Pose.from_cpp(np.array(a)) for a in fine_tune_result["cameras_from_target"]
    ]

    optimized_intrinsics = replace(
        curr_intrinsics,
        dx_grid=fine_tune_result["dx_grid"],
        dy_grid=fine_tune_result["dy_grid"],
    )

    return optimized_intrinsics, optimized_cameras_from_target


def _compute_fov_from_opencv(
    opencv_model: OpenCV, buffer_deg: float = 2.0
) -> tuple[float, float]:
    return (
        opencv_model.fov_deg_x + buffer_deg,
        opencv_model.fov_deg_y + buffer_deg,
    )


def _calibrate_pinhole_splined(
    target_points: np.ndarray,
    detections: list[Detection],
    config: PinholeSplinedConfig,
    num_stddevs_outlier_threshold: float | None,
) -> CalibrationResult[PinholeSplined]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    opencv_config = OpenCVConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        initial_focal_length=config.initial_focal_length,
        included_distoriton_coefficients=OpenCVConfig.FULL_12,
    )

    LOG.info("Calibrating seed opencv model...")
    start_time = default_timer()
    opencv_calibration_result = _opencv_calibrate(
        target_points, detections, opencv_config, None
    )
    end_time = default_timer()
    LOG.info(f"OpenCV seed model ready in {end_time - start_time:.1f}s")

    opencv_model = opencv_calibration_result.optimized_camera_model

    fov_deg_x, fov_deg_y = _compute_fov_from_opencv(opencv_model)
    LOG.info(f"Computed FOV from OpenCV model: {fov_deg_x:.1f}° x {fov_deg_y:.1f}°")

    cpp_config = lbb.PinholeSplinedConfig(
        config.image_width,
        config.image_height,
        fov_deg_x,
        fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
    )

    LOG.info("Calculating matching spline model...")
    start_time = default_timer()
    out_dict = lbb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(), cpp_config
    )
    end_time = default_timer()
    LOG.info(f"Matching spline model ready in {end_time - start_time:.1f}s")

    x_knots = out_dict["x_knots"]
    y_knots = out_dict["y_knots"]

    prior_model = PinholeSplined(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=opencv_model.fx,
        fy=opencv_model.fy,
        cx=opencv_model.cx,
        cy=opencv_model.cy,
        dx_grid=x_knots,
        dy_grid=y_knots,
        num_knots_x=config.num_knots_x,
        num_knots_y=config.num_knots_y,
        fov_deg_x=fov_deg_x,
        fov_deg_y=fov_deg_y,
    )

    cameras_from_target = opencv_calibration_result.optimized_cameras_T_target

    def optimize_fn(intrinsics, cameras, tp, dets):
        LOG.info("Running full optimization...")
        start = default_timer()
        result = _pinhole_splined_refine_inner(intrinsics, cameras, tp, dets)
        LOG.info(f"Performed full optimization in {default_timer() - start:.2f}s")
        return result

    curr_intrinsics, curr_cameras, curr_detections = _run_with_outlier_filtering(
        optimize_fn,
        prior_model,
        cameras_from_target,
        target_points,
        detections,
        num_stddevs_outlier_threshold,
    )

    detection_infos = _compute_detection_infos(
        curr_intrinsics,
        curr_cameras,
        detections,
        curr_detections if curr_detections is not detections else None,
        target_points,
    )

    metadata = PinholeSplinedCalibrationMetadata(
        seed_opencv_distortion_params=opencv_model.distortion_coeffs
    )

    curr_intrinsics = replace(curr_intrinsics, calibration_metadata=metadata)

    _log_residual_stats(detection_infos)
    return CalibrationResult(
        optimized_camera_model=curr_intrinsics,
        optimized_cameras_T_target=curr_cameras,
        detection_infos=detection_infos,
    )


@overload
def calibrate_camera(
    target_points: np.ndarray,
    detections: list[Detection],
    camera_model_config: PinholeSplinedConfig,
    num_stddevs_outlier_threshold: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[PinholeSplined]: ...


@overload
def calibrate_camera(
    target_points: np.ndarray,
    detections: list[Detection],
    camera_model_config: OpenCVConfig,
    num_stddevs_outlier_threshold: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[OpenCV]: ...


def calibrate_camera(
    target_points: np.ndarray,
    detections: list[Detection],
    camera_model_config: CameraModelConfig,
    num_stddevs_outlier_threshold: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    if isinstance(camera_model_config, PinholeSplinedConfig):
        return _calibrate_pinhole_splined(
            target_points, detections, camera_model_config, num_stddevs_outlier_threshold
        )

    if isinstance(camera_model_config, OpenCVConfig):
        return _opencv_calibrate(
            target_points, detections, camera_model_config, num_stddevs_outlier_threshold
        )

    raise RuntimeError("Invalid config")
