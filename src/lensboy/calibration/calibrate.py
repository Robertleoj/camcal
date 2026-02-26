import logging
from dataclasses import dataclass, replace
from typing import Generic, TypeVar, overload

import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig
from lensboy.camera_models.opencv import OpenCV, OpenCVConfig
from lensboy.camera_models.pinhole_splined import PinholeSplined, PinholeSplinedConfig
from lensboy.geometry.pose import Pose

LOG = logging.getLogger(__name__)


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


def _mad(arr: np.ndarray):
    return 1.4826 * np.median(np.abs(arr - np.median(arr)))


def _filter_outliers_single_detection(
    detection: Detection,
    residuals: np.ndarray,
    num_stddevs_outlier_threshold: float,
) -> Detection:
    residual_norms = np.linalg.norm(residuals, axis=1)
    residual_mad = _mad(residual_norms)

    outlier_mask = residual_norms > (num_stddevs_outlier_threshold * residual_mad)
    inlier_mask = ~outlier_mask

    filtered_point_indices = detection.target_point_indices[inlier_mask]
    filtered_points = detection.detected_points_in_image[inlier_mask]

    return Detection(filtered_point_indices, filtered_points)


def _filter_outliers(
    detections: list[Detection],
    residuals: np.ndarray,
    num_stddevs_outlier_threshold: float,
) -> list[Detection]:
    filtered_detections = []

    for detection, residuals_detection in zip(detections, residuals):
        filtered_detection = _filter_outliers_single_detection(
            detection, residuals_detection, num_stddevs_outlier_threshold
        )

        filtered_detections.append(filtered_detection)

    return filtered_detections


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

    result = lbb.calibrate_opencv(
        intrinsics_initial_value=params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_target=cameras_from_target_in,
        target_points=list(target_points),
        detections=[d.to_cpp() for d in detections],
    )

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
    num_cameras = len(detections)

    initial_intrinsics = config.get_initial_value()

    # TODO: get initial poses with PnP
    initial_cameras_from_target = [Pose.from_tz(100) for _ in range(num_cameras)]

    if num_stddevs_outlier_threshold is None:
        optimized_intrinsics, optimized_cameras_from_target = _opencv_calibrate_inner(
            initial_intrinsics,
            config,
            initial_cameras_from_target,
            target_points,
            detections,
        )

        return CalibrationResult()

    original_detections = detections
    curr_detections = detections
    curr_intrinsics = initial_intrinsics
    curr_cameras_from_target = initial_cameras_from_target

    while True:
        curr_intrinsics, curr_cameras_from_target = _opencv_calibrate_inner(
            curr_intrinsics,
            config,
            curr_cameras_from_target,
            target_points,
            curr_detections,
        )

        projection_results = [
            _project_and_calculate_residuals(
                target_points, cam_from_target, detection, curr_intrinsics
            )
            for cam_from_target, detection in zip(curr_cameras_from_target, detections)
        ]

        new_detections = _filter_outliers(
            curr_detections,
            calibration_result.detection_infos,
            num_stddevs_outlier_threshold,
        )

        if all(
            len(new_det) == len(old_det)
            for new_det, old_det in zip(new_detections, curr_detections)
        ):
            break

        curr_detections = new_detections

    original_detections_detection_infos = [
        _compute_detection_info(
            target_points, cam_from_target, detection, curr_intrinsics
        )
        for cam_from_target, detection in zip(
            curr_cameras_from_target, original_detections
        )
    ]

    return CalibrationResult(
        optimized_camera_model=curr_intrinsics,
        optimized_cameras_T_target=curr_cameras_from_target,
        detection_infos=original_detections_detection_infos,
    )


def _pinhole_splined_refine_inner(
    curr_intrinsics: PinholeSplined,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    detections: list[Detection],
) -> CalibrationResult[PinholeSplined]:
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

    fine_tuned_dx_grid = fine_tune_result["dx_grid"]
    fine_tuned_dy_grid = fine_tune_result["dy_grid"]

    optimized_intrinsics = replace(
        curr_intrinsics, dx_grid=fine_tuned_dx_grid, dy_grid=fine_tuned_dy_grid
    )

    detection_infos = [
        _compute_detection_info(
            target_points, cam_from_target, detection, optimized_intrinsics
        )
        for cam_from_target, detection in zip(optimized_cameras_from_target, detections)
    ]

    return CalibrationResult(
        optimized_camera_model=optimized_intrinsics,
        optimized_cameras_T_target=optimized_cameras_from_target,
        detection_infos=detection_infos,
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

    opencv_calibration_result = _opencv_calibrate(
        target_points, detections, opencv_config, None
    )

    opencv_model = opencv_calibration_result.optimized_camera_model

    out_dict = lbb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(), config._cpp_config()
    )

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
        fov_deg_x=config.fov_deg_x,
        fov_deg_y=config.fov_deg_y,
    )

    cameras_from_target = opencv_calibration_result.optimized_cameras_T_target

    if num_stddevs_outlier_threshold is None:
        return _pinhole_splined_refine_inner(
            prior_model, cameras_from_target, target_points, detections
        )

    original_detections = detections
    curr_detections = detections
    curr_intrinsics = prior_model
    curr_cameras_from_target = cameras_from_target

    while True:
        calibration_result = _pinhole_splined_refine_inner(
            curr_intrinsics,
            curr_cameras_from_target,
            target_points,
            curr_detections,
        )

        curr_intrinsics = calibration_result.optimized_camera_model
        curr_cameras_from_target = calibration_result.optimized_cameras_T_target

        new_detections = _filter_outliers(
            curr_detections,
            calibration_result.detection_infos,
            num_stddevs_outlier_threshold,
        )

        if all(
            len(new_det) == len(old_det)
            for new_det, old_det in zip(new_detections, curr_detections)
        ):
            break

        curr_detections = new_detections

    original_detections_detection_infos = [
        _compute_detection_info(
            target_points, cam_from_target, detection, curr_intrinsics
        )
        for cam_from_target, detection in zip(
            curr_cameras_from_target, original_detections
        )
    ]

    return CalibrationResult(
        optimized_camera_model=curr_intrinsics,
        optimized_cameras_T_target=curr_cameras_from_target,
        detection_infos=original_detections_detection_infos,
    )


@overload
def calibrate_camera(
    target_points,
    detections,
    camera_model_config: PinholeSplinedConfig,
) -> CalibrationResult[PinholeSplined]: ...


@overload
def calibrate_camera(
    target_points,
    detections,
    camera_model_config: OpenCVConfig,
) -> CalibrationResult[OpenCV]: ...


def calibrate_camera(
    target_points: np.ndarray,
    detections: list[Detection],
    camera_model_config: CameraModelConfig,
    num_stddevs_outlier_threshold: float | None = None,
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
