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
    point_ids: np.ndarray
    points: np.ndarray

    def __post_init__(self):
        assert self.point_ids.ndim == 1, f"Expected 1D point_ids, got {self.point_ids.ndim}D"
        assert np.issubdtype(self.point_ids.dtype, np.integer), (
            f"Expected integer dtype for point_ids, got {self.point_ids.dtype}"
        )
        assert self.points.ndim == 2 and self.points.shape[1] == 2, f"Expected (N, 2) points, got {self.points.shape}"
        assert np.issubdtype(self.points.dtype, np.floating), (
            f"Expected floating dtype for points, got {self.points.dtype}"
        )

    def to_cpp(self) -> tuple[list[int], list[np.ndarray]]:
        return (self.point_ids.tolist(), list(self.points))


T = TypeVar("T", bound=CameraModel)


@dataclass
class CalibrationResult(Generic[T]):
    optimized_camera_model: T
    optimized_cameras_T_target: list[Pose]


def _opencv_calibrate_inner(
    curr_intrinsics: OpenCV,
    config: OpenCVConfig,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    detections: list[Detection],
) -> CalibrationResult[OpenCV]:
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

    optimized_cameras_from_target: list[Pose] = [Pose.from_cpp(np.array(a)) for a in result["cameras_from_target"]]

    return CalibrationResult(
        optimized_camera_model=optimized_intrinsics,
        optimized_cameras_T_target=optimized_cameras_from_target,
    )


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

    result = _opencv_calibrate_inner(initial_intrinsics, config, initial_cameras_from_target, target_points, detections)

    return result


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

    optimized_cameras_from_target = [Pose.from_cpp(np.array(a)) for a in fine_tune_result["cameras_from_target"]]

    fine_tuned_dx_grid = fine_tune_result["dx_grid"]
    fine_tuned_dy_grid = fine_tune_result["dy_grid"]

    optimized_model = replace(curr_intrinsics, dx_grid=fine_tuned_dx_grid, dy_grid=fine_tuned_dy_grid)

    return CalibrationResult(
        optimized_camera_model=optimized_model, optimized_cameras_T_target=optimized_cameras_from_target
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

    opencv_calibration_result = _opencv_calibrate(target_points, detections, opencv_config, None)

    opencv_model = opencv_calibration_result.optimized_camera_model

    out_dict = lbb.get_matching_spline_distortion_model(opencv_model.distortion_coeffs.tolist(), config._cpp_config())

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

    result = _pinhole_splined_refine_inner(prior_model, cameras_from_target, target_points, detections)

    return result


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
        return _calibrate_pinhole_splined(target_points, detections, camera_model_config, num_stddevs_outlier_threshold)

    if isinstance(camera_model_config, OpenCVConfig):
        return _opencv_calibrate(target_points, detections, camera_model_config, num_stddevs_outlier_threshold)

    raise RuntimeError("Invalid config")
