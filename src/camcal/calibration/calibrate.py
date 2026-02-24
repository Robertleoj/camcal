import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from jaxtyping import Float, Int

from camcal import camcal_bindings as cb
from camcal.camera_models.base_model import CameraModel, CameraModelConfig
from camcal.camera_models.opencv import OpenCV, OpenCVConfig
from camcal.camera_models.pinhole_splined import PinholeSplined, PinholeSplinedConfig
from camcal.geometry.pose import Pose

LOG = logging.getLogger(__name__)


@dataclass
class Detection:
    point_ids: Int[np.ndarray, " N"]
    points: Float[np.ndarray, "N 2"]

    def to_cpp(self) -> tuple[list[int], list[Float[np.ndarray, "2"]]]:
        return (self.point_ids.tolist(), list(self.points))


@dataclass
class CalibrationResult:
    optimized_camera_model: CameraModel
    optimized_cameras_from_world: list[Pose]


def _opencv_calibrate(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    config: OpenCVConfig,
) -> CalibrationResult:
    num_cameras = len(detections)

    initial_intrinsics = config.get_initial_value()
    initial_params = initial_intrinsics.params()

    mask = config.optimize_mask()
    if mask is None:
        intrinsics_param_optimize_mask = np.ones(
            len(initial_params), dtype=bool
        ).tolist()
    else:
        intrinsics_param_optimize_mask = mask.tolist()

    # TODO: get initial poses with PnP
    cameras_from_world = [
        np.array([0, 0, 0, 0, 0, 100], dtype=np.float32) for _ in range(num_cameras)
    ]

    result = cb.calibrate_opencv(
        intrinsics_initial_value=initial_params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_world=cameras_from_world,
        target_points=list(target_points),
        detections=[d.to_cpp() for d in detections],
    )

    optimized_intrinsics = initial_intrinsics.with_params(result["intrinsics"])

    cameras_from_world = [
        Pose.from_cpp(np.array(a)) for a in result["cameras_from_world"]
    ]

    return CalibrationResult(
        optimized_camera_model=optimized_intrinsics,
        optimized_cameras_from_world=cameras_from_world,
    )


def _calibrate_pinhole_splined(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    config: PinholeSplinedConfig,
) -> CalibrationResult:
    opencv_config = OpenCVConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        initial_focal_length=config.initial_focal_length,
        included_distoriton_coefficients=OpenCVConfig.FULL_12,
    )

    # calibrate using full opencv distortion model
    opencv_calibration_result = _opencv_calibrate(
        target_points, detections, opencv_config
    )

    # Optimize splined distortion to match opencv distortion model
    opencv_model = cast(OpenCV, opencv_calibration_result.optimized_camera_model)

    out_dict = cb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(), config._cpp_config()
    )

    x_knots = out_dict["x_knots"]
    y_knots = out_dict["y_knots"]

    print(x_knots)
    print(y_knots)

    output_model = PinholeSplined(
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

    return CalibrationResult(
        output_model, opencv_calibration_result.optimized_cameras_from_world
    )

    # Final, full bundle adjustment to fine-tune the spline model.
    # should have a prior to stay at the opencv distortion values


def calibrate_camera(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    camera_model_config: CameraModelConfig,
) -> CalibrationResult:
    if isinstance(camera_model_config, PinholeSplinedConfig):
        return _calibrate_pinhole_splined(
            target_points, detections, camera_model_config
        )

    if isinstance(camera_model_config, OpenCVConfig):
        return _opencv_calibrate(target_points, detections, camera_model_config)

    raise RuntimeError("Invalid config")
