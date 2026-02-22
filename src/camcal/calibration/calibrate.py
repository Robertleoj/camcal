from dataclasses import dataclass, replace

import numpy as np
from jaxtyping import Float, Int
from typing import cast

from camcal import camcal_bindings as cb
from camcal.camera_models.base_model import CameraModel, CameraModelConfig
from camcal.camera_models.pinhole_splined import PinholeSplinedConfig
from camcal.camera_models.basic import PinholeConfig
from camcal.geometry.pose import Pose
import logging

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


def _initial_pose_estimate(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    camera_model_config: PinholeSplinedConfig,
):
    LOG.info("Estimating initial poses")

    num_cameras = len(detections)

    pinhole_config = PinholeConfig(
        image_height=camera_model_config.image_height,
        image_width=camera_model_config.image_width,
        initial_focal_length=camera_model_config.initial_focal_length,
    )

    initial_intrinsics = pinhole_config.get_initial_value()
    initial_params = initial_intrinsics.params()
    intrinsics_param_optimize_mask = np.ones(len(initial_params), dtype=bool).tolist()

    result = cb.calibrate_camera(
        camera_model_name=initial_intrinsics._camera_model_name(),
        config=initial_intrinsics.get_cpp_config(),
        intrinsics_initial_value=initial_params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_world=[
            np.array([0, 0, 0, 0, 0, 100], dtype=np.float32) for _ in range(num_cameras)
        ],
        target_points=list(target_points),
        detections=[d.to_cpp() for d in detections],
    )

    optimized_intrinsics = initial_intrinsics.with_params(result["intrinsics"])

    cameras_from_world = [
        Pose.from_cpp(np.array(a)) for a in result["cameras_from_world"]
    ]

    return replace(
        camera_model_config, initial_focal_length=optimized_intrinsics.fx
    ), cameras_from_world


def calibrate_camera(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    camera_model_config: CameraModelConfig,
) -> CalibrationResult:
    num_cameras = len(detections)

    initial_intrinsics = camera_model_config.get_initial_value()
    initial_params = initial_intrinsics.params()
    mask = camera_model_config.optimize_mask()
    if mask is None:
        intrinsics_param_optimize_mask = np.ones(
            len(initial_params), dtype=bool
        ).tolist()
    else:
        intrinsics_param_optimize_mask = mask.tolist()

    cameras_from_world = [
        np.array([0, 0, 0, 0, 0, 100], dtype=np.float32) for _ in range(num_cameras)
    ]

    if isinstance(camera_model_config, PinholeSplinedConfig):
        camera_model_config, cameras_from_world_poses = _initial_pose_estimate(
            target_points, detections, camera_model_config
        )

        cameras_from_world = [p.to_cpp() for p in cameras_from_world_poses]

    result = cb.calibrate_camera(
        camera_model_name=initial_intrinsics._camera_model_name(),
        config=initial_intrinsics.get_cpp_config(),
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
