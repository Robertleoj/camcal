from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int

from camcal import camcal_bindings as cb
from camcal.camera_models.base_model import CameraModel, CameraModelConfig
from camcal.geometry.pose import Pose


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


def calibrate_camera(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    camera_model_config: CameraModelConfig,
) -> CalibrationResult:
    num_cameras = len(detections)

    initial_intrinsics = camera_model_config.get_initial_value()
    initial_params = initial_intrinsics.params()
    intrinsics_param_optimize_mask = np.ones(len(initial_params), dtype=bool).tolist()

    result = cb.calibrate_camera(
        camera_model_name=initial_intrinsics._camera_model_name(),
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

    return CalibrationResult(
        optimized_camera_model=optimized_intrinsics,
        optimized_cameras_from_world=cameras_from_world,
    )
