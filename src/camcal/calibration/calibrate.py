from dataclasses import dataclass
from jaxtyping import Float, Int
import numpy as np

from camcal.camera_models.base_model import CameraModel, CameraModelConfig
from camcal import camcal_bindings as cb


@dataclass
class Detection:
    point_ids: Int[np.ndarray, " N"]
    points: Float[np.ndarray, "N 2"]

    def to_cpp(self) -> tuple[list[int], list[Float[np.ndarray, "2"]]]:
        return (self.point_ids.tolist(), list(self.points))


@dataclass
class CalibrationResult:
    camera_model: CameraModel


def calibrate_camera(
    target_points: Float[np.ndarray, "N 3"],
    detections: list[Detection],
    camera_model_config: CameraModelConfig,
) -> CalibrationResult:
    # figure out which optimization function to call

    # call
    # camcal_bindings.

    cb.calibrate_camera(
        camera_model_name="opncv",
        intrinsics_initial_value=[0, 0, 0, 0],
        intrinsics_param_optimize_mask=[True, True, True, True],
        camera_poses_world=[np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)],
        target_points=list(target_points),
        detections=[d.to_cpp() for d in detections],
    )

    """
    intrinsics: 
    (
        model name, e.g. "opencv5"
        initial_value: list[float]
        optimize_mask: list[bool]
    )

    poses: 
    [
        initial_value
    ] (Frames)

    target_points: N x 3
    detections: 

    Returns
    optimized_intrinsics: list[float]
    optimized_poses: list[Pose]



    """
