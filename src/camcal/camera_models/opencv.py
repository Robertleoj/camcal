from dataclasses import dataclass
import numpy as np
from jaxtyping import Float

from camcal.camera_models.base_model import CameraModel, CameraModelConfig


@dataclass
class OpenCV5Config(CameraModelConfig):
    pass


@dataclass
class OpenCV5(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float

    distortion_coeffs: Float[np.ndarray, "5"]
