from dataclasses import dataclass
from camcal.camera_models.base_model import CameraModel, CameraModelConfig

@dataclass
class PinholeConfig(CameraModelConfig):
    pass


@dataclass
class Pinhole(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float
