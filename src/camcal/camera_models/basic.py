from camcal.camera_models.base_model import CameraModel, CameraModelConfig


class PinholeConfig(CameraModelConfig):
    pass


class Pinhole(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float
