from __future__ import annotations

from dataclasses import dataclass, replace

from camcal.camera_models.base_model import CameraModel, CameraModelConfig


@dataclass
class PinholeConfig(CameraModelConfig):
    initial_focal_length: float

    def get_initial_value(self) -> Pinhole:
        return Pinhole(
            image_height=self.image_height,
            image_width=self.image_width,
            fx=self.initial_focal_length,
            fy=self.initial_focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
        )

    @staticmethod
    def camera_model_class():
        return Pinhole


@dataclass
class Pinhole(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def _camera_model_name():
        return "pinhole"

    @staticmethod
    def config_class():
        return PinholeConfig

    def params(self):
        return [self.fx, self.fy, self.cx, self.cy]

    def with_params(self, params: list[float]) -> Pinhole:
        assert len(params) == 4

        fx, fy, cx, cy = params

        return replace(self, fx=fx, fy=fy, cx=cx, cy=cy)
