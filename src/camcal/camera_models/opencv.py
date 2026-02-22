from __future__ import annotations

from dataclasses import dataclass, replace, field

import numpy as np
from jaxtyping import Float, Bool

from camcal.camera_models.base_model import CameraModel, CameraModelConfig
from enum import Enum, auto


K1, K2, P1, P2, K3, K4, K5, K6, S1, S2, S3, S4 = range(12)


def _mask(*idx: int) -> Bool[np.ndarray, "12"]:
    m = np.zeros(12, dtype=bool)
    m[list(idx)] = True
    return m


@dataclass
class OpenCVConfig(CameraModelConfig):
    initial_focal_length: float
    included_distoriton_coefficients: Bool[np.ndarray, " 12"] = field(
        default_factory=lambda: OpenCVConfig.STANDARD
    )

    STANDARD = _mask(K1, K2, P1, P2, K3)
    RADIAL_6 = _mask(K1, K2, K3, K4, K5, K6)
    TANGENTIAL = _mask(P1, P2)
    THIN_PRISM = _mask(S1, S2, S3, S4)
    FULL_12 = _mask(*range(12))

    def optimize_mask(self) -> Bool[np.ndarray, " 16"]:
        mask = np.zeros(16, dtype=bool)

        mask[:4] = True
        mask[4:] = self.included_distoriton_coefficients
        return mask

    def get_initial_value(self) -> OpenCV:
        return OpenCV(
            image_height=self.image_height,
            image_width=self.image_width,
            fx=self.initial_focal_length,
            fy=self.initial_focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            distortion_coeffs=np.zeros(12, dtype=np.float64),
        )

    @staticmethod
    def camera_model_class():
        return OpenCV


@dataclass
class OpenCV(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float

    distortion_coeffs: Float[np.ndarray, " 12"]

    @staticmethod
    def _camera_model_name() -> str:
        return "opencv"

    @staticmethod
    def config_class():
        return OpenCVConfig

    def params(self):
        return [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs]

    def with_params(self, params: list[float]) -> CameraModel:
        assert len(params) == 4 + 12

        fx, fy, cx, cy = params[:4]

        distortion_coeffs = np.array(params[4:])

        return replace(
            self, fx=fx, fy=fy, cx=cx, cy=cy, distortion_coeffs=distortion_coeffs
        )
