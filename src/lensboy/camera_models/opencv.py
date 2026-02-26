from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from lensboy.camera_models.base_model import CameraModel, CameraModelConfig

K1, K2, P1, P2, K3, K4, K5, K6, S1, S2, S3, S4 = range(12)


def _mask(*idx: int) -> np.ndarray:
    m = np.zeros(12, dtype=bool)
    if len(idx) > 0:
        m[list(idx)] = True
    return m


@dataclass
class OpenCVConfig(CameraModelConfig):
    image_height: int
    image_width: int

    initial_focal_length: float
    included_distoriton_coefficients: np.ndarray = field(
        default_factory=lambda: OpenCVConfig.STANDARD
    )

    NONE = _mask()
    STANDARD = _mask(K1, K2, P1, P2, K3)
    RADIAL_6 = _mask(K1, K2, K3, K4, K5, K6)
    TANGENTIAL = _mask(P1, P2)
    THIN_PRISM = _mask(S1, S2, S3, S4)
    FULL_12 = _mask(*range(12))

    def __post_init__(self):
        assert self.included_distoriton_coefficients.shape == (12,), (
            f"Expected (12,) mask, got {self.included_distoriton_coefficients.shape}"
        )
        assert self.included_distoriton_coefficients.dtype == np.bool_, (
            f"Expected bool dtype, got {self.included_distoriton_coefficients.dtype}"
        )

    def optimize_mask(self) -> np.ndarray:
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


@dataclass
class OpenCV(CameraModel):
    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    distortion_coeffs: np.ndarray

    def __post_init__(self):
        assert self.distortion_coeffs.shape == (12,), (
            f"Expected (12,) distortion_coeffs, got {self.distortion_coeffs.shape}"
        )
        assert np.issubdtype(self.distortion_coeffs.dtype, np.floating), (
            f"Expected floating dtype, got {self.distortion_coeffs.dtype}"
        )

    def _params(self):
        return [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs]

    def _with_params(self, params: list[float]) -> OpenCV:
        assert len(params) == 4 + 12

        fx, fy, cx, cy = params[:4]

        distortion_coeffs = np.array(params[4:])

        return replace(
            self, fx=fx, fy=fy, cx=cx, cy=cy, distortion_coeffs=distortion_coeffs
        )

    def project_points(
        self,
        points_in_cam: np.ndarray,
    ) -> np.ndarray:
        assert points_in_cam.ndim == 2 and points_in_cam.shape[1] == 3, (
            f"Expected (N, 3) array, got {points_in_cam.shape}"
        )
        assert np.issubdtype(points_in_cam.dtype, np.floating), (
            f"Expected floating dtype, got {points_in_cam.dtype}"
        )
        return cv2.projectPoints(
            points_in_cam,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=self.K(),
            distCoeffs=self.distortion_coeffs,
        )[0].reshape(-1, 2)

    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
