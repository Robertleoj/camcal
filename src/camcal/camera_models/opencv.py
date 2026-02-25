from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from jaxtyping import Bool, Float

from camcal.camera_models.base_model import CameraModel, CameraModelConfig

K1, K2, P1, P2, K3, K4, K5, K6, S1, S2, S3, S4 = range(12)


def _mask(*idx: int) -> Bool[np.ndarray, 12]:
    m = np.zeros(12, dtype=bool)
    if len(idx) > 0:
        m[list(idx)] = True
    return m


@dataclass
class OpenCVConfig(CameraModelConfig):
    initial_focal_length: float
    included_distoriton_coefficients: Bool[np.ndarray, " 12"] = field(
        default_factory=lambda: OpenCVConfig.STANDARD
    )

    NONE = _mask()
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

    def params(self):
        return [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs]

    def with_params(self, params: list[float]) -> CameraModel:
        assert len(params) == 4 + 12

        fx, fy, cx, cy = params[:4]

        distortion_coeffs = np.array(params[4:])

        return replace(
            self, fx=fx, fy=fy, cx=cx, cy=cy, distortion_coeffs=distortion_coeffs
        )

    def project_points(
        self,
        points_in_cam: Float[np.ndarray, "N 3"],
    ) -> Float[np.ndarray, "N 2"]:
        camera_matrix = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
        )

        return cv2.projectPoints(
            points_in_cam,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=camera_matrix,
            distCoeffs=self.distortion_coeffs,
        )[0].reshape(-1, 2)

    def get_undistortion_maps(
        self,
        *,
        image_height: int,
        image_width: int,
        new_camera_matrix: Float[np.ndarray, "3 3"] | None = None,
        alpha: float = 0.0,
        center_principal_point: bool = False,
        m1type: int = cv2.CV_32FC1,
    ) -> tuple[
        Float[np.ndarray, "3 3"],
        Float[np.ndarray, "H W"],
        Float[np.ndarray, "H W"],
    ]:
        H = int(image_height)
        W = int(image_width)
        if H <= 0 or W <= 0:
            raise ValueError("image_height and image_width must be > 0")

        K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        D = self.distortion_coeffs.astype(np.float64, copy=False).reshape(-1, 1)

        if new_camera_matrix is None:
            K_new, _roi = cv2.getOptimalNewCameraMatrix(
                K,
                D,
                (W, H),
                alpha,
                newImgSize=(W, H),
                centerPrincipalPoint=center_principal_point,
            )
        else:
            K_new = np.asarray(new_camera_matrix, dtype=np.float64)
            if K_new.shape != (3, 3):
                raise ValueError(f"new_camera_matrix must be (3,3), got {K_new.shape}")

        map_x, map_y = cv2.initUndistortRectifyMap(
            cameraMatrix=K,
            distCoeffs=D,
            R=np.eye(3, dtype=np.float64),
            newCameraMatrix=K_new,
            size=(W, H),
            m1type=m1type,
        )

        # Ensure nice dtypes (OpenCV already returns float32 for CV_32FC1)
        map_x = np.asarray(map_x)
        map_y = np.asarray(map_y)
        return K_new, map_x, map_y
