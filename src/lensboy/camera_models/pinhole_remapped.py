from dataclasses import dataclass

import cv2
import numpy as np

from lensboy.camera_models.base_model import CameraModel


@dataclass
class PinholeRemapped(CameraModel):
    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    map_x: np.ndarray
    map_y: np.ndarray

    input_image_width: int
    input_image_height: int

    def __post_init__(self):
        assert self.map_x.ndim == 2, f"Expected 2D map_x, got {self.map_x.ndim}D"
        assert np.issubdtype(self.map_x.dtype, np.floating), (
            f"Expected floating dtype for map_x, got {self.map_x.dtype}"
        )
        assert self.map_y.ndim == 2, f"Expected 2D map_y, got {self.map_y.ndim}D"
        assert np.issubdtype(self.map_y.dtype, np.floating), (
            f"Expected floating dtype for map_y, got {self.map_y.dtype}"
        )

    def undistort(
        self,
        image: np.ndarray,
        *,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int | float | tuple[int, int, int] | tuple[int, int, int, int] = 0,
    ) -> np.ndarray:
        if image.ndim not in (2, 3):
            raise ValueError(f"image must be HxW or HxWxC, got {image.shape}")

        map_x = np.asarray(self.map_x, dtype=np.float32, order="C")
        map_y = np.asarray(self.map_y, dtype=np.float32, order="C")

        h, w = image.shape[:2]
        if (h, w) != (self.input_image_height, self.input_image_width):
            raise ValueError(
                f"image shape {(h, w)} != model image size "
                f"{(self.input_image_height, self.input_image_width)}"
            )

        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=interpolation,
            borderMode=border_mode,
            borderValue=border_value,
        )

    def project_points_undistorted(self, points_in_cam: np.ndarray) -> np.ndarray:
        assert points_in_cam.ndim == 2 and points_in_cam.shape[1] == 3, (
            f"Expected (N, 3) array, got {points_in_cam.shape}"
        )
        assert np.issubdtype(points_in_cam.dtype, np.floating), (
            f"Expected floating dtype, got {points_in_cam.dtype}"
        )
        points_cam = np.asarray(points_in_cam, dtype=np.float64)

        X = points_cam[:, 0]
        Y = points_cam[:, 1]
        Z = points_cam[:, 2]

        x = X / Z
        y = Y / Z

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return np.stack([u, v], axis=1)
