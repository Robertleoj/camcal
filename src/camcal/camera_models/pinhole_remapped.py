from dataclasses import dataclass

import cv2
import numpy as np
from jaxtyping import Float

from camcal.camera_models.base_model import CameraModel


@dataclass
class PinholeRemapped(CameraModel):
    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    map_x: Float[np.ndarray, "H W"]
    map_y: Float[np.ndarray, "H W"]

    input_image_width: int
    input_image_height: int

    def undistort(
        self,
        image: np.ndarray,
        *,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_CONSTANT,
        border_value: int
        | float
        | tuple[int, int, int]
        | tuple[int, int, int, int] = 0,
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
