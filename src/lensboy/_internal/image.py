import cv2
import numpy as np


def to_gray(
    img: np.ndarray,
) -> np.ndarray:
    assert img.dtype == np.uint8, f"Expected uint8 image, got {img.dtype}"
    assert img.ndim in (2, 3), f"Expected 2D or 3D image, got {img.ndim}D"
    if len(img.shape) == 2 or img.shape[2] == 1:
        return img.reshape(img.shape[:2])

    if img.shape[2] != 3:
        raise ValueError("Image must have one or three channels")

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
