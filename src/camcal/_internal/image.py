import cv2
import numpy as np
from jaxtyping import UInt8


def to_gray(
    img: UInt8[np.ndarray, "H W"] | UInt8[np.ndarray, "H W C"],
) -> UInt8[np.ndarray, "H W"]:
    if len(img.shape) == 2 or img.shape[2] == 1:
        return img.reshape(img.shape[:2])

    if img.shape[2] != 3:
        raise ValueError("Image must have one or three channels")

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
