import cv2
import numpy as np
from dataclasses import dataclass

from camcal._internal.image import to_gray

from jaxtyping import Float32, Int32


@dataclass
class CharucoDetection:
    charuco_ids: Int32[np.ndarray, " N"]
    charuco_corners: Float32[np.ndarray, "N 2"]


def detect_charuco(img: np.ndarray, board: cv2.aruco.CharucoBoard) -> CharucoDetection:
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.minMarkers = 1

    refine_params = cv2.aruco.RefineParameters()

    charuco_detector = cv2.aruco.CharucoDetector(
        board, charucoParams=charuco_params, refineParams=refine_params
    )

    gray = to_gray(img)

    (charuco_corners, charuco_ids, _marker_corners, _marker_ids) = (
        charuco_detector.detectBoard(gray)
    )

    return CharucoDetection(charuco_ids.squeeze(), charuco_corners.squeeze(1))
