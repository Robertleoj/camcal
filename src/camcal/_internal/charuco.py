import cv2
import numpy as np

from camcal._internal.image import to_gray
from camcal.calibration.calibrate import Detection


def detect_charuco(img: np.ndarray, board: cv2.aruco.CharucoBoard) -> Detection:
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

    return Detection(charuco_ids.squeeze(), charuco_corners.squeeze(1))
