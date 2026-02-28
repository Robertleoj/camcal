import cv2
import numpy as np

from lensboy._internal.progress import progress
from lensboy.analysis.image import to_gray
from lensboy.calibration.calibrate import Frame


def _detect_charuco(img: np.ndarray, board: cv2.aruco.CharucoBoard) -> Frame:
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

    return Frame(charuco_ids.squeeze(), charuco_corners.squeeze(1))


def extract_frames_from_charuco(
    board: cv2.aruco.CharucoBoard, images: list[np.ndarray]
) -> tuple[np.ndarray, list[Frame]]:
    """Detect ChArUco corners in a batch of images.

    Args:
        board: The ChArUco board definition.
        images: Calibration images, each of shape (H, W) or (H, W, C).

    Returns:
        target_points: 3D corner coordinates from the board definition, shape (N, 3).
        frames: Per-image frames, one per input image.
    """
    frames: list[Frame] = []

    for img in progress(images, desc="Detecting charuco"):
        frame = _detect_charuco(img, board)
        frames.append(frame)

    target_points = np.array(board.getChessboardCorners())

    return target_points, frames
