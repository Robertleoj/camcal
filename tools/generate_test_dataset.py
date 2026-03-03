"""Generate a test dataset from the wide-angle charuco images.

Extracts charuco detections and saves them as an npz file so tests
can run without needing the raw images.

Usage:
    python tools/generate_test_dataset.py
"""

from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

import lensboy as lb

IMG_DIR = Path("data/images/wide_angle_charuco_private")
OUTPUT_PATH = Path("data/test_datasets/wide_angle_charuco.npz")


def main():
    """Extract charuco detections and save as a test dataset."""
    img_paths = sorted(IMG_DIR.glob("*.png"))
    assert len(img_paths) > 0, f"No images found in {IMG_DIR}"
    imgs = [iio.imread(pth) for pth in img_paths]

    board = cv2.aruco.CharucoBoard(
        (14, 9),
        40,
        30,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    )
    target_points, frames, _image_indices = lb.extract_frames_from_charuco(board, imgs)

    img_height, img_width = imgs[0].shape[:2]

    frame_arrays = {}
    for i, f in enumerate(frames):
        frame_arrays[f"frame_{i}_indices"] = f.target_point_indices
        frame_arrays[f"frame_{i}_detections"] = f.detected_points_in_image

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        target_points=target_points,
        num_frames=np.array(len(frames)),
        image_height=np.array(img_height),
        image_width=np.array(img_width),
        **frame_arrays,
    )
    print(f"Saved test dataset to {OUTPUT_PATH}")
    print(f"  {len(frames)} frames")


if __name__ == "__main__":
    main()
