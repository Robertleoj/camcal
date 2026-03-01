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
    target_points, frames = lb.extract_frames_from_charuco(board, imgs)

    img_height, img_width = imgs[0].shape[:2]

    # Pack frames into arrays for storage
    all_indices = [f.target_point_indices for f in frames]
    all_detections = [f.detected_points_in_image for f in frames]
    frame_lengths = np.array([len(f.target_point_indices) for f in frames])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        target_points=target_points,
        frame_indices=np.concatenate(all_indices),
        frame_detections=np.concatenate(all_detections),
        frame_lengths=frame_lengths,
        image_height=np.array(img_height),
        image_width=np.array(img_width),
    )
    print(f"Saved test dataset to {OUTPUT_PATH}")
    print(f"  {len(frames)} frames, {sum(frame_lengths)} total detections")


if __name__ == "__main__":
    main()
