"""Calibrate from real images and write the solved dataset into the library.

Reads charuco images, runs a full calibration, and writes:
- src/lensboy/_calibration_data.json

This JSON is loaded by lensboy.demo at runtime to generate synthetic images.

Usage:
    python tools/generate_solved_dataset.py
"""

import json
from pathlib import Path

import cv2
import imageio.v3 as iio

import lensboy as lb

IMG_DIR = Path("data/images/wide_angle_charuco_private")
OUTPUT_PATH = Path("src/lensboy/_demo_calibration.json")


def _serialize_warp(warp: lb.TargetWarp) -> dict:
    wc = warp.warp_coordinates
    pose = wc.target_from_warp_frame
    return {
        "warp_frame_rotvec": pose.rotvec.tolist(),
        "warp_frame_trans": pose.translation.tolist(),
        "x_scale": wc.x_scale,
        "y_scale": wc.y_scale,
        "coefficients": list(warp.object_warp),
    }


def main():
    """Calibrate from real images and save the result into the library."""
    img_paths = sorted(IMG_DIR.glob("*.png"))
    assert len(img_paths) > 0, f"No images found in {IMG_DIR}"
    imgs = [iio.imread(pth) for pth in img_paths]

    board = cv2.aruco.CharucoBoard(
        (14, 9),
        40,
        30,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
    )
    target_points, frames, _image_indices = lb.extract_frames_from_charuco(
        board, imgs
    )

    img_h, img_w = imgs[0].shape[:2]
    config = lb.OpenCVConfig(
        image_height=img_h,
        image_width=img_w,
        included_distortion_coefficients=lb.OpenCVConfig.FULL_14,
    )
    result = lb.calibrate_camera(
        target_points, frames, camera_model_config=config
    )

    calibration: dict = {
        "board": {
            "size": list(board.getChessboardSize()),
            "square_length": float(board.getSquareLength()),
            "marker_length": float(board.getMarkerLength()),
            "dictionary_id": cv2.aruco.DICT_5X5_100,
        },
        "camera_model": result.camera_model.to_json(),
        "image_width": img_w,
        "image_height": img_h,
        "poses": [
            {
                "rotvec": p.rotvec.tolist(),
                "trans": p.translation.tolist(),
            }
            for p in result.cameras_from_target
        ],
    }
    if result.target_warp is not None:
        calibration["target_warp"] = _serialize_warp(result.target_warp)

    OUTPUT_PATH.write_text(json.dumps(calibration, indent=2))

    sigma = result.residual_sigma_map()
    print(f"Saved to {OUTPUT_PATH}")
    print(f"  {len(frames)} views, residual sigma: {sigma:.4f}px")


if __name__ == "__main__":
    main()
