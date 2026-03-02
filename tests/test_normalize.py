"""Round-trip tests: project then normalize and check we get back the original."""

from pathlib import Path

import numpy as np

from lensboy.camera_models.opencv import OpenCV
from lensboy.camera_models.pinhole_remapped import PinholeRemapped
from lensboy.camera_models.pinhole_splined import PinholeSplined

DATA = Path(__file__).parent.parent / "data/test_datasets"


def _random_points_in_cam(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.5, 0.5, (n, 2))
    z = rng.uniform(1.0, 5.0, (n, 1))
    return np.hstack([xy * z, z])


def test_pinhole_remapped_roundtrip() -> None:
    model = PinholeRemapped(
        image_width=640,
        image_height=480,
        fx=500.0,
        fy=500.0,
        cx=320.0,
        cy=240.0,
        map_x=np.zeros((480, 640), dtype=np.float32),
        map_y=np.zeros((480, 640), dtype=np.float32),
        input_image_width=640,
        input_image_height=480,
    )
    points = _random_points_in_cam()
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / points[:, 2:3]
    np.testing.assert_allclose(normalized, expected, atol=1e-10)


def test_opencv_roundtrip() -> None:
    model = OpenCV.load(DATA / "opencv.json")
    points = _random_points_in_cam()
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / points[:, 2:3]
    np.testing.assert_allclose(normalized, expected, atol=1e-6)


def test_spline_roundtrip() -> None:
    model = PinholeSplined.load(DATA / "spline.json")
    points = _random_points_in_cam()
    pixels = model.project_points(points)
    normalized = model.normalize_points(pixels)

    expected = points / points[:, 2:3]
    np.testing.assert_allclose(normalized, expected, atol=1e-6)
