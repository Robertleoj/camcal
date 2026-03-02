from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from lensboy.camera_models.base_model import CameraModel


def _sample_disk(
    center: np.ndarray, radius: float, num_rings: int = 20
) -> np.ndarray:
    """Sample pixel coordinates on a disk using polar coordinates.

    The number of angular samples per ring scales with the ring radius
    to give roughly uniform spatial density.

    Args:
        center: Disk center in pixel coordinates, shape (2,).
        radius: Disk radius in pixels.
        num_rings: Number of concentric rings (excluding the center point).

    Returns:
        Pixel coordinates, shape (N, 2).
    """
    points = [center.copy()]
    for i in range(1, num_rings + 1):
        r = radius * i / num_rings
        num_angular = max(6, round(2 * np.pi * i))
        theta = np.linspace(0, 2 * np.pi, num_angular, endpoint=False)
        ring = center + r * np.column_stack([np.cos(theta), np.sin(theta)])
        points.append(ring)
    return np.concatenate(points).reshape(-1, 2)


def _objective(
    params: np.ndarray,
    points: np.ndarray,
    unit_vectors: np.ndarray,
    optimize_translation: bool,
) -> float:
    """Negative cosine-alignment cost for the implied transformation.

    Args:
        params: Rotation vector (3,), optionally followed by translation (3,).
        points: Unprojected points from model_a, shape (N, 3).
        unit_vectors: Unit direction vectors from model_b, shape (N, 3).
        optimize_translation: Whether params includes a translation component.

    Returns:
        Negative sum of cosine similarities (to be minimized).
    """
    rotvec = params[:3]
    R = Rotation.from_rotvec(rotvec).as_matrix()
    transformed = points @ R.T
    if optimize_translation:
        transformed = transformed + params[3:6]
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    normalized = transformed / norms
    return -np.sum(unit_vectors * normalized)


def find_matching_implied_transformation(
    model_a: CameraModel,
    model_b: CameraModel,
    distance: float | None = None,
    radius: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the implied coordinate-frame transformation between two camera models.

    Finds the pose R, t that best aligns the viewing directions of model_a
    with those of model_b by maximizing::

        sum_i  v_i^T  normalized(R @ p_i + t)

    where p_i are 3D points unprojected from model_a at the given distance,
    and v_i are unit vectors unprojected from model_b.

    When distance is None, translation is not optimized (equivalent to
    points at infinity where translation has no effect), and the returned
    translation is zero.

    Args:
        model_a: First camera model.
        model_b: Second camera model. Must have the same image dimensions as model_a.
        distance: Distance at which to place unprojected points from model_a.
            If None, only rotation is optimized (no translation).
        radius: Pixel radius from image center within which to sample.
            If None, defaults to 3/4 of the half-short-side.

    Returns:
        Rotation matrix (3, 3) and translation vector (3,).
    """
    assert (
        model_a.image_width == model_b.image_width
        and model_a.image_height == model_b.image_height
    ), (
        f"Image dimensions must match: "
        f"{model_a.image_width}x{model_a.image_height} vs "
        f"{model_b.image_width}x{model_b.image_height}"
    )

    optimize_translation = distance is not None
    d = distance if distance is not None else 1.0

    if radius is None:
        radius = 0.75 * min(model_a.image_width, model_a.image_height) / 2

    # 1. Sample the imager in a disk around the center
    center = np.array(
        [(model_a.image_width - 1) / 2, (model_a.image_height - 1) / 2]
    )
    pixels = _sample_disk(center, radius)

    # 2. Unproject model_a pixels to points at distance d
    rays_a = model_a.normalize_points(pixels)  # (N, 3), z=1
    norms_a = np.linalg.norm(rays_a, axis=1, keepdims=True)
    points = rays_a * (d / norms_a)  # (N, 3), ||p_i|| = d

    # 3. Unproject model_b pixels to unit vectors
    rays_b = model_b.normalize_points(pixels)  # (N, 3), z=1
    unit_vectors = rays_b / np.linalg.norm(rays_b, axis=1, keepdims=True)  # (N, 3)

    # 4. Find R, t that maximizes sum_i v_i^T normalized(R @ p_i + t)
    x0 = np.zeros(6 if optimize_translation else 3)
    result = minimize(
        _objective,
        x0,
        args=(points, unit_vectors, optimize_translation),
        method="L-BFGS-B",
    )

    R = Rotation.from_rotvec(result.x[:3]).as_matrix()
    t = result.x[3:6] if optimize_translation else np.zeros(3)
    return R, t
