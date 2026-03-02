from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lensboy.geometry.pose import Pose


def rot_euler(pose: Pose, seq: str = "xyz", degrees: bool = True) -> np.ndarray:
    """Euler angle decomposition of a pose's rotation.

    Args:
        pose: The pose to decompose.
        seq: Axis sequence, e.g. ``"xyz"``, ``"zyx"``.
        degrees: If True, return angles in degrees.

    Returns:
        Euler angles, shape (3,).
    """
    return Rotation.from_matrix(pose.rotmat).as_euler(seq, degrees=degrees)
