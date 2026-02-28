from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _to_homogeneous(points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 3, (
        f"Expected (N, 3) array, got {points.shape}"
    )
    assert np.issubdtype(points.dtype, np.floating), (
        f"Expected floating dtype, got {points.dtype}"
    )
    if points.shape[1] == 4:
        return points

    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones])


@dataclass
class Pose:
    """A rigid body transform (rotation + translation) in 3D space.

    Attributes:
        matrix: 4x4 transformation matrix, shape (4, 4).
    """

    matrix: np.ndarray

    def __post_init__(self):
        assert self.matrix.shape == (4, 4), (
            f"Expected (4, 4) matrix, got {self.matrix.shape}"
        )
        assert np.issubdtype(self.matrix.dtype, np.floating), (
            f"Expected floating dtype, got {self.matrix.dtype}"
        )

    @staticmethod
    def from_rotvec_trans(
        *,
        rotvec: np.ndarray | None = None,
        trans: np.ndarray | None = None,
    ) -> Pose:
        """Create a Pose from a rotation vector and translation.

        Args:
            rotvec: Rotation vector, shape (3,). Identity rotation if None.
            trans: Translation vector, shape (3,). Zero translation if None.

        Returns:
            The resulting Pose.
        """
        if rotvec is not None:
            rotvec = rotvec.flatten().astype(float)
            assert rotvec.shape == (3,), f"Expected (3,) rotvec, got {rotvec.shape}"

        if trans is not None:
            trans = trans.flatten().astype(float)
            assert trans.shape == (3,), f"Expected (3,) trans, got {trans.shape}"

        rotmat = None
        if rotvec is not None:
            rotmat = cv2.Rodrigues(rotvec)[0]

        return Pose.from_rotmat_trans(rotmat=rotmat, trans=trans)

    @staticmethod
    def from_rotmat_trans(
        *,
        rotmat: np.ndarray | None = None,
        trans: np.ndarray | None = None,
    ) -> Pose:
        """Create a Pose from a rotation matrix and translation.

        Args:
            rotmat: Rotation matrix, shape (3, 3). Identity rotation if None.
            trans: Translation vector, shape (3,). Zero translation if None.

        Returns:
            The resulting Pose.
        """
        if rotmat is not None:
            assert rotmat.shape == (3, 3), f"Expected (3, 3) rotmat, got {rotmat.shape}"
            assert np.issubdtype(rotmat.dtype, np.floating), (
                f"Expected floating dtype, got {rotmat.dtype}"
            )
        if trans is not None:
            assert trans.shape == (3,), f"Expected (3,) trans, got {trans.shape}"
            assert np.issubdtype(trans.dtype, np.floating), (
                f"Expected floating dtype, got {trans.dtype}"
            )
        mat = np.eye(4)
        if rotmat is not None:
            mat[:3, :3] = rotmat

        if trans is not None:
            mat[:3, 3] = trans

        return Pose(mat)

    @staticmethod
    def from_tx(x: float):
        """Create a pure x-axis translation Pose."""
        return Pose.from_rotvec_trans(trans=np.array([x, 0, 0]))

    @staticmethod
    def from_ty(y: float):
        """Create a pure y-axis translation Pose."""
        return Pose.from_rotvec_trans(trans=np.array([0, y, 0]))

    @staticmethod
    def from_tz(z: float):
        """Create a pure z-axis translation Pose."""
        return Pose.from_rotvec_trans(trans=np.array([0, 0, z]))

    @property
    def rotvec(self) -> np.ndarray:
        """Rotation vector, shape (3,)."""
        rvec, J = cv2.Rodrigues(self.rotmat)
        return rvec.squeeze()

    @property
    def rotmat(self) -> np.ndarray:
        """Rotation matrix, shape (3, 3)."""
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        """Translation vector, shape (3,)."""
        return self.matrix[:3, 3].squeeze()

    def _to_cpp(self) -> np.ndarray:
        """Serialise to the C++ bindings representation, shape (6,)."""
        rotvec = self.rotvec
        trans = self.translation

        return np.concatenate([rotvec, trans], 0)

    @staticmethod
    def _from_cpp(cpp_arr: np.ndarray) -> Pose:
        """Deserialise from the C++ bindings representation.

        Args:
            cpp_arr: Concatenated [rotvec, translation], shape (6,).

        Returns:
            The resulting Pose.
        """
        assert cpp_arr.shape == (6,), f"Expected (6,) array, got {cpp_arr.shape}"
        assert np.issubdtype(cpp_arr.dtype, np.floating), (
            f"Expected floating dtype, got {cpp_arr.dtype}"
        )
        rotvec = cpp_arr[:3]
        trans = cpp_arr[3:]

        return Pose.from_rotvec_trans(rotvec=rotvec, trans=trans)

    def inverse(self):
        """Return the inverse transform."""
        new_rotmat = self.rotmat.T
        new_trans = -new_rotmat @ self.translation
        return Pose.from_rotmat_trans(rotmat=new_rotmat, trans=new_trans)

    def apply(self, points: np.ndarray):
        """Apply this transform to a batch of 3D points.

        Args:
            points: Shape (N, 3).

        Returns:
            Transformed points, shape (N, 3).
        """
        assert points.ndim == 2 and points.shape[1] == 3, (
            f"Expected (N, 3) array, got {points.shape}"
        )
        assert np.issubdtype(points.dtype, np.floating), (
            f"Expected floating dtype, got {points.dtype}"
        )
        points_homo = _to_homogeneous(points)  # (N, 4)

        # (4,4) @ (4,N) -> (4,N) -> transpose -> (N,4)
        transformed_points_homo = (self.matrix @ points_homo.T).T

        return transformed_points_homo[:, :3]

    def apply1(self, point: np.ndarray):
        """Apply this transform to a single 3D point.

        Args:
            point: Shape (3,).

        Returns:
            Transformed point, shape (3,).
        """
        assert point.shape == (3,), f"Expected (3,) point, got {point.shape}"
        assert np.issubdtype(point.dtype, np.floating), (
            f"Expected floating dtype, got {point.dtype}"
        )
        return self.apply(point[None, :])[0]
