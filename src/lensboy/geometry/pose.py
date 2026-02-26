from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 3, f"Expected (N, 3) array, got {points.shape}"
    assert np.issubdtype(points.dtype, np.floating), f"Expected floating dtype, got {points.dtype}"
    if points.shape[1] == 4:
        return points

    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones])


@dataclass
class Pose:
    matrix: np.ndarray

    def __post_init__(self):
        assert self.matrix.shape == (4, 4), f"Expected (4, 4) matrix, got {self.matrix.shape}"
        assert np.issubdtype(self.matrix.dtype, np.floating), f"Expected floating dtype, got {self.matrix.dtype}"

    @staticmethod
    def from_rotvec_trans(
        *,
        rotvec: np.ndarray | None = None,
        trans: np.ndarray | None = None,
    ) -> Pose:
        if rotvec is not None:
            assert rotvec.shape == (3,), f"Expected (3,) rotvec, got {rotvec.shape}"
            assert np.issubdtype(rotvec.dtype, np.floating), f"Expected floating dtype, got {rotvec.dtype}"
        if trans is not None:
            assert trans.shape == (3,), f"Expected (3,) trans, got {trans.shape}"
            assert np.issubdtype(trans.dtype, np.floating), f"Expected floating dtype, got {trans.dtype}"
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
        if rotmat is not None:
            assert rotmat.shape == (3, 3), f"Expected (3, 3) rotmat, got {rotmat.shape}"
            assert np.issubdtype(rotmat.dtype, np.floating), f"Expected floating dtype, got {rotmat.dtype}"
        if trans is not None:
            assert trans.shape == (3,), f"Expected (3,) trans, got {trans.shape}"
            assert np.issubdtype(trans.dtype, np.floating), f"Expected floating dtype, got {trans.dtype}"
        mat = np.eye(4)
        if rotmat is not None:
            mat[:3, :3] = rotmat

        if trans is not None:
            mat[:3, 3] = trans

        return Pose(mat)

    @property
    def rotvec(self) -> np.ndarray:
        rvec, J = cv2.Rodrigues(self.rotmat)
        return rvec.squeeze()

    @property
    def rotmat(self) -> np.ndarray:
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> np.ndarray:
        return self.matrix[:3, 3].squeeze()

    def to_cpp(self) -> np.ndarray:
        rotvec = self.rotvec
        trans = self.translation

        return np.concatenate([rotvec, trans], 0)

    @staticmethod
    def from_cpp(cpp_arr: np.ndarray) -> Pose:
        assert cpp_arr.shape == (6,), f"Expected (6,) array, got {cpp_arr.shape}"
        assert np.issubdtype(cpp_arr.dtype, np.floating), f"Expected floating dtype, got {cpp_arr.dtype}"
        rotvec = cpp_arr[:3]
        trans = cpp_arr[3:]

        return Pose.from_rotvec_trans(rotvec=rotvec, trans=trans)

    def inverse(self):
        new_rotmat = self.rotmat.T
        new_trans = -new_rotmat @ self.translation
        return Pose.from_rotmat_trans(rotmat=new_rotmat, trans=new_trans)

    def apply(self, points: np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 3, f"Expected (N, 3) array, got {points.shape}"
        assert np.issubdtype(points.dtype, np.floating), f"Expected floating dtype, got {points.dtype}"
        points_homo = to_homogeneous(points)  # (N, 4)

        # (4,4) @ (4,N) -> (4,N) -> transpose -> (N,4)
        transformed_points_homo = (self.matrix @ points_homo.T).T

        return transformed_points_homo[:, :3]

    def apply1(self, point: np.ndarray):
        assert point.shape == (3,), f"Expected (3,) point, got {point.shape}"
        assert np.issubdtype(point.dtype, np.floating), f"Expected floating dtype, got {point.dtype}"
        return self.apply(point[None, :])[0]
