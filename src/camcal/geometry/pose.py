from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cv2
import einops


from jaxtyping import Float


def to_homogeneous(points: Float[np.ndarray, " N 3"]) -> Float[np.ndarray, " N 4"]:
    if points.shape[1] == 4:
        return points

    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones])


@dataclass
class Pose:
    matrix: Float[np.ndarray, "4 4"]

    @staticmethod
    def from_rotvec_trans(
        *,
        rotvec: Float[np.ndarray, " 3"] | None = None,
        trans: Float[np.ndarray, " 3"] | None = None,
    ) -> Pose:
        rotmat = None
        if rotvec is not None:
            rotmat = cv2.Rodrigues(rotvec)[0]

        return Pose.from_rotmat_trans(rotmat=rotmat, trans=trans)

    @staticmethod
    def from_rotmat_trans(
        *,
        rotmat: Float[np.ndarray, " 3 3"] | None = None,
        trans: Float[np.ndarray, " 3"] | None = None,
    ) -> Pose:
        mat = np.eye(4)
        if rotmat is not None:
            mat[:3, :3] = rotmat

        if trans is not None:
            mat[:3, 3] = trans

        return Pose(mat)

    @property
    def rotvec(self) -> Float[np.ndarray, " 3"]:
        rvec, J = cv2.Rodrigues(self.rotmat)
        return rvec.squeeze()

    @property
    def rotmat(self) -> Float[np.ndarray, " 3 3"]:
        return self.matrix[:3, :3].copy()

    @property
    def translation(self) -> Float[np.ndarray, " 3"]:
        return self.matrix[:3, 3].squeeze()

    def to_cpp(self) -> Float[np.ndarray, " 6"]:
        rotvec = self.rotvec
        trans = self.translation

        return np.concatenate([rotvec, trans], 0)

    @staticmethod
    def from_cpp(cpp_arr: Float[np.ndarray, " 6"]) -> Pose:
        rotvec = cpp_arr[:3]
        trans = cpp_arr[3:]

        return Pose.from_rotvec_trans(rotvec=rotvec, trans=trans)

    def inverse(self):
        new_rotmat = self.rotmat.T
        new_trans = -new_rotmat @ self.translation
        return Pose.from_rotmat_trans(rotmat=new_rotmat, trans=new_trans)

    def apply(self, points: Float[np.ndarray, " N 3"]):
        points_homo = to_homogeneous(points)
        transformed_points_homo = einops.einsum(
            self.matrix, points_homo, "h d, n d -> n h"
        )

        return transformed_points_homo[:, :3]

    def apply1(self, point: Float[np.ndarray, " 3"]):
        return self.apply(point[None, :])[0]
