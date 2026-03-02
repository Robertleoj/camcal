from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from lensboy.camera_models.base_model import CameraModel, CameraModelConfig

K1, K2, P1, P2, K3, K4, K5, K6, S1, S2, S3, S4 = range(12)


def _mask(*idx: int) -> np.ndarray:
    m = np.zeros(12, dtype=bool)
    if len(idx) > 0:
        m[list(idx)] = True
    return m


@dataclass
class OpenCVConfig(CameraModelConfig):
    """Configuration for fitting an OpenCV pinhole + distortion model.

    Preset boolean masks for common distortion subsets are available as class
    attributes: NONE, STANDARD, RADIAL_6, TANGENTIAL, THIN_PRISM, FULL_12.

    Attributes:
        image_height: Image height in pixels.
        image_width: Image width in pixels.
        initial_focal_length: Initial focal length guess in pixels.
        included_distoriton_coefficients: Boolean mask selecting which of the 12
            OpenCV distortion coefficients to optimise, shape (12,).
    """

    image_height: int
    image_width: int

    initial_focal_length: float
    included_distoriton_coefficients: np.ndarray = field(
        default_factory=lambda: OpenCVConfig.STANDARD
    )

    NONE = _mask()
    STANDARD = _mask(K1, K2, P1, P2, K3)
    RADIAL_6 = _mask(K1, K2, K3, K4, K5, K6)
    TANGENTIAL = _mask(P1, P2)
    THIN_PRISM = _mask(S1, S2, S3, S4)
    FULL_12 = _mask(*range(12))

    def __post_init__(self):
        assert self.included_distoriton_coefficients.shape == (12,), (
            f"Expected (12,) mask, got {self.included_distoriton_coefficients.shape}"
        )
        assert self.included_distoriton_coefficients.dtype == np.bool_, (
            f"Expected bool dtype, got {self.included_distoriton_coefficients.dtype}"
        )

    def optimize_mask(self) -> np.ndarray:
        """Return the optimization parameter mask over [fx, fy, cx, cy, *distortion].

        Returns:
            Boolean mask of shape (16,); the first 4 entries (intrinsics) are
            always True.
        """
        mask = np.zeros(16, dtype=bool)

        mask[:4] = True
        mask[4:] = self.included_distoriton_coefficients
        return mask

    def get_initial_value(self) -> OpenCV:
        """Construct the initial OpenCV model from this config."""
        return OpenCV(
            image_height=self.image_height,
            image_width=self.image_width,
            fx=self.initial_focal_length,
            fy=self.initial_focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            distortion_coeffs=np.zeros(12, dtype=np.float64),
        )


@dataclass
class OpenCV(CameraModel):
    """OpenCV pinhole + distortion camera model.

    Attributes:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        fx: Focal length along x in pixels.
        fy: Focal length along y in pixels.
        cx: Principal point x in pixels.
        cy: Principal point y in pixels.
        distortion_coeffs: Full 12-parameter OpenCV distortion vector, shape (12,).
    """

    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    distortion_coeffs: np.ndarray

    def __post_init__(self):
        assert self.distortion_coeffs.shape == (12,), (
            f"Expected (12,) distortion_coeffs, got {self.distortion_coeffs.shape}"
        )
        assert np.issubdtype(self.distortion_coeffs.dtype, np.floating), (
            f"Expected floating dtype, got {self.distortion_coeffs.dtype}"
        )

    def _params(self):
        return [self.fx, self.fy, self.cx, self.cy, *self.distortion_coeffs]

    def _with_params(self, params: list[float]) -> OpenCV:
        assert len(params) == 4 + 12

        fx, fy, cx, cy = params[:4]

        distortion_coeffs = np.array(params[4:])

        return replace(
            self, fx=fx, fy=fy, cx=cx, cy=cy, distortion_coeffs=distortion_coeffs
        )

    def normalize_points(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera-frame points with z=1.

        Uses cv2.undistortPoints to invert the full OpenCV distortion model.

        Args:
            pixel_coords: Shape (N, 2).

        Returns:
            Normalized points in camera frame, shape (N, 3) with z=1.
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        assert pts.ndim == 2 and pts.shape[1] == 2, (
            f"Expected (N, 2) array, got {pts.shape}"
        )
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-14)
        undistorted = cv2.undistortPointsIter(
            pts.reshape(-1, 1, 2),
            self.K(),
            self.distortion_coeffs,
            R=None,  # type: ignore
            P=None,  # type: ignore
            criteria=criteria,
        ).reshape(-1, 2)
        return np.column_stack([undistorted, np.ones(len(undistorted))])

    def project_points(
        self,
        points_in_cam: np.ndarray,
    ) -> np.ndarray:
        """Project 3D camera-frame points to pixel coordinates.

        Args:
            points_in_cam: Shape (N, 3).

        Returns:
            Projected pixel coordinates, shape (N, 2).
        """
        assert points_in_cam.ndim == 2 and points_in_cam.shape[1] == 3, (
            f"Expected (N, 3) array, got {points_in_cam.shape}"
        )
        assert np.issubdtype(points_in_cam.dtype, np.floating), (
            f"Expected floating dtype, got {points_in_cam.dtype}"
        )
        return cv2.projectPoints(
            points_in_cam,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=self.K(),
            distCoeffs=self.distortion_coeffs,
        )[0].reshape(-1, 2)

    def K(self):
        """Return the 3x3 camera intrinsics matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    @property
    def fov_deg_x(self) -> float:
        """Horizontal field of view in degrees.

        Computed by undistorting the image border pixels and measuring the
        angular span of the resulting normalised coordinates.
        """
        return self._compute_fov()[0]

    @property
    def fov_deg_y(self) -> float:
        """Vertical field of view in degrees.

        Computed by undistorting the image border pixels and measuring the
        angular span of the resulting normalised coordinates.
        """
        return self._compute_fov()[1]

    def save(self, path: Path | str) -> None:
        """Serialize the model to a JSON file.

        Args:
            path: Destination file path.
        """
        Path(path).write_text(json.dumps(self.to_json(), indent=4))

    @staticmethod
    def load(path: Path | str) -> OpenCV:
        """Load a model from a JSON file written by save().

        Args:
            path: Path to the JSON file.

        Returns:
            Reconstructed model.
        """
        return OpenCV.from_json(json.loads(Path(path).read_text()))

    def to_json(self) -> dict:
        """Serialize the model to a JSON-compatible dict.

        Returns:
            Dict with all model parameters. Distortion coefficients are stored
            as a list of length 12.
        """
        return {
            "type": "opencv",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_coeffs": self.distortion_coeffs.tolist(),
        }

    @staticmethod
    def from_json(data: dict) -> OpenCV:
        """Reconstruct a model from a dict produced by to_json().

        Args:
            data: Dict with all model parameters.

        Returns:
            Reconstructed model.
        """
        return OpenCV(
            image_width=data["image_width"],
            image_height=data["image_height"],
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            distortion_coeffs=np.array(data["distortion_coeffs"], dtype=np.float64),
        )

    def _compute_fov(self) -> tuple[float, float]:
        n = 200
        xs = np.linspace(0, self.image_width, n)
        ys = np.linspace(0, self.image_height, n)
        border = np.vstack(
            [
                np.column_stack([xs, np.zeros(n)]),
                np.column_stack([xs, np.full(n, self.image_height)]),
                np.column_stack([np.zeros(n), ys]),
                np.column_stack([np.full(n, self.image_width), ys]),
            ]
        ).astype(np.float64)

        undistorted = cv2.undistortPoints(
            border.reshape(-1, 1, 2),
            self.K(),
            self.distortion_coeffs,
        ).reshape(-1, 2)

        nx, ny = undistorted[:, 0], undistorted[:, 1]
        fov_x = float(np.degrees(np.arctan(np.max(nx)) + np.arctan(-np.min(nx))))
        fov_y = float(np.degrees(np.arctan(np.max(ny)) + np.arctan(-np.min(ny))))
        return fov_x, fov_y
