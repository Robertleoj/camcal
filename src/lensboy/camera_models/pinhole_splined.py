from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig
from lensboy.camera_models.pinhole_remapped import PinholeRemapped


@dataclass
class PinholeSplinedConfig(CameraModelConfig):
    image_height: int
    image_width: int

    initial_focal_length: float

    fov_deg_x: float
    fov_deg_y: float

    num_knots_x: int
    num_knots_y: int

    def get_initial_value(self) -> PinholeSplined:
        return PinholeSplined(
            image_height=self.image_height,
            image_width=self.image_width,
            fx=self.initial_focal_length,
            fy=self.initial_focal_length,
            cx=self.image_width / 2,
            cy=self.image_height / 2,
            fov_deg_x=self.fov_deg_x,
            fov_deg_y=self.fov_deg_y,
            num_knots_x=self.num_knots_x,
            num_knots_y=self.num_knots_y,
            dx_grid=np.zeros((self.num_knots_x, self.num_knots_y), dtype=float),
            dy_grid=np.zeros((self.num_knots_x, self.num_knots_y), dtype=float),
        )

    def _cpp_config(self) -> lbb.PinholeSplinedConfig:
        return lbb.PinholeSplinedConfig(
            self.image_width,
            self.image_height,
            self.fov_deg_x,
            self.fov_deg_y,
            self.num_knots_x,
            self.num_knots_y,
        )


@dataclass
class PinholeSplined(CameraModel):
    image_width: int
    image_height: int

    fx: float
    fy: float
    cx: float
    cy: float

    dx_grid: np.ndarray
    dy_grid: np.ndarray

    num_knots_x: int
    num_knots_y: int

    fov_deg_x: float
    fov_deg_y: float

    def __post_init__(self):
        assert self.dx_grid.ndim == 2, f"Expected 2D dx_grid, got {self.dx_grid.ndim}D"
        assert np.issubdtype(self.dx_grid.dtype, np.floating), f"Expected floating dtype for dx_grid, got {self.dx_grid.dtype}"
        assert self.dy_grid.ndim == 2, f"Expected 2D dy_grid, got {self.dy_grid.ndim}D"
        assert np.issubdtype(self.dy_grid.dtype, np.floating), f"Expected floating dtype for dy_grid, got {self.dy_grid.dtype}"

    @staticmethod
    def _camera_model_name() -> str:
        return "pinhole_splined"

    def _cpp_config(self) -> lbb.PinholeSplinedConfig:
        return lbb.PinholeSplinedConfig(
            self.image_width,
            self.image_height,
            self.fov_deg_x,
            self.fov_deg_y,
            self.num_knots_x,
            self.num_knots_y,
        )

    def _cpp_params(self) -> lbb.PinholeSplinedIntrinsicsParameters:
        return lbb.PinholeSplinedIntrinsicsParameters(
            self._k4(), self.dx_grid, self.dy_grid
        )

    def project_points(
        self,
        points_in_cam: np.ndarray,
    ) -> np.ndarray:
        assert points_in_cam.ndim == 2 and points_in_cam.shape[1] == 3, (
            f"Expected (N, 3) array, got {points_in_cam.shape}"
        )
        assert np.issubdtype(points_in_cam.dtype, np.floating), (
            f"Expected floating dtype, got {points_in_cam.dtype}"
        )
        return lbb.project_pinhole_splined_points(
            self._cpp_config(),
            self._cpp_params(),
            points_in_camera=points_in_cam,
        )

    def _k4(self):
        return (self.fx, self.fy, self.cx, self.cy)

    def get_pinhole_model(
        self,
        new_k4: tuple[float, float, float, float] | None = None,
        new_image_size_wh: tuple[int, int] | None = None,
    ) -> PinholeRemapped:
        if new_k4 is not None:
            k4 = new_k4
        else:
            k4 = self._k4()

        if new_image_size_wh is not None:
            image_wh = new_image_size_wh
        else:
            image_wh = (self.image_width, self.image_height)

        map_x, map_y = lbb.make_undistortion_maps_pinhole_splined(
            self._cpp_config(),
            self._cpp_params(),
            np.array(k4, dtype=float),
            image_wh,
        )

        return PinholeRemapped(
            image_width=image_wh[0],
            image_height=image_wh[1],
            input_image_width=self.image_width,
            input_image_height=self.image_height,
            fx=k4[0],
            fy=k4[1],
            cx=k4[2],
            cy=k4[3],
            map_x=map_x,
            map_y=map_y,
        )
