from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from functools import cached_property
from jaxtyping import Float

from camcal import camcal_bindings as cb
from camcal.camera_models.base_model import CameraModel, CameraModelConfig


@dataclass
class PinholeSplinedConfig(CameraModelConfig):
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

    def _cpp_config(self) -> cb.PinholeSplinedConfig:
        return cb.PinholeSplinedConfig(
            self.image_width,
            self.image_height,
            self.fov_deg_x,
            self.fov_deg_y,
            self.num_knots_x,
            self.num_knots_y,
        )


@dataclass
class PinholeSplined(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float

    dx_grid: Float[np.ndarray, "Ky Kx"]
    dy_grid: Float[np.ndarray, "Ky Kx"]

    num_knots_x: int
    num_knots_y: int

    fov_deg_x: float
    fov_deg_y: float

    @staticmethod
    def _camera_model_name() -> str:
        return "pinhole_splined"

    def params(self):
        return [
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            *self.dx_grid.ravel().tolist(),
            *self.dy_grid.ravel().tolist(),
        ]

    def with_params(self, params: list[float]) -> PinholeSplined:
        fx, fy, cx, cy = params[:4]

        params = params[4:]

        total_knots_per_map = self.num_knots_x * self.num_knots_y

        x_knots_list = params[:total_knots_per_map]
        params = params[total_knots_per_map:]
        y_knots_list = params[:total_knots_per_map]

        x_knots = np.array(x_knots_list).reshape(self.num_knots_y, self.num_knots_x)
        y_knots = np.array(y_knots_list).reshape(self.num_knots_y, self.num_knots_x)

        return replace(
            self,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            undistortion_knots_x=x_knots,
            undistortion_knots_y=y_knots,
        )

    def _cpp_config(self) -> cb.PinholeSplinedConfig:
        return cb.PinholeSplinedConfig(
            self.image_width,
            self.image_height,
            self.fov_deg_x,
            self.fov_deg_y,
            self.num_knots_x,
            self.num_knots_y,
        )

    def _k4(self) -> Float[np.ndarray, " 4"]:
        return np.array([self.fx, self.fy, self.cx, self.cy], dtype=float)

    def _cpp_params(self) -> cb.PinholeSplinedIntrinsicsParameters:
        return cb.PinholeSplinedIntrinsicsParameters(
            self._k4(), self.dx_grid, self.dy_grid
        )

    def project_points(
        self,
        points_in_cam: Float[np.ndarray, "N 3"],
    ) -> Float[np.ndarray, "N 2"]:
        return cb.project_pinhole_splined_points(
            self._cpp_config(),
            self._cpp_params(),
            points_in_camera=points_in_cam,
        )

    def _get_K(self) -> Float[np.ndarray, "3 3"]:
        return np.array(
            [[self.fx, 0, self.cx], [self.fy, 0, self.cy], [0, 0, 1]], dtype=float
        )

    def get_undistortion_maps(
        self, *args, **kwargs
    ) -> tuple[
        Float[np.ndarray, "3 3"], Float[np.ndarray, "H w"], Float[np.ndarray, "H w"]
    ]:
        K = self._get_K()
        map_x, map_y = cb.make_undistortion_maps_pinhole_splined(
            self._cpp_config(), self._cpp_params()
        )

        return K, map_x, map_y
