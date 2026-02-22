from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from jaxtyping import Bool, Float

from camcal import camcal_bindings as cb
from camcal.camera_models.base_model import CameraModel, CameraModelConfig
from camcal.camera_models.basic import Pinhole


@dataclass
class PinholeSplinedConfig(CameraModelConfig):
    initial_focal_length: float
    num_knots_x: int
    num_knots_y: int

    fov_deg_x: float
    fov_deg_y: float

    @staticmethod
    def camera_model_class():
        return PinholeSplined

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
            undistortion_knots_x=np.zeros(
                (self.num_knots_x, self.num_knots_y), dtype=float
            ),
            undistortion_knots_y=np.zeros(
                (self.num_knots_x, self.num_knots_y), dtype=float
            ),
        )


@dataclass
class PinholeSplined(CameraModel):
    fx: float
    fy: float
    cx: float
    cy: float

    undistortion_knots_x: Float[np.ndarray, "Kx Ky"]
    undistortion_knots_y: Float[np.ndarray, "Kx Ky"]

    num_knots_x: int
    num_knots_y: int

    fov_deg_x: float
    fov_deg_y: float

    @staticmethod
    def _camera_model_name() -> str:
        return "pinhole_splined"

    @staticmethod
    def config_class():
        return PinholeSplinedConfig

    def params(self):
        return [
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            *self.undistortion_knots_x.ravel().tolist(),
            *self.undistortion_knots_y.ravel().tolist(),
        ]

    def get_cpp_config(self) -> cb.ModelConfig:
        return cb.ModelConfig(
            double_params={"fov_deg_x": self.fov_deg_x, "fov_deg_y": self.fov_deg_y},
            int_params={
                "image_height": self.image_height,
                "image_width": self.image_width,
                "num_knots_x": self.num_knots_x,
                "num_knots_y": self.num_knots_y,
            },
        )

    def with_params(self, params: list[float]) -> PinholeSplined:
        fx, fy, cx, cy = params[:4]

        params = params[4:]

        total_knots_per_map = self.num_knots_x * self.num_knots_y

        x_knots_list = params[:total_knots_per_map]
        params = params[total_knots_per_map:]
        y_knots_list = params[:total_knots_per_map]

        x_knots = np.array(x_knots_list).reshape(self.num_knots_x, self.num_knots_y)
        y_knots = np.array(y_knots_list).reshape(self.num_knots_x, self.num_knots_y)

        return replace(
            self,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            undistortion_knots_x=x_knots,
            undistortion_knots_y=y_knots,
        )
