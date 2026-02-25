from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float


@dataclass
class CameraModelConfig(ABC):
    image_height: int
    image_width: int

    @abstractmethod
    def get_initial_value(self) -> CameraModel: ...


@dataclass
class CameraModel(ABC):
    image_height: int
    image_width: int

    @staticmethod
    @abstractmethod
    def _camera_model_name() -> str: ...

    @abstractmethod
    def params(self) -> list[float]: ...

    @abstractmethod
    def with_params(
        self,
        params: list[float],
    ) -> CameraModel: ...

    @abstractmethod
    def project_points(
        self,
        points_in_cam: Float[np.ndarray, "N 3"],
    ) -> Float[np.ndarray, "N 2"]: ...
