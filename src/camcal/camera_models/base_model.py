from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CameraModelConfig(ABC):
    image_height: int
    image_width: int

    @abstractmethod
    def get_initial_value(self) -> CameraModel: ...

    @staticmethod
    @abstractmethod
    def camera_model_class() -> type[CameraModel]: ...


@dataclass
class CameraModel(ABC):
    image_height: int
    image_width: int

    @staticmethod
    @abstractmethod
    def _camera_model_name() -> str: ...

    @staticmethod
    @abstractmethod
    def config_class() -> type[CameraModelConfig]: ...

    @abstractmethod
    def params(self) -> list[float]: ...

    @abstractmethod
    def with_params(self, params: list[float]) -> CameraModel: ...
