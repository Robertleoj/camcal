from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CameraModelConfig(ABC):
    @abstractmethod
    def get_initial_value(self) -> CameraModel: ...


@dataclass
class CameraModel(ABC):
    pass
