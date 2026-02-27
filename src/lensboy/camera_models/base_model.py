from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass
class CameraModelConfig(ABC):
    pass


@dataclass
class CameraModel(ABC):
    pass
