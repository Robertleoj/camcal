from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraModelConfig(ABC):
    pass


@dataclass
class CameraModel(ABC):
    image_width: int
    image_height: int

    @abstractmethod
    def project_points(self, points_in_cam: np.ndarray) -> np.ndarray: ...
