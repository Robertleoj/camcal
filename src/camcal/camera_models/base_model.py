from dataclasses import dataclass


@dataclass
class CameraModelConfig:
    image_height: int
    image_width: int


@dataclass
class CameraModel:
    image_height: int
    image_width: int
