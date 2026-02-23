"""
CamCal for camera calibration
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['ModelConfig', 'add', 'calibrate_opencv', 'get_matching_spline_distortion_model']
class ModelConfig:
    def __init__(self, double_params: collections.abc.Mapping[str, typing.SupportsFloat] = {}, int_params: collections.abc.Mapping[str, typing.SupportsInt] = {}) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @property
    def double_params(self) -> dict[str, float]:
        ...
    @double_params.setter
    def double_params(self, arg0: collections.abc.Mapping[str, typing.SupportsFloat]) -> None:
        ...
    @property
    def int_params(self) -> dict[str, int]:
        ...
    @int_params.setter
    def int_params(self, arg0: collections.abc.Mapping[str, typing.SupportsInt]) -> None:
        ...
def add(a: typing.SupportsInt, b: typing.SupportsInt) -> int:
    """
    Add two integers together - test
    """
def calibrate_opencv(intrinsics_initial_value: collections.abc.Sequence[typing.SupportsFloat], intrinsics_param_optimize_mask: collections.abc.Sequence[bool], cameras_from_world: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[6, 1]"]], target_points: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]], detections: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]]]) -> dict:
    ...
def get_matching_spline_distortion_model(opencv_distortion_params: collections.abc.Sequence[typing.SupportsFloat], fov_deg_x: typing.SupportsFloat, fov_deg_y: typing.SupportsFloat, num_knots_x: typing.SupportsInt, num_knots_y: typing.SupportsInt) -> dict:
    ...
