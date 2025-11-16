"""
CamCal for camera calibration
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['add', 'calibrate_camera']
def add(a: typing.SupportsInt, b: typing.SupportsInt) -> int:
    """
    Add two integers together - test
    """
def calibrate_camera(camera_model_name: str, intrinsics_initial_value: collections.abc.Sequence[typing.SupportsFloat], intrinsics_param_optimize_mask: collections.abc.Sequence[bool], cameras_from_world: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[6, 1]"]], target_points: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]], detections: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]]]) -> dict:
    ...
