"""
CamCal for camera calibration
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['add', 'calibrate_opencv', 'get_matching_spline_distortion_model', 'project_pinhole_splined_points']
def add(a: typing.SupportsInt, b: typing.SupportsInt) -> int:
    """
    Add two integers together - test
    """
def calibrate_opencv(intrinsics_initial_value: collections.abc.Sequence[typing.SupportsFloat], intrinsics_param_optimize_mask: collections.abc.Sequence[bool], cameras_from_world: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[6, 1]"]], target_points: collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]], detections: collections.abc.Sequence[tuple[collections.abc.Sequence[typing.SupportsInt], collections.abc.Sequence[typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[2, 1]"]]]]) -> dict:
    ...
def get_matching_spline_distortion_model(opencv_distortion_params: collections.abc.Sequence[typing.SupportsFloat], fov_deg_x: typing.SupportsFloat, fov_deg_y: typing.SupportsFloat, num_knots_x: typing.SupportsInt, num_knots_y: typing.SupportsInt) -> dict:
    ...
def project_pinhole_splined_points(fov_deg_x: typing.SupportsFloat, fov_deg_y: typing.SupportsFloat, num_knots_x: typing.SupportsInt, num_knots_y: typing.SupportsInt, k4: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dx_grid: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], dy_grid: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], points_in_camera: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> numpy.typing.NDArray[numpy.float64]:
    """
    Vectorized pinhole+splined projection over points_in_camera.
    
    Args:
      fov_deg_x, fov_deg_y: FOV in degrees
      num_knots_x, num_knots_y: knot grid size
      k4: numpy array shape (4,) -> [fx, fy, cx, cy]
      dx_grid: numpy array shape (num_knots_y, num_knots_x), C-order (row-major)
      dy_grid: numpy array shape (num_knots_y, num_knots_x), C-order (row-major)
      points_in_camera: numpy array shape (N, 3), C-order
    
    Returns:
      numpy array shape (N, 2)
    """
