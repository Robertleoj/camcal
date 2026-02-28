import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from timeit import default_timer
from typing import Generic, TypeVar, overload

import cv2
import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy.camera_models.base_model import CameraModel, CameraModelConfig
from lensboy.camera_models.opencv import OpenCV, OpenCVConfig
from lensboy.camera_models.pinhole_splined import (
    PinholeSplined,
    PinholeSplinedConfig,
)
from lensboy.geometry.pose import Pose

LOG = logging.getLogger(__name__)

DEFAULT_OUTLIER_THRESHOLD = 3.0
MAX_OUTLIER_FILTER_PASSES = 2


@dataclass
class Frame:
    """Detected calibration target points in a single image.

    Attributes:
        target_point_indices: Index into the target point array for each detection,
            shape (N,).
        detected_points_in_image: Corresponding pixel coordinates, shape (N, 2).
    """

    target_point_indices: np.ndarray
    detected_points_in_image: np.ndarray

    def __post_init__(self):
        assert (
            self.target_point_indices.shape[0] == self.detected_points_in_image.shape[0]
        ), (
            "Expected target_point_indices to have the "
            "length shape as detected_points_in_image"
        )

        assert self.target_point_indices.ndim == 1, (
            f"Expected 1D target_point_indices, got {self.target_point_indices.ndim}D"
        )
        assert np.issubdtype(self.target_point_indices.dtype, np.integer), (
            "Expected integer dtype for target_point_indices, ",
            f"got {self.target_point_indices.dtype}",
        )

        assert (
            self.detected_points_in_image.ndim == 2
            and self.detected_points_in_image.shape[1] == 2
        ), f"Expected (N, 2) points in image, got {self.detected_points_in_image.shape}"
        assert np.issubdtype(self.detected_points_in_image.dtype, np.floating), (
            "Expected floating dtype for points, "
            f"got {self.detected_points_in_image.dtype}"
        )

    def _to_cpp(self) -> tuple[list[int], list[np.ndarray]]:
        return (self.target_point_indices.tolist(), list(self.detected_points_in_image))

    def __len__(self):
        return self.target_point_indices.shape[0]


T = TypeVar("T", bound=CameraModel)
_IntrinsicsT = TypeVar("_IntrinsicsT", OpenCV, PinholeSplined)


@dataclass
class FrameInfo:
    """Per-image reprojection diagnostics computed after calibration.

    Attributes:
        projected_points: Model-projected pixel coordinates, shape (N, 2).
        residuals: Pixel-space residuals (detected minus projected), shape (N, 2).
        inlier_mask: Boolean mask indicating inlier points, shape (N,).
    """

    # N x 2
    projected_points: np.ndarray

    # N x 2
    residuals: np.ndarray

    # N
    inlier_mask: np.ndarray


@dataclass
class _OptimizationState(Generic[_IntrinsicsT]):
    intrinsics: _IntrinsicsT
    cameras_from_target: list[Pose]
    frames: list[Frame]
    warp_kxy: tuple[float, float] | None


@dataclass
class WarpCoordinates:
    """Maps target points into a planar frame scaled to [-1, 1] for warp estimation.

    Attributes:
        target_from_warp_frame: The target should be coplanar with the xy plane
            of this frame.
        x_scale: Half-extent of the target along the warp x-axis, in target units.
        y_scale: Half-extent of the target along the warp y-axis, in target units.
    """

    target_from_warp_frame: Pose
    x_scale: float
    y_scale: float

    def _to_cpp(self) -> lbb.WarpCoordinates:
        """Serialise to the C++ bindings representation."""
        return lbb.WarpCoordinates(
            target_from_warp_frame=self.target_from_warp_frame._to_cpp(),
            x_scale=self.x_scale,
            y_scale=self.y_scale,
        )

    @staticmethod
    def _from_cpp(cpp: lbb.WarpCoordinates) -> "WarpCoordinates":
        """Deserialise from the C++ bindings representation."""
        return WarpCoordinates(
            target_from_warp_frame=Pose._from_cpp(cpp.target_from_warp_frame),
            x_scale=cpp.x_scale,
            y_scale=cpp.y_scale,
        )


@dataclass
class TargetWarp:
    """Quadratic warp applied to the calibration target to model slight non-planarity.

    The warp displaces each point along the target normal by
    kx*(1 - x²) + ky*(1 - y²), where x and y are scaled to [-1, 1].

    Attributes:
        warp_coordinates: Frame and scale used to map target points to [-1, 1].
        object_warp: Quadratic warp coefficients (kx, ky).
    """

    warp_coordinates: WarpCoordinates
    object_warp: tuple[float, float]

    def warp_target(self, target_points: np.ndarray) -> np.ndarray:
        """Apply the quadratic warp to 3D target points.

        Args:
            target_points: Shape (N, 3).

        Returns:
            Warped points in the target frame, shape (N, 3).
        """
        points_in_warp = self.warp_coordinates.target_from_warp_frame.inverse().apply(
            target_points
        )
        x_in_warp = points_in_warp[:, 0]
        y_in_warp = points_in_warp[:, 1]

        scaled_x_in_warp = x_in_warp / self.warp_coordinates.x_scale
        scaled_y_in_warp = y_in_warp / self.warp_coordinates.y_scale

        kx, ky = self.object_warp

        z = kx * (1 - scaled_x_in_warp**2) + ky * (1 - scaled_y_in_warp**2)

        warped_points_in_warp = points_in_warp.copy()
        warped_points_in_warp[:, 2] = z

        warped_points_in_target = self.warp_coordinates.target_from_warp_frame.apply(
            warped_points_in_warp
        )

        return warped_points_in_target


@dataclass
class CalibrationResult(Generic[T]):
    """Output of camera calibration.

    Attributes:
        optimized_camera_model: The calibrated camera model.
        optimized_cameras_T_target: One pose per image (camera-from-target).
        frame_infos: Per-image reprojection diagnostics, one per input image.
        warp_info: Estimated target warp, or None if not estimated.
    """

    optimized_camera_model: T
    optimized_cameras_T_target: list[Pose]
    frame_infos: list[FrameInfo]
    warp_info: TargetWarp | None = None


def _project_and_calculate_residuals(
    target_points: np.ndarray,
    camera_from_target: Pose,
    frame: Frame,
    model: OpenCV | PinholeSplined,
    target_warp: TargetWarp | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    point_indices = frame.target_point_indices

    points_in_target = target_points[point_indices]
    if target_warp is not None:
        points_in_target = target_warp.warp_target(points_in_target)
    points_in_camera = camera_from_target.apply(points_in_target)

    projected_points_in_image = model.project_points(points_in_camera)

    residuals = frame.detected_points_in_image - projected_points_in_image

    return projected_points_in_image, residuals


def _mad_sigma_1d(x: np.ndarray) -> float:
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    # 1.4826 = 1 / Phi^{-1}(0.75)  (MAD->sigma for 1D normal)
    return 1.4826 * mad


def _robust_sigma_xy(residuals: list[np.ndarray]) -> float:
    R = np.concatenate(residuals, axis=0)  # (M,2)
    sx = _mad_sigma_1d(R[:, 0])
    sy = _mad_sigma_1d(R[:, 1])
    # combine to one sigma (assume roughly same scale)
    return float(np.sqrt(0.5 * (sx * sx + sy * sy)))


def _radius_threshold_from_k(k: float) -> float:
    # If x,y ~ N(0, sigma^2), then r^2/sigma^2 ~ chi2(df=2).
    # A 1D ±kσ "inlier probability" is p = 2*Phi(k)-1.
    # For df=2: chi2 CDF is 1-exp(-x/2), so quantile is x = -2 ln(1-p).
    # Threshold radius = sqrt(x) * sigma.
    p1 = 0.5 * (1.0 + math.erf(k / math.sqrt(2.0)))
    p = 2.0 * p1 - 1.0
    x = -2.0 * np.log(1.0 - p)
    return float(np.sqrt(x))


def _filter_outliers(
    frames: list,
    residuals: list[np.ndarray],
    k: float = 3.5,
    sigma_floor_px: float = 0.25,  # prevents collapse
) -> list:
    sigma = max(_robust_sigma_xy(residuals), sigma_floor_px)
    gate = _radius_threshold_from_k(k) * sigma  # radius in pixels

    filtered = []
    for frame, r in zip(frames, residuals):
        inlier_mask = np.linalg.norm(r, axis=1) <= gate
        filtered.append(
            type(frame)(
                frame.target_point_indices[inlier_mask],
                frame.detected_points_in_image[inlier_mask],
            )
        )
    return filtered


def _opencv_calibrate_inner(
    curr_intrinsics: OpenCV,
    config: OpenCVConfig,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    frames: list[Frame],
    warp_coordinates: WarpCoordinates | None = None,
    warp_kxy: tuple[float, float] | None = None,
) -> tuple[OpenCV, list[Pose], tuple[float, float] | None]:
    params = curr_intrinsics._params()
    mask = config.optimize_mask()
    intrinsics_param_optimize_mask = mask.tolist()

    cameras_from_target_in = [p._to_cpp() for p in curr_cameras_from_target]

    LOG.info("Running full optimization...")
    start_time = default_timer()
    result = lbb.calibrate_opencv(
        intrinsics_initial_value=params,
        intrinsics_param_optimize_mask=intrinsics_param_optimize_mask,
        cameras_from_target=cameras_from_target_in,
        target_points=list(target_points),
        frames=[f._to_cpp() for f in frames],
        warp_coordinates=(
            warp_coordinates._to_cpp() if warp_coordinates is not None else None
        ),
        warp_kxy_initial=list(warp_kxy) if warp_kxy is not None else [0.0, 0.0],
    )
    end_time = default_timer()
    LOG.info(f"Ran optimizer in {end_time - start_time:.2f}s")

    optimized_intrinsics = curr_intrinsics._with_params(result["intrinsics"])

    optimized_cameras_from_target: list[Pose] = [
        Pose._from_cpp(np.array(a)) for a in result["cameras_from_target"]
    ]

    out_kxy: tuple[float, float] | None = None
    if warp_coordinates is not None:
        kxy_arr = np.array(result["warp_kxy"])
        out_kxy = (float(kxy_arr[0]), float(kxy_arr[1]))

    return optimized_intrinsics, optimized_cameras_from_target, out_kxy


def _compute_frame_infos(
    intrinsics: OpenCV | PinholeSplined,
    cameras_from_target: list[Pose],
    original_frames: list[Frame],
    filtered_frames: list[Frame] | None,
    target_points: np.ndarray,
    target_warp: TargetWarp | None = None,
) -> list[FrameInfo]:
    frame_infos: list[FrameInfo] = []
    for i in range(len(cameras_from_target)):
        projected, residuals = _project_and_calculate_residuals(
            target_points,
            cameras_from_target[i],
            original_frames[i],
            intrinsics,
            target_warp,
        )

        if filtered_frames is not None:
            inlier_mask = np.isin(
                original_frames[i].target_point_indices,
                filtered_frames[i].target_point_indices,
            )
        else:
            inlier_mask = np.ones(len(original_frames[i]), dtype=bool)

        frame_infos.append(FrameInfo(projected, residuals, inlier_mask))

    return frame_infos


def _log_residual_stats(frame_infos: list[FrameInfo]) -> None:
    inlier_norms = np.concatenate(
        [
            np.linalg.norm(fi.residuals[fi.inlier_mask], axis=1)
            for fi in frame_infos
            if fi.inlier_mask.any()
        ]
    )
    LOG.info(
        f"Residuals (inliers): mean={np.mean(inlier_norms):.3f}px, "
        f"worst={np.max(inlier_norms):.3f}px"
    )


def _run_with_outlier_filtering(
    optimize_fn: Callable[
        [_OptimizationState[_IntrinsicsT]], _OptimizationState[_IntrinsicsT]
    ],
    initial_state: _OptimizationState[_IntrinsicsT],
    target_points: np.ndarray,
    outlier_threshold_stddevs: float | None,
    warp_coordinates: WarpCoordinates | None = None,
) -> _OptimizationState[_IntrinsicsT]:
    total_observations = sum(len(f) for f in initial_state.frames)
    state = initial_state

    for i in range(MAX_OUTLIER_FILTER_PASSES + 1):
        state = optimize_fn(state)

        if outlier_threshold_stddevs is None or i == MAX_OUTLIER_FILTER_PASSES:
            break

        curr_target_warp = (
            TargetWarp(warp_coordinates, state.warp_kxy)
            if warp_coordinates is not None and state.warp_kxy is not None
            else None
        )
        curr_residuals = [
            _project_and_calculate_residuals(
                target_points, cam, frame, state.intrinsics, curr_target_warp
            )[1]
            for cam, frame in zip(state.cameras_from_target, state.frames)
        ]

        new_frames = _filter_outliers(
            state.frames, curr_residuals, outlier_threshold_stddevs
        )

        if all(
            len(new_frame) == len(old_frame)
            for new_frame, old_frame in zip(new_frames, state.frames)
        ):
            break

        total_remaining = sum(len(f) for f in new_frames)
        total_outliers = total_observations - total_remaining
        LOG.info(
            f"Threw out some outliers, now have {total_outliers}/{total_observations}"
            f" ({total_outliers / total_observations * 100:.1f}%) - going again..."
        )

        state = replace(state, frames=new_frames)

    return state


def _initialize_poses_with_pnp(
    initial_intrinsics: OpenCV,
    target_points: np.ndarray,
    frames: list[Frame],
) -> list[Pose]:
    K = initial_intrinsics.K()
    dist_coeffs = np.zeros(5, dtype=np.float64)

    poses = []
    for frame in frames:
        obj_pts = target_points[frame.target_point_indices].astype(np.float64)
        img_pts = frame.detected_points_in_image.astype(np.float64)

        if len(obj_pts) >= 4:
            success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist_coeffs)
            if success:
                poses.append(
                    Pose.from_rotvec_trans(rotvec=rvec.flatten(), trans=tvec.flatten())
                )
                continue

        LOG.warning("PnP init failed for frame, using fallback pose")
        poses.append(Pose.from_tz(100))

    return poses


_PLANARITY_RATIO_THRESHOLD = 0.1
_RECT_FIT_RATIO_THRESHOLD = 0.85


def _make_warp_coordinates(target_points: np.ndarray) -> WarpCoordinates | None:
    centroid = target_points.mean(axis=0)
    centered = target_points - centroid
    _, s, Vt = np.linalg.svd(centered, full_matrices=False)

    planarity_ratio = s[2] / s[1] if s[1] > 1e-10 else np.inf
    if planarity_ratio > _PLANARITY_RATIO_THRESHOLD:
        LOG.warning(
            "Target warp can only be estimated with a planar target "
            f"(planarity ratio {planarity_ratio:.3f} > {_PLANARITY_RATIO_THRESHOLD}). "
            "Skipping warp estimation."
        )
        return None

    x_in_plane = Vt[0]
    y_in_plane = Vt[1]
    points_2d = centered @ np.column_stack([x_in_plane, y_in_plane])

    pts32 = points_2d.astype(np.float32)
    rect = cv2.minAreaRect(pts32)
    rect_w, rect_h = float(rect[1][0]), float(rect[1][1])
    rect_area = rect_w * rect_h
    hull_area = float(cv2.contourArea(cv2.convexHull(pts32)))

    use_rect = rect_area > 1e-10 and hull_area / rect_area > _RECT_FIT_RATIO_THRESHOLD

    if use_rect:
        box = cv2.boxPoints(rect).astype(float)
        e0 = box[1] - box[0]
        e1 = box[3] - box[0]
        u = e0 / np.linalg.norm(e0)
        v = e1 / np.linalg.norm(e1)
        cx2, cy2 = float(rect[0][0]), float(rect[0][1])
        x_scale = float(np.linalg.norm(e0) / 2.0)
        y_scale = float(np.linalg.norm(e1) / 2.0)
    else:
        LOG.info("Target is not rectangular; falling back to PCA for warp frame axes.")
        eigvals, eigvecs = np.linalg.eigh(np.cov(points_2d.T))
        order = np.argsort(eigvals)[::-1]
        u = eigvecs[:, order[0]]
        v = eigvecs[:, order[1]]
        proj_u = points_2d @ u
        proj_v = points_2d @ v
        cu = (proj_u.max() + proj_u.min()) / 2.0
        cv_val = (proj_v.max() + proj_v.min()) / 2.0
        center_2d = cu * u + cv_val * v
        cx2, cy2 = float(center_2d[0]), float(center_2d[1])
        x_scale = float((proj_u.max() - proj_u.min()) / 2.0)
        y_scale = float((proj_v.max() - proj_v.min()) / 2.0)

    x_hat = u[0] * x_in_plane + u[1] * y_in_plane
    y_hat = v[0] * x_in_plane + v[1] * y_in_plane
    z_hat = np.cross(x_hat, y_hat)

    center_3d = centroid + cx2 * x_in_plane + cy2 * y_in_plane
    R = np.column_stack([x_hat, y_hat, z_hat])

    return WarpCoordinates(
        target_from_warp_frame=Pose.from_rotmat_trans(rotmat=R, trans=center_3d),
        x_scale=x_scale,
        y_scale=y_scale,
    )


def _opencv_calibrate(
    target_points: np.ndarray,
    frames: list[Frame],
    config: OpenCVConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
) -> CalibrationResult[OpenCV]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    initial_intrinsics = config.get_initial_value()
    LOG.info("Computing initial poses with PnP...")
    initial_cameras_from_target = _initialize_poses_with_pnp(
        initial_intrinsics, target_points, frames
    )

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(state: _OptimizationState[OpenCV]) -> _OptimizationState[OpenCV]:
        intrinsics, cameras, kxy = _opencv_calibrate_inner(
            state.intrinsics,
            config,
            state.cameras_from_target,
            target_points,
            state.frames,
            warp_coordinates,
            state.warp_kxy,
        )
        return replace(
            state, intrinsics=intrinsics, cameras_from_target=cameras, warp_kxy=kxy
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(initial_intrinsics, initial_cameras_from_target, frames, None),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
    )

    warp_info = None
    if warp_coordinates is not None and state.warp_kxy is not None:
        warp_info = TargetWarp(
            warp_coordinates=warp_coordinates, object_warp=state.warp_kxy
        )
        kx, ky = state.warp_kxy
        LOG.info(f"Target warp optimized at kx={kx:.2f}, ky={ky:.2f}")

    frame_infos = _compute_frame_infos(
        state.intrinsics,
        state.cameras_from_target,
        frames,
        state.frames if state.frames is not frames else None,
        target_points,
        warp_info,
    )

    _log_residual_stats(frame_infos)
    return CalibrationResult(
        optimized_camera_model=state.intrinsics,
        optimized_cameras_T_target=state.cameras_from_target,
        frame_infos=frame_infos,
        warp_info=warp_info,
    )


def _pinhole_splined_refine_inner(
    curr_intrinsics: PinholeSplined,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    frames: list[Frame],
    warp_coordinates: WarpCoordinates | None,
    warp_kxy: tuple[float, float] | None = None,
) -> tuple[PinholeSplined, list[Pose], tuple[float, float] | None]:
    fine_tune_result = lbb.fine_tune_pinhole_splined(
        model_config=curr_intrinsics._cpp_config(),
        intrinsics_parameters=curr_intrinsics._cpp_params(),
        cameras_from_target=[pose._to_cpp() for pose in curr_cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in frames],
        warp_coordinates=(
            warp_coordinates._to_cpp() if warp_coordinates is not None else None
        ),
        warp_kxy_initial=list(warp_kxy) if warp_kxy is not None else [0.0, 0.0],
    )

    optimized_cameras_from_target = [
        Pose._from_cpp(np.array(a)) for a in fine_tune_result["cameras_from_target"]
    ]

    optimized_intrinsics = replace(
        curr_intrinsics,
        dx_grid=fine_tune_result["dx_grid"],
        dy_grid=fine_tune_result["dy_grid"],
    )

    out_kxy: tuple[float, float] | None = None
    if warp_coordinates is not None:
        kxy_arr = np.array(fine_tune_result["warp_kxy"])
        out_kxy = (float(kxy_arr[0]), float(kxy_arr[1]))

    return optimized_intrinsics, optimized_cameras_from_target, out_kxy


def _compute_fov_from_opencv(
    opencv_model: OpenCV, buffer_deg: float = 2.0
) -> tuple[float, float]:
    return (
        opencv_model.fov_deg_x + buffer_deg,
        opencv_model.fov_deg_y + buffer_deg,
    )


def _calibrate_pinhole_splined(
    target_points: np.ndarray,
    frames: list[Frame],
    config: PinholeSplinedConfig,
    outlier_threshold_stddevs: float | None,
    estimate_target_warp: bool,
) -> CalibrationResult[PinholeSplined]:
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    opencv_config = OpenCVConfig(
        image_height=config.image_height,
        image_width=config.image_width,
        initial_focal_length=config.initial_focal_length,
        included_distoriton_coefficients=OpenCVConfig.FULL_12,
    )

    LOG.info("Calibrating seed opencv model...")
    start_time = default_timer()
    opencv_calibration_result = _opencv_calibrate(
        target_points, frames, opencv_config, None, estimate_target_warp=False
    )
    end_time = default_timer()
    LOG.info(f"OpenCV seed model ready in {end_time - start_time:.1f}s")

    opencv_model = opencv_calibration_result.optimized_camera_model

    fov_deg_x, fov_deg_y = _compute_fov_from_opencv(opencv_model)
    LOG.info(f"Computed FOV from OpenCV model: {fov_deg_x:.1f}° x {fov_deg_y:.1f}°")

    cpp_config = lbb.PinholeSplinedConfig(
        config.image_width,
        config.image_height,
        fov_deg_x,
        fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
    )

    LOG.info("Calculating matching spline model...")
    start_time = default_timer()
    out_dict = lbb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(), cpp_config
    )
    end_time = default_timer()
    LOG.info(f"Matching spline model ready in {end_time - start_time:.1f}s")

    x_knots = out_dict["x_knots"]
    y_knots = out_dict["y_knots"]

    prior_model = PinholeSplined(
        image_height=config.image_height,
        image_width=config.image_width,
        fx=opencv_model.fx,
        fy=opencv_model.fy,
        cx=opencv_model.cx,
        cy=opencv_model.cy,
        dx_grid=x_knots,
        dy_grid=y_knots,
        num_knots_x=config.num_knots_x,
        num_knots_y=config.num_knots_y,
        fov_deg_x=fov_deg_x,
        fov_deg_y=fov_deg_y,
    )

    cameras_from_target = opencv_calibration_result.optimized_cameras_T_target

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(
        state: _OptimizationState[PinholeSplined],
    ) -> _OptimizationState[PinholeSplined]:
        LOG.info("Running full optimization...")
        start = default_timer()
        intrinsics, cameras, kxy = _pinhole_splined_refine_inner(
            state.intrinsics,
            state.cameras_from_target,
            target_points,
            state.frames,
            warp_coordinates,
            state.warp_kxy,
        )
        LOG.info(f"Performed full optimization in {default_timer() - start:.2f}s")
        return replace(
            state, intrinsics=intrinsics, cameras_from_target=cameras, warp_kxy=kxy
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(prior_model, cameras_from_target, frames, None),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
    )

    warp_info = None
    if warp_coordinates is not None and state.warp_kxy is not None:
        warp_info = TargetWarp(
            warp_coordinates=warp_coordinates, object_warp=state.warp_kxy
        )

        kx, ky = state.warp_kxy
        LOG.info(f"Target warp optimized at kx={kx:.2f}, ky={ky:.2f}")

    frame_infos = _compute_frame_infos(
        state.intrinsics,
        state.cameras_from_target,
        frames,
        state.frames if state.frames is not frames else None,
        target_points,
        warp_info,
    )

    final_intrinsics = replace(
        state.intrinsics, seed_opencv_distortion_parameters=opencv_model.distortion_coeffs
    )

    _log_residual_stats(frame_infos)
    return CalibrationResult(
        optimized_camera_model=final_intrinsics,
        optimized_cameras_T_target=state.cameras_from_target,
        frame_infos=frame_infos,
        warp_info=warp_info,
    )


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: PinholeSplinedConfig,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[PinholeSplined]: ...


@overload
def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: OpenCVConfig,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult[OpenCV]: ...


def calibrate_camera(
    target_points: np.ndarray,
    frames: list[Frame],
    camera_model_config: CameraModelConfig,
    estimate_target_warp: bool = True,
    outlier_threshold_stddevs: float | None = DEFAULT_OUTLIER_THRESHOLD,
) -> CalibrationResult:
    """Calibrate a camera from a set of per-image frames.

    Target warp estimation requires a planar target; it will be skipped
    automatically if the target points are not sufficiently coplanar.

    Args:
        target_points: 3D target point coordinates, shape (N, 3).
        frames: Per-image frames, one per calibration image.
        camera_model_config: Specifies the camera model to fit.
        estimate_target_warp: Whether to estimate a quadratic warp of the target
            to account for slight non-planarity.
        outlier_threshold_stddevs: Sigma threshold for outlier rejection.
            Pass None to disable.

    Returns:
        Calibration result containing the optimised model and per-image diagnostics.
    """
    assert target_points.ndim == 2 and target_points.shape[1] == 3, (
        f"Expected (N, 3) target_points, got {target_points.shape}"
    )
    assert np.issubdtype(target_points.dtype, np.floating), (
        f"Expected floating dtype for target_points, got {target_points.dtype}"
    )
    if isinstance(camera_model_config, PinholeSplinedConfig):
        return _calibrate_pinhole_splined(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
        )

    if isinstance(camera_model_config, OpenCVConfig):
        return _opencv_calibrate(
            target_points,
            frames,
            camera_model_config,
            outlier_threshold_stddevs,
            estimate_target_warp,
        )

    raise RuntimeError("Invalid config")
