from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from timeit import default_timer
from typing import TYPE_CHECKING, Generic, TypeVar, overload

if TYPE_CHECKING:
    from matplotlib.figure import Figure

import cv2
import numpy as np

from lensboy import lensboy_bindings as lbb
from lensboy._logging import log, warn
from lensboy.camera_models.base_model import CameraModelConfig
from lensboy.camera_models.opencv import OpenCV, OpenCVConfig
from lensboy.camera_models.pinhole_splined import (
    PinholeSplined,
    PinholeSplinedConfig,
)
from lensboy.geometry.pose import Pose

DEFAULT_OUTLIER_THRESHOLD = 5.0
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

    def __repr__(self) -> str:
        return f"Frame({len(self)} detections)"

    def __len__(self):
        return self.target_point_indices.shape[0]


_IntrinsicsT = TypeVar("_IntrinsicsT", OpenCV, PinholeSplined)


@dataclass
class FrameDiagnostics:
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

    def __repr__(self) -> str:
        n = len(self.residuals)
        n_inliers = int(np.count_nonzero(self.inlier_mask))
        if n_inliers > 0:
            mean_res = float(
                np.mean(np.linalg.norm(self.residuals[self.inlier_mask], axis=1))
            )
        else:
            mean_res = 0.0
        return (
            f"FrameDiagnostics({n_inliers}/{n} inliers, mean_residual={mean_res:.3f}px)"
        )


@dataclass
class _OptimizationState(Generic[_IntrinsicsT]):
    intrinsics: _IntrinsicsT
    cameras_from_target: list[Pose]
    frames: list[Frame]
    warp_coeffs: tuple[float, float, float, float, float] | None


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
    def _from_cpp(cpp: lbb.WarpCoordinates) -> WarpCoordinates:
        """Deserialise from the C++ bindings representation."""
        return WarpCoordinates(
            target_from_warp_frame=Pose._from_cpp(cpp.target_from_warp_frame),
            x_scale=cpp.x_scale,
            y_scale=cpp.y_scale,
        )


@dataclass
class TargetWarp:
    """Legendre-polynomial warp applied to the calibration target to model non-planarity.

    The warp displaces each point along the target normal by
    a*P2(x) + b*P2(y) + c*P2(x)*P2(y) + d*P4(x) + e*P4(y),
    where x and y are scaled to [-1, 1] and P2, P4 are Legendre polynomials.

    Attributes:
        warp_coordinates: Frame and scale used to map target points to [-1, 1].
        object_warp: Legendre warp coefficients (a, b, c, d, e).
    """

    warp_coordinates: WarpCoordinates
    object_warp: tuple[float, float, float, float, float]

    def __repr__(self) -> str:
        c = self.object_warp
        return (
            f"TargetWarp(coeffs=["
            f"{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}, "
            f"{c[3]:.4f}, {c[4]:.4f}])"
        )

    def warp_target(self, target_points: np.ndarray) -> np.ndarray:
        """Apply the Legendre warp to 3D target points.

        Args:
            target_points: Shape (N, 3).

        Returns:
            Warped points in the target frame, shape (N, 3).
        """
        return lbb.warp_target_points(
            self.warp_coordinates._to_cpp(),
            np.array(self.object_warp),
            target_points,
        )

    def max_deflection(self, target_points: np.ndarray) -> float:
        """Peak-to-peak z-deflection caused by the warp, in target units.

        Args:
            target_points: Original 3D target coordinates, shape (N, 3).

        Returns:
            The spread (max minus min) of warp-induced z change.
        """
        warp_frame_from_target = self.warp_coordinates.target_from_warp_frame.inverse()
        z_original = warp_frame_from_target.apply(target_points)[:, 2]
        warped = self.warp_target(target_points)
        z_warped = warp_frame_from_target.apply(warped)[:, 2]
        dz = z_warped - z_original
        return float(np.max(dz) - np.min(dz))


@dataclass
class CalibrationResult(Generic[_IntrinsicsT]):
    """Output of camera calibration.

    Attributes:
        camera_model: The calibrated camera model.
        cameras_from_target: One pose per image (camera-from-target).
        frame_diagnostics: Per-image reprojection diagnostics, one per input image.
        frames: Input detection frames used for calibration.
        target_points: 3D calibration target points, shape (M, 3).
        target_warp: Estimated target warp, or None if not estimated.
    """

    camera_model: _IntrinsicsT
    cameras_from_target: list[Pose]
    frame_diagnostics: list[FrameDiagnostics]
    frames: list[Frame]
    target_points: np.ndarray
    target_warp: TargetWarp | None = None

    def __repr__(self) -> str:
        model = self.camera_model
        sigma = self.residual_sigma_map()
        n_out = self.num_outliers()
        n_det = self.num_detections()
        return (
            f"CalibrationResult(\n"
            f"  model={model!r},\n"
            f"  frames={len(self.frames)}, "
            f"detections={n_det}, outliers={n_out},\n"
            f"  residual_sigma={sigma:.4f}px"
            f"{',' if self.target_warp is not None else ''}\n"
            + (
                f"  target_warp={self.target_warp!r}\n"
                if self.target_warp is not None
                else ""
            )
            + ")"
        )

    def residual_sigma_map(self) -> float:
        """Robust MAP estimate of residual standard deviation over inliers.

        Uses the MAD-based estimator (sigma = 1.4826 * MAD) on all inlier
        residual components (x and y combined).

        Returns:
            Estimated residual standard deviation in pixels.
        """
        inlier_vals = np.concatenate(
            [fi.residuals[fi.inlier_mask] for fi in self.frame_diagnostics]
        ).ravel()
        mu = float(np.median(inlier_vals))
        mad = float(np.median(np.abs(inlier_vals - mu)))
        return 1.4826 * mad

    def num_outliers(self) -> int:
        """Count the total number of outlier detections across all frames.

        Returns:
            Total outlier count.
        """
        return sum(
            int(np.count_nonzero(~fi.inlier_mask)) for fi in self.frame_diagnostics
        )

    def num_detections(self) -> int:
        """Count the total number of detections across all frames.

        Returns:
            Total detection count (inliers + outliers).
        """
        return sum(len(fi.residuals) for fi in self.frame_diagnostics)

    # -- Plot forwarding methods --
    # These require the `analysis` extra (pip install lensboy[analysis]).

    def plot_detection_coverage(
        self,
        *,
        title: str = "Coverage",
        s: float = 6.0,
        grid_cells: int = 20,
        return_figure: bool = False,
    ) -> Figure | None:
        """Scatter-plot all detected points with empty grid cells highlighted.

        Divides the image into a grid and shades cells with no detections,
        making coverage gaps easy to spot.

        Args:
            title: Plot title.
            s: Marker size passed to ``ax.scatter``.
            grid_cells: Number of grid cells along the longer image axis.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_detection_coverage

        return plot_detection_coverage(
            self.frames,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            title=title,
            s=s,
            grid_cells=grid_cells,
            return_figure=return_figure,
        )

    def plot_distortion_grid(
        self,
        *,
        grid_step_norm: float = 0.05,
        fov_fraction: float | None = None,
        ux_max: float | None = None,
        uy_max: float | None = None,
        cmap_name: str = "jet",
        show_spline_knots: bool = False,
        return_figure: bool = False,
    ) -> Figure | None:
        """Project a regular grid through a camera model to visualize distortion.

        Builds a grid in normalized (tan-angle) space from the model's FOV, projects
        it, and clips to the image bounds.

        Args:
            grid_step_norm: Spacing between grid lines in normalized coordinates.
            fov_fraction: Fraction of the full FOV to sample (0, 1].
            ux_max: Upper bound in normalized x, mirrored to negative.
            uy_max: Upper bound in normalized y, mirrored to negative.
            cmap_name: Matplotlib colormap name.
            show_spline_knots: When True and the model is a PinholeSplined,
                overlay the spline control points on both panels.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_distortion_grid

        return plot_distortion_grid(
            self.camera_model,
            grid_step_norm=grid_step_norm,
            fov_fraction=fov_fraction,
            ux_max=ux_max,
            uy_max=uy_max,
            cmap_name=cmap_name,
            show_spline_knots=show_spline_knots,
            return_figure=return_figure,
        )

    def plot_residuals(
        self,
        *,
        bins: int = 100,
        n_sigma: float = 6.0,
        axis_range: float | None = None,
        title: str = "Reprojection residuals",
        return_figure: bool = False,
    ) -> Figure | None:
        """Per-component histogram and 2D scatter of reprojection residuals.

        Top-left: inlier histogram with a 1D Gaussian fit overlaid.
        Bottom-left: 2D scatter of inlier residuals with fitted 2D Gaussian
        contours.  Both left panels are trimmed to ±``n_sigma`` standard
        deviations.  Right column: full-range scatter highlighting outliers
        in red.

        Args:
            bins: Number of histogram bins.
            n_sigma: Number of fitted-Gaussian standard deviations for axis limits.
            axis_range: Fixed symmetric axis limit (±value) for the histogram and
                2D scatter plots. The full-range plot is unaffected. Auto-scaled
                from n_sigma if None.
            title: Overall figure title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_residuals

        return plot_residuals(
            self.frame_diagnostics,
            bins=bins,
            n_sigma=n_sigma,
            axis_range=axis_range,
            title=title,
            return_figure=return_figure,
        )

    def plot_residual_vectors(
        self,
        *,
        title: str = "Residual vectors",
        scale: float = 10.0,
        scale_by_magnitude: bool = True,
        color_by: str = "magnitude",
        return_figure: bool = False,
    ) -> Figure | None:
        """Quiver plot of reprojection residual vectors over the image plane.

        Each arrow is placed at the detected point location with direction and
        length given by the residual.

        Args:
            title: Plot title.
            scale: Multiplier applied to arrow lengths for visibility.
            scale_by_magnitude: When False, all arrows are drawn with uniform
                length (direction only).
            color_by: ``"magnitude"`` colours by residual norm, ``"angle"``
                colours by residual direction using a cyclic colormap.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_residual_vectors

        return plot_residual_vectors(
            self.frames,
            self.frame_diagnostics,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            title=title,
            scale=scale,
            scale_by_magnitude=scale_by_magnitude,
            color_by=color_by,
            return_figure=return_figure,
        )

    def plot_residual_grid(
        self,
        *,
        grid_cells: int = 40,
        arrow_scale: float = 100.0,
        heatmap_max: float | None = None,
        title: str = "Residual grid",
        return_figure: bool = False,
    ) -> Figure | None:
        """Binned residual summary showing per-cell magnitude and mean direction.

        The image plane is divided into a grid. Each cell is coloured by the mean
        inlier residual magnitude and has an arrow showing the mean residual
        vector, revealing spatial bias patterns.

        Args:
            grid_cells: Number of grid cells along the longer image axis.
            arrow_scale: Multiplier applied to the mean-residual arrows.
            heatmap_max: Upper limit for the colour scale. Auto-scaled if None.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_residual_grid

        return plot_residual_grid(
            self.frames,
            self.frame_diagnostics,
            image_width=self.camera_model.image_width,
            image_height=self.camera_model.image_height,
            grid_cells=grid_cells,
            arrow_scale=arrow_scale,
            heatmap_max=heatmap_max,
            title=title,
            return_figure=return_figure,
        )

    def plot_target_and_poses(
        self,
        *,
        triad_scale: float = 20.0,
        title: str = "Target and camera poses",
        return_figure: bool = False,
    ) -> Figure | None:
        """3D scatter of the calibration target with camera poses shown as triads.

        Each camera is drawn as a coordinate-frame triad (X=red, Y=green, Z=blue)
        at the camera position in the target reference frame.

        Args:
            triad_scale: Length of each triad axis arrow in target units.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_target_and_poses

        return plot_target_and_poses(
            self.target_points,
            self.cameras_from_target,
            triad_scale=triad_scale,
            title=title,
            return_figure=return_figure,
        )

    def plot_target_warp(
        self,
        *,
        grid_res: int = 300,
        contour_levels: int = 15,
        title: str = "Target warp",
        return_figure: bool = False,
    ) -> Figure | None:
        """Contour plot of the target warp z-displacement viewed from above.

        Evaluates the warp function over a dense grid in the warp frame's xy plane
        and shows filled contours of the z height, with target point positions
        scattered on top.

        Args:
            grid_res: Number of grid samples along each axis.
            contour_levels: Number of contour lines.
            title: Plot title.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.

        Raises:
            ValueError: If no target warp was estimated.
        """
        from lensboy.analysis.plots import plot_target_warp

        if self.target_warp is None:
            raise ValueError("No target warp was estimated in this calibration.")
        return plot_target_warp(
            self.target_points,
            self.target_warp,
            grid_res=grid_res,
            contour_levels=contour_levels,
            title=title,
            return_figure=return_figure,
        )

    def plot_worst_residual_frames(
        self,
        images: list[np.ndarray],
        *,
        n: int = 5,
        scale: float = 10.0,
        title: str = "Worst residual frames",
        include_outliers: bool = True,
        return_figure: bool = False,
    ) -> Figure | None:
        """Show the frames with the largest residuals, with residual vectors overlaid.

        Frames are ranked by their single worst (max-magnitude) residual and the
        top ``n`` are displayed in a single-column figure.  Each subplot shows the
        image with quiver arrows from detected points in the direction and
        magnitude of the residual, coloured by magnitude.

        Args:
            images: Source images corresponding to each frame, shape (H, W) or (H, W, 3).
            n: Number of worst frames to display.
            scale: Multiplier applied to arrow lengths for visibility.
            title: Overall figure title.
            include_outliers: Whether to include outlier points. When False, only
                inlier points (per ``FrameDiagnostics.inlier_mask``) are shown and used
                for ranking.
            return_figure: If True, return the figure instead of calling ``plt.show()``.

        Returns:
            The figure if ``return_figure`` is True, otherwise None.
        """
        from lensboy.analysis.plots import plot_worst_residual_frames

        return plot_worst_residual_frames(
            self.frame_diagnostics,
            self.frames,
            images,
            n=n,
            scale=scale,
            title=title,
            include_outliers=include_outliers,
            return_figure=return_figure,
        )


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

    residuals = projected_points_in_image - frame.detected_points_in_image

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


def _filter_outliers(
    frames: list,
    residuals: list[np.ndarray],
    k: float,
    sigma_floor_px: float = 0.05,  # prevents collapse
) -> list:
    sigma = max(_robust_sigma_xy(residuals), sigma_floor_px)
    gate = k * sigma

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
    warp_coeffs: tuple[float, float, float, float, float] | None = None,
) -> tuple[OpenCV, list[Pose], tuple[float, float, float, float, float] | None]:
    params = curr_intrinsics._params()
    mask = config.optimize_mask()
    intrinsics_param_optimize_mask = mask.tolist()

    cameras_from_target_in = [p._to_cpp() for p in curr_cameras_from_target]

    log("Running full optimization...")
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
        warp_coeffs_initial=list(warp_coeffs) if warp_coeffs is not None else [0.0] * 5,
    )
    end_time = default_timer()
    log(f"Ran optimizer in {end_time - start_time:.2f}s")

    optimized_intrinsics = curr_intrinsics._with_params(result["intrinsics"])

    optimized_cameras_from_target: list[Pose] = [
        Pose._from_cpp(np.array(a)) for a in result["cameras_from_target"]
    ]

    out_coeffs: tuple[float, float, float, float, float] | None = None
    if warp_coordinates is not None:
        arr = np.array(result["warp_coeffs"])
        out_coeffs = (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
            float(arr[3]),
            float(arr[4]),
        )

    return optimized_intrinsics, optimized_cameras_from_target, out_coeffs


def _compute_frame_diagnostics(
    intrinsics: OpenCV | PinholeSplined,
    cameras_from_target: list[Pose],
    original_frames: list[Frame],
    filtered_frames: list[Frame] | None,
    target_points: np.ndarray,
    target_warp: TargetWarp | None = None,
) -> list[FrameDiagnostics]:
    frame_diagnostics: list[FrameDiagnostics] = []
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

        frame_diagnostics.append(FrameDiagnostics(projected, residuals, inlier_mask))

    return frame_diagnostics


def _log_residual_stats(frame_diagnostics: list[FrameDiagnostics]) -> None:
    inlier_norms = np.concatenate(
        [
            np.linalg.norm(fi.residuals[fi.inlier_mask], axis=1)
            for fi in frame_diagnostics
            if fi.inlier_mask.any()
        ]
    )
    log(
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
    original_frames = initial_state.frames
    total_observations = sum(len(f) for f in original_frames)
    state = initial_state

    for i in range(MAX_OUTLIER_FILTER_PASSES + 1):
        non_empty_mask = [len(f) > 0 for f in state.frames]
        if not any(non_empty_mask):
            raise ValueError(
                "All points in all frames were marked as outliers; "
                "calibration cannot continue."
            )

        if all(non_empty_mask):
            state = optimize_fn(state)
        else:
            n_empty = sum(not m for m in non_empty_mask)
            log(f"Skipping {n_empty} frame(s) with no inlier points")
            active_frames = [f for f, m in zip(state.frames, non_empty_mask) if m]
            active_poses = [
                p for p, m in zip(state.cameras_from_target, non_empty_mask) if m
            ]
            optimized = optimize_fn(
                replace(state, frames=active_frames, cameras_from_target=active_poses)
            )
            # Merge optimized poses back, keeping old poses for empty frames
            full_poses = list(state.cameras_from_target)
            j = 0
            for idx, m in enumerate(non_empty_mask):
                if m:
                    full_poses[idx] = optimized.cameras_from_target[j]
                    j += 1
            state = replace(
                optimized, frames=state.frames, cameras_from_target=full_poses
            )

        if outlier_threshold_stddevs is None or i == MAX_OUTLIER_FILTER_PASSES:
            break

        curr_target_warp = (
            TargetWarp(warp_coordinates, state.warp_coeffs)
            if warp_coordinates is not None and state.warp_coeffs is not None
            else None
        )
        # Compute residuals on the original (unfiltered) frames so that
        # previously-rejected points can be recovered if they now fit.
        curr_residuals = [
            _project_and_calculate_residuals(
                target_points, cam, frame, state.intrinsics, curr_target_warp
            )[1]
            for cam, frame in zip(state.cameras_from_target, original_frames)
        ]

        new_frames = _filter_outliers(
            original_frames, curr_residuals, outlier_threshold_stddevs
        )

        if all(
            np.array_equal(new_frame.target_point_indices, old_frame.target_point_indices)
            for new_frame, old_frame in zip(new_frames, state.frames)
        ):
            break

        total_remaining = sum(len(f) for f in new_frames)
        total_outliers = total_observations - total_remaining
        pct = total_outliers / total_observations * 100
        log(
            f"Outlier filtering: {total_outliers}/{total_observations}"
            f" ({pct:.1f}%) outliers - going again..."
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

        warn("PnP init failed for frame, using fallback pose")
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
        warn(
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
        log("Target is not rectangular; falling back to PCA for warp frame axes.")
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
    log("Computing initial poses with PnP...")
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
            state.warp_coeffs,
        )
        return replace(
            state, intrinsics=intrinsics, cameras_from_target=cameras, warp_coeffs=kxy
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(initial_intrinsics, initial_cameras_from_target, frames, None),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
    )

    target_warp = None
    if warp_coordinates is not None and state.warp_coeffs is not None:
        target_warp = TargetWarp(
            warp_coordinates=warp_coordinates, object_warp=state.warp_coeffs
        )
        deflection = target_warp.max_deflection(target_points)
        log(f"Target warp max deflection: {deflection:.4f} (target units)")

    frame_diagnostics = _compute_frame_diagnostics(
        state.intrinsics,
        state.cameras_from_target,
        frames,
        state.frames if state.frames is not frames else None,
        target_points,
        target_warp,
    )

    _log_residual_stats(frame_diagnostics)
    return CalibrationResult(
        camera_model=state.intrinsics,
        cameras_from_target=state.cameras_from_target,
        frame_diagnostics=frame_diagnostics,
        frames=frames,
        target_points=target_points,
        target_warp=target_warp,
    )


def _pinhole_splined_refine_inner(
    curr_intrinsics: PinholeSplined,
    curr_cameras_from_target: list[Pose],
    target_points: np.ndarray,
    frames: list[Frame],
    warp_coordinates: WarpCoordinates | None,
    warp_coeffs: tuple[float, float, float, float, float] | None = None,
) -> tuple[PinholeSplined, list[Pose], tuple[float, float, float, float, float] | None]:
    fine_tune_result = lbb.fine_tune_pinhole_splined(
        model_config=curr_intrinsics._cpp_config(),
        intrinsics_parameters=curr_intrinsics._cpp_params(),
        cameras_from_target=[pose._to_cpp() for pose in curr_cameras_from_target],
        target_points=list(target_points),
        frames=[f._to_cpp() for f in frames],
        warp_coordinates=(
            warp_coordinates._to_cpp() if warp_coordinates is not None else None
        ),
        warp_coeffs_initial=list(warp_coeffs) if warp_coeffs is not None else [0.0] * 5,
    )

    optimized_cameras_from_target = [
        Pose._from_cpp(np.array(a)) for a in fine_tune_result["cameras_from_target"]
    ]

    optimized_intrinsics = replace(
        curr_intrinsics,
        dx_grid=fine_tune_result["dx_grid"],
        dy_grid=fine_tune_result["dy_grid"],
    )

    out_coeffs: tuple[float, float, float, float, float] | None = None
    if warp_coordinates is not None:
        arr = np.array(fine_tune_result["warp_coeffs"])
        out_coeffs = (
            float(arr[0]),
            float(arr[1]),
            float(arr[2]),
            float(arr[3]),
            float(arr[4]),
        )

    return optimized_intrinsics, optimized_cameras_from_target, out_coeffs


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
        included_distortion_coefficients=OpenCVConfig.FULL_14,
    )

    log("Calibrating seed opencv model...")
    start_time = default_timer()
    opencv_calibration_result = _opencv_calibrate(
        target_points, frames, opencv_config, None, estimate_target_warp=False
    )
    end_time = default_timer()
    log(f"OpenCV seed model ready in {end_time - start_time:.1f}s")

    opencv_model = opencv_calibration_result.camera_model

    fov_deg_x, fov_deg_y = _compute_fov_from_opencv(opencv_model)
    log(f"Computed FOV from OpenCV model: {fov_deg_x:.1f}° x {fov_deg_y:.1f}°")

    cpp_config = lbb.PinholeSplinedConfig(
        config.image_width,
        config.image_height,
        fov_deg_x,
        fov_deg_y,
        config.num_knots_x,
        config.num_knots_y,
    )

    log("Calculating matching spline model...")
    start_time = default_timer()
    out_dict = lbb.get_matching_spline_distortion_model(
        opencv_model.distortion_coeffs.tolist(), cpp_config
    )
    end_time = default_timer()
    log(f"Matching spline model ready in {end_time - start_time:.1f}s")

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

    cameras_from_target = opencv_calibration_result.cameras_from_target

    warp_coordinates = None
    if estimate_target_warp:
        warp_coordinates = _make_warp_coordinates(target_points)

    def optimize_fn(
        state: _OptimizationState[PinholeSplined],
    ) -> _OptimizationState[PinholeSplined]:
        log("Running full optimization...")
        start = default_timer()
        intrinsics, cameras, kxy = _pinhole_splined_refine_inner(
            state.intrinsics,
            state.cameras_from_target,
            target_points,
            state.frames,
            warp_coordinates,
            state.warp_coeffs,
        )
        log(f"Performed full optimization in {default_timer() - start:.2f}s")
        return replace(
            state, intrinsics=intrinsics, cameras_from_target=cameras, warp_coeffs=kxy
        )

    state = _run_with_outlier_filtering(
        optimize_fn,
        _OptimizationState(prior_model, cameras_from_target, frames, None),
        target_points,
        outlier_threshold_stddevs,
        warp_coordinates=warp_coordinates,
    )

    target_warp = None
    if warp_coordinates is not None and state.warp_coeffs is not None:
        target_warp = TargetWarp(
            warp_coordinates=warp_coordinates, object_warp=state.warp_coeffs
        )
        deflection = target_warp.max_deflection(target_points)
        log(f"Target warp max deflection: {deflection:.4f} (target units)")

    frame_diagnostics = _compute_frame_diagnostics(
        state.intrinsics,
        state.cameras_from_target,
        frames,
        state.frames if state.frames is not frames else None,
        target_points,
        target_warp,
    )

    final_intrinsics = replace(
        state.intrinsics, seed_opencv_distortion_parameters=opencv_model.distortion_coeffs
    )

    _log_residual_stats(frame_diagnostics)
    return CalibrationResult(
        camera_model=final_intrinsics,
        cameras_from_target=state.cameras_from_target,
        frame_diagnostics=frame_diagnostics,
        frames=frames,
        target_points=target_points,
        target_warp=target_warp,
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
        estimate_target_warp: Whether to estimate a Legendre-polynomial warp
            of the target to account for slight non-planarity.
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
