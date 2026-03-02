Don't go too deep without exposing and sanity checking your approach with the user.
Reveal your thinking and processing as much as possible!

Try to avoid thinking deeply unless explicity instructed to do so.

Don't worry about linting errors if `ruff check --fix` and `ruff format` will fix them.
But do not run any of the linter checks on the terminal yourself, I'll handle that.

# Docstrings

Write docstrings for public functions/methods.

For functions, they should be on this format

```
"""<short description>

<optional long description>

Args:
    arg1: <description>
    arg2: <description>

Returns:
    <description>
    <description>
"""
```

Never say types in descriptions, as they are already annotated. Mention the shape of all numpy arrays.
Try not to duplicate information inside the docstring too much, and don't just repeat the variable names.

# Comments

Keep comments minimal, only add comments when the code needs explaining why it's doing something.

# Plots

Follow the color scheme of the existing plots in the plots.py file.
Making the type checker happy is less important in the plots if the plotting library APIs make it hard to make the types work. In this case the usage of `type: ignore` is fine.

# Project Overview

**lensboy** - a camera calibration library (package name `lensboy`, repo dir `camcal`).

- Python package with C++ pybind11 extensions (Ceres-based optimization)
- Build: scikit-build-core, CMakeLists.txt at repo root, C++ in `cpp_src/`
- Dependencies: numpy, opencv-contrib-python. Optional: matplotlib (for `analysis` extras)
- Python 3.11+

# Source Layout

```
src/lensboy/
├── __init__.py                  # Public API re-exports
├── lensboy_bindings.pyi         # Stub for C++ pybind11 module
├── calibration/
│   └── calibrate.py             # calibrate_camera(), Frame, CalibrationResult, TargetWarp, etc.
├── camera_models/
│   ├── base_model.py            # ABC: CameraModel (project_points, normalize_points), CameraModelConfig
│   ├── opencv.py                # OpenCV (pinhole + 12-param distortion), OpenCVConfig
│   ├── pinhole_splined.py       # PinholeSplined (pinhole + 2D B-spline distortion), PinholeSplinedConfig
│   └── pinhole_remapped.py      # PinholeRemapped (undistorted pinhole with remap tables)
├── common_targets/
│   └── charuco.py               # extract_frames_from_charuco()
├── geometry/
│   └── pose.py                  # Pose (4x4 rigid transform, rotvec/rotmat constructors)
├── analysis/
│   ├── __init__.py              # Lazy import guard for matplotlib
│   ├── plots.py                 # All visualization functions (plot_residuals, plot_distortion_grid, etc.)
│   └── image.py                 # to_gray(), to_color()
├── _internal/
│   ├── paths.py                 # repo_root()
│   ├── progress.py              # Custom progress bar (no tqdm dependency)
│   └── privacy.py               # privatize_images() - mask non-target regions
cpp_src/                         # C++ pybind11 extensions (Ceres optimizer)
tests/
├── test_integration.py
└── test_normalize.py
tools/
└── generate_test_dataset.py
```

# Key Architecture

- **Camera models** all inherit `CameraModel` ABC with `project_points(N,3)->(N,2)` and `normalize_points(N,2)->(N,3)`.
- **OpenCV model**: standard pinhole + up to 12 OpenCV distortion params. Config has boolean mask for which distortion coeffs to optimize. All Python, uses cv2 for projection.
- **PinholeSplined model**: pinhole + 2D B-spline distortion grids (dx_grid, dy_grid). Projection/normalization calls C++ bindings. Calibration seeds from an OpenCV fit, then refines with spline model.
- **PinholeRemapped**: output model from PinholeSplined - stores precomputed remap tables for undistortion. Simple pinhole project/normalize (no distortion).
- **Calibration pipeline** (`calibrate_camera`): takes target_points (N,3) + list of Frame + config -> CalibrationResult. Supports outlier filtering (MAD-based) and target warp estimation (Legendre polynomials for non-planar targets).
- **Pose**: 4x4 homogeneous transform. Constructors: `from_rotvec_trans`, `from_rotmat_trans`, `from_tx/ty/tz`. C++ bridge uses 6-vector (rotvec||trans).
- **Frame**: pairs `target_point_indices` (N,) with `detected_points_in_image` (N,2).
- All models have `save()`/`load()` and `to_json()`/`from_json()` for serialization.
- C++ bindings module: `lensboy.lensboy_bindings` (aliased `lbb`). Key functions: `calibrate_opencv`, `fine_tune_pinhole_splined`, `get_matching_spline_distortion_model`, `project_pinhole_splined_points`, `normalize_pinhole_splined_points`, `make_undistortion_maps_pinhole_splined`.