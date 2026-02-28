#pragma once

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <vector>
#include "./cameramodels.hpp"
#include "./type_defs.hpp"

namespace lensboy {

namespace py = pybind11;

py::dict calibrate_opencv(
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections,
    std::optional<WarpCoordinates> warp_coordinates = std::nullopt,
    std::array<double, 2> warp_kxy_initial = {0.0, 0.0}
);

py::dict get_matching_spline_distortion_model(
    std::vector<double>& opencv_distortion_params,
    PinholeSplinedConfig& model_config
);

py::dict fine_tune_pinhole_splined(
    PinholeSplinedConfig& model_config,
    PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_target,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections,
    std::optional<WarpCoordinates> warp_coordinates = std::nullopt,
    std::array<double, 2> warp_kxy_initial = {0.0, 0.0}
);

}  // namespace lensboy