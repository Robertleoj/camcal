#pragma once

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "./cameramodels.hpp"
#include "./type_defs.hpp"

namespace camcal {

namespace py = pybind11;

py::dict calibrate_opencv(
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
);

py::dict get_matching_spline_distortion_model(
    std::vector<double>& opencv_distortion_params,
    PinholeSplinedConfig& model_config
);

py::dict fine_tune_pinhole_splined(
    PinholeSplinedConfig& model_config,
    PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
);

}  // namespace camcal