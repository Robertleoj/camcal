#pragma once

#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include "./type_defs.hpp"

namespace camcal {

namespace py = pybind11;

py::dict calibrate_camera(
    std::string camera_model_name,
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
);

}  // namespace camcal