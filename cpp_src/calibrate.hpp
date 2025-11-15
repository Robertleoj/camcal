#pragma once

// #include <Eigen/src/Core/Matrix.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace camcal {

namespace py = pybind11;

template <typename T>
using Vec2 = Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec6 = Eigen::Matrix<T, 6, 1>;

py::dict calibrate_camera(
    std::string camera_model_name,
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& camera_poses_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
);

}  // namespace camcal