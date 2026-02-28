#pragma once
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>

namespace lensboy {

template <typename T>
using Vec2 = Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec6 = Eigen::Matrix<T, 6, 1>;

struct WarpCoordinates {
    Vec6<double> target_from_warp_frame;  // [rx, ry, rz, tx, ty, tz]
    double x_scale;
    double y_scale;
};

}  // namespace lensboy