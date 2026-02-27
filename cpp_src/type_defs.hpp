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
    Vec3<double> center_in_target;
    Vec3<double> x_axis;
    Vec3<double> y_axis;
};

}  // namespace lensboy