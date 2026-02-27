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
    Vec2<double> center_in_target;
    Vec2<double> x_axis;
    Vec2<double> y_axis;
};

}  // namespace lensboy