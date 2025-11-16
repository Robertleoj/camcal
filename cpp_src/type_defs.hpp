#pragma once
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>

namespace camcal {

template <typename T>
using Vec2 = Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec6 = Eigen::Matrix<T, 6, 1>;

}