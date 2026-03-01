#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "./type_defs.hpp"

namespace lensboy {

template <typename T>
Eigen::Matrix<T, 3, 1> transform_point(
    const Vec6<T>& pose,         // [rx ry rz tx ty tz]
    const Vec3<T>& point_target  // [x y z]
) {
    Vec3<T> rotated;
    ceres::AngleAxisRotatePoint(
        pose.data(),
        point_target.data(),
        rotated.data()
    );

    rotated[0] += pose[3];
    rotated[1] += pose[4];
    rotated[2] += pose[5];

    return rotated;
}

template <typename T>
Vec3<T> apply_warp_to_target_point(
    const Vec3<T>& p_target,
    const WarpCoordinates& warp,
    const T* const kxy
) {
    const Vec3<double> rv = warp.target_from_warp_frame.head<3>();
    const Vec3<double> center = warp.target_from_warp_frame.tail<3>();

    const double angle = rv.norm();
    const Eigen::Matrix3d R =
        angle < 1e-10 ? Eigen::Matrix3d::Identity()
                      : Eigen::AngleAxisd(angle, rv / angle).toRotationMatrix();

    const Vec3<double> x_hat = R.col(0);
    const Vec3<double> y_hat = R.col(1);
    const Vec3<double> z_hat = R.col(2);

    const Vec3<T> d = p_target - center.cast<T>();

    const T wx = T(x_hat[0]) * d[0] + T(x_hat[1]) * d[1] + T(x_hat[2]) * d[2];
    const T wy = T(y_hat[0]) * d[0] + T(y_hat[1]) * d[1] + T(y_hat[2]) * d[2];

    const T xs = wx / T(warp.x_scale);
    const T ys = wy / T(warp.y_scale);
    const T z_warp = kxy[0] * (T(1.0) - xs * xs) + kxy[1] * (T(1.0) - ys * ys);

    Vec3<T> result = center.cast<T>();
    result[0] += T(x_hat[0]) * wx + T(y_hat[0]) * wy + T(z_hat[0]) * z_warp;
    result[1] += T(x_hat[1]) * wx + T(y_hat[1]) * wy + T(z_hat[1]) * z_warp;
    result[2] += T(x_hat[2]) * wx + T(y_hat[2]) * wy + T(z_hat[2]) * z_warp;
    return result;
}

}  // namespace lensboy