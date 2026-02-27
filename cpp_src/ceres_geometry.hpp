#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
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

}  // namespace lensboy