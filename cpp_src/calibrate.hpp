#pragma once

// #include <Eigen/src/Core/Matrix.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace camcal {

using Vec2 = Eigen::Matrix<double, 2, 1>;
using Vec3 = Eigen::Matrix<double, 3, 1>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

void calibrate_camera(
    std::string camera_model_name,
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6>& camera_poses_world,
    std::vector<Vec3>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2> > >&
        detections
);

}  // namespace camcal