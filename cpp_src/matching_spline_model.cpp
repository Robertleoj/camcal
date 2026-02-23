#include <ceres/ceres.h>
#include "./calibrate.hpp"
#include "./utils.hpp"

namespace camcal {
py::dict get_matching_spline_distortion_model(
    std::vector<double>& opencv_distortion_params,
    double fov_deg_x,
    double fov_deg_y,
    uint32_t num_knots_x,
    uint32_t num_knots_y
) {
    double fov_rad_x = fov_deg_x * M_PI / 180.0;
    double fov_rad_y = fov_deg_y * M_PI / 180.0;

    double half_x_range = std::tan(fov_rad_x / 2.0);
    double half_y_range = std::tan(fov_rad_y / 2.0);

    double x_range_start = -half_x_range;
    double x_range_end = +half_x_range;
    double y_range_start = -half_y_range;
    double y_range_end = +half_y_range;

    uint32_t num_samples_x = num_knots_x * 20;
    uint32_t num_samples_y = num_knots_y * 20;

    auto x_knots = vector_mat<double>(num_knots_x, num_knots_y, 0);
    auto y_knots = vector_mat<double>(num_knots_x, num_knots_y, 0);

    ceres::Problem problem;
}
}  // namespace camcal
