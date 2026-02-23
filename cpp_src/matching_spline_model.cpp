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

    // add all the control points
    for (size_t x_knot_idx = 0; x_knot_idx < num_knots_x; x_knot_idx++) {
        for (size_t y_knot_idx = 0; y_knot_idx < num_knots_y; y_knot_idx++) {
            problem.AddParameterBlock(&x_knots[x_knot_idx][y_knot_idx], 1);
            problem.AddParameterBlock(&y_knots[x_knot_idx][y_knot_idx], 1);
        }
    }

    const double inv_x_span = 1 / (x_range_end - x_range_start);
    const double inv_y_span = 1 / (y_range_end - y_range_start);

    for (size_t x_sample_idx = 0; x_sample_idx < num_samples_x;
         x_sample_idx++) {
        for (size_t y_sample_idx = 0; y_sample_idx < num_samples_y;
             y_sample_idx++) {
            // get normalized point
            double x_proportion = static_cast<double>(x_sample_idx) /
                                  (static_cast<double>(num_samples_x) - 1);
            double y_proportion = static_cast<double>(y_sample_idx) /
                                  (static_cast<double>(num_samples_y) - 1);

            double x_normalized =
                x_range_start + (x_range_end - x_range_start) * x_proportion;
            double y_normalized =
                x_range_start + (x_range_end - x_range_start) * x_proportion;

            Vec2<double> normalized_point(x_normalized, y_normalized);

            Vec2<double> opencv_distorted_point;
            // get opencv distorted point (target value)
            distort_opencv(
                opencv_distortion_params.data(),
                normalized_point,
                opencv_distorted_point
            );

            // get correct knot indices
            double x_spline =
                1.0 + (x_normalized - x_range_start) *
                          (static_cast<double>(num_knots_x) - 3.0) * inv_x_span;

            double y_spline =
                1.0 + (y_normalized - y_range_start) *
                          (static_cast<double>(num_knots_y) - 3.0) * inv_y_span;

            // create the factor
        }
    }
}
}  // namespace camcal
