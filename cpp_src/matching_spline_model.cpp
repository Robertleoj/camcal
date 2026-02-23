#include <ceres/ceres.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>
#include "./calibrate.hpp"
#include "./utils.hpp"

namespace camcal {

struct DistortionError {
    DistortionError(
        const double opencv_distorted_x,
        const double opencv_distorted_y,
        const double x_normalized,
        const double y_normalized,
        const double spline_u,
        const double spline_v
    )
        : opencv_distorted_x(opencv_distorted_x),
          opencv_distorted_y(opencv_distorted_y),
          x_normalized(x_normalized),
          y_normalized(y_normalized),
          spline_u(spline_u),
          spline_v(spline_v) {}

    template <typename T>
    bool operator()(
        const T* const dx_x1y1,
        const T* const dx_x2y1,
        const T* const dx_x3y1,
        const T* const dx_x4y1,
        const T* const dx_x1y2,
        const T* const dx_x2y2,
        const T* const dx_x3y2,
        const T* const dx_x4y2,
        const T* const dx_x1y3,
        const T* const dx_x2y3,
        const T* const dx_x3y3,
        const T* const dx_x4y3,
        const T* const dx_x1y4,
        const T* const dx_x2y4,
        const T* const dx_x3y4,
        const T* const dx_x4y4,

        const T* const dy_x1y1,
        const T* const dy_x2y1,
        const T* const dy_x3y1,
        const T* const dy_x4y1,
        const T* const dy_x1y2,
        const T* const dy_x2y2,
        const T* const dy_x3y2,
        const T* const dy_x4y2,
        const T* const dy_x1y3,
        const T* const dy_x2y3,
        const T* const dy_x3y3,
        const T* const dy_x4y3,
        const T* const dy_x1y4,
        const T* const dy_x2y4,
        const T* const dy_x3y4,
        const T* const dy_x4y4,

        T* residuals
    ) const {
        T wx[4], wy[4];
        cubic_bspline_basis_uniform(T(spline_u), wx);
        cubic_bspline_basis_uniform(T(spline_v), wy);

        T grid_x[4][4] = {
            {dx_x1y1[0], dx_x2y1[0], dx_x3y1[0], dx_x4y1[0]},
            {dx_x1y2[0], dx_x2y2[0], dx_x3y2[0], dx_x4y2[0]},
            {dx_x1y3[0], dx_x2y3[0], dx_x3y3[0], dx_x4y3[0]},
            {dx_x1y4[0], dx_x2y4[0], dx_x3y4[0], dx_x4y4[0]}
        };

        T grid_y[4][4] = {
            {dy_x1y1[0], dy_x2y1[0], dy_x3y1[0], dy_x4y1[0]},
            {dy_x1y2[0], dy_x2y2[0], dy_x3y2[0], dy_x4y2[0]},
            {dy_x1y3[0], dy_x2y3[0], dy_x3y3[0], dy_x4y3[0]},
            {dy_x1y4[0], dy_x2y4[0], dy_x3y4[0], dy_x4y4[0]}
        };

        T dx = T(0);
        T dy = T(0);

        for (int b = 0; b < 4; ++b) {
            const T wyb = wy[b];
            for (int a = 0; a < 4; ++a) {
                const T w = wyb * wx[a];
                dx += grid_x[b][a] * w;
                dy += grid_y[b][a] * w;
            }
        }

        T spline_distorted_x = T(x_normalized) + dx;
        T spline_distorted_y = T(y_normalized) + dy;

        residuals[0] = spline_distorted_x - T(opencv_distorted_x);
        residuals[1] = spline_distorted_y - T(opencv_distorted_y);
        return true;
    }

    static ceres::CostFunction* Create(
        const double opencv_distorted_x,
        const double opencv_distorted_y,
        const double x_normalized,
        const double y_normalized,
        const double spline_u,
        const double spline_v
    ) {
        // clang-format off
        using Cost = ceres::AutoDiffCostFunction<
            DistortionError,
            2,  // residuals
            1,1,1,1, 1,1,1,1,
            1,1,1,1, 1,1,1,1,
            1,1,1,1, 1,1,1,1,
            1,1,1,1, 1,1,1,1
        >;
        // clang-format on

        return new Cost(new DistortionError(
            opencv_distorted_x,
            opencv_distorted_y,
            x_normalized,
            y_normalized,
            spline_u,
            spline_v
        ));
    }

    double spline_u;
    double spline_v;
    double x_normalized;
    double y_normalized;
    double opencv_distorted_x;
    double opencv_distorted_y;
};

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

    uint32_t num_samples_x = num_knots_x * 5;
    uint32_t num_samples_y = num_knots_y * 5;

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
                y_range_start + (y_range_end - y_range_start) * y_proportion;

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

            const double eps = 1e-12;
            const double x_max = static_cast<double>(num_knots_x) - 2.0;
            const double y_max = static_cast<double>(num_knots_y) - 2.0;
            x_spline = std::min(std::max(x_spline, 1.0), x_max - eps);
            y_spline = std::min(std::max(y_spline, 1.0), y_max - eps);

            uint32_t ix = static_cast<uint32_t>(std::floor(x_spline));
            uint32_t iy = static_cast<uint32_t>(std::floor(y_spline));

            double u = x_spline - static_cast<double>(ix);
            double v = y_spline - static_cast<double>(iy);

            ceres::CostFunction* cost = DistortionError::Create(
                opencv_distorted_point.x(),
                opencv_distorted_point.y(),
                x_normalized,
                y_normalized,
                u,
                v
            );

            // clang-format off
            problem.AddResidualBlock(
                cost,
                nullptr,

                // dx control points (x_knots): (ix-1..ix+2) x (iy-1..iy+2)
                &x_knots[ix - 1][iy - 1], &x_knots[ix + 0][iy - 1],
                &x_knots[ix + 1][iy - 1], &x_knots[ix + 2][iy - 1],

                &x_knots[ix - 1][iy + 0], &x_knots[ix + 0][iy + 0],
                &x_knots[ix + 1][iy + 0], &x_knots[ix + 2][iy + 0],

                &x_knots[ix - 1][iy + 1], &x_knots[ix + 0][iy + 1],
                &x_knots[ix + 1][iy + 1], &x_knots[ix + 2][iy + 1],

                &x_knots[ix - 1][iy + 2], &x_knots[ix + 0][iy + 2],
                &x_knots[ix + 1][iy + 2], &x_knots[ix + 2][iy + 2],

                // dy control points (y_knots)
                &y_knots[ix - 1][iy - 1], &y_knots[ix + 0][iy - 1],
                &y_knots[ix + 1][iy - 1], &y_knots[ix + 2][iy - 1],

                &y_knots[ix - 1][iy + 0], &y_knots[ix + 0][iy + 0],
                &y_knots[ix + 1][iy + 0], &y_knots[ix + 2][iy + 0],

                &y_knots[ix - 1][iy + 1], &y_knots[ix + 0][iy + 1],
                &y_knots[ix + 1][iy + 1], &y_knots[ix + 2][iy + 1],

                &y_knots[ix - 1][iy + 2], &y_knots[ix + 0][iy + 2],
                &y_knots[ix + 1][iy + 2], &y_knots[ix + 2][iy + 2]
            );
            // clang-format on
        }
    }

    spdlog::info("Created problem");
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = 10'000;

    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    spdlog::info("Solve done");
    spdlog::info(summary.BriefReport());

    // Convert to NumPy arrays (N x M)
    py::array_t<double> x_array({num_knots_x, num_knots_y});
    py::array_t<double> y_array({num_knots_x, num_knots_y});

    auto x_buf = x_array.mutable_unchecked<2>();
    auto y_buf = y_array.mutable_unchecked<2>();

    for (size_t i = 0; i < num_knots_x; ++i) {
        for (size_t j = 0; j < num_knots_y; ++j) {
            x_buf(i, j) = x_knots[i][j];
            y_buf(i, j) = y_knots[i][j];
        }
    }

    py::dict out;
    out["x_knots"] = x_array;
    out["y_knots"] = y_array;

    // Optional but smart — return ranges so Python evaluates spline correctly
    out["x_range_start"] = x_range_start;
    out["x_range_end"] = x_range_end;
    out["y_range_start"] = y_range_start;
    out["y_range_end"] = y_range_end;

    return out;
}
}  // namespace camcal
