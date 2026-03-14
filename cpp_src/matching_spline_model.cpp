#include <ceres/ceres.h>
#include <pybind11/numpy.h>
#include <spdlog/spdlog.h>
#include "./utils.hpp"
#include "cameramodels.hpp"

namespace lensboy {

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
        // dx control points: y-major naming + ordering (y1 row, then y2, ...)
        const T* const dx_y1x1,
        const T* const dx_y1x2,
        const T* const dx_y1x3,
        const T* const dx_y1x4,
        const T* const dx_y2x1,
        const T* const dx_y2x2,
        const T* const dx_y2x3,
        const T* const dx_y2x4,
        const T* const dx_y3x1,
        const T* const dx_y3x2,
        const T* const dx_y3x3,
        const T* const dx_y3x4,
        const T* const dx_y4x1,
        const T* const dx_y4x2,
        const T* const dx_y4x3,
        const T* const dx_y4x4,

        // dy control points: y-major naming + ordering
        const T* const dy_y1x1,
        const T* const dy_y1x2,
        const T* const dy_y1x3,
        const T* const dy_y1x4,
        const T* const dy_y2x1,
        const T* const dy_y2x2,
        const T* const dy_y2x3,
        const T* const dy_y2x4,
        const T* const dy_y3x1,
        const T* const dy_y3x2,
        const T* const dy_y3x3,
        const T* const dy_y3x4,
        const T* const dy_y4x1,
        const T* const dy_y4x2,
        const T* const dy_y4x3,
        const T* const dy_y4x4,

        T* residuals
    ) const {
        T wx[4], wy[4];
        cubic_bspline_basis_uniform(T(spline_u), wx);
        cubic_bspline_basis_uniform(T(spline_v), wy);

        // local grids are indexed [y][x]
        T grid_x[4][4] = {
            {dx_y1x1[0], dx_y1x2[0], dx_y1x3[0], dx_y1x4[0]},
            {dx_y2x1[0], dx_y2x2[0], dx_y2x3[0], dx_y2x4[0]},
            {dx_y3x1[0], dx_y3x2[0], dx_y3x3[0], dx_y3x4[0]},
            {dx_y4x1[0], dx_y4x2[0], dx_y4x3[0], dx_y4x4[0]}
        };

        T grid_y[4][4] = {
            {dy_y1x1[0], dy_y1x2[0], dy_y1x3[0], dy_y1x4[0]},
            {dy_y2x1[0], dy_y2x2[0], dy_y2x3[0], dy_y2x4[0]},
            {dy_y3x1[0], dy_y3x2[0], dy_y3x3[0], dy_y3x4[0]},
            {dy_y4x1[0], dy_y4x2[0], dy_y4x3[0], dy_y4x4[0]}
        };

        T dx = T(0);
        T dy = T(0);

        // y-major accumulation: b = y, a = x
        for (int b = 0; b < 4; ++b) {
            const T wyb = wy[b];
            for (int a = 0; a < 4; ++a) {
                const T w = wyb * wx[a];
                dx += grid_x[b][a] * w;
                dy += grid_y[b][a] * w;
            }
        }

        const T spline_distorted_x = T(x_normalized) + dx;
        const T spline_distorted_y = T(y_normalized) + dy;

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
    PinholeSplinedConfig& model_config,
    double image_bound_x,
    double image_bound_y
) {
    const double fov_rad_x = model_config.fov_deg_x * M_PI / 180.0;
    const double fov_rad_y = model_config.fov_deg_y * M_PI / 180.0;

    // Sampling range in normalized (pinhole) space
    const double half_x_pinhole = std::tan(fov_rad_x / 2.0);
    const double half_y_pinhole = std::tan(fov_rad_y / 2.0);

    // Spline domain is in stereographic space
    const double half_x_stereo = stereo_half_range(fov_rad_x);
    const double half_y_stereo = stereo_half_range(fov_rad_y);

    const uint32_t num_samples_x = model_config.num_knots_x;
    const uint32_t num_samples_y = model_config.num_knots_y;

    // y-major storage: knots[y][x]
    auto x_knots = vector_mat<double>(
        model_config.num_knots_y,
        model_config.num_knots_x,
        0
    );
    auto y_knots = vector_mat<double>(
        model_config.num_knots_y,
        model_config.num_knots_x,
        0
    );

    ceres::Problem problem;

    // add all control points (y-major)
    for (size_t y = 0; y < model_config.num_knots_y; ++y) {
        for (size_t x = 0; x < model_config.num_knots_x; ++x) {
            problem.AddParameterBlock(&x_knots[y][x], 1);
            problem.AddParameterBlock(&y_knots[y][x], 1);
        }
    }

    const int Nx = static_cast<int>(model_config.num_knots_x);
    const int Ny = static_cast<int>(model_config.num_knots_y);

    const double inv_x_span = 1.0 / (2.0 * half_x_stereo);
    const double inv_y_span = 1.0 / (2.0 * half_y_stereo);

    // Track which cells have in-image distortion residuals
    std::vector<bool> cell_has_data(Nx * Ny, false);

    for (size_t y_sample_idx = 0; y_sample_idx < num_samples_y;
         ++y_sample_idx) {
        for (size_t x_sample_idx = 0; x_sample_idx < num_samples_x;
             ++x_sample_idx) {
            const double x_proportion =
                static_cast<double>(x_sample_idx) /
                (static_cast<double>(num_samples_x) - 1);
            const double y_proportion =
                static_cast<double>(y_sample_idx) /
                (static_cast<double>(num_samples_y) - 1);

            // Sample in normalized (pinhole) space
            const double x_normalized =
                -half_x_pinhole + 2.0 * half_x_pinhole * x_proportion;
            const double y_normalized =
                -half_y_pinhole + 2.0 * half_y_pinhole * y_proportion;

            // Convert to stereographic for spline lookup
            double x_stereo, y_stereo;
            normalized_to_stereographic(
                x_normalized,
                y_normalized,
                x_stereo,
                y_stereo
            );

            // Spline coords in knot index space (stereographic domain)
            double x_spline = 1.0 + (x_stereo + half_x_stereo) *
                                        (static_cast<double>(Nx) - 3.0) *
                                        inv_x_span;

            double y_spline = 1.0 + (y_stereo + half_y_stereo) *
                                        (static_cast<double>(Ny) - 3.0) *
                                        inv_y_span;

            // Clamp to the interior range where the 4x4 support patch
            // has unique knot indices. Outside this range, clamping would
            // cause duplicate parameter block pointers which Ceres forbids.
            const double eps = 1e-12;
            const double x_min = 1.0;
            const double y_min = 1.0;
            const double x_max = static_cast<double>(Nx) - 2.0;
            const double y_max = static_cast<double>(Ny) - 2.0;

            if (x_spline < x_min || x_spline > x_max - eps ||
                y_spline < y_min || y_spline > y_max - eps) {
                continue;
            }

            const uint32_t ix = static_cast<uint32_t>(std::floor(x_spline));
            const uint32_t iy = static_cast<uint32_t>(std::floor(y_spline));

            // Skip distortion matching outside the image bounds
            if (std::abs(x_normalized) > image_bound_x ||
                std::abs(y_normalized) > image_bound_y) {
                continue;
            }

            cell_has_data[iy * Nx + ix] = true;

            Vec2<double> normalized_point(x_normalized, y_normalized);

            Vec2<double> opencv_distorted_point;
            distort_opencv(
                opencv_distortion_params.data(),
                normalized_point,
                opencv_distorted_point
            );

            const double u = x_spline - static_cast<double>(ix);
            const double v = y_spline - static_cast<double>(iy);

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

                // dx control points (y-major): y first, then x
                &x_knots[iy - 1][ix - 1], &x_knots[iy - 1][ix + 0],
                &x_knots[iy - 1][ix + 1], &x_knots[iy - 1][ix + 2],

                &x_knots[iy + 0][ix - 1], &x_knots[iy + 0][ix + 0],
                &x_knots[iy + 0][ix + 1], &x_knots[iy + 0][ix + 2],

                &x_knots[iy + 1][ix - 1], &x_knots[iy + 1][ix + 0],
                &x_knots[iy + 1][ix + 1], &x_knots[iy + 1][ix + 2],

                &x_knots[iy + 2][ix - 1], &x_knots[iy + 2][ix + 0],
                &x_knots[iy + 2][ix + 1], &x_knots[iy + 2][ix + 2],

                // dy control points (y-major): y first, then x
                &y_knots[iy - 1][ix - 1], &y_knots[iy - 1][ix + 0],
                &y_knots[iy - 1][ix + 1], &y_knots[iy - 1][ix + 2],

                &y_knots[iy + 0][ix - 1], &y_knots[iy + 0][ix + 0],
                &y_knots[iy + 0][ix + 1], &y_knots[iy + 0][ix + 2],

                &y_knots[iy + 1][ix - 1], &y_knots[iy + 1][ix + 0],
                &y_knots[iy + 1][ix + 1], &y_knots[iy + 1][ix + 2],

                &y_knots[iy + 2][ix - 1], &y_knots[iy + 2][ix + 0],
                &y_knots[iy + 2][ix + 1], &y_knots[iy + 2][ix + 2]
            );
            // clang-format on
        }
    }

    // Smoothness priors for cells without in-image data
    const double sqrt_lambda = std::sqrt(model_config.smoothness_lambda);
    for (int cy = 0; cy < Ny; cy++) {
        for (int cx = 0; cx < Nx; cx++) {
            if (cell_has_data[cy * Nx + cx]) {
                continue;
            }

            // Horizontal: 4-knot stencil along rows
            if (cx - 1 >= 0 && cx + 2 < Nx) {
                for (int row = cy; row <= cy + 1 && row < Ny; row++) {
                    auto* cost = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost,
                        nullptr,
                        &x_knots[row][cx - 1],
                        &x_knots[row][cx],
                        &x_knots[row][cx + 1],
                        &x_knots[row][cx + 2]
                    );
                    auto* cost_y = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost_y,
                        nullptr,
                        &y_knots[row][cx - 1],
                        &y_knots[row][cx],
                        &y_knots[row][cx + 1],
                        &y_knots[row][cx + 2]
                    );
                }
            }

            // Vertical: 4-knot stencil along columns
            if (cy - 1 >= 0 && cy + 2 < Ny) {
                for (int col = cx; col <= cx + 1 && col < Nx; col++) {
                    auto* cost = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost,
                        nullptr,
                        &x_knots[cy - 1][col],
                        &x_knots[cy][col],
                        &x_knots[cy + 1][col],
                        &x_knots[cy + 2][col]
                    );
                    auto* cost_y = new ceres::
                        AutoDiffCostFunction<KnotSmoothness, 1, 1, 1, 1, 1>(
                            new KnotSmoothness{sqrt_lambda}
                        );
                    problem.AddResidualBlock(
                        cost_y,
                        nullptr,
                        &y_knots[cy - 1][col],
                        &y_knots[cy][col],
                        &y_knots[cy + 1][col],
                        &y_knots[cy + 2][col]
                    );
                }
            }
        }
    }

    SPDLOG_DEBUG("Created problem");
    ceres::Solver::Options options;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = 10'000;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    SPDLOG_DEBUG("Solve done");
    SPDLOG_DEBUG(summary.BriefReport());

    // Return NumPy arrays in y-major shape (Y, X)
    py::array_t<double> x_array(
        {model_config.num_knots_y, model_config.num_knots_x}
    );
    py::array_t<double> y_array(
        {model_config.num_knots_y, model_config.num_knots_x}
    );

    auto x_buf = x_array.mutable_unchecked<2>();
    auto y_buf = y_array.mutable_unchecked<2>();

    for (size_t y = 0; y < model_config.num_knots_y; ++y) {
        for (size_t x = 0; x < model_config.num_knots_x; ++x) {
            x_buf(y, x) = x_knots[y][x];
            y_buf(y, x) = y_knots[y][x];
        }
    }

    py::dict out;
    out["x_knots"] = x_array;
    out["y_knots"] = y_array;

    out["x_range_start"] = -half_x_stereo;
    out["x_range_end"] = half_x_stereo;
    out["y_range_start"] = -half_y_stereo;
    out["y_range_end"] = half_y_stereo;

    return out;
}

}  // namespace lensboy