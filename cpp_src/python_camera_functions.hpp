#include <spdlog/spdlog.h>
#include "./pybind_utils.hpp"
#include "cameramodels.hpp"

#include <cstdint>

namespace lensboy {

static py::array_t<double> project_pinhole_splined_pywrapper(
    lensboy::PinholeSplinedConfig& model_config,
    lensboy::PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        points_in_camera
) {
    // --- grids: must match model_config dimensions ---
    auto dxb = intrinsics.dx_grid.request();
    auto dyb = intrinsics.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dxb.shape[1] == model_config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dyb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dyb.shape[1] == model_config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    // --- points: shape (N, 3) ---
    auto pb = points_in_camera.request();
    require(pb.ndim == 2, "points_in_camera must be a 2D numpy array");
    require(pb.shape[1] == 3, "points_in_camera must have shape (N, 3)");

    const ssize_t N = pb.shape[0];

    const double* k4p = static_cast<const double*>(intrinsics.k4.request().ptr);
    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);
    const double* P = static_cast<const double*>(pb.ptr);

    // Output: (N, 2), C contiguous
    py::array_t<double> out({N, (ssize_t)2});
    auto ob = out.request();
    auto* O = static_cast<double*>(ob.ptr);

    py::gil_scoped_release release;

    for (ssize_t i = 0; i < N; ++i) {
        Vec3<double> p;
        p[0] = P[i * 3 + 0];
        p[1] = P[i * 3 + 1];
        p[2] = P[i * 3 + 2];

        Vec2<double> r;
        project_pinhole_splined<double>(
            &model_config,
            k4p,  // fx, fy, cx, cy
            dxp,  // row-major contiguous (C-order)
            dyp,  // row-major contiguous (C-order)
            p,
            r
        );

        O[i * 2 + 0] = r[0];
        O[i * 2 + 1] = r[1];
    }

    return out;
}

static py::tuple make_undistortion_maps_pinhole_splined(
    lensboy::PinholeSplinedConfig& model_config,
    lensboy::PinholeSplinedIntrinsicsParameters& intrinsics,
    py::array_t<double, py::array::c_style | py::array::forcecast> k4,
    std::pair<int, int> image_size_wh
) {
    // --- grids: must match model_config dimensions ---
    auto dxb = intrinsics.dx_grid.request();
    auto dyb = intrinsics.dy_grid.request();
    require(
        (uint32_t)dxb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dxb.shape[1] == model_config.num_knots_x,
        "dx_grid must have shape (num_knots_y, num_knots_x)"
    );
    require(
        (uint32_t)dyb.shape[0] == model_config.num_knots_y &&
            (uint32_t)dyb.shape[1] == model_config.num_knots_x,
        "dy_grid must have shape (num_knots_y, num_knots_x)"
    );

    const int Nx = (int)model_config.num_knots_x;
    const int Ny = (int)model_config.num_knots_y;

    auto k4b_in = intrinsics.k4.request();
    require(
        k4b_in.ndim == 1 && k4b_in.shape[0] == 4,
        "intrinsics.k4 must have shape (4,)"
    );
    const double* k4_in = static_cast<const double*>(k4b_in.ptr);

    const double fx_in = k4_in[0];
    const double fy_in = k4_in[1];
    const double cx_in = k4_in[2];
    const double cy_in = k4_in[3];
    require(
        fx_in != 0.0 && fy_in != 0.0,
        "intrinsics.k4 fx/fy must be non-zero"
    );

    // Undistorted/output camera k4 (controls output view)
    double k4_out_storage[4];

    auto k4b = k4.request();
    require(k4b.ndim == 1 && k4b.shape[0] == 4, "new_k4 must have shape (4,)");
    const double* p = static_cast<const double*>(k4b.ptr);
    for (int i = 0; i < 4; ++i) {
        k4_out_storage[i] = p[i];
    }
    const double* k4_out = k4_out_storage;

    const double fx_out = k4_out[0];
    const double fy_out = k4_out[1];
    const double cx_out = k4_out[2];
    const double cy_out = k4_out[3];
    require(fx_out != 0.0 && fy_out != 0.0, "new_k4 fx/fy must be non-zero");

    const double* dxp = static_cast<const double*>(dxb.ptr);
    const double* dyp = static_cast<const double*>(dyb.ptr);

    int W = image_size_wh.first;
    int H = image_size_wh.second;

    require(W > 0 && H > 0, "Image width/height must be > 0");
    py::array_t<float> map_x({(ssize_t)H, (ssize_t)W});
    py::array_t<float> map_y({(ssize_t)H, (ssize_t)W});
    auto mx = map_x.request();
    auto my = map_y.request();
    float* MX = static_cast<float*>(mx.ptr);
    float* MY = static_cast<float*>(my.ptr);

    // Precompute spline domain mapping terms
    const double fov_rad_x = model_config.fov_deg_x * M_PI / 180.0;
    const double fov_rad_y = model_config.fov_deg_y * M_PI / 180.0;
    const double half_x_range = std::tan(fov_rad_x / 2.0);
    const double half_y_range = std::tan(fov_rad_y / 2.0);

    const double x_range_start = -half_x_range;
    const double x_range_end = +half_x_range;
    const double y_range_start = -half_y_range;
    const double y_range_end = +half_y_range;

    const double inv_x_span = 1.0 / (x_range_end - x_range_start);
    const double inv_y_span = 1.0 / (y_range_end - y_range_start);

    for (int y = 0; y < H; ++y) {
        const double y_norm = (double(y) - cy_out) / fy_out;
        for (int x = 0; x < W; ++x) {
            const double x_norm = (double(x) - cx_out) / fx_out;

            const double x_spline =
                1.0 + (x_norm - x_range_start) * double(Nx - 3) * inv_x_span;
            const double y_spline =
                1.0 + (y_norm - y_range_start) * double(Ny - 3) * inv_y_span;

            const double dx = eval_bspline2d_uniform_cubic_clamped(
                dxp,
                Nx,
                Ny,
                x_spline,
                y_spline
            );
            const double dy = eval_bspline2d_uniform_cubic_clamped(
                dyp,
                Nx,
                Ny,
                x_spline,
                y_spline
            );

            const double x_dist = x_norm + dx;
            const double y_dist = y_norm + dy;

            const int idx = y * W + x;
            MX[idx] = (float)(fx_in * x_dist + cx_in);
            MY[idx] = (float)(fy_in * y_dist + cy_in);
        }
    }

    return py::make_tuple(map_x, map_y);
}

}  // namespace lensboy