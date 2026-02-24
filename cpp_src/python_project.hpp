// pybind_project_pinhole_splined.cpp
#include "./pybind_utils.hpp"
#include "cameramodels.hpp"

#include <cstdint>
#include <stdexcept>

namespace camcal {

static py::array_t<double> project_pinhole_splined_pywrapper(
    camcal::PinholeSplinedConfig& model_config,
    camcal::PinholeSplinedIntrinsicsParameters& intrinsics,
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

    const auto* k4p = static_cast<const double*>(intrinsics.k4.request().ptr);
    const auto* dxp = static_cast<const double*>(dxb.ptr);
    const auto* dyp = static_cast<const double*>(dyb.ptr);
    const auto* P = static_cast<const double*>(pb.ptr);

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
            model_config.fov_deg_x,
            model_config.fov_deg_y,
            model_config.num_knots_x,
            model_config.num_knots_y,
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

}  // namespace camcal