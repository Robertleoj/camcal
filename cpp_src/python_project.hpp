// pybind_project_pinhole_splined.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "cameramodels.hpp"

#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

namespace camcal {
static void require(
    bool cond,
    const char* msg
) {
    if (!cond) {
        throw std::invalid_argument(msg);
    }
}

static py::array_t<double> project_pinhole_splined_pywrapper(
    camcal::PinholeSplinedConfig& model_config,
    py::array_t<double, py::array::c_style | py::array::forcecast> k4,
    py::array_t<double, py::array::c_style | py::array::forcecast> dx_grid,
    py::array_t<double, py::array::c_style | py::array::forcecast> dy_grid,
    py::array_t<double, py::array::c_style | py::array::forcecast>
        points_in_camera
) {
    // --- k4: shape (4,) ---
    auto k4b = k4.request();
    require(k4b.ndim == 1, "k4 must be a 1D numpy array");
    require(k4b.shape[0] == 4, "k4 must have shape (4,)");

    // --- grids: shape (num_knots_y, num_knots_x), row-major (C contiguous) ---
    auto dxb = dx_grid.request();
    auto dyb = dy_grid.request();
    require(dxb.ndim == 2, "dx_grid must be a 2D numpy array");
    require(dyb.ndim == 2, "dy_grid must be a 2D numpy array");
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

    const auto* k4p = static_cast<const double*>(k4b.ptr);
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