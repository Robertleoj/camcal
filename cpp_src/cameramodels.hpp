#pragma once
#include <ceres/jet.h>
#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <cmath>
#include "./type_defs.hpp"

namespace py = pybind11;
namespace lensboy {

struct PinholeSplinedConfig {
    uint32_t image_width;
    uint32_t image_height;
    double fov_deg_x;
    double fov_deg_y;
    uint32_t num_knots_x;
    uint32_t num_knots_y;
};

struct PinholeSplinedIntrinsicsParameters {
    py::array_t<double, py::array::c_style | py::array::forcecast> k4;
    py::array_t<double, py::array::c_style | py::array::forcecast> dx_grid;
    py::array_t<double, py::array::c_style | py::array::forcecast> dy_grid;
};

template <typename T>
void project_pinhole(
    const T* const intrinsics,
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    Vec3<T> normalized_point = point_in_camera / point_in_camera[2];

    T fx = intrinsics[0];
    T fy = intrinsics[1];
    T cx = intrinsics[2];
    T cy = intrinsics[3];

    result << (normalized_point[0] * fx) + cx, (normalized_point[1] * fy) + cy;
}

template <typename T>
void distort_opencv(
    const T* const distortion_parameters,
    const Vec2<T>& normalized_point,
    Vec2<T>& result
) {
    // Distortion coeffs in OpenCV order:
    // (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4)
    const T k1 = distortion_parameters[0];
    const T k2 = distortion_parameters[1];
    const T p1 = distortion_parameters[2];
    const T p2 = distortion_parameters[3];
    const T k3 = distortion_parameters[4];
    const T k4 = distortion_parameters[5];
    const T k5 = distortion_parameters[6];
    const T k6 = distortion_parameters[7];
    const T s1 = distortion_parameters[8];
    const T s2 = distortion_parameters[9];
    const T s3 = distortion_parameters[10];
    const T s4 = distortion_parameters[11];

    const T x = normalized_point[0];
    const T y = normalized_point[1];

    const T r2 = x * x + y * y;
    const T r4 = r2 * r2;
    const T r6 = r4 * r2;

    // OpenCV rational radial model:
    // radial = (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6)
    const T radial_num = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
    const T radial_den = T(1) + k4 * r2 + k5 * r4 + k6 * r6;
    const T radial = radial_num / radial_den;

    const T x_radial = x * radial;
    const T y_radial = y * radial;

    // Tangential (Brown-Conrady, same as OpenCV)
    const T x_tan = T(2) * p1 * x * y + p2 * (r2 + T(2) * x * x);
    const T y_tan = p1 * (r2 + T(2) * y * y) + T(2) * p2 * x * y;

    // Thin prism distortion (OpenCV s1..s4)
    const T x_prism = s1 * r2 + s2 * r4;
    const T y_prism = s3 * r2 + s4 * r4;

    const T x_distorted = x_radial + x_tan + x_prism;
    const T y_distorted = y_radial + y_tan + y_prism;

    result << x_distorted, y_distorted;
}

template <typename T>
void project_opencv(
    const T* const
        intrinsics,  // fx, fy, cx, cy, k1, k2, p1, p2, k3..k6, s1..s4
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    Vec2<T> normalized(
        point_in_camera[0] / point_in_camera[2],
        point_in_camera[1] / point_in_camera[2]
    );
    Vec2<T> distorted_normalized;

    distort_opencv(intrinsics + 4, normalized, distorted_normalized);

    const T fx = intrinsics[0];
    const T fy = intrinsics[1];
    const T cx = intrinsics[2];
    const T cy = intrinsics[3];

    result << fx * distorted_normalized[0] + cx,
        fy * distorted_normalized[1] + cy;
}

inline int clamp_int(
    int v,
    int lo,
    int hi
) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename T>
static inline void cubic_bspline_basis_uniform(
    const T& u,
    T w[4]
) {
    // u in [0,1)
    const T u2 = u * u;
    const T u3 = u2 * u;
    // weights for control indices offsets [-1,0,1,2] relative to cell index
    w[0] = (T(1) - T(3) * u + T(3) * u2 - u3) / T(6);  // (1-u)^3 / 6
    w[1] = (T(4) - T(6) * u2 + T(3) * u3) / T(6);      // (3u^3 - 6u^2 + 4)/6
    w[2] = (T(1) + T(3) * u + T(3) * u2 - T(3) * u3) /
           T(6);       // (-3u^3 + 3u^2 + 3u + 1)/6
    w[3] = u3 / T(6);  // u^3 / 6
}

template <typename T>
static inline int clamp_int(
    int v,
    int lo,
    int hi
) {
    return v < lo ? lo : (v > hi ? hi : v);
}

template <typename T>
static inline T clamp_T(
    const T& v,
    const T& lo,
    const T& hi
) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Overload for double: returns the value directly
inline double scalar_value(
    const double& x
) {
    return x;
}

// Jet specialization
template <typename T, int N>
inline double scalar_value(
    const ceres::Jet<T, N>& x
) {
    return x.a;
}

template <typename T>
static inline T eval_bspline2d_uniform_cubic_clamped(
    const T* grid,  // row-major, size Ny*Nx
    int Nx,
    int Ny,
    const T& x_spline,  // spline coordinate (control points at integer coords)
    const T& y_spline
) {
    // We assume "clamped by edge replication" boundary behavior:
    // indices outside [0..Nx-1]/[0..Ny-1] are clamped.
    //
    // For cubic, valid interior is [1, Nx-2) in spline coords for non-clamped;
    // but with clamping we can evaluate anywhere and it'll replicate edges.
    T gx = x_spline;
    T gy = y_spline;

    // Keep floor() stable near the upper edge if gx is exactly integer at
    // boundary. (Not strictly necessary but avoids ix == Nx-1 leading to
    // neighborhood beyond.)
    const T eps = T(1e-12);
    gx = clamp_T(gx, T(0), T(Nx - 1) - eps);
    gy = clamp_T(gy, T(0), T(Ny - 1) - eps);

    const int ix = static_cast<int>(std::floor(scalar_value(gx)));
    const int iy = static_cast<int>(std::floor(scalar_value(gy)));

    const T u = gx - T(ix);
    const T v = gy - T(iy);

    T wx[4], wy[4];
    cubic_bspline_basis_uniform(u, wx);
    cubic_bspline_basis_uniform(v, wy);

    // neighborhood indices in each dimension: (i-1 .. i+2)
    const int xs[4] = {
        clamp_int(ix - 1, 0, Nx - 1),
        clamp_int(ix + 0, 0, Nx - 1),
        clamp_int(ix + 1, 0, Nx - 1),
        clamp_int(ix + 2, 0, Nx - 1)
    };
    const int ys[4] = {
        clamp_int(iy - 1, 0, Ny - 1),
        clamp_int(iy + 0, 0, Ny - 1),
        clamp_int(iy + 1, 0, Ny - 1),
        clamp_int(iy + 2, 0, Ny - 1)
    };

    // tensor-product sum: wy^T * patch * wx
    T acc = T(0);
    for (int b = 0; b < 4; ++b) {
        const int yy = ys[b];
        const T wyb = wy[b];
        const int row0 = yy * Nx;
        for (int a = 0; a < 4; ++a) {
            const int xx = xs[a];
            acc += grid[row0 + xx] * (wyb * wx[a]);
        }
    }
    return acc;
}

template <typename T>
void project_pinhole_splined(
    PinholeSplinedConfig* config,
    const T* const k4,  // fx, fy, cx, cy
    const T* const dx_grid,
    const T* const dy_grid,
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    // --- pinhole normalized coords
    const T x_normalized = point_in_camera[0] / point_in_camera[2];
    const T y_normalized = point_in_camera[1] / point_in_camera[2];

    const int Nx = static_cast<int>(config->num_knots_x);
    const int Ny = static_cast<int>(config->num_knots_y);

    const double fov_rad_x = config->fov_deg_x * M_PI / 180.0;
    const double fov_rad_y = config->fov_deg_y * M_PI / 180.0;

    // We define the spline domain over the normalized pinhole plane such that
    // x_normalized in [-tan(fov_x/2), +tan(fov_x/2)] maps across the interior
    // of the spline grid (with clamping outside).
    const double half_x_range = std::tan(fov_rad_x / 2.0);
    const double half_y_range = std::tan(fov_rad_y / 2.0);

    const T x_range_start = T(-half_x_range);
    const T x_range_end = T(+half_x_range);
    const T y_range_start = T(-half_y_range);
    const T y_range_end = T(+half_y_range);

    const T fx = k4[0];
    const T fy = k4[1];
    const T cx = k4[2];
    const T cy = k4[3];

    const T inv_x_span = T(1) / (x_range_end - x_range_start);
    const T inv_y_span = T(1) / (y_range_end - y_range_start);

    const T x_spline =
        T(1) + (x_normalized - x_range_start) * T(Nx - 3) * inv_x_span;
    const T y_spline =
        T(1) + (y_normalized - y_range_start) * T(Ny - 3) * inv_y_span;

    // --- evaluate spline surfaces
    const T dx = eval_bspline2d_uniform_cubic_clamped(
        dx_grid,
        Nx,
        Ny,
        x_spline,
        y_spline
    );
    const T dy = eval_bspline2d_uniform_cubic_clamped(
        dy_grid,
        Nx,
        Ny,
        x_spline,
        y_spline
    );

    // --- apply distortion in normalized plane
    const T x_distorted = x_normalized + dx;
    const T y_distorted = y_normalized + dy;

    // --- intrinsics to pixels
    result[0] = fx * x_distorted + cx;
    result[1] = fy * y_distorted + cy;
}

}  // namespace lensboy
