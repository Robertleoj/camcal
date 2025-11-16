#pragma once
#include <fmt/format.h>
#include "./type_defs.hpp"

namespace camcal {

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
void project_opencv(
    const T* const intrinsics,  // fx, fy, cx, cy, k1..k6, s1..s4
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    // Normalized coordinates
    T x = point_in_camera[0] / point_in_camera[2];
    T y = point_in_camera[1] / point_in_camera[2];

    // Intrinsics
    const T fx = intrinsics[0];
    const T fy = intrinsics[1];
    const T cx = intrinsics[2];
    const T cy = intrinsics[3];

    // Distortion coeffs in OpenCV order:
    // (k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4)
    const T k1 = intrinsics[4];
    const T k2 = intrinsics[5];
    const T p1 = intrinsics[6];
    const T p2 = intrinsics[7];
    const T k3 = intrinsics[8];
    const T k4 = intrinsics[9];
    const T k5 = intrinsics[10];
    const T k6 = intrinsics[11];
    const T s1 = intrinsics[12];
    const T s2 = intrinsics[13];
    const T s3 = intrinsics[14];
    const T s4 = intrinsics[15];

    // r^2 etc.
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
    // x_prism = s1 * r^2 + s2 * r^4
    // y_prism = s3 * r^2 + s4 * r^4
    const T x_prism = s1 * r2 + s2 * r4;
    const T y_prism = s3 * r2 + s4 * r4;

    const T x_distorted = x_radial + x_tan + x_prism;
    const T y_distorted = y_radial + y_tan + y_prism;

    // Back to pixels
    result << fx * x_distorted + cx, fy * y_distorted + cy;
}

template <typename T>
void project(
    const std::string& camera_model_name,
    const T* const intrinsics,
    const Vec3<T>& point_in_camera,
    Vec2<T>& result
) {
    if (camera_model_name == "pinhole") {
        project_pinhole<T>(intrinsics, point_in_camera, result);
        return;
    }

    if (camera_model_name == "opencv") {
        project_opencv<T>(intrinsics, point_in_camera, result);
        return;
    }

    throw std::runtime_error(
        fmt::format("Unknown camera model {}", camera_model_name)
    );
}

}  // namespace camcal