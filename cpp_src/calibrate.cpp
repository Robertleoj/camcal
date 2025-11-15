#include "calibrate.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fmt/format.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>

namespace camcal {

constexpr size_t pinhole_num_params = 4;
constexpr size_t opencv_num_params = 4 + 12;

template <typename T>
Eigen::Matrix<T, 3, 1> transform_point(
    const Vec6<T>& pose,        // [rx ry rz tx ty tz]
    const Vec3<T>& point_world  // [x y z]
) {
    Vec3<T> rotated;
    ceres::AngleAxisRotatePoint(
        pose.data(),
        point_world.data(),
        rotated.data()
    );

    // Add translation
    rotated[0] += pose[3];
    rotated[1] += pose[4];
    rotated[2] += pose[5];

    return rotated;
}

template <typename T>
void project_pinhole(
    const T* const intrinsics,
    const Vec6<T>& camera_from_world,
    const Vec3<T>& point_in_world,
    Vec2<T>& result
) {
    Vec3<T> point_in_camera =
        transform_point(camera_from_world, point_in_world);

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
    const Vec6<T>& camera_from_world,
    const Vec3<T>& point_in_world,
    Vec2<T>& result
) {
    // Transform world -> camera
    Vec3<T> point_in_camera =
        transform_point(camera_from_world, point_in_world);

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
    const Vec6<T>& camera_from_world,
    const Vec3<T>& point_in_world,
    Vec2<T>& result
) {
    if (camera_model_name == "pinhole") {
        project_pinhole<T>(
            intrinsics,
            camera_from_world,
            point_in_world,
            result
        );
        return;
    }

    if (camera_model_name == "opencv") {
        project_opencv<T>(
            intrinsics,
            camera_from_world,
            point_in_world,
            result
        );
        return;
    }

    throw std::runtime_error(
        fmt::format("Unknown camera model {}", camera_model_name)
    );
}

struct ReprojectionError {
    ReprojectionError(
        const std::string& camera_model_name,
        const double observed_x,
        const double observed_y
    )
        : observed_x(observed_x),
          observed_y(observed_y),
          camera_model_name(camera_model_name) {}

    template <typename T>
    bool operator()(
        const T* const intrinsics,
        const T* const camera_from_world,
        const T* const point_in_world,
        T* residuals
    ) const {
        Vec6<T> eigen_camera_from_world(camera_from_world);
        Vec3<T> eigen_point_in_world(point_in_world);

        Vec2<T> image_point;

        project(
            this->camera_model_name,
            intrinsics,
            eigen_camera_from_world,
            eigen_point_in_world,
            image_point
        );

        residuals[0] = image_point[0] - observed_x;
        residuals[1] = image_point[1] - observed_y;

        return true;
    }

    static ceres::CostFunction* create(
        const std::string& camera_model_name,
        const double observed_x,
        const double observed_y
    ) {
        if (camera_model_name == "pinhole") {
            // 4 parameters in intrinsics
            return new ceres::AutoDiffCostFunction<
                ReprojectionError,
                2,
                pinhole_num_params,
                6,
                3>(
                new ReprojectionError(camera_model_name, observed_x, observed_y)
            );
        }

        if (camera_model_name == "opencv") {
            return new ceres::AutoDiffCostFunction<
                ReprojectionError,
                2,
                opencv_num_params,
                6,
                3>(
                new ReprojectionError(camera_model_name, observed_x, observed_y)
            );
        }

        throw std::runtime_error(
            fmt::format("camera model {} does not exist", camera_model_name)
        );
    }

    std::string camera_model_name;
    double observed_x;
    double observed_y;
};

struct OptimizationState {
    std::vector<double> intrinsics;
    std::vector<std::vector<double>> cameras_to_world;
    std::vector<std::vector<double>> target_points;

    static OptimizationState from_calibrate_camera_input(
        std::vector<double>& intrinsics_initial_value,
        std::vector<Vec6<double>>& camera_poses_world,
        std::vector<Vec3<double>>& target_points
    ) {
        std::vector<std::vector<double>> camera_poses_out;
        for (auto& vec : camera_poses_world) {
            camera_poses_out.push_back(
                std::vector<double>(vec.data(), vec.data() + vec.size())
            );
        }

        std::vector<std::vector<double>> target_points_out;
        for (auto& vec : target_points) {
            target_points_out.push_back(
                std::vector<double>(vec.data(), vec.data() + vec.size())
            );
        }

        return OptimizationState{
            intrinsics_initial_value,
            std::move(camera_poses_out),
            std::move(target_points_out),
        };
    }

    py::dict make_dict() {
        py::dict result;

        result["intrinsics"] = this->intrinsics;
        result["cameras_to_world"] = this->cameras_to_world;

        return result;
    }
};

py::dict calibrate_camera(
    std::string camera_model_name,
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& camera_poses_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
) {
    ceres::Problem problem;

    OptimizationState state = OptimizationState::from_calibrate_camera_input(
        intrinsics_initial_value,
        camera_poses_world,
        target_points
    );

    problem.AddParameterBlock(state.intrinsics.data(), state.intrinsics.size());
    std::vector<int> fixed_intrinsics_param_indices;
    for (size_t param_idx = 0; param_idx < state.intrinsics.size();
         param_idx++) {
        bool should_optimize = intrinsics_param_optimize_mask[param_idx];
        if (!should_optimize) {
            fixed_intrinsics_param_indices.push_back(param_idx);
        }
    }

    auto* manifold = new ceres::SubsetManifold(
        state.intrinsics.size(),
        fixed_intrinsics_param_indices
    );
    problem.SetManifold(state.intrinsics.data(), manifold);

    for (auto& cam : state.cameras_to_world) {
        problem.AddParameterBlock(cam.data(), cam.size());
    }

    for (auto& pt : state.target_points) {
        problem.AddParameterBlock(pt.data(), pt.size());

        // don't optimize target points
        problem.SetParameterBlockConstant(pt.data());
    }

    spdlog::info("Added parameter blocks");

    size_t num_cameras = detections.size();

    for (size_t camera_idx = 0; camera_idx < num_cameras; camera_idx++) {
        auto& target_point_indices = std::get<0>(detections[camera_idx]);
        auto& observations = std::get<1>(detections[camera_idx]);

        auto& camera_pose = state.cameras_to_world[camera_idx];

        size_t num_observations = observations.size();

        for (size_t observation_idx = 0; observation_idx < num_observations;
             observation_idx++) {
            auto& observation = observations[observation_idx];
            auto& target =
                state.target_points[target_point_indices[observation_idx]];

            problem.AddResidualBlock(
                ReprojectionError::create(
                    camera_model_name,
                    observation(0, 0),
                    observation(1, 0)
                ),
                nullptr,
                state.intrinsics.data(),
                camera_pose.data(),
                target.data()
            );
        }
    }

    spdlog::info("Added residual blocks");

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.use_nonmonotonic_steps = true;
    options.max_num_iterations = 10'000;

    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return state.make_dict();
}

}  // namespace camcal