#include "./calibrate.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fmt/format.h>
#include <pybind11/stl.h>
#include <spdlog/spdlog.h>
#include "./cameramodels.hpp"

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

        Vec3<T> eigen_point_in_cam =
            transform_point(eigen_camera_from_world, eigen_point_in_world);

        Vec2<T> image_point;
        project(
            this->camera_model_name,
            intrinsics,
            eigen_point_in_cam,
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
    std::vector<std::vector<double>> cameras_from_world;
    std::vector<std::vector<double>> target_points;

    static OptimizationState from_calibrate_camera_input(
        std::vector<double>& intrinsics_initial_value,
        std::vector<Vec6<double>>& cameras_from_world,
        std::vector<Vec3<double>>& target_points
    ) {
        std::vector<std::vector<double>> camera_poses_out;
        for (auto& vec : cameras_from_world) {
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
        result["cameras_from_world"] = this->cameras_from_world;

        return result;
    }
};

py::dict calibrate_camera(
    std::string camera_model_name,
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6<double>>& cameras_from_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
) {
    ceres::Problem problem;

    OptimizationState state = OptimizationState::from_calibrate_camera_input(
        intrinsics_initial_value,
        cameras_from_world,
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

    for (auto& cam : state.cameras_from_world) {
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

        auto& camera_pose = state.cameras_from_world[camera_idx];

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