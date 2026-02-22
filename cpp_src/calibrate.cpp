#include "./calibrate.hpp"
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
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
        ModelConfig* model_config,
        const double observed_x,
        const double observed_y
    )
        : observed_x(observed_x),
          observed_y(observed_y),
          model_config(model_config),
          camera_model_name(camera_model_name) {}

    template <typename T>
    bool operator()(
        T const* const* params,
        T* residuals
    ) const {
        const T* const intrinsics = params[0];
        const T* const camera_from_world = params[1];
        const T* const point_in_world = params[2];

        Vec6<T> eigen_camera_from_world(camera_from_world);
        Vec3<T> eigen_point_in_world(point_in_world);

        Vec3<T> eigen_point_in_cam =
            transform_point(eigen_camera_from_world, eigen_point_in_world);

        Vec2<T> image_point;
        project(
            this->camera_model_name,
            this->model_config,
            intrinsics,
            eigen_point_in_cam,
            image_point
        );

        residuals[0] = image_point[0] - T(observed_x);
        residuals[1] = image_point[1] - T(observed_y);
        return true;
    }

    static ceres::CostFunction* create(
        const std::string& camera_model_name,
        ModelConfig* model_config,
        const double observed_x,
        const double observed_y
    ) {
        // runtime intrinsics size
        int intrinsics_size = 0;

        if (camera_model_name == "pinhole") {
            intrinsics_size = pinhole_num_params;  // 4
        } else if (camera_model_name == "opencv") {
            intrinsics_size = opencv_num_params;
        } else if (camera_model_name == "pinhole_splined") {
            const int nx =
                static_cast<int>(model_config->int_params.at("num_knots_x"));
            const int ny =
                static_cast<int>(model_config->int_params.at("num_knots_y"));

            // Your packing: [fx,fy,cx,cy] + dx_grid(nx*ny) + dy_grid(nx*ny)
            intrinsics_size = 4 + 2 * (nx * ny);
        } else {
            throw std::runtime_error(
                fmt::format("camera model {} does not exist", camera_model_name)
            );
        }

        // The '4' here is the number of residuals? No. It's the max # of
        // parameter blocks for internal bookkeeping; you can set it to 3 or 4.
        // We'll use 3 blocks.
        using CostT = ceres::DynamicAutoDiffCostFunction<ReprojectionError, 4>;

        auto* cost = new CostT(new ReprojectionError(
            camera_model_name,
            model_config,
            observed_x,
            observed_y
        ));

        cost->AddParameterBlock(intrinsics_size);  // intrinsics (runtime sized)
        cost->AddParameterBlock(6);                // camera_from_world
        cost->AddParameterBlock(3);                // point_in_world
        cost->SetNumResiduals(2);

        return cost;
    }

    double observed_x;
    double observed_y;
    ModelConfig* model_config;
    std::string camera_model_name;
};

struct SplineZeroPrior {
    SplineZeroPrior(
        int nx,
        int ny,
        double lambda
    )
        : nx(nx),
          ny(ny),
          sqrt_lambda(std::sqrt(lambda)) {}

    template <typename T>
    bool operator()(
        T const* const* params,
        T* residuals
    ) const {
        const T* const intrinsics = params[0];

        const int n = nx * ny;
        const int offset = 4;  // [fx, fy, cx, cy] then dx(n) then dy(n)

        // dx grid
        for (int i = 0; i < n; ++i) {
            residuals[i] = T(sqrt_lambda) * intrinsics[offset + i];
        }
        // dy grid
        for (int i = 0; i < n; ++i) {
            residuals[n + i] = T(sqrt_lambda) * intrinsics[offset + n + i];
        }

        return true;
    }

    static ceres::CostFunction* create(
        int nx,
        int ny,
        double lambda,
        int intrinsics_size
    ) {
        const int n = nx * ny;
        const int num_residuals = 2 * n;

        using CostT = ceres::DynamicAutoDiffCostFunction<SplineZeroPrior, 4>;
        auto* cost = new CostT(new SplineZeroPrior(nx, ny, lambda));
        cost->AddParameterBlock(intrinsics_size);
        cost->SetNumResiduals(num_residuals);
        return cost;
    }

    int nx;
    int ny;
    double sqrt_lambda;
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
    ModelConfig& model_config,
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
                    &model_config,
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

    if (camera_model_name == "pinhole_splined") {
        const int nx =
            static_cast<int>(model_config.int_params.at("num_knots_x"));
        const int ny =
            static_cast<int>(model_config.int_params.at("num_knots_y"));

        // Regularization strength (tweak as needed)
        // If you want, wire this into
        // model_config.double_params["spline_zero_prior_lambda"]
        const double lambda = 1e-5;

        const int intrinsics_size = static_cast<int>(state.intrinsics.size());

        problem.AddResidualBlock(
            SplineZeroPrior::create(nx, ny, lambda, intrinsics_size),
            nullptr,
            state.intrinsics.data()
        );

        spdlog::info("Added spline zero prior: lambda={}", lambda);
    }

    spdlog::info("Added residual blocks");

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.use_nonmonotonic_steps = false;
    options.max_num_iterations = 10'000;

    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return state.make_dict();
}

}  // namespace camcal