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
        Vec2<T> image_point;

        if (camera_model_name == "pinhole_splined") {
            const T* const k4 = params[0];  // [fx,fy,cx,cy]
            const T* const x_knots = params[1];
            const T* const y_knots = params[2];
            const T* const camera_from_world = params[3];
            const T* const point_in_world = params[4];

            Vec6<T> eigen_camera_from_world(camera_from_world);
            Vec3<T> eigen_point_in_world(point_in_world);

            Vec3<T> eigen_point_in_cam =
                transform_point(eigen_camera_from_world, eigen_point_in_world);

            project_pinhole_splined(
                this->model_config,
                k4,
                x_knots,
                y_knots,
                eigen_point_in_cam,
                image_point
            );
        } else {
            const T* const intrinsics = params[0];
            const T* const camera_from_world = params[1];
            const T* const point_in_world = params[2];

            Vec6<T> eigen_camera_from_world(camera_from_world);
            Vec3<T> eigen_point_in_world(point_in_world);

            Vec3<T> eigen_point_in_cam =
                transform_point(eigen_camera_from_world, eigen_point_in_world);

            project(
                this->camera_model_name,
                this->model_config,
                intrinsics,
                eigen_point_in_cam,
                image_point
            );
        }

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
        using CostT = ceres::DynamicAutoDiffCostFunction<ReprojectionError, 5>;
        auto* cost = new CostT(new ReprojectionError(
            camera_model_name,
            model_config,
            observed_x,
            observed_y
        ));

        if (camera_model_name == "pinhole") {
            cost->AddParameterBlock((int)pinhole_num_params);  // intrinsics
            cost->AddParameterBlock(6);
            cost->AddParameterBlock(3);
        } else if (camera_model_name == "opencv") {
            cost->AddParameterBlock((int)opencv_num_params);  // intrinsics
            cost->AddParameterBlock(6);
            cost->AddParameterBlock(3);
        } else if (camera_model_name == "pinhole_splined") {
            const int nx = (int)model_config->int_params.at("num_knots_x");
            const int ny = (int)model_config->int_params.at("num_knots_y");
            const int n = nx * ny;

            cost->AddParameterBlock(4);  // [fx,fy,cx,cy]
            cost->AddParameterBlock(n);  // x_knots
            cost->AddParameterBlock(n);  // y_knots
            cost->AddParameterBlock(6);  // camera_from_world
            cost->AddParameterBlock(3);  // point_in_world
        } else {
            throw std::runtime_error(
                fmt::format("camera model {} does not exist", camera_model_name)
            );
        }

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
        const T* const x = params[0];
        const T* const y = params[1];
        const int n = nx * ny;

        for (int i = 0; i < n; ++i) {
            residuals[i] = T(sqrt_lambda) * x[i];
        }
        for (int i = 0; i < n; ++i) {
            residuals[n + i] = T(sqrt_lambda) * y[i];
        }
        return true;
    }

    static ceres::CostFunction* create(
        int nx,
        int ny,
        double lambda
    ) {
        const int n = nx * ny;
        using CostT = ceres::DynamicAutoDiffCostFunction<SplineZeroPrior, 2>;
        auto* cost = new CostT(new SplineZeroPrior(nx, ny, lambda));
        cost->AddParameterBlock(n);  // x_knots
        cost->AddParameterBlock(n);  // y_knots
        cost->SetNumResiduals(2 * n);
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

py::dict fine_tune_pinhole_splined(
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

    double* intr = state.intrinsics.data();
    const int intr_size = (int)state.intrinsics.size();

    double* k4 = intr;  // [0..3]

    double* x_knots = nullptr;
    double* y_knots = nullptr;
    int n_knots = 0;

    if (camera_model_name == "pinhole_splined") {
        const int nx = (int)model_config.int_params.at("num_knots_x");
        const int ny = (int)model_config.int_params.at("num_knots_y");
        n_knots = nx * ny;

        x_knots = intr + 4;
        y_knots = intr + 4 + n_knots;

        // 3 separate parameter blocks
        problem.AddParameterBlock(k4, 4);
        problem.AddParameterBlock(x_knots, n_knots);
        problem.AddParameterBlock(y_knots, n_knots);

        // SubsetManifold for each block using the *same* optimize mask you
        // already have
        std::vector<int> fixed_k4, fixed_x, fixed_y;
        fixed_k4.reserve(4);
        fixed_x.reserve(n_knots);
        fixed_y.reserve(n_knots);

        // [fx,fy,cx,cy]
        for (int i = 0; i < 4; ++i) {
            if (!intrinsics_param_optimize_mask[i]) {
                fixed_k4.push_back(i);
            }
        }
        // x_knots live at [4 .. 4+n_knots)
        for (int i = 0; i < n_knots; ++i) {
            if (!intrinsics_param_optimize_mask[4 + i]) {
                fixed_x.push_back(i);
            }
        }
        // y_knots live at [4+n_knots .. 4+2*n_knots)
        for (int i = 0; i < n_knots; ++i) {
            if (!intrinsics_param_optimize_mask[4 + n_knots + i]) {
                fixed_y.push_back(i);
            }
        }

        problem.SetManifold(k4, new ceres::SubsetManifold(4, fixed_k4));
        problem.SetManifold(
            x_knots,
            new ceres::SubsetManifold(n_knots, fixed_x)
        );
        problem.SetManifold(
            y_knots,
            new ceres::SubsetManifold(n_knots, fixed_y)
        );
    } else {
        // old behavior: single intrinsics block
        problem.AddParameterBlock(intr, intr_size);

        std::vector<int> fixed_intrinsics_param_indices;
        fixed_intrinsics_param_indices.reserve(intr_size);
        for (int i = 0; i < intr_size; ++i) {
            if (!intrinsics_param_optimize_mask[i]) {
                fixed_intrinsics_param_indices.push_back(i);
            }
        }

        problem.SetManifold(
            intr,
            new ceres::SubsetManifold(intr_size, fixed_intrinsics_param_indices)
        );
    }

    std::vector<int> fixed_intrinsics_param_indices;
    for (size_t param_idx = 0; param_idx < state.intrinsics.size();
         param_idx++) {
        bool should_optimize = intrinsics_param_optimize_mask[param_idx];
        if (!should_optimize) {
            fixed_intrinsics_param_indices.push_back(param_idx);
        }
    }

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

            if (camera_model_name == "pinhole_splined") {
                problem.AddResidualBlock(
                    ReprojectionError::create(
                        camera_model_name,
                        &model_config,
                        observation(0, 0),
                        observation(1, 0)
                    ),
                    nullptr,
                    k4,
                    x_knots,
                    y_knots,
                    camera_pose.data(),
                    target.data()
                );
            } else {
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
            SplineZeroPrior::create(nx, ny, lambda),
            nullptr,
            x_knots,
            y_knots
        );
        spdlog::info("Added spline zero prior: lambda={}", lambda);
    }

    spdlog::info("Added residual blocks");

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.use_nonmonotonic_steps = false;
    options.max_num_iterations = 10'000;

    options.num_threads =
        std::min(std::max(1, (int)std::thread::hardware_concurrency()), 16);

    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    return state.make_dict();
}

}  // namespace camcal