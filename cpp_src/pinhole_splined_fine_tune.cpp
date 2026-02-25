#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/rotation.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <vector>

#include "./calibrate.hpp"
#include "./cameramodels.hpp"
#include "./pybind_utils.hpp"

namespace camcal {

struct SplineMap {
    int Nx = 0;
    int Ny = 0;
    double half_x = 0.0;
    double half_y = 0.0;
    double x_scale = 0.0;
    double y_scale = 0.0;

    explicit SplineMap(
        const PinholeSplinedConfig& cfg
    ) {
        Nx = (int)cfg.num_knots_x;
        Ny = (int)cfg.num_knots_y;

        const double fov_rad_x = cfg.fov_deg_x * M_PI / 180.0;
        const double fov_rad_y = cfg.fov_deg_y * M_PI / 180.0;
        half_x = std::tan(fov_rad_x / 2.0);
        half_y = std::tan(fov_rad_y / 2.0);

        x_scale = (Nx - 3) / (2.0 * half_x);
        y_scale = (Ny - 3) / (2.0 * half_y);
    }

    inline void project_to_spline_coords(
        const double* cam6,
        const Vec3<double>& pw,
        double& gx,
        double& gy,
        double& x_n,
        double& y_n
    ) const {
        double pc[3];
        ceres::AngleAxisRotatePoint(cam6, pw.data(), pc);
        pc[0] += cam6[3];
        pc[1] += cam6[4];
        pc[2] += cam6[5];

        const double inv_z = 1.0 / pc[2];
        x_n = pc[0] * inv_z;
        y_n = pc[1] * inv_z;

        const double x_s_raw = 1.0 + (x_n + half_x) * x_scale;
        const double y_s_raw = 1.0 + (y_n + half_y) * y_scale;

        constexpr double eps = 1e-12;
        gx = std::max(0.0, std::min(x_s_raw, Nx - 1.0 - eps));
        gy = std::max(0.0, std::min(y_s_raw, Ny - 1.0 - eps));
    }

    inline void cell_index(
        const double* cam6,
        const Vec3<double>& pw,
        int& ix,
        int& iy
    ) const {
        double gx, gy, xn, yn;
        project_to_spline_coords(cam6, pw, gx, gy, xn, yn);
        ix = (int)gx;
        iy = (int)gy;
    }

    inline void support_indices_4x4(
        int ix,
        int iy,
        std::array<int, 16>& flat
    ) const {
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            const int yy = clamp_int(iy + b - 1, 0, Ny - 1);
            for (int a = 0; a < 4; a++) {
                const int xx = clamp_int(ix + a - 1, 0, Nx - 1);
                flat[idx++] = yy * Nx + xx;
            }
        }
    }
};

class KnotPrior2D final : public ceres::CostFunction {
   public:
    KnotPrior2D(
        double sqrt_lambda,
        double x0,
        double y0
    )
        : s_(sqrt_lambda),
          x0_(x0),
          y0_(y0) {
        mutable_parameter_block_sizes()->push_back(1);
        mutable_parameter_block_sizes()->push_back(1);
        set_num_residuals(2);
    }

    bool Evaluate(
        double const* const* params,
        double* residuals,
        double** jacobians
    ) const override {
        const double x = params[0][0];
        const double y = params[1][0];

        residuals[0] = s_ * (x - x0_);
        residuals[1] = s_ * (y - y0_);

        if (!jacobians) {
            return true;
        }

        // Each jacobian is a dense [2 x 1] block (Ceres expects row-major)
        if (jacobians[0]) {
            jacobians[0][0] = s_;   // d r0 / dx
            jacobians[0][1] = 0.0;  // d r1 / dx
        }
        if (jacobians[1]) {
            jacobians[1][0] = 0.0;  // d r0 / dy
            jacobians[1][1] = s_;   // d r1 / dy
        }
        return true;
    }

   private:
    double s_;
    double x0_, y0_;
};

class ReprojectionErrorAnalyticalScalarKnots final
    : public ceres::CostFunction {
   public:
    using Jet6 = ceres::Jet<double, 6>;

    ReprojectionErrorAnalyticalScalarKnots(
        const SplineMap& map,
        const double* k4,
        const Vec3<double>& point_world,
        int ix0,
        int iy0,
        double obs_x,
        double obs_y
    )
        : map_(map),
          obs_x_(obs_x),
          obs_y_(obs_y),
          fx_(k4[0]),
          fy_(k4[1]),
          cx_(k4[2]),
          cy_(k4[3]),
          pw_(point_world),
          ix0_(ix0),
          iy0_(iy0) {
        mutable_parameter_block_sizes()->push_back(6);
        for (int i = 0; i < 16; i++) {
            mutable_parameter_block_sizes()->push_back(1);  // dx
        }
        for (int i = 0; i < 16; i++) {
            mutable_parameter_block_sizes()->push_back(1);  // dy
        }
        set_num_residuals(2);
    }

    bool Evaluate(
        double const* const* params,
        double* residuals,
        double** jacobians
    ) const override {
        const double* cam = params[0];

        // Transform point (double)
        double gx, gy, x_n, y_n;
        map_.project_to_spline_coords(cam, pw_, gx, gy, x_n, y_n);

        // Use the FIXED cell origin (ix0_,iy0_)
        const double u = gx - (double)ix0_;
        const double v = gy - (double)iy0_;

        // Basis weights
        double wx[4], wy[4];
        cubic_bspline_basis_uniform(u, wx);
        cubic_bspline_basis_uniform(v, wy);

        // Spline eval using scalar blocks
        double dx = 0.0, dy = 0.0;
        int idx = 0;
        for (int b = 0; b < 4; b++) {
            for (int a = 0; a < 4; a++) {
                const double w = wy[b] * wx[a];
                dx += params[1 + idx][0] * w;
                dy += params[17 + idx][0] * w;
                idx++;
            }
        }

        residuals[0] = fx_ * (x_n + dx) + cx_ - obs_x_;
        residuals[1] = fy_ * (y_n + dy) + cy_ - obs_y_;

        if (!jacobians) {
            return true;
        }

        // Knot jacobians: each is [2x1]
        for (int i = 0; i < 16; i++) {
            const int b = i / 4;
            const int a = i % 4;
            const double w = wy[b] * wx[a];

            if (jacobians[1 + i]) {
                jacobians[1 + i][0] = fx_ * w;  // dr0/ddx
                jacobians[1 + i][1] = 0.0;      // dr1/ddx
            }
            if (jacobians[17 + i]) {
                jacobians[17 + i][0] = 0.0;      // dr0/ddy
                jacobians[17 + i][1] = fy_ * w;  // dr1/ddy
            }
        }

        // --- Camera jacobian [2x6] via Jet6 ---
        if (jacobians[0]) {
            Jet6 cam_j[6];
            for (int i = 0; i < 6; i++) {
                cam_j[i].a = cam[i];
                cam_j[i].v.setZero();
                cam_j[i].v[i] = 1.0;
            }

            Jet6 pw_j[3];
            for (int i = 0; i < 3; i++) {
                pw_j[i].a = pw_[i];
                pw_j[i].v.setZero();
            }

            Jet6 pc_j[3];
            ceres::AngleAxisRotatePoint(cam_j, pw_j, pc_j);
            pc_j[0] += cam_j[3];
            pc_j[1] += cam_j[4];
            pc_j[2] += cam_j[5];

            const Jet6 inv_z_j = Jet6(1.0) / pc_j[2];
            const Jet6 xn_j = pc_j[0] * inv_z_j;
            const Jet6 yn_j = pc_j[1] * inv_z_j;

            const Jet6 xs_j =
                Jet6(1.0) + (xn_j + Jet6(map_.half_x)) * Jet6(map_.x_scale);
            const Jet6 ys_j =
                Jet6(1.0) + (yn_j + Jet6(map_.half_y)) * Jet6(map_.y_scale);

            constexpr double eps = 1e-12;
            const Jet6 gx_j =
                clamp_T(xs_j, Jet6(0.0), Jet6(map_.Nx - 1.0 - eps));
            const Jet6 gy_j =
                clamp_T(ys_j, Jet6(0.0), Jet6(map_.Ny - 1.0 - eps));

            const Jet6 u_j = gx_j - Jet6((double)ix0_);
            const Jet6 v_j = gy_j - Jet6((double)iy0_);

            Jet6 wx_j[4], wy_j[4];
            cubic_bspline_basis_uniform(u_j, wx_j);
            cubic_bspline_basis_uniform(v_j, wy_j);

            Jet6 dx_j(0.0), dy_j(0.0);
            int idx2 = 0;
            for (int b = 0; b < 4; b++) {
                for (int a = 0; a < 4; a++) {
                    const Jet6 w = wy_j[b] * wx_j[a];
                    dx_j += Jet6(params[1 + idx2][0]) * w;
                    dy_j += Jet6(params[17 + idx2][0]) * w;
                    idx2++;
                }
            }

            const Jet6 r0_j =
                Jet6(fx_) * (xn_j + dx_j) + Jet6(cx_) - Jet6(obs_x_);
            const Jet6 r1_j =
                Jet6(fy_) * (yn_j + dy_j) + Jet6(cy_) - Jet6(obs_y_);

            double* J = jacobians[0];
            for (int j = 0; j < 6; j++) {
                J[j] = r0_j.v[j];
                J[6 + j] = r1_j.v[j];
            }
        }

        return true;
    }

   private:
    const SplineMap& map_;
    double obs_x_, obs_y_;
    double fx_, fy_, cx_, cy_;
    Vec3<double> pw_;
    int ix0_, iy0_;
};

struct ObservationRecord {
    size_t cam_idx;
    int pt_idx;
    double obs_x;
    double obs_y;
    int ix;
    int iy;
};

struct CellChangeCallback final : public ceres::IterationCallback {
    CellChangeCallback(
        const SplineMap& map,
        const std::vector<Vec6<double>>& cams,
        const std::vector<Vec3<double>>& pts,
        std::vector<ObservationRecord>& obs
    )
        : map_(map),
          cams_(cams),
          pts_(pts),
          obs_(obs) {}

    ceres::CallbackReturnType operator()(const ceres::IterationSummary&) override {
        for (auto& r : obs_) {
            int nix, niy;
            map_.cell_index(cams_[r.cam_idx].data(), pts_[r.pt_idx], nix, niy);
            if (nix != r.ix || niy != r.iy) {
                changed_ = true;
                return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
            }
        }
        return ceres::SOLVER_CONTINUE;
    }

    bool changed() const { return changed_; }

   private:
    const SplineMap& map_;
    const std::vector<Vec6<double>>& cams_;
    const std::vector<Vec3<double>>& pts_;
    std::vector<ObservationRecord>& obs_;
    bool changed_ = false;
};

static inline void BuildProblem(
    ceres::Problem& problem,
    const PinholeSplinedConfig& cfg,
    const SplineMap& map,
    const double* k4p,
    double* dxp,
    double* dyp,
    const std::vector<Vec6<double>>& cameras_from_world,
    const std::vector<Vec3<double>>& target_points,
    const std::vector<
        std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections,
    std::vector<double*>& dx_blocks,
    std::vector<double*>& dy_blocks,
    std::vector<ObservationRecord>& obs_records,
    const std::vector<double>& dx0,
    const std::vector<double>& dy0,
    double sqrt_lambda
) {
    const int nx = (int)cfg.num_knots_x;
    const int ny = (int)cfg.num_knots_y;
    const int n_knots = nx * ny;

    // k4 constant
    problem.AddParameterBlock(const_cast<double*>(k4p), 4);
    problem.SetParameterBlockConstant(const_cast<double*>(k4p));

    // per-knot blocks (size 1)
    dx_blocks.resize(n_knots);
    dy_blocks.resize(n_knots);
    for (int i = 0; i < n_knots; i++) {
        dx_blocks[i] = dxp + i;
        dy_blocks[i] = dyp + i;
        problem.AddParameterBlock(dx_blocks[i], 1);
        problem.AddParameterBlock(dy_blocks[i], 1);
    }

    // camera blocks
    for (auto& cam : cameras_from_world) {
        problem.AddParameterBlock(const_cast<double*>(cam.data()), 6);
    }

    // points constant
    for (auto& pt : target_points) {
        problem.AddParameterBlock(const_cast<double*>(pt.data()), 3);
        problem.SetParameterBlockConstant(const_cast<double*>(pt.data()));
    }

    // priors: cheap + exact
    for (int i = 0; i < n_knots; i++) {
        problem.AddResidualBlock(
            new KnotPrior2D(sqrt_lambda, dx0[i], dy0[i]),
            nullptr,
            dx_blocks[i],
            dy_blocks[i]
        );
    }

    // reprojection residuals (wired to correct 16 knots for each observation)
    obs_records.clear();
    const size_t num_cams = detections.size();
    for (size_t cam_idx = 0; cam_idx < num_cams; cam_idx++) {
        auto& ids = std::get<0>(detections[cam_idx]);
        auto& obs = std::get<1>(detections[cam_idx]);
        auto& cam6 = cameras_from_world[cam_idx];

        for (size_t oi = 0; oi < obs.size(); oi++) {
            const int pt_idx = ids[oi];
            const auto& pw = target_points[pt_idx];
            const double ox = obs[oi](0, 0);
            const double oy = obs[oi](1, 0);

            int ix, iy;
            map.cell_index(cam6.data(), pw, ix, iy);

            std::array<int, 16> flat{};
            map.support_indices_4x4(ix, iy, flat);

            // Create cost with fixed cell (ix,iy)
            auto* cost = new ReprojectionErrorAnalyticalScalarKnots(
                map,
                k4p,
                pw,
                ix,
                iy,
                ox,
                oy
            );

            // Build parameter list: cam + 16 dx + 16 dy
            std::array<double*, 33> blocks{};
            blocks[0] = const_cast<double*>(cam6.data());
            for (int i = 0; i < 16; i++) {
                blocks[1 + i] = dx_blocks[flat[i]];
            }
            for (int i = 0; i < 16; i++) {
                blocks[17 + i] = dy_blocks[flat[i]];
            }

            problem.AddResidualBlock(
                cost,
                nullptr,
                blocks[0],
                blocks[1],
                blocks[2],
                blocks[3],
                blocks[4],
                blocks[5],
                blocks[6],
                blocks[7],
                blocks[8],
                blocks[9],
                blocks[10],
                blocks[11],
                blocks[12],
                blocks[13],
                blocks[14],
                blocks[15],
                blocks[16],
                blocks[17],
                blocks[18],
                blocks[19],
                blocks[20],
                blocks[21],
                blocks[22],
                blocks[23],
                blocks[24],
                blocks[25],
                blocks[26],
                blocks[27],
                blocks[28],
                blocks[29],
                blocks[30],
                blocks[31],
                blocks[32]
            );

            obs_records.push_back(
                ObservationRecord{cam_idx, pt_idx, ox, oy, ix, iy}
            );
        }
    }
}

py::dict fine_tune_pinhole_splined(
    camcal::PinholeSplinedConfig& model_config,
    camcal::PinholeSplinedIntrinsicsParameters& intrinsics_parameters,
    std::vector<Vec6<double>>& cameras_from_world,
    std::vector<Vec3<double>>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2<double>>>>&
        detections
) {
    auto dxb = intrinsics_parameters.dx_grid.request();
    auto dyb = intrinsics_parameters.dy_grid.request();
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

    double* k4p = static_cast<double*>(intrinsics_parameters.k4.request().ptr);
    double* dxp = static_cast<double*>(dxb.ptr);
    double* dyp = static_cast<double*>(dyb.ptr);

    const int nx = (int)model_config.num_knots_x;
    const int ny = (int)model_config.num_knots_y;
    const int n_knots = nx * ny;

    // freeze “initial” prior anchors (true initial values)
    std::vector<double> dx0(n_knots), dy0(n_knots);
    for (int i = 0; i < n_knots; i++) {
        dx0[i] = dxp[i];
        dy0[i] = dyp[i];
    }

    const double lambda = 1e-5;
    const double sqrt_lambda = std::sqrt(lambda);

    SplineMap map(model_config);

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::ITERATIVE_SCHUR;

    options.preconditioner_type = ceres::SCHUR_JACOBI;

    options.minimizer_progress_to_stdout = true;
    options.update_state_every_iteration = true;

    constexpr int max_rebuilds = 25;

    std::vector<double*> dx_blocks, dy_blocks;
    std::vector<ObservationRecord> obs_records;

    ceres::Solver::Summary last_summary;

    for (int outer = 0; outer < max_rebuilds; outer++) {
        ceres::Problem problem;

        BuildProblem(
            problem,
            model_config,
            map,
            k4p,
            dxp,
            dyp,
            cameras_from_world,
            target_points,
            detections,
            dx_blocks,
            dy_blocks,
            obs_records,
            dx0,
            dy0,
            sqrt_lambda
        );

        CellChangeCallback
            cb(map, cameras_from_world, target_points, obs_records);
        options.callbacks.clear();
        options.callbacks.push_back(&cb);

        spdlog::info(
            "Solve pass {} (residuals wired for current cells)...",
            outer
        );

        ceres::Solve(options, &problem, &last_summary);

        if (!cb.changed()) {
            spdlog::info(
                "No cell changes detected. Done after {} rebuild(s).",
                outer
            );
            break;
        }
        spdlog::info("Cell change detected -> rebuilding problem.");
    }

    py::dict out;
    out["dx_grid"] = intrinsics_parameters.dx_grid;
    out["dy_grid"] = intrinsics_parameters.dy_grid;

    std::vector<std::vector<double>> poses_out;
    poses_out.reserve(cameras_from_world.size());
    for (auto& cam : cameras_from_world) {
        poses_out.push_back(
            std::vector<double>(cam.data(), cam.data() + cam.size())
        );
    }
    out["cameras_from_world"] = poses_out;
    return out;
}

}  // namespace camcal