#include "calibrate.hpp"

namespace camcal {

void calibrate_camera(
    std::string camera_model_name,
    std::vector<double>& intrinsics_initial_value,
    std::vector<bool>& intrinsics_param_optimize_mask,
    std::vector<Vec6>& camera_poses_world,
    std::vector<Vec3>& target_points,
    std::vector<std::tuple<std::vector<int32_t>, std::vector<Vec2> > >&
        detections
) {}

}  // namespace camcal