# %%
try:
    from IPython import get_ipython


    get_ipython().run_line_magic('load_ext', 'autoreload') # type: ignore
    get_ipython().run_line_magic('autoreload', '2') # type: ignore
except:
    pass

# %%
from camcal.calibration.calibrate import calibrate_camera
from camcal.camera_models import OpenCVConfig, PinholeSplinedConfig, PinholeSplined
import cv2
import numpy as np
import imageio.v3 as iio
import mediapy
from camcal.calibration.calibrate import Detection
from tqdm import tqdm
import slamd
import matplotlib.pyplot as plt
from camcal._internal.charuco import detect_charuco
from camcal._internal.paths import repo_root


# %%
# Load images
img_directory = repo_root() / "data" / "images" / "wide_angle_charuco"
img_paths = img_directory.glob("*.png")

imgs = [iio.imread(pth) for pth in img_paths]

# %%
mediapy.show_image(imgs[0], width=1000)

# %%
# detect charuco
board = cv2.aruco.CharucoBoard(
    (14, 9), 40, 30, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
)

# %%
# Detect target in images
detections = []
for img in tqdm(imgs):
    detection = detect_charuco(img, board)
    detections.append(detection)


# %%
img_height, img_width = imgs[0].shape[:2]
img_height, img_width

# %%
# camera_model_config = OpenCVConfig(
#     image_height=img_height,
#     image_width=img_width,
#     initial_focal_length=1000,
#     included_distoriton_coefficients=OpenCVConfig.FULL_12
# )
camera_model_config = PinholeSplinedConfig(
    image_height=img_height,
    image_width=img_width,
    initial_focal_length=1000,
    num_knots_x=25,
    num_knots_y=20,
    fov_deg_x=160.0,
    fov_deg_y=140.0
)


# %%
# N x 3
obj_points = np.array(board.getChessboardCorners())
print(obj_points.shape, obj_points.dtype)

# %%
calibration_result = calibrate_camera(
    obj_points,
    detections,
    camera_model_config=camera_model_config
)


# %%
calibration_result

# %%
if 'vis' not in globals():
    vis = slamd.Visualizer("calibrate_test", port=4893)

    scene = vis.scene("calibrate_test")

# %%
scene.set_object(
    "/target_points",
    slamd.geom.PointCloud(
        obj_points,
        colors=(255, 0, 0),
        radii=2
    )
)

# %%
for i, pose in enumerate(calibration_result.optimized_cameras_T_target):
    scene.set_object(
        f"/camera_{i}",
        slamd.geom.Triad(pose=pose.inverse().matrix, scale=20),
    )

# %%
test_sample_idx = 0

debug_img = cv2.cvtColor(imgs[test_sample_idx].copy(), cv2.COLOR_GRAY2RGB)

mediapy.show_image(debug_img, width=1000)

# %%
intrinsics = calibration_result.optimized_camera_model

# %%
print(intrinsics)

# %%
pinhole_model = intrinsics.get_pinhole_model(new_k4=(800, 800, 1500, 1000), new_image_size_wh=(3000, 2000))
undistorted = pinhole_model.undistort(debug_img)

mediapy.show_image(undistorted, width=1000)


# %%
def draw_points(img, points, color=(0, 255, 0), r=4, thickness=-1):
    for (x, y) in points:
        cv2.circle(img, (int(x), int(y)), r, color, thickness)
    return img


# %%
detection = detections[test_sample_idx]

debug_img = draw_points(debug_img, detection.points, color=(255, 0, 0), r=4)
mediapy.show_image(debug_img, width=1000)

# %%
camera_pose = calibration_result.optimized_cameras_T_target[test_sample_idx]

# %%
intrinsics.dx_grid

# %%
projected = []
undistorted_projected = []

for pt_idx in detection.point_ids:
    pt_target = obj_points[pt_idx]
    pt_cam = camera_pose.apply1(pt_target)


    img_pt = intrinsics.project_points(
        pt_cam[None, :],
    )

    img_pt_undistorted = pinhole_model.project_points_undistorted(pt_cam[None, :])

    projected.append(img_pt.squeeze())
    undistorted_projected.append(img_pt_undistorted.squeeze())




# %%
debug_img = draw_points(debug_img, np.array(projected), color=(0, 255, 0), r=4)
debug_img_undistorted = draw_points(undistorted, np.array(undistorted_projected), color=(0, 255, 0), r=4)

mediapy.show_images([debug_img, debug_img_undistorted], columns=1, width=1000)

# %%
residuals = []

for i, detection in enumerate(detections):
    detection: Detection

    indices = detection.point_ids

    points_in_target = obj_points[indices]

    camera_pose = calibration_result.optimized_cameras_T_target[i]
    points_in_cam = camera_pose.apply(points_in_target)


    projected = intrinsics.project_points(
        points_in_cam.astype(np.float32)
    )


    measured = detection.points

    delta = measured - projected

    x_deltas = delta[:, 0]
    y_deltas = delta[:, 1]

    residuals.extend(x_deltas.tolist())
    residuals.extend(y_deltas.tolist())


residuals = np.array(residuals)
print(residuals.shape)

# %%
no_outliers = residuals[np.abs(residuals) < 1.5]
plt.hist(no_outliers, bins=100)

# %%
std = np.std(residuals)
print(std)
