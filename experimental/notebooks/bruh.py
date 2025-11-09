# %%
from IPython import get_ipython

import camcal.camera_models # type: ignore
get_ipython().run_line_magic('load_ext', 'autoreload') # type: ignore
get_ipython().run_line_magic('autoreload', '2') # type: ignore

# %%
import cv2
import imageio.v3 as iio
import mediapy
from tqdm import tqdm
import camcal
from camcal._internal.charuco import detect_charuco
from camcal._internal.paths import repo_root


# %%
# Load images
img_directory = repo_root() / "data" / "images" / "wide_angle_charuco"
img_paths = img_directory.glob("*.png")

imgs = [iio.imread(pth) for pth in img_paths]

# %%
mediapy.show_image(imgs[0], width=500)

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
    detections.append((detection.charuco_ids, detection.charuco_corners))

# %%
detections[0][0].shape

# %%
img_height, img_width = imgs[0].shape[:2]
img_height, img_width

# %%
camera_model_config = camcal.camera_models.PinholeConfig(
    image_height=img_height,
    image_width=img_width
)


# %%
# N x 3
obj_points = board.getChessboardCorners()

"""
detections = [
    (
        point_indices, # M (int, which obj_points do the img_points correspond to)
        img_points     # M x 2
    ),
]
"""

"""
camcal.calibrate_camera(
    obj_points,
    detections
    config=camera_model_config
)
"""


# %%
# calibration_result = camcal.calibrate_camera(detections, camera_model_config)
#
# camera_model = calibration_result.camera_model
# camera_model.save("model.cam")

# %%
# cam = CameraModel.load("model.cam")
# cam.project()
# cam.deproject()
