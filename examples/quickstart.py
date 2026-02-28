# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # LensBoy quickstart
#
# This notebook shows you how to use lensboy to calibrate a camera, and verify it using the analysis tools.

# %%
import logging

logging.basicConfig(
    level=logging.INFO,
)
import lensboy as lb
import lensboy.analysis as lba
import cv2
import imageio.v3 as iio
import mediapy
import slamd
from pathlib import Path


# %% [markdown]
# First, load the images, and look at an example image.

# %%
img_directory = Path("../data/images/wide_angle_charuco_private")
img_paths = img_directory.glob("*.png")
imgs = [iio.imread(pth) for pth in img_paths]
mediapy.show_image(imgs[0], width=1000)

# %% [markdown]
# Don't worry about the black region outside the target - I used a previous calibration to black it out for privacy reasons.
#
# Here I'm using a charuco board to calibrate a wide-angle camera. lensboy makes it relatively easy to extract charuco detections.

# %%
board = cv2.aruco.CharucoBoard(
    (14, 9), 40, 30, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
)
target_points, frames = lb.extract_frames_from_charuco(board, imgs)

# %%
img_height, img_width = imgs[0].shape[:2]
img_height, img_width

# %% [markdown]
# Let's see how well our data covers the image space:

# %%
lba.plot_detection_coverage(frames, image_width=img_width, image_height=img_height)

# %% [markdown]
# Pretty good, especially for such a wide-angle camera.
#

# %% [markdown]
#
# # Calibrate (OpenCV)
#
# We'll start by fitting an opencv-compatible model to this camera. 
# OpenCV models are compatible with almost all standard tools, and are perfectly fine for most applications.
#
# I'll go ahead and use all available distortion coefficients.

# %%
camera_model_config = lb.OpenCVConfig(
    image_height=img_height,
    image_width=img_width,
    # from lens & sensor specs - does not need to be accurate
    initial_focal_length=1000,
    # all distortion coefficients
    included_distoriton_coefficients=lb.OpenCVConfig.FULL_12
)


# %%
calibration_result = lb.calibrate_camera(
    target_points,
    frames,
    camera_model_config=camera_model_config
)


# %% [markdown]
# We can see a few things from the logs:
# * Outlier filtering was performed. 0.3% is pretty good, I would start to worry if it went over a few percent.
# * The optimizer estimated some target warp. This is useful, as flat boards like these are almost never completely flat.
#

# %% [markdown]
# # Analyze result
# Let's now do a bit of analysis on the output of the calibration.
#
#
# First, let's check the geometry that the solver ended up with:

# %%
lba.plot_target_and_poses(
    target_points,
    calibration_result.optimized_cameras_T_target
)

# %% [markdown]
# Looks about right. Let's sanity check that the reprojected target points look good in a random image:

# %%
test_sample_idx = 0

projected_drawn = lba.draw_points_in_image(
    calibration_result.frame_infos[test_sample_idx].projected_points,
    image=imgs[test_sample_idx]
)

mediapy.show_image(projected_drawn, width=1000)

# %% [markdown]
# Looks good to me. Now let's visualize the lens distortion that the solve ended up with:

# %%
intrinsics = calibration_result.optimized_camera_model
lba.plot_distortion_grid(
    intrinsics,
    grid_step_norm=0.05,
    cmap_name="jet",
    fov_fraction=0.7
)

# %% [markdown]
# The extreme distortion makes sense after looking at the images.
#
# Let's now plot the residuals of the solve:

# %%
lba.plot_residual_histogram(calibration_result.frame_infos)

# %% [markdown]
# The residuals look pretty gaussian, and the spread is tight. Also, the 2D residuals look roughly isotropic, which is a good sign.
#
# There are some pretty bad outliers - let's see where they happen:

# %%
lba.plot_worst_residual_frames(
    calibration_result.frame_infos,
    frames,
    imgs,
    scale=50,
    n=5
)

# %% [markdown]
# This looks like some issue with the charuco detector - we will not worry about it here, since they are filtered out anyway.
#
# Let's see if we can find any pattern in the residuals. The following plot is useful for this.

# %%
lba.plot_residual_grid(
    frames,
    calibration_result.frame_infos,
    image_width=img_width,
    image_height=img_height,
    grid_cells=50,
)

# %% [markdown]
# This plot shows the average magnitude of residuals in each cell, and plots the average residual as a vector.
#
# Looks like the residuals are well-behaved in the center region, and become increasingly erroatic at the edges. 
# My guesses for reasons for this are
# * OpenCV models degrade slightly at the edges 
# * The charuco detector might have some biases under extreme distortion
# * We have less data to constrain the model in the edge regions. 
#
# Since everything looks good in most of the image, I am happy with this model.

# %% [markdown]
# This is another (slighly less) useful plot function to inspect the residuals:

# %%
lba.plot_residual_vectors(
    frames,
    calibration_result.frame_infos,
    image_width=img_width,
    image_height=img_height,
    scale=10,
    scale_by_magnitude=False,
    color_by = 'angle'
)

# %% [markdown]
# As mentioned earlier, the solver estimated the warp of the calibration target. We can take a look at this warp estimation:

# %%
target_warp = calibration_result.warp_info

if target_warp is not None:
    lba.plot_target_warp(
        target_points,
        target_warp
    )

# %% [markdown]
# So the maximum deflection is about 0.5mm at the center. Not so bad, but if we solve without the warp, we get a worse fit:

# %%

calibration_result_no_warp = lb.calibrate_camera(
    target_points,
    frames,
    camera_model_config=camera_model_config,
    estimate_target_warp=False
)
lba.plot_residual_histogram(calibration_result_no_warp.frame_infos)

# %% [markdown]
# Comparing to the previous plot, this is clearly much worse.

# %% [markdown]
# # Save/load/use
#
# When we are happy with our model, we can save it:

# %%
model_path = Path("../data/camera_models/opencv.json")
intrinsics.save(model_path)

# %% [markdown]
# And then load it. I recommend just extracting the parameters to use in your application, this library is focused on calibration, not runtime use.

# %%
recovered_intrinsics = lb.OpenCV.load(model_path)

# grab the params to use in your application
image_width = recovered_intrinsics.image_width
image_height = recovered_intrinsics.image_height
K = recovered_intrinsics.K()
dist_coeffs = recovered_intrinsics.distortion_coeffs

# %% [markdown]
# # Spline models
#
# If opencv models are not capable of modeling your lens with adequate precision, lensboy offers much more flexible and powerful models that use B-splines to model lens distortion.
#
# Let's calibratre such a lens model here.

# %%
spline_model_config = lb.PinholeSplinedConfig(
    img_height,
    img_width,
    initial_focal_length=1000,
    num_knots_x=30,
    num_knots_y=20
)

# %%
spline_calibration_result = lb.calibrate_camera(
    target_points,
    frames,
    camera_model_config=spline_model_config
)

# %% [markdown]
# As you can see, this optimization takes much longer because of the increased number of parameters.
#
# Let's look at the residual analysis:

# %%
lba.plot_residual_histogram(spline_calibration_result.frame_infos)

# %% [markdown]
# We clearly see a better fit than the opencv model. However, I'm using a lens with good optics and clean distortion, so the difference is not too pronounced here.
#
# Let's take a look at the distortion:

# %%
spline_intrinsics = spline_calibration_result.optimized_camera_model
lba.plot_distortion_grid(
    spline_intrinsics,
    grid_step_norm=0.05,
    cmap_name="jet",
    fov_fraction=0.7
)

# %% [markdown]
# Since the spline is such a powerful model, it tends to behave unpredictably on the edges, where there is no data. Let's see the residual grid:

# %%
lba.plot_residual_grid(
    frames,
    spline_calibration_result.frame_infos,
    image_width=img_width,
    image_height=img_height,
    grid_cells=50,
)

# %% [markdown]
# Here the power of the spline model is clear - it is able to capture the lens distortion much better on the edges of the image. 

# %% [markdown]
# # Save/load/use
# You can save and load spline models just like opencv models:
#

# %%
spline_intrinsics.save(model_path)
spline_intrinsics = lb.PinholeSplined.load(model_path)

# %% [markdown]
# Since lensboy's spline model is not supported anywhere you need to use it a bit differently.
#
# The difference in use is that you need to undistort your images before using the camera model for most applications. 
#
# For this purpose, we support converting the splined lens model to simple pinhole intrniscs along with undistortion maps:

# %%
pinhole_remapped = spline_intrinsics.get_pinhole_model()

# %% [markdown]
# I'll grab one image that I have not blacked out to illustrate the undistortion

# %%
display_img = iio.imread(Path("../data/images/wide_angle_charuco/030.png"))


# %% [markdown]
# Let's look at an undistorted image:

# %%
undistorted = pinhole_remapped.undistort(display_img)
mediapy.show_image(undistorted, width=1000)

# %% [markdown]
# Looks like a pure pinhole image. However, we are cutting off a large part of the field of view of the original image here. The reason is that `get_pinhole_model` defaults to using the same focal length and principal point as the spline model. 
#
# This makes it necessary to tune the export a bit. You can provide custom pinhole parameters and image sizes into `get_pinhole_model`:

# %%
pinhole_remapped = spline_intrinsics.get_pinhole_model(k4=(800, 800, 1500, 1000), image_size_wh=(3000, 2000))
undistorted = pinhole_remapped.undistort(display_img)
mediapy.show_image(undistorted, width=1000)

# %% [markdown]
# This preserves more of the field of view. Additional functions you can use are `get_pinhole_model_fov` if you have a specific fov in mind, or `get_pinhole_model_alpha`, which calculates your pinhole parameters based on how much data you are okay with losing, trading off with black areas:

# %%
pinhole_remapped_alpha_1 = spline_intrinsics.get_pinhole_model_alpha(alpha=1.0)
undistorted_alpha_1 = pinhole_remapped_alpha_1.undistort(display_img)

pinhole_remapped_alpha_0 = spline_intrinsics.get_pinhole_model_alpha(alpha=0.0)
undistorted_alpha_0 = pinhole_remapped_alpha_0.undistort(display_img)
mediapy.show_images([undistorted_alpha_1, undistorted_alpha_0], width=1000, columns=1)

# %% [markdown]
# After choosing your undistortion model, the attributes of the `PinholeRemapped` are everything you need to use this cameramodel with other tools. 

# %% [markdown]
# I would recommend storing the `PinholeSplined` model on disk or database or whatever, and then only exporting to a `PinholeRemapped` after loading it in your application, since the remaps can be quite big, while the splined model is small.
