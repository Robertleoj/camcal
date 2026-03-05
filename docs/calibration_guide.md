# A Practical Guide to Camera Calibration

This is a practical guide on how to calibrate a camera using lensboy. I'll walk through all the steps you need to get a quality calibration.

Throughout the guide, I'll be calibrating this camera as an example:

<img src="./media/calibration_docs/setup/lucid_1.png" width="300"> <img src="./media/calibration_docs/setup/lucid_2.png" width="300"> <img src="./media/calibration_docs/setup/lucid_3.png" width="300">

In my use case, I mount two of these next to each other to perform close-range stereo vision that requires sub-millimeter accuracy. They will be placed on the end-effector of the masonry robots we make at [Monumental](https://www.monumental.co/), and are used to scan the bricks the robot places. It's a crucial part of the robot, and needs to be as precise and reliable as possible.

I'm using an ultra wide angle lens, which makes for a good example, as they can be tricky to calibrate.

## 1. What is Camera Calibration?

A camera is a sensor that captures a detailed projection of the 3D world onto a 2D plane. To make the most of this tremendously useful sensor, we often require a precise mathematical model of the projection.

Each camera is unique - the exact mapping depends on the physical properties of your specific lens and sensor assembly. All camera lenses introduce some level of nonlinear distortion, which must be modeled.

<img src="./media/calibration_docs/projection_diagram.png">

**Camera intrinsics calibration** is the process of finding the parameters of a mathematical function that describes this 3D-to-2D mapping. Once you have this function, you can:

- **Undistort images** - remove lens distortion to get straight lines and accurate measurements
- **Measure in 3D** - go from pixel coordinates back to real-world coordinates
- **Localize** - use known world features and camera observations to locate the camera in 3D
- **Combine multiple cameras** - stereo vision, multi-camera rigs, etc.

The calibration function is typically a simple linear **pinhole model** (focal length, principal point) plus a **distortion model** that captures how your lens deviates from an ideal pinhole. Different distortion models exist with varying numbers of parameters - finding the right level of complexity for your lens is a key part of the calibration process.

## 2. Preparing Your Lens

Before you calibrate, your lens needs to be in its final mechanical state. Calibration captures the exact geometry of the optics at the moment you collect data - if anything moves afterward, the calibration is invalid.

**Focus and tune** - set your focus distance, aperture, and zoom to match your application. If you have a lens with variable focus, get the image looking exactly how you need it at your working distance.

To focus my camera, I will point it at a high-contrast image, and position it at my working distance, a bit over 165mm. I will then use the variance of the Laplacian to measure how sharp my image is, and tune it to get the highest value.

<img src="./media/calibration_docs/setup/focus_board.png" width=400> <img src="./media/calibration_docs/setup/focus_distance.png" width=400>

**Lock everything down** - once the lens is tuned, make sure nothing can shift. Use set screws if your lens has them. For critical applications, apply a small amount of Loctite to the focus and zoom rings. Even a tiny rotation of the focus ring changes the calibration.

My camera will be experiencing vibrations, and the end effector experiences the occasional impact, which will propagate to my camera. I need to be extra careful that my lens does not move under these conditions. I use Loctite to lock down all the threads, and a set screw with Loctite for the focusing thread:

<img src="./media/calibration_docs/setup/set_screw.png" width=700>

A lens that drifts between calibration and deployment will silently degrade your results. It can be hard to detect when the camera is out in the field, so prevention is your best option, and first line of defense.

## 3. Choosing a Calibration Target

The calibration target is a physical object with known geometry that you image from varying positions. We will then use the known geometry and the corresponding detections in the images to solve for the camera parameters.

**ChArUco boards** are a good default choice. A ChArUco board combines a checkerboard pattern with ArUco markers. The ArUco markers let each corner be uniquely identified even when the board is partially occluded, while the checkerboard corners provide sub-pixel accurate detections. OpenCV has support for ChArUco board detection, which `lensboy` wraps in a convenient utility function.

<img src="./media/calibration_docs/charuco_example.png" width=800>

**Why checkerboard corners?** Checkerboard corners are where four squares meet, forming a saddle point in image intensity. This saddle-point geometry is very stable for sub-pixel detection - the corner location is well-defined regardless of lighting angle, slight blur, or exposure variation. Other targets (like grids of circles or dots) rely on detecting quad or blob edges, which are more sensitive to light bleed and threshold effects.

The target should be rigid, precisely manufactured, and dense enough to cover a large portion of the camera's field of view.

A great default is to buy a ChArUco target from [calib.io](https://calib.io/) (not sponsored). This is what I use for almost all my intrinsics calibration needs.

I will be using a 600mm x 400mm ChArUco board from calib.io with a 9 x 14 grid for my camera.

<img src="./media/calibration_docs/setup/charuco.png" width=1000>

## 4. Collecting Calibration Data

Data collection is absolutely crucial for a good calibration. This is what the optimizer uses to compute the intrinsics parameters, and your calibration is only as good as your data. There are a few specific things you should aim for:

**Cover the entire image plane.** Move the target around so that detections land in every region of the frame - center, edges, and especially corners. If you want your projection function to be accurate in an area of the image, it needs to be well covered by observations.

**Take close-ups, and vary your angles.** Most of your images should be angled close-ups. Angled samples are essential for accurately solving the intrinsics - head-on views provide weak constraints on focal length and principal point. Close-ups dramatically decrease the projection uncertainty. [This great study](https://mrcal.secretsauce.net/docs-2.0/tour-choreography.html) demonstrates why you should take angled close-ups.

**How many images?** 50-100 is a reasonable range. More images help when fitting complex models, but there are diminishing returns. It's better to have 40 well-distributed images than 200 that all look the same.

**Ensure quality images.** Avoid motion blur, and keep the lighting good. You want your features detected as precisely as possible. However, you should still opt for close-ups even if your image is slightly out of focus at close range.

I've converged on a pretty simple pattern that I'll use again for my camera. I use 6 main positions for my camera and take 10 images in each, rotating the camera up and down. These are the positions:

<img src="./media/calibration_docs/setup/upper_left.png" width=400> <img src="./media/calibration_docs/setup/upper_center.png" width=400>

<img src="./media/calibration_docs/setup/upper_right.png" width=400><img src="./media/calibration_docs/setup/bottom_left.png" width=400>

<img src="./media/calibration_docs/setup/bottom_center.png" width=400> <img src="./media/calibration_docs/setup/bottom_right.png" width=400>

Here are some examples of the images from each position:

<img src="./media/calibration_docs/setup/top_left_img.jpg" width=400> <img src="./media/calibration_docs/setup/top_img.jpg" width=400>

<img src="./media/calibration_docs/setup/top_right_img.jpg" width=400><img src="./media/calibration_docs/setup/bottom_left_img.jpg" width=400>

<img src="./media/calibration_docs/setup/bottom_img.jpg" width=400> <img src="./media/calibration_docs/setup/bottom_right_img.jpg" width=400>

These are all angled close-ups with varying angles, and I end up with good coverage. This has worked well for me for a while.

## 5. Detecting Keypoints

With your images collected, the next step is to detect the features in the images. Each type of target requires a matching detector. I will be using lensboy's `extract_frames_from_charuco()` to detect my ChArUco board. It's just a simple wrapper for OpenCV's ChArUco detector.

```python
board = cv2.aruco.CharucoBoard(
    size=(14, 9),
    squareLength=40,  # mm
    markerLength=30,  # mm
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100),
)

target_points, frames, used_indices = lb.extract_frames_from_charuco(board, images)
```

The board definition must match the physical target you used - same number of squares, same dictionary, and correct square/marker sizes in whatever unit you want to work in (typically millimeters). The relevant details are usually printed on ChArUco boards.

Here is a visualization of the detected corners on a couple of the images:

<img src="./media/calibration_docs/detection_1.png" width=1000>
<img src="./media/calibration_docs/detection_2.png" width=1000>

Let's use `plot_detection_coverage()` to see how well I did in terms of coverage:

<img src="./media/calibration_docs/coverage.png" width=1000>

Looks like I did pretty well - I'm only missing the corners of the image, but it's very hard to capture data there on such a wide-angle lens, and I won't be using the data from there anyway.

If you see that you don't have much data in an area of the image you will be using, you need to take more samples to make sure to cover them.

## 6. First Calibration Run

You'll want to choose the distortion model according to your camera and application. I would say there are two main variables that control how you should choose your lens model:

- The distortion characteristics of the lens
- Your accuracy needs

Some lenses have extreme amounts of distortion like the one I'm using now. This requires a distortion model capable of modeling this amount of distortion. Each group of distortion parameters in OpenCV models is intended to model a specific type of distortion, and you can choose your distortion parameters according to the characteristics of your lens+sensor setup. See [this page](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) for a detailed explanation of OpenCV-type distortion. However, in my experience, including more distortion parameters doesn't really adversely affect your calibration quality. The solver will just set them to zero if they are not applicable.

All lenses deviate by some amount from the ideal lenses described by the OpenCV distortion model. By how much and in what way depends on your specific lens. If you want your distortion model to model these imperfections well, the OpenCV models are not sufficient, and you'll need to use a spline-based distortion model, which you can do with lensboy. These use B-spline grids to model the distortion, and are extremely flexible.

I have a wide-angle lens with extreme distortion. I have high accuracy needs, so I suspect I will need to use a spline-based model. However, let's start by fitting an OpenCV-style lens model to my lens to see how well it works.

For my first experiment, I'll use the 6 radial parameters $k_1,\ldots,k_6$, the tangential parameters $p_1,p_2$, and the thin prism parameters $s_1,\ldots,s_4$. We can fit this model with `lensboy` as follows:

```python
config = lb.OpenCVConfig(
    image_height=image_height,
    image_width=image_width,
    initial_focal_length=1000,
    included_distortion_coefficients=(
        lb.OpenCVConfig.RADIAL_6
        | lb.OpenCVConfig.TANGENTIAL
        | lb.OpenCVConfig.THIN_PRISM
    ),
)

result = lb.calibrate_camera(target_points, frames, config)
```

For `initial_focal_length`, a rough estimate is fine - the optimizer will refine it. If you know your sensor width and lens focal length in mm, you can compute it as `focal_length_mm * image_width / sensor_width_mm`.

The logs of the solver were

```
Computing initial poses with PnP...
Running full optimization...
Ran optimizer in 0.24s
Outlier filtering: 300/5955 (5.0%) outliers - going again...
Running full optimization...
Ran optimizer in 0.18s
Outlier filtering: 310/5955 (5.2%) outliers - going again...
Running full optimization...
Ran optimizer in 0.16s
Target warp max deflection: 0.3542 (target units)
Residuals (inliers): mean=0.220px, worst=1.094px
```

You might notice two things:

**Outlier filtering:** lensboy automatically filters outliers when fitting the lens model. The reason for this is that you often have erroneous or noisy data in your dataset, and including them will corrupt your fit. You can control the aggressiveness of the outlier filtering by tweaking `outlier_threshold_stddevs`, and turn it off entirely by passing `None`. However, the default value of `3` provides a good balance and works well for me. I see that about 5% of my data was filtered out, which is normal - we'll see later that this is mostly due to the ChArUco detector struggling under extreme distortion. I'd start to worry if it goes over 10%.

**Target warp estimation:** No matter how precisely manufactured, your target will never be perfectly flat - it will have some kind of warping. Because of this, lensboy automatically estimates the warping of your target, which usually results in better fits. This feature is not available for very non-planar targets. You can disable this feature by setting `estimate_target_warp` to `False`.

We also see the mean reprojection error of the inliers. This is the average norm of the difference between your measurements and the reprojected target points, given the warping of the target, the optimized camera poses, and the intrinsics model. The magnitude depends on many things, but most importantly **the quality of your data** and the **quality of the lens model fit**. The residuals grow if your data is noisy, and if your lens model underfits (is not powerful enough to capture your lens distortion).

## 7. Analyzing Calibration Quality

Let's analyze the calibration a bit to see if we actually have a good fit.

There are two main things you should think about when analyzing the quality of your calibration

- **Underfitting:** does your intrinsics model adequately capture your real camera projection? This happens if you choose a model that cannot capture your camera projection to your desired degree of accuracy. In this case, no matter how good your data is, your intrinsics will have systematic errors.

- **Overfitting:** This is when the lens model starts fitting to noise in your data, and you will again get systematic errors in the intrinsics model. This happens when you choose a powerful model, but do not have enough high-quality data to constrain it properly. Two things will happen: first, the model will have too much freedom in between data points, and will behave unpredictably in those areas. The second is that the model will optimize itself to exactly match individual noisy observations.

### The residual plot

Your first step after fitting an intrinsics model should almost always be to look at the residual distribution for which you can use `plot_residuals()`. Let's take a look at the residuals from the calibration we fit earlier:

<img src="./media/calibration_docs/first_model_residuals.png" width="1000">

This looks about as I'd expect. What you should look out for:

- **Histogram should be roughly normal.** If your histogram does not look like a normal distribution, something is going systematically wrong, and you need to debug it.
- **2D residuals should be isotropic.** The 2D residual distribution in the bottom left should be radially symmetric - you should not be able to see much of a pattern. Again, if this is not the case, you need to figure out what's causing the irregularity.

The gaussian MAD $\sigma$ is a robust estimate of the standard deviation of the data - it represents the distribution better than a raw standard deviation. When it comes to this number, lower is better until we start overfitting.

To show you an example of a plot where something is going wrong, here is a residual plot from where I attempted to calibrate a camera using april tags instead of a charuco board:

<img src="./media/calibration_docs/april_tags_residuals.png" width="1000">

Looking at this plot, you should see that the 2D distribution is not radially symmetric - it has these four "arms" reaching out. It turned out that this is because april tags are individual squares that are detected using quad detection:

<img src="./media/calibration_docs/april_tag.png" width="200">

However, different brightnesses can lead to it being detected slightly smaller or bigger, explaining the "arms" in the residual plot. This is a good reason you should opt for a checkerboard pattern instead of tags like this - they don't have this kind of variance.

### Worst frames

It's useful to inspect the frames with the largest residuals to understand where your model struggles. lensboy provides `plot_worst_residual_frames()` for this. Let's look at the 3 worst frames:

<img src="./media/calibration_docs/first_model_worst_3_residuals.png" width="800">

Looking at these images, I see that the largest residuals are caused by the ChArUco detector struggling under the extreme distortion. I won't worry about this, as these samples are filtered out as outliers.

### The residual grid

The residual plot is a great sanity check, but it deletes all spatial information - do the residuals behave differently in different regions of the image?

To analyze this, lensboy provides `plot_residual_grid()`. It bins the residuals in a grid over the image. Each grid cell is then colored according to the **mean norm of the residuals** in that bin, and shows the **mean residual** as a vector emanating from the center of the cell.

This gives you information about two things:

- **Are the residuals larger in some places than others?** If the residuals are systematically larger in some areas, this indicates underfitting or increased detection noise in those areas. Most commonly, it's the former, and you need to choose a more powerful model.
- **Do the residuals have directional biases anywhere in the image?** If this is the case, it is again very likely your model underfits the lens, and you need to choose a more powerful model.

When looking at the residual grid, it is important to only focus on the areas where you have plenty of data, and expect the lens model to be well constrained. It will usually look particularly messy towards the edges where data is sparse, and this is usually expected - your data doesn't constrain the model well there, so it will not fit well there.

Let's look at the residual grid for the model we fit earlier:

<img src="./media/calibration_docs/first_model_residual_grid.png" width="1000">

[insert analysis of the residual grid - this should look reasonable for the stronger OpenCV model]

### Target warp

As mentioned earlier, lensboy estimates the warp of near-planar targets by default. I went for a 5-parameter model that should work well for most targets, and has worked well for me. It can be useful to visualize the estimated warp with `plot_target_warp()`.

Let's take a look at the estimated warp for the model we fit earlier:

<img src="./media/calibration_docs/strong_opencv_target_warp.png" width="700">

The warp estimation has a bowl shape that I see often for charuco boards. The spread is small (about 0.5mm), but still enough to matter.

If we fit a model without enabling the target warp, and plot the residuals, we see that we get a wider residual distribution:

<img src="./media/calibration_docs/strong_opencv_no_warp_residuals.png" width="1000">

We have a higher MAD $\sigma$, and more outliers. Most of the time, you should enable target warp estimation.

### The distortion pattern

lensboy provides the plot `plot_distortion_grid()` to visualize the projection function that your intrinsics define. This doesn't provide much concrete information about the quality of the fit, but is useful for your intuitional understanding of how the distortion model of your camera works.

Let's look at this plot for our camera model:

<img src="./media/calibration_docs/strong_opencv_distortion_grid.png" width="1000">

The left side shows a grid at the $z=1$ in camera frame, and the right shows how that grid is transformed into image space. We can clearly see that my wide-angle lens introduces a large amount of distortion.

### Cross-validation via model differencing

How can we know whether our camera model is overfitting to the data?

In principle, overfitting is defined by two things:

- The model behaves erratically between data points, where it is less constrained
- The model flexes to exactly match noisy data. This means it will make incorrect predictions on data that has no error.

A key insight is that because the model behaves erratically between data points, and it bends to noise in the data, you should get different projection models based on the specific dataset you fit them on, even if they are from the same distribution.

This means we can diagnose overfitting by splitting our dataset into two parts, fit a model on each part, and compare the models. If they differ a lot, we are likely overfitting.

To compare two different lens models, we can sample a grid on the image of the first model, and unproject it. We then project it into the second lens model, and look at the difference in the pixel values.

A small complexity in this approach is that different camera models, even of the same camera, imply a different camera frame relative to the physical camera. We need to find the difference between these two camera frames to be able to fairly compare the models. This is explained in detail in [this mrcal article](https://mrcal.secretsauce.net/differencing.html), so I won't go into the details here. One result of this is that you need to choose a (possibly infinite) distance at which to compare the models.

This model comparison is easily done with lensboy using the `plot_projection_diff()` plot. I'll start by splitting my dataset into two sets:

```python
frames_a = frames[0::2]
frames_b = frames[1::2]
```

Now let's fit two instances of the same model on the two sets:

```python
model_a = lb.calibrate_camera(target_points, frames_a, config)
model_b = lb.calibrate_camera(target_points, frames_b, config)
```

Now that we have the two models, let's take a look at `plot_projection_diff()`:

<img src="./media/calibration_docs/spline_30x20_projection_diff.png" width="1200">

The left side shows the magnitude of the projection difference between the models. This looks pretty reasonable! The models differ by less than 0.2 pixels in most of the image, so I don't suspect heavy overfitting.

The right side shows the pattern of the projection difference. In reality the differences are usually imperceptibly small, so they are exaggerated.

One thing to look out for is the "fit circle". Its interior is the area of the image we use to find the difference between the implied camera frames of the models. This should only cover areas of the image where you expect good intrinsics. If it goes out of that area, this plot will not be realistic, and you need to adjust the `radius` argument to `plot_projection_diff()`.

## 8. Splined Models

The OpenCV model from section 7 fits my lens reasonably well, but can we do better? For my application I want to push for maximum precision, so let's see if a more flexible model can reduce the residuals further.

Spline-based models use B-spline grids to model distortion, and so can model more arbitrary distortion patterns. However, they are also more prone to overfitting due to their flexibility, and thus require more data to constrain properly.

We can configure a spline model in lensboy with `PinholeSplinedConfig`. You control how flexible the model is by tuning the spline grid density.

I'll train a splined model using a 30x20 grid as my starting point:

```python
config = lb.PinholeSplinedConfig(
    image_height=image_height,
    image_width=image_width,
    initial_focal_length=1000,
    num_knots_x=30,
    num_knots_y=20,
)

result = lb.calibrate_camera(target_points, frames, config)
```

Let's take a look at `plot_residuals()` and `plot_residual_grid()`:

<img src="./media/calibration_docs/spline_30x20_residuals.png" width="1000">

<img src="./media/calibration_docs/spline_30x20_residual_grid.png" width="700">

As expected with such a flexible model, it fits our data even better. Our MAD $\sigma$ is even lower, and the residual grid looks even tighter.

Let's take a look at how this spline model's projection function looks like with `plot_distortion_grid()`. I'll show the spline knots by setting `show_spline_knots` to `True`:

<img src="./media/calibration_docs/spline_30x20_distortion_grid.png" width="1000">

The distortion looks very similar to the OpenCV models, except at the edges, where it behaves a bit erratically - this is standard for a spline-based model, as it is very flexible and relatively underconstrained at the edges.

## 9. Putting It All Together

Now let's use the diagnostic tools from section 7 to systematically compare models with increasing complexity. For each model, I'll show the residual plot, residual grid, and cross-validation model difference.

#### OpenCV RADIAL_6

Let's start with a simpler OpenCV model using only the 6 radial distortion parameters, to see what underfitting looks like in practice.

```python
config = lb.OpenCVConfig(
    ...,
    included_distortion_coefficients=lb.OpenCVConfig.RADIAL_6,
)
```

<img src="./media/calibration_docs/opencv_radial_6/residuals.png" width="700">
<img src="./media/calibration_docs/opencv_radial_6/residual_grid.png" width="700">
<img src="./media/calibration_docs/opencv_radial_6/projection_diff.png" width="1200">

The residual grid clearly shows the hallmarks of underfitting - the residuals grow systematically away from the center, and there are obvious directional biases across the image. The cross-validation models match very closely, which makes sense: a simple model doesn't have enough freedom to overfit.

#### OpenCV RADIAL_6 + TANGENTIAL + THIN_PRISM

This is the OpenCV model we fit in section 6.

```python
config = lb.OpenCVConfig(
    ...,
    included_distortion_coefficients=(
        lb.OpenCVConfig.RADIAL_6
        | lb.OpenCVConfig.TANGENTIAL
        | lb.OpenCVConfig.THIN_PRISM
    ),
)
```

<img src="./media/calibration_docs/opencv_radial_6_tangential_thin_prism/residuals.png" width="700">
<img src="./media/calibration_docs/opencv_radial_6_tangential_thin_prism/residual_grid.png" width="700">
<img src="./media/calibration_docs/opencv_radial_6_tangential_thin_prism/projection_diff.png" width="1200">

Like we saw before, this model fits our lens relatively well, and looking at the cross-validation plot, we are also clearly not overfitting - the model differences are extremely small.

If my application did not require extreme accuracy, this model would do just fine.

#### Spline 20x15

Now for a pretty lean spline model with only a 20x15 grid:

```python
config = lb.PinholeSplinedConfig(
    ...,
    num_knots_x=20,
    num_knots_y=15,
)
```

<img src="./media/calibration_docs/spline_20x15/residuals.png" width="700">
<img src="./media/calibration_docs/spline_20x15/residual_grid.png" width="700">
<img src="./media/calibration_docs/spline_20x15/projection_diff.png" width="1200">

This one is interesting: the fit is worse than the previous OpenCV model, but the cross-validation difference is still higher than for the OpenCV models.

The reason for this is likely that the model is underpowered, but doesn't contain the inductive biases that the OpenCV models have. So the underfitting does not save it from having higher variance - with sparse knots, the spline surface has enough freedom to warp in different directions depending on which observations it happens to see, but not enough resolution to actually capture the true distortion. I won't be using this one.

#### Spline 40x30

Let's look at a larger spline model with a 40x30 grid:

```python
config = lb.PinholeSplinedConfig(
    ...,
    num_knots_x=40,
    num_knots_y=30,
)
```

<img src="./media/calibration_docs/spline_40x30/residuals.png" width="800">
<img src="./media/calibration_docs/spline_40x30/residual_grid.png" width="700">
<img src="./media/calibration_docs/spline_40x30/projection_diff.png" width="1200">

This is the first model that outperforms both OpenCV models in how well it fits the data. The projection difference plot shows that the cross-validation models match pretty well, indicating that the model is not overfitting.

#### Spline 50x35

Let's try an even larger spline grid, 50x35:

```python
config = lb.PinholeSplinedConfig(
    ...,
    num_knots_x=50,
    num_knots_y=35,
)
```

<img src="./media/calibration_docs/spline_50x35/residuals.png" width="1000">
<img src="./media/calibration_docs/spline_50x35/residual_grid.png" width="700">
<img src="./media/calibration_docs/spline_50x35/projection_diff.png" width="1200">

The model does not fit the data any better than the 40x30 spline model, but the projection difference is starting to increase slightly, indicating that we are moving towards overfitting.

#### Final choice

The choice is between the `OpenCV RADIAL_6 + TANGENTIAL + THIN_PRISM` model and the `Spline 40x30` model.

For my application I want to bias towards precision, so I'll choose the `Spline 40x30` model.

Another application that I use the same lens+sensor combination is localization from detections of a known world map. There, the precision requirements are slightly more lenient, so for that application I choose the OpenCV model for its simplicity of use.

## 10.
