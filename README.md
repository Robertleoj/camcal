<p align="center">
  <img src="media/logo.png" alt="lensboy" width="400">
</p>

# lensboy

Camera calibration for vision engineers. Extremely simple to use, and maximally powerful.

Supports OpenCV camera models and spline-based distortion models for lenses that OpenCV can't handle.

<p align="center">
  <img src="media/showcase_3.png" width="700"><br>
  <img src="media/showcase_4.png" width="345"> <img src="media/showcase_1.png" width="345"><br>
  <img src="media/showcase_2.png" width="345"> <img src="media/showcase_5.png" width="345">
</p>

## Quick example

```python
import lensboy as lb

# detect calibration target in images
target_points, frames = lb.extract_frames_from_charuco(board, imgs)

# calibrate
result = lb.calibrate_camera(
    target_points, frames,
    camera_model_config=lb.OpenCVConfig(
        image_height=h, image_width=w, initial_focal_length=1000,
    ),
)

# save
result.optimized_camera_model.save("camera.json")
```

Need more accuracy? Just swap the config — same API, way more powerful:

```python
result = lb.calibrate_camera(
    target_points, frames,
    camera_model_config=lb.PinholeSplinedConfig(
        image_height=h, image_width=w, initial_focal_length=1000,
    ),
)
```

## Why lensboy

Even for standard OpenCV models, lensboy gives you better calibrations than raw `cv2.calibrateCamera`:

- **Automatic outlier filtering** removes bad detections
- **Target warp estimation** compensates for non-flat calibration boards
- **Analysis tools** to verify your calibration is actually good

For wide-angle lenses where OpenCV's polynomial distortion model isn't enough, lensboy offers spline-based distortion models that can capture arbitrary distortion patterns. This approach is inspired by [mrcal](https://mrcal.secretsauce.net/), but lensboy is designed to be easier to use and trivial to install.

## Install

For calibration time, includes analysis and plotting tools:

```bash
pip install lensboy[analysis]
```

For loading and using the camera models:

```bash
pip install lensboy
```

## Getting started

See the [quickstart notebook](examples/quickstart.ipynb) for a full walkthrough covering both OpenCV and spline models.

## Spline models

When OpenCV's polynomial distortion model can't fully capture your lens, switch to a spline model. These use B-spline grids instead of polynomial coefficients, and can capture arbitrary distortion patterns.

The calibrated spline model can be converted to a pinhole model with undistortion maps, so you can use it anywhere:

```python
pinhole = spline_model.get_pinhole_model()
undistorted = pinhole.undistort(image)
```
