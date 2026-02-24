#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "./python_project.hpp"
#include "calibrate.hpp"
#include "cameramodels.hpp"

namespace py = pybind11;

int add(
    int a,
    int b
) {
    return a + b;
}

// Define the Python module
PYBIND11_MODULE(
    camcal_bindings,
    m
) {
    m.doc() = "CamCal for camera calibration";
    m.def(
        "add",
        &add,
        "Add two integers together - test",
        py::arg("a"),
        py::arg("b")
    );

    m.def(
        "calibrate_opencv",
        &camcal::calibrate_opencv,
        py::arg("intrinsics_initial_value"),
        py::arg("intrinsics_param_optimize_mask"),
        py::arg("cameras_from_world"),
        py::arg("target_points"),
        py::arg("detections")
    );

    m.def(
        "get_matching_spline_distortion_model",
        &camcal::get_matching_spline_distortion_model,
        py::arg("opencv_distortion_params"),
        py::arg("model_config")
    );

    m.def(
        "project_pinhole_splined_points",
        &camcal::project_pinhole_splined_pywrapper,
        py::arg("model_config"),
        py::arg("k4"),
        py::arg("dx_grid"),
        py::arg("dy_grid"),
        py::arg("points_in_camera"),
        R"doc(
Vectorized pinhole+splined projection over points_in_camera.

Args:
  fov_deg_x, fov_deg_y: FOV in degrees
  num_knots_x, num_knots_y: knot grid size
  k4: numpy array shape (4,) -> [fx, fy, cx, cy]
  dx_grid: numpy array shape (num_knots_y, num_knots_x), C-order (row-major)
  dy_grid: numpy array shape (num_knots_y, num_knots_x), C-order (row-major)
  points_in_camera: numpy array shape (N, 3), C-order

Returns:
  numpy array shape (N, 2)
)doc"
    );

    py::class_<camcal::PinholeSplinedConfig>(m, "PinholeSplinedConfig")
        .def(
            py::init<double, double, uint32_t, uint32_t>(),
            py::arg("fov_deg_x"),
            py::arg("fov_deg_y"),
            py::arg("num_knots_x"),
            py::arg("num_knots_y")
        )
        .def_readwrite("fov_deg_x", &camcal::PinholeSplinedConfig::fov_deg_x)
        .def_readwrite("fov_deg_y", &camcal::PinholeSplinedConfig::fov_deg_y)
        .def_readwrite(
            "num_knots_x",
            &camcal::PinholeSplinedConfig::num_knots_x
        )
        .def_readwrite(
            "num_knots_y",
            &camcal::PinholeSplinedConfig::num_knots_y
        )
        .def("__repr__", [](const camcal::PinholeSplinedConfig& self) {
            std::ostringstream oss;
            oss << "PinholeSplinedConfig("
                << "fov_deg_x=" << self.fov_deg_x
                << ", fov_deg_y=" << self.fov_deg_y
                << ", num_knots_x=" << self.num_knots_x
                << ", num_knots_y=" << self.num_knots_y << ")";
            return oss.str();
        });
}
