#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "calibrate.hpp"
#include "cameramodels.hpp"
#include "./python_project.hpp"

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

    py::class_<camcal::ModelConfig>(m, "ModelConfig")
        .def(
            py::init(
                [](const std::unordered_map<std::string, double>& double_params,
                   const std::unordered_map<std::string, uint32_t>& int_params
                ) {
                    camcal::ModelConfig cfg;
                    cfg.double_params = double_params;
                    cfg.int_params = int_params;
                    return cfg;
                }
            ),
            py::arg("double_params") =
                std::unordered_map<std::string, double>{},
            py::arg("int_params") = std::unordered_map<std::string, uint32_t>{}
        )
        // expose as normal Python dict-like fields
        .def_readwrite("double_params", &camcal::ModelConfig::double_params)
        .def_readwrite("int_params", &camcal::ModelConfig::int_params)

        // optional: nice repr so printing it doesn't suck
        .def("__repr__", [](const camcal::ModelConfig& self) {
            return "<ModelInfo double_params=" +
                   py::repr(py::cast(self.double_params)).cast<std::string>() +
                   " int_params=" +
                   py::repr(py::cast(self.int_params)).cast<std::string>() +
                   ">";
        });

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
        py::arg("fov_deg_x"),
        py::arg("fov_deg_y"),
        py::arg("num_knots_x"),
        py::arg("num_knots_y")
    );

    m.def(
        "project_pinhole_splined_points",
        &camcal::project_pinhole_splined_pywrapper,
        py::arg("fov_deg_x"),
        py::arg("fov_deg_y"),
        py::arg("num_knots_x"),
        py::arg("num_knots_y"),
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
}
