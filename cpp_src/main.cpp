#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "calibrate.hpp"

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
        "calibrate_camera",
        &camcal::calibrate_camera,
        py::arg("camera_model_name"),
        py::arg("intrinsics_initial_value"),
        py::arg("intrinsics_param_optimize_mask"),
        py::arg("camera_poses_world"),
        py::arg("target_points"),
        py::arg("detections")
    );
}
