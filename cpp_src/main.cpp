#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
        "calibrate_camera",
        &camcal::calibrate_camera,
        py::arg("camera_model_name"),
        py::arg("config"),
        py::arg("intrinsics_initial_value"),
        py::arg("intrinsics_param_optimize_mask"),
        py::arg("cameras_from_world"),
        py::arg("target_points"),
        py::arg("detections")
    );
}
