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
        "fine_tune_pinhole_splined",
        &camcal::fine_tune_pinhole_splined,
        py::arg("model_config"),
        py::arg("intrinsics_parameters"),
        py::arg("cameras_from_world"),
        py::arg("target_points"),
        py::arg("detections")
    );

    m.def(
        "project_pinhole_splined_points",
        &camcal::project_pinhole_splined_pywrapper,
        py::arg("model_config"),
        py::arg("intrinsics"),
        py::arg("points_in_camera")
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

    py::class_<camcal::PinholeSplinedIntrinsicsParameters>(
        m,
        "PinholeSplinedIntrinsicsParameters"
    )
        .def(
            py::init([](py::array_t<double> k4,
                        py::array_t<double> dx_grid,
                        py::array_t<double> dy_grid) {
                using A = py::
                    array_t<double, py::array::c_style | py::array::forcecast>;
                auto k4_ = A(k4);
                auto dx_ = A(dx_grid);
                auto dy_ = A(dy_grid);

                auto k4b = k4_.request();
                if (k4b.ndim != 1 || k4b.shape[0] != 4) {
                    throw py::value_error("k4 must have shape (4,)");
                }

                auto dxb = dx_.request();
                auto dyb = dy_.request();
                if (dxb.ndim != 2) {
                    throw py::value_error("dx_grid must be a 2D array");
                }
                if (dyb.ndim != 2) {
                    throw py::value_error("dy_grid must be a 2D array");
                }

                return camcal::PinholeSplinedIntrinsicsParameters{
                    k4_,
                    dx_,
                    dy_
                };
            }),
            py::arg("k4"),
            py::arg("dx_grid"),
            py::arg("dy_grid")
        )
        .def_readwrite("k4", &camcal::PinholeSplinedIntrinsicsParameters::k4)
        .def_readwrite(
            "dx_grid",
            &camcal::PinholeSplinedIntrinsicsParameters::dx_grid
        )
        .def_readwrite(
            "dy_grid",
            &camcal::PinholeSplinedIntrinsicsParameters::dy_grid
        )
        .def(
            "__repr__",
            [](const camcal::PinholeSplinedIntrinsicsParameters& self) {
                auto dxb = self.dx_grid.request();
                std::ostringstream oss;
                oss << "PinholeSplinedIntrinsicsParameters("
                    << "dx_grid_shape=(" << dxb.shape[0] << ", " << dxb.shape[1]
                    << "))";
                return oss.str();
            }
        );
}
