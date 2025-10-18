#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(int a, int b) { return a + b; }

// Define the Python module
PYBIND11_MODULE(camcal_bindings, m) {
  m.doc() = "CamCal for camera calibration";
  m.def("add", &add, "Add two integers together - test", py::arg("a"),
        py::arg("b"));
}
