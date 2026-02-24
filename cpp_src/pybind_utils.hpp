#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace camcal {
static void require(
    bool cond,
    const char* msg
) {
    if (!cond) {
        throw std::invalid_argument(msg);
    }
}
}