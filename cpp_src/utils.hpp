#pragma once
#include <stdint.h>
#include <vector>

namespace camcal {
template <typename T>
inline std::vector<std::vector<T>> vector_mat(
    uint32_t n,
    uint32_t m,
    T initial_value
) {
    std::vector<std::vector<T>> mat(n, std::vector<T>(m, initial_value));
    return mat;
}
}  // namespace camcal