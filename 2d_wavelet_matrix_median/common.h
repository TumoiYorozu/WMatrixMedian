#pragma once

namespace wavelet_matrix_median {
    using namespace std;

    template <typename T>
    constexpr T div_ceil(T a, T b) {
        return (a + b - 1) / b;
    }
} // end namespace wavelet_median
