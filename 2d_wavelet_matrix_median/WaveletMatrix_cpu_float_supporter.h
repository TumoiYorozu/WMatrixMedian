#pragma once

#include <vector>
#include <algorithm>

#include <execution> // for Parallelism TS

namespace wavelet_matrix_median {
using std::vector;

template<typename ValT = float, typename IdxT = uint32_t>
struct WMMedianFloatSupporterCPU {
    WMMedianFloatSupporterCPU() {}
    vector<std::pair<ValT, IdxT>> table;
    void sort_and_set(const ValT* src, const int H, const int W, const int src_step, vector<IdxT> &dst) {
        table.reserve(H * W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                table.emplace_back(src[y * src_step + x], y * W + x);
            }
        }
        sort(std::execution::par, table.begin(), table.end());
        // sort(table.begin(), table.end());
        dst.resize(H * W);
        for (int i = 0; i < H * W; ++i) {
            dst[table[i].second] = i;
        }
    }
};

} // end namespace wavelet_matrix_median
