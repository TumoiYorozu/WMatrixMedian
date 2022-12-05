#include <cstdint>
#include <vector>
#include <algorithm>

#include "WaveletMatrix_cpu_1_simple.h"
#include "WaveletMatrix_cpu_2_parallel.h"
#include "WaveletMatrix_cpu_3_parallel2.h"
#include "WaveletMatrix_cpu_float_supporter.h"

namespace wavelet_matrix_median {

    
template<class WMatrix2DMedianImpl, typename T>
void wm_median_cpu_core(const T *src, const int radius, const int H, const int W, const int src_step, T *dst, const int dst_step) {
    const int exH= H + 2 * radius;
    const int exW= W + 2 * radius;
    if constexpr(is_same<T, float>() == false) {
        WMatrix2DMedianImpl wm2d_median;
        wm2d_median.construct(src, exH, exW, src_step);
        wm2d_median.median_cut_border(radius, dst, dst_step);
    
    } else {
        using IdxT = uint32_t;
        const int val_bit_len = WaveletMatrix<>::get_bit_len(exH * exW);
        WMatrix2DMedianImpl wm2d_median;
        WMMedianFloatSupporterCPU<T, IdxT> float_supporter;
        vector<IdxT> buf;
        float_supporter.sort_and_set(src, exH, exW, src_step, buf);
        wm2d_median.construct(buf.data(), exH, exW, exW, val_bit_len);

        IdxT* dst_as_idxT = reinterpret_cast<IdxT*>(dst);
        for(int y = 0; y < H; ++y) {
            for(int x = 0; x < W; ++x) {
                dst_as_idxT[y * dst_step + x] = -1;
            }
        }
        wm2d_median.median_cut_border(radius, dst_as_idxT, dst_step);
        for(int y = 0; y < H; ++y) {
            for(int x = 0; x < W; ++x) {
                dst[y * dst_step + x] = float_supporter.table[dst_as_idxT[y * dst_step + x]].first;
            }
        }
    }
}

template<typename T>
void wm_naive_cpu_median(const T *src, const int radius, const int H, const int W, const int src_step, T *dst, const int dst_step) {
    using XIdxT = uint16_t;
    if constexpr(is_same<T, float>() == false) {
        wm_median_cpu_core<WM_2DMedian<T, WaveletMatrix<XIdxT>>>(src, radius, H, W, src_step, dst, dst_step);
    } else {
        using IdxT = uint32_t;
        wm_median_cpu_core<WM_2DMedian<IdxT, WaveletMatrix<XIdxT>>>(src, radius, H, W, src_step, dst, dst_step);
    }
}

template<typename T>
void wm_parallel_cpu_median(const T *src, const int radius, const int H, const int W, const int src_step, T *dst, const int dst_step) {
    using XIdxT = uint16_t;
    if constexpr(is_same<T, float>() == false) {
        wm_median_cpu_core<WM_2DMedianParallel<T, WaveletMatrixParallel<XIdxT>>>(src, radius, H, W, src_step, dst, dst_step);
    } else {
        using IdxT = uint32_t;
        wm_median_cpu_core<WM_2DMedianParallel<IdxT, WaveletMatrixParallel<XIdxT>>>(src, radius, H, W, src_step, dst, dst_step);
    }
}

template<typename T>
void wm_parallel2_cpu_median(const T *src, const int radius, const int H, const int W, const int src_step, T *dst, const int dst_step) {
    using XIdxT = uint16_t;
    if constexpr(is_same<T, float>() == false) {
        wm_median_cpu_core<WM_2DMedianParallel2<T, WaveletMatrixParallel2<XIdxT>>>(src, radius, H, W, src_step, dst, dst_step);
    } else {
        using IdxT = uint32_t;
        wm_median_cpu_core<WM_2DMedianParallel2<IdxT, WaveletMatrixParallel2<XIdxT>>>(src, radius, H, W, src_step, dst, dst_step);
    }
}

} // end namespace wavelet_median
