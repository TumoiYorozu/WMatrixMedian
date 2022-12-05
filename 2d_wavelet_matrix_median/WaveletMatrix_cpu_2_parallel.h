#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <x86intrin.h>
#include <atomic>
#include <omp.h>

#include "WaveletMatrix_cpu_bitvector.h"
#include "WaveletMatrix_cpu_1_simple.h"


namespace wavelet_matrix_median {
using std::vector;


template <typename T = uint16_t, typename IdxType = uint32_t>

struct WaveletMatrixParallel : WaveletMatrix<T, IdxType> {
    using wm = WaveletMatrix<T, IdxType>;
    WaveletMatrixParallel(IdxType _n = 0) : wm(_n){}
    void construct(const IdxType data_bit_len) {
        wm::bit_len = data_bit_len;
        wm::BV.assign(wm::bit_len, wm::size);
        
        const int m = omp_get_max_threads();
        const int p_original = (wm::size + m - 1) / m; // = ceil(size/m)
        const int p = (p_original + wm::WORD_SIZE - 1) / wm::WORD_SIZE * wm::WORD_SIZE;
        //Round up to multiples of 32 because BitVector does not support atomic.

        vector<std::atomic<IdxType>> Zeros(m + 1), Zeros_next(m + 1);
        #pragma omp parallel
        {
            const int t = omp_get_thread_num() + 1; // Change thread id to 1-origin
            const int r = p * (t - 1);
            const int s = std::min<int>(p * t, wm::size);
            for(int j = r; j < s; ++j) {
                if ((wm::S[j] >> (wm::bit_len - 1) & 1) == 0) {
                    Zeros[t]++;
                }
            }
        }
        vector<T> S_next(wm::size);
        for (int i = wm::bit_len - 1; i >= 0; --i) {
            
            for(int t = 1; t <= m; ++t) {
                Zeros[t] += Zeros[t - 1];
            }

            std::fill(Zeros_next.begin(), Zeros_next.end(), 0);
            #pragma omp parallel
            {
                const int t = omp_get_thread_num() + 1; // Change thread id to 1-origin
                IdxType I0 = Zeros[t - 1];
                IdxType I1 = Zeros[m] + p * (t - 1) - I0;
                const int r = p * (t - 1);
                const int s = std::min<int>(p * t, wm::size);
                for(int j = r; j < s; ++j) {
                    const auto v = wm::S[j];
                    if ((v >> i & 1) == 0) { // 0
                        if ((v >> (i - 1) & 1) == 0) {
                            Zeros_next[I0/p + 1].fetch_add(1);
                        }
                        wm::BV[i].nset(j);
                        S_next[I0++] = v;
                    }
                    else {                   // 1
                        if ((v >> (i - 1) & 1) == 0) {
                            Zeros_next[I1/p + 1].fetch_add(1);
                        }
                        S_next[I1++] = v;
                    }
                }
            }
            wm::BV[i].construct();
            wm::S.swap(S_next);
            Zeros.swap(Zeros_next);
        }
        wm::S.clear();
    }
};




template<typename ValT, class WaveletMatrixImpl>
struct WM_2DMedianParallel : WM_2DMedian<ValT, WaveletMatrixImpl> {
    using wm = WM_2DMedian<ValT, WaveletMatrixImpl>;
    WM_2DMedianParallel() : wm(){}
    using typename wm::XIdxT;
    using typename wm::XYIdxT;

    WM_2DMedianParallel* construct(const ValT* src, const int _H, const int _W, const int src_step, const int _val_bit_len = wm::MAX_VAL_BIT_LEN) {
        
        wm::val_bit_len = _val_bit_len;
        wm::H = _H;
        wm::W = _W;
        const int HW = wm::H * wm::W;
        const int w_bit_len = WaveletMatrixImpl::get_bit_len(wm::W); // w=7, 7 = 111(2) -> bit_len=3;  w=8 -> bit_len=4
        assert(wm::W < 65535); // That is, less than 65534.

        vector<std::pair<ValT, XIdxT>> I(HW); // mathcal{I}_i. buffer
        const int m = omp_get_max_threads();
        const int p = (HW + m - 1) / m; // = ceil(HW/m)
        vector<std::atomic<XYIdxT>> Zeros(m + 1), Zeros_next(m + 1);

        #pragma omp parallel
        {
            const int t = omp_get_thread_num() + 1; // Change thread id to 1-origin
            const int r = p * (t - 1);
            const int s = std::min(p * t, HW);
            for(int j = r; j < s; ++j) {
                const int x = j % wm::W;
                const int y = j / wm::W;
                const auto v = src[y * src_step + x];
                I[j] = {v, x};
                if ((v >> (wm::val_bit_len - 1) & 1) == 0) {
                    Zeros[t]++;
                }
            }
        }
        
        vector<std::pair<ValT, XIdxT>> I_next(HW);
        for (int i = wm::val_bit_len - 1; i >= 0; --i) {

            for(int t = 1; t <= m; ++t) {
                Zeros[t] += Zeros[t - 1];
            }
            std::fill(Zeros_next.begin(), Zeros_next.end(), 0);
            wm::X_wm[i] = WaveletMatrixImpl(HW);


            #pragma omp parallel
            {
                const int t = omp_get_thread_num() + 1; // Change thread id to 1-origin
                XYIdxT I0 = Zeros[t - 1];
                XYIdxT I1 = Zeros[m] + p * (t - 1) - I0;
                const int r = p * (t - 1);
                const int s = std::min(p * t, HW);
                for(int j = r; j < s; ++j) {
                    const auto vx = I[j];
                    const auto v = vx.first;
                    const auto x = vx.second;
                    if ((v >> i & 1) == 0) { // 0
                        if ((v >> (i - 1) & 1) == 0) {
                            Zeros_next[I0/p + 1].fetch_add(1);
                        }
                        wm::X_wm[i].set_preconstruct(j, x);
                        I_next[I0++] = vx;
                    }
                    else {                   // 1
                        if ((v >> (i - 1) & 1) == 0) {
                            Zeros_next[I1/p + 1].fetch_add(1);
                        }
                        wm::X_wm[i].set_preconstruct(j, wm::W);
                        I_next[I1++] = vx;
                    }
                }
            }
            wm::X_wm[i].construct(w_bit_len);
            I.swap(I_next);
            Zeros.swap(Zeros_next);
        }
        return this;
    }
    
    void median_cut_border(const int r, ValT *dst, const int dst_step) const {
        const int diameter = 2 * r + 1;
        const int median_idx = diameter * diameter / 2;
        
        #pragma omp parallel for
        for(int y = 0; y <= wm::H - diameter; ++y) {
            for(int x = 0; x <= wm::W - diameter; ++x) {
                const ValT res = wm::quantile2d(x, x + diameter, y, y + diameter, median_idx);
                dst[y * dst_step + x] = res;
            }
        }
    }
};

} // end namespace wavelet_matrix_median
