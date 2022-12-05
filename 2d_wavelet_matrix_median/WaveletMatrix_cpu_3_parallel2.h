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

struct WaveletMatrixParallel2 : WaveletMatrix<T, IdxType> {
    using wm = WaveletMatrix<T, IdxType>;
    WaveletMatrixParallel2(IdxType _n = 0) : wm(_n){}
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
                IdxType Zeros00 = 0, Zeros01 = 0, Zeros10 = 0, Zeros11 = 0;
                const IdxType I0_threshold = (I0 / p + 1) * p;
                const IdxType I1_threshold = (I1 / p + 1) * p;

                const int r = p * (t - 1);
                const int s = std::min<int>(p * t, wm::size);
                for(int j = r; j < s; ++j) {
                    const auto v = wm::S[j];
                    if ((v >> i & 1) == 0) { // 0
                        if ((v >> (i - 1) & 1) == 0) {
                            if (I0 < I0_threshold) {
                                Zeros00++;
                            } else {
                                Zeros01++;
                            }
                        }
                        wm::BV[i].nset(j);
                        S_next[I0++] = v;
                    }
                    else {                   // 1
                        if ((v >> (i - 1) & 1) == 0) {
                            if (I1 < I1_threshold) {
                                Zeros10++;
                            } else {
                                Zeros11++;
                            }
                        }
                        S_next[I1++] = v;
                    }
                }
                Zeros_next[I0_threshold/ p + 0].fetch_add(Zeros00);
                Zeros_next[I0_threshold/ p + 1].fetch_add(Zeros01);
                Zeros_next[I1_threshold/ p + 0].fetch_add(Zeros10);
                Zeros_next[I1_threshold/ p + 1].fetch_add(Zeros11);
            }
            wm::BV[i].construct();
            wm::S.swap(S_next);
            Zeros.swap(Zeros_next);
        }
        wm::S.clear();
    }
};




template<typename ValT, class WaveletMatrixImpl>
struct WM_2DMedianParallel2 : WM_2DMedian<ValT, WaveletMatrixImpl> {
    using wm = WM_2DMedian<ValT, WaveletMatrixImpl>;
    WM_2DMedianParallel2() : wm(){}
    using typename wm::XIdxT;
    using typename wm::XYIdxT;
    BitVector<XYIdxT> X_bv[wm::MAX_VAL_BIT_LEN];

    WM_2DMedianParallel2* construct(const ValT* src, const int _H, const int _W, const int src_step, const int _val_bit_len = wm::MAX_VAL_BIT_LEN) {
        
        wm::val_bit_len = _val_bit_len;
        wm::H = _H;
        wm::W = _W;
        const int HW = wm::H * wm::W;
        const int w_bit_len = WaveletMatrixImpl::get_bit_len(wm::W); // w=7, 7 = 111(2) -> bit_len=3;  w=8 -> bit_len=4
        assert(wm::W < 65535); // That is, less than 65534.

        vector<std::pair<ValT, XIdxT>> I(HW); // mathcal{I}_i. buffer
        const int m = omp_get_max_threads();
        const int p_original = (HW + m - 1) / m; // = ceil(HW/m)
        const int p = (p_original + WaveletMatrixImpl::WORD_SIZE - 1) / WaveletMatrixImpl::WORD_SIZE * WaveletMatrixImpl::WORD_SIZE;
        //Round up to multiples of 32 because BitVector does not support atomic.

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
            X_bv[i] = BitVector<XYIdxT>(HW);

            #pragma omp parallel
            {
                const int t = omp_get_thread_num() + 1; // Change thread id to 1-origin
                XYIdxT I0 = Zeros[t - 1];
                XYIdxT I1 = Zeros[m] + p * (t - 1) - I0;
                XYIdxT Zeros00 = 0, Zeros01 = 0, Zeros10 = 0, Zeros11 = 0;
                const XYIdxT I0_threshold = (I0 / p + 1) * p;
                const XYIdxT I1_threshold = (I1 / p + 1) * p;

                const int r = p * (t - 1);
                const int s = std::min(p * t, HW);
                for(int j = r; j < s; ++j) {
                    const auto vx = I[j];
                    const auto v = vx.first;
                    const auto x = vx.second;
                    if ((v >> i & 1) == 0) { // 0
                        if ((v >> (i - 1) & 1) == 0) {
                            if (I0 < I0_threshold) {
                                Zeros00++;
                            } else {
                                Zeros01++;
                            }
                        }
                        wm::X_wm[i].set_preconstruct(j, x);
                        X_bv[i].nset(j);
                        I_next[I0++] = vx;
                    }
                    else {                   // 1
                        if ((v >> (i - 1) & 1) == 0) {
                            if (I1 < I1_threshold) {
                                Zeros10++;
                            } else {
                                Zeros11++;
                            }
                        }
                        wm::X_wm[i].set_preconstruct(j, wm::W);
                        I_next[I1++] = vx;
                    }
                }
                Zeros_next[I0_threshold/ p + 0].fetch_add(Zeros00);
                Zeros_next[I0_threshold/ p + 1].fetch_add(Zeros01);
                Zeros_next[I1_threshold/ p + 0].fetch_add(Zeros10);
                Zeros_next[I1_threshold/ p + 1].fetch_add(Zeros11);
            }
            wm::X_wm[i].construct(w_bit_len);
            X_bv[i].construct();
            I.swap(I_next);
            Zeros.swap(Zeros_next);
        }
        return this;
    }

    // [x0, x1), [y0, y1)
    inline ValT quantile2d(const XIdxT x0, const XIdxT x1, const int y0, const int y1, XYIdxT k) const {
        assert(0 <= x0 && x0 <= wm::W);
        assert(0 <= x1 && x1 <= wm::W);
        assert(0 <= y0 && y0 <= wm::H);
        assert(0 <= y1 && y1 <= wm::H);
        XYIdxT L = y0 * wm::W;
        XYIdxT R = y1 * wm::W;
        ValT res = 0;
        for (int i = wm::val_bit_len - 1; i >= 0; --i) {
            const XYIdxT L_num = X_bv[i].rank0(L); // instead of range_freq(0, L, 0, wm::W);
            const XYIdxT R_num = X_bv[i].rank0(R); // instead of range_freq(0, R, 0, wm::W);

            const int num = wm::X_wm[i].range_freq(L, R, x0, x1);
            if (k < num) {
                L = L_num;
                R = R_num;
            }
            else {
                k -= num;
                const XYIdxT zeros = X_bv[i].rank0(wm::H * wm::W); // instead of range_freq(0, H*W, 0, W);
                L = L - L_num + zeros;
                R = R - R_num + zeros;
                res |= (ValT)1 << i;
            }
        }
        return res;
    }
    
    void median_cut_border(const int r, ValT *dst, const int dst_step) const {
        const int diameter = 2 * r + 1;
        const int median_idx = diameter * diameter / 2;
        
        #pragma omp parallel for
        for(int y = 0; y <= wm::H - diameter; ++y) {
            for(int x = 0; x <= wm::W - diameter; ++x) {
                const ValT res = quantile2d(x, x + diameter, y, y + diameter, median_idx);
                dst[y * dst_step + x] = res;
            }
        }
    }
};

} // end namespace wavelet_matrix_median
