#pragma once

#include <cassert>
#include <vector>
#include <algorithm>
#include <x86intrin.h>

#include "WaveletMatrix_cpu_bitvector.h"

namespace wavelet_matrix_median {
using std::vector;
using std::is_same;

template <typename T = uint16_t, typename IdxType = uint32_t>
struct WaveletMatrix {
    using T_Type = T;
    using WordT = uint32_t;
    static constexpr int WORD_SIZE = 32;
    static_assert(WORD_SIZE == BitVector<IdxType>::WORD_SIZE);
    static_assert(is_same<WordT, typename BitVector<IdxType>::WordT>());

    IdxType size, bit_len;
    vector<T> S; // Src input. Destroyed after construction.
    vector<BitVector<IdxType>> BV;
    WaveletMatrix(IdxType _n = 0) {
        size = (_n + WORD_SIZE - 1) / WORD_SIZE * WORD_SIZE;
        S.resize(size);
    }
    static constexpr int get_bit_len(uint32_t val) {
        if (val == 0) return 0;
        return 32 - __builtin_clz(val);
    }
    inline void set_preconstruct(const IdxType i, const T& val) {
        S[i] = val;
    }
    void construct(const IdxType data_bit_len) {
        bit_len = data_bit_len;
        BV.assign(bit_len, size);

        vector<T> first, last;
        first.reserve(size);
        last.reserve(size);
        for (int i = bit_len - 1; i >= 0; --i) {
            first.clear();
            last.clear();
            for(int j = 0; j < size; ++j) {
                const auto v = S[j];
                if ((v >> i & 1) == 0) { // 0
                    BV[i].nset(j);
                    first.emplace_back(v);
                }
                else {                   // 1
                    last.emplace_back(v);
                }
            }
            BV[i].construct();
            S.swap(first);
            S.insert(S.end(), last.begin(), last.end());
        }
        S.clear();
    }

    // Number of values in interval [L, R) which is less than x
    IdxType less_freq(IdxType L, IdxType R, const T x) const {
        assert(0 <= L && L <= size);
        assert(0 <= R && R <= size);
        assert(0 <= x && x < (1u << bit_len));
        IdxType res = 0;
        for (int i = bit_len - 1; i >= 0; --i) {
            if (((x >> i) & 1) == 0) {
                L = BV[i].rank0(L);
                R = BV[i].rank0(R);
            }
            else {
                res += BV[i].rank0(R) - BV[i].rank0(L);
                L = L - BV[i].rank0(L) + BV[i].rank0(size);
                R = R - BV[i].rank0(R) + BV[i].rank0(size);
            }
        }
        return res;
    }

    // Number of values contained in interval [x0, x1) in the value range of [y0, y1)
    IdxType range_freq(const IdxType x0, const IdxType x1, const T y0, const T y1) const {
        return less_freq(x0, x1, y1) - less_freq(x0, x1, y0);
    }
};




template<typename ValT, class WaveletMatrixImpl>
struct WM_2DMedian {
    WM_2DMedian() {}
    using XIdxT = uint16_t;
    using XYIdxT = int32_t;
    static_assert(is_same<ValT, uint32_t>() || is_same<ValT, uint16_t>() || is_same<ValT, uint8_t>());
    static_assert(is_same<XIdxT, typename WaveletMatrixImpl::T_Type>());
    static constexpr int MAX_VAL_BIT_LEN = 8 *sizeof(ValT);
    WaveletMatrixImpl X_wm[MAX_VAL_BIT_LEN];
    int val_bit_len = -1;
    int H, W;

    WM_2DMedian* construct(const ValT* src, const int _H, const int _W, const int src_step, const int _val_bit_len = MAX_VAL_BIT_LEN) {
        val_bit_len = _val_bit_len;
        H = _H;
        W = _W;
        const int HW = H * W;
        const int w_bit_len = WaveletMatrixImpl::get_bit_len(W); // w=7, 7 = 111(2) -> bit_len=3;  w=8 -> bit_len=4
        assert(W < 65535); // That is, less than 65534.

        vector<std::pair<ValT, XIdxT>> I; // mathcal{I}_i. buffer
        vector<std::pair<ValT, XIdxT>> first, last;
        I.reserve(HW);
        first.reserve(HW);
        last.reserve(HW);

        for(int y = 0; y < H; ++y) {
            for(int x = 0; x < W; ++x) {
                I.emplace_back(src[y * src_step + x], x);
            }
        }
        
        for (int i = val_bit_len - 1; i >= 0; --i) {
            first.clear();
            last.clear();
            X_wm[i] = WaveletMatrixImpl(HW);
            for(int j = 0; j < HW; ++j) {
                const auto v = I[j].first;
                const auto x = I[j].second;
                if ((v >> i & 1) == 0) { // 0
                    X_wm[i].set_preconstruct(j, x);
                    first.emplace_back(v, x);
                }
                else {                   // 1
                    X_wm[i].set_preconstruct(j, W);
                    last.emplace_back(v, x);
                }
            }
            I.swap(first);
            I.insert(I.end(), last.begin(), last.end());
            X_wm[i].construct(w_bit_len);
        }
        return this;
    }
    
    // [x0, x1), [y0, y1)
    inline ValT quantile2d(const XIdxT x0, const XIdxT x1, const int y0, const int y1, XYIdxT k) const {
        assert(0 <= x0 && x0 <= W);
        assert(0 <= x1 && x1 <= W);
        assert(0 <= y0 && y0 <= H);
        assert(0 <= y1 && y1 <= H);
        XYIdxT L = y0 * W;
        XYIdxT R = y1 * W;
        ValT res = 0;
        for (int i = val_bit_len - 1; i >= 0; --i) {
            const XYIdxT L_num = X_wm[i].range_freq(0, L, 0, W);
            const XYIdxT R_num = X_wm[i].range_freq(0, R, 0, W);
            const int num = X_wm[i].range_freq(L, R, x0, x1);
            if (k < num) {
                L = L_num;
                R = R_num;
            }
            else {
                k -= num;
                const XYIdxT zeros = X_wm[i].range_freq(0, H * W, 0, W);
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
        for(int y = 0; y <= H - diameter; ++y) {
            for(int x = 0; x <= W - diameter; ++x) {
                const ValT res = quantile2d(x, x + diameter, y, y + diameter, median_idx);
                dst[y * dst_step + x] = res;
            }
        }
    }
};

} // end namespace wavelet_matrix_median
