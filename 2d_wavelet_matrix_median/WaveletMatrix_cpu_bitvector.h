#pragma once

#include <vector>
#include <cstdint>
#include <x86intrin.h>


namespace wavelet_matrix_median {
using std::vector;

template <typename IdxType = uint32_t>
struct BitVector {
    static constexpr int WORD_SIZE = 32;
    using WordT = uint32_t;

    inline static int bitCount32(uint32_t bits) {
#if _MSC_VER >= 1920 || __INTEL_COMPILER
        return (int)_mm_popcnt_u32(bits);
#else
        return __builtin_popcount(bits);
#endif
    }

    // The number of 1's present from the least significant bit to the i-th bit of x.
    inline static uint32_t rank(const uint32_t x, const uint32_t i) {
        return bitCount32((uint32_t)_bzhi_u32(x, i));
    }
    struct chunk_t {
        WordT    nbit;
        IdxType  nsum;
    };
    vector<chunk_t> chunk;
    BitVector(const IdxType size = 0) : chunk(size / WORD_SIZE + 1) {}

    // When the i-th is set to 0, it is reversed and set to 1 internally.
    inline void nset(const IdxType i) {
        chunk[i / WORD_SIZE].nbit |= (WordT)1 << (i % WORD_SIZE);
    }
    void construct() {
        for (IdxType i = 0; i < chunk.size() - 1; ++i) {
            chunk[i + 1].nsum = chunk[i].nsum + bitCount32(chunk[i].nbit);
        }
    }
    inline IdxType rank0(const IdxType i) const {
        const auto e = chunk[i / WORD_SIZE];
        return e.nsum + rank(e.nbit, i % WORD_SIZE);
    }
    inline IdxType rank1(const IdxType i) const {
        return i - rank0(i);
    }
};

} // end namespace wavelet_median
